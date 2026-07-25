// CUDA GPU backend for the Brouwer-Zimmermann exponential enumeration.
//
// Compilation: this translation unit is compiled by CMake ONLY when a CUDA toolkit
// is found (nvcc present). On machines without CUDA it is never built, and the weak
// `cuda_backend()` stub in src/backend_cpu.cpp is used instead. The strong definition
// below overrides that weak stub when this file is compiled and linked in.
//
// Provenance / testing: this kernel is a 1:1 port of the WORKING, VALIDATED Metal
// backend (src/metal/backend_metal.mm). It is "GPU-tested-by-construction": NO NVIDIA
// hardware was available at authoring time, so it was written to mirror the Metal
// kernel's per-combination math exactly rather than being run. Any change here MUST be
// kept in lockstep with the Metal kernel, which is the reference.
//
// Algorithm (identical to Metal): one GPU thread owns a contiguous slice of combination
// indices. The combinatorial-number-system (CNS) unranking of the slice's start index
// gives the thread its first weight-d combination of the K message rows; `next_comb`
// steps through the slice. For each combination, and for every information-set generator
// (gamma), it XORs the d selected bit-packed rows (stride u64 words each) into a codeword,
// applies the logical detector, popcounts, tracks a per-thread minimum, and folds it into
// a single global atomicMin.
//
// PERFORMANCE (mirrors the Metal fix): the per-thread scratch arrays are the throughput
// bottleneck -- sized to the worst case they spill to local memory and collapse occupancy.
// So the kernel is TEMPLATED on a compile-time stride (the codeword loops unroll and
// `cw[STRIDE]` stays in registers) and on a small pos[] length bucket. The host dispatch
// instantiates the variant matching (plan.stride, pos_bucket(plan.d)).

#include "distfind/backend.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace distfind {

namespace {

constexpr int MAX_WORDS = 16;  // codeword <= 1024 bits on the GPU path (else CPU fallback)
constexpr int MAX_D     = 32;  // combination weight <= 32 on the GPU path (else CPU fallback)

// A weight level with fewer than this many candidate codewords (combinations * gammas)
// is cheaper on the multicore CPU than paying GPU launch latency for. Tunable via env.
// Lower than the historical 1<<20: the variant kernel makes the GPU pay off much sooner.
static u64 gpu_min_work() {
    static u64 v = [] {
        const char* e = std::getenv("DISTFIND_GPU_MIN_WORK");
        return e ? (u64)std::strtoull(e, nullptr, 10) : (u64)(1u << 18);
    }();
    return v;
}

// Smallest pos[] bucket that holds `d` (mirrors the Metal host helper).
static int pos_bucket(int d) {
    if (d <= 8)  return 8;
    if (d <= 16) return 16;
    return 32;
}

// Scalar launch parameters, passed to the kernel by value. `total` and `chunk` are 64-bit
// to match the host's u64 combination counts (C(K,d) can exceed 2^32).
struct Params {
    uint32_t n, stride, K, d, num_gamma, kcheck, binom_maxK;
    uint64_t total;
    uint64_t chunk;
    uint32_t current_best;
};

// Flat binomial table lookup: binom(n,k) = B[n*(maxK+1)+k], saturating to 0 out of range.
__device__ __forceinline__ unsigned long long
binom(const unsigned long long* B, unsigned int maxK, int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0ull;
    return B[(unsigned int)n * (maxK + 1u) + (unsigned int)k];
}

// One thread per contiguous chunk of combination indices. Templated on the compile-time
// stride (STRIDE) and pos[] length (POSN) so the codeword scratch stays in registers.
template<int STRIDE, int POSN>
__global__ void bz_enumerate(
    const unsigned long long* __restrict__ gamma,
    const unsigned long long* __restrict__ chk,
    const unsigned long long* __restrict__ binomB,
    Params P,
    unsigned int* __restrict__ result)
{
    unsigned long long total = P.total;
    unsigned long long start = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x)
                               * P.chunk;
    unsigned int best = P.current_best;

    // Guard the work (rather than returning early) so the trailing atomicMin always runs.
    if (start < total) {
        unsigned long long end = start + P.chunk;
        if (end > total) end = total;

        int K = (int)P.K, d = (int)P.d;

        // --- combinatorial-number-system unrank of `start` into the first combination ---
        int pos[POSN];
        {
            unsigned long long r = start; int x = 0;
            for (int i = 0; i < d; ++i) {
                for (;;) {
                    unsigned long long cnt = binom(binomB, P.binom_maxK, K - 1 - x, d - 1 - i);
                    if (r < cnt) { pos[i] = x; ++x; break; }
                    r -= cnt; ++x;
                }
            }
        }

        unsigned long long cw[STRIDE];
        for (unsigned long long it = start; it < end; ++it) {
            for (unsigned int g = 0; g < P.num_gamma; ++g) {
                const unsigned long long* G =
                    gamma + (unsigned long long)g * (unsigned long long)K * STRIDE;
                #pragma unroll
                for (int w = 0; w < STRIDE; ++w) cw[w] = 0ull;
                for (int i = 0; i < d; ++i) {
                    const unsigned long long* row = G + (unsigned long long)pos[i] * STRIDE;
                    #pragma unroll
                    for (int w = 0; w < STRIDE; ++w) cw[w] ^= row[w];
                }
                bool logical = (P.kcheck == 0u);
                for (unsigned int c = 0; c < P.kcheck && !logical; ++c) {
                    unsigned long long acc = 0ull;
                    const unsigned long long* cr = chk + (unsigned long long)c * STRIDE;
                    #pragma unroll
                    for (int w = 0; w < STRIDE; ++w) acc ^= (cr[w] & cw[w]);
                    if (__popcll(acc) & 1) logical = true;
                }
                if (logical) {
                    unsigned int wt = 0;
                    #pragma unroll
                    for (int w = 0; w < STRIDE; ++w) wt += (unsigned int)__popcll(cw[w]);
                    if (wt < best) best = wt;
                }
            }
            // advance to the next combination (lexicographic next_comb)
            int j = d - 1;
            while (j >= 0 && pos[j] == K - d + j) --j;
            if (j < 0) break;
            ++pos[j];
            for (int t = j + 1; t < d; ++t) pos[t] = pos[t - 1] + 1;
        }
    }

    // Per-thread fold into the single global result. Relaxed semantics are fine: atomicMin
    // is itself atomic and we only ever take a minimum (commutative/associative).
    atomicMin(result, best);
}

// ---- compile-time dispatch over (stride, posn) -----------------------------------
template<int STRIDE>
static void launch_stride(int posn, unsigned int blocks, unsigned int threads,
                          const unsigned long long* g, const unsigned long long* c,
                          const unsigned long long* b, const Params& P, unsigned int* r) {
    switch (posn) {
        case 8:  bz_enumerate<STRIDE, 8><<<blocks, threads>>>(g, c, b, P, r);  break;
        case 16: bz_enumerate<STRIDE, 16><<<blocks, threads>>>(g, c, b, P, r); break;
        default: bz_enumerate<STRIDE, 32><<<blocks, threads>>>(g, c, b, P, r); break;
    }
}

static bool launch(int stride, int posn, unsigned int blocks, unsigned int threads,
                   const unsigned long long* g, const unsigned long long* c,
                   const unsigned long long* b, const Params& P, unsigned int* r) {
    switch (stride) {
        case 1:  launch_stride<1>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 2:  launch_stride<2>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 3:  launch_stride<3>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 4:  launch_stride<4>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 5:  launch_stride<5>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 6:  launch_stride<6>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 7:  launch_stride<7>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 8:  launch_stride<8>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 9:  launch_stride<9>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 10: launch_stride<10>(posn, blocks, threads, g, c, b, P, r); return true;
        case 11: launch_stride<11>(posn, blocks, threads, g, c, b, P, r); return true;
        case 12: launch_stride<12>(posn, blocks, threads, g, c, b, P, r); return true;
        case 13: launch_stride<13>(posn, blocks, threads, g, c, b, P, r); return true;
        case 14: launch_stride<14>(posn, blocks, threads, g, c, b, P, r); return true;
        case 15: launch_stride<15>(posn, blocks, threads, g, c, b, P, r); return true;
        case 16: launch_stride<16>(posn, blocks, threads, g, c, b, P, r); return true;
        default: return false;  // out of range -> caller falls back to CPU
    }
}

struct CudaBackend : Backend {
    bool ok = false;
    std::mutex mtx;

    // Device-buffer cache for (gamma, check, binom), keyed on the plan's unique per-solve
    // token (NOT host pointer identity: host vectors are freed between solves and often
    // reallocated at the same address with different contents -- the stale-buffer bug).
    u64 buf_key = ~0ull;
    unsigned long long* g_buf = nullptr;
    unsigned long long* c_buf = nullptr;
    unsigned long long* b_buf = nullptr;
    unsigned int* r_buf = nullptr;   // persistent 1-word result buffer

    CudaBackend() {
        int count = 0;
        cudaError_t e = cudaGetDeviceCount(&count);
        ok = (e == cudaSuccess && count > 0);
        if (ok) cudaMalloc((void**)&r_buf, sizeof(unsigned int));
    }

    ~CudaBackend() override {
        if (g_buf) cudaFree(g_buf);
        if (c_buf) cudaFree(c_buf);
        if (b_buf) cudaFree(b_buf);
        if (r_buf) cudaFree(r_buf);
    }

    std::string name() const override { return "cuda"; }
    bool available() const override { return ok; }

    // (Re)upload `bytes` from host `ptr` into a device buffer. A zero-length input still
    // gets a 1-element allocation so the kernel always receives a valid pointer.
    void upload(const u64* ptr, size_t bytes, unsigned long long*& slot) {
        if (slot) { cudaFree(slot); slot = nullptr; }
        size_t alloc = bytes ? bytes : sizeof(unsigned long long);
        cudaMalloc((void**)&slot, alloc);
        if (bytes) cudaMemcpy(slot, ptr, bytes, cudaMemcpyHostToDevice);
    }

    int enumerate(const EnumPlan& plan) override {
        if (plan.stride > MAX_WORDS || plan.d > MAX_D || !ok)
            return cpu_backend()->enumerate(plan);

        // number of combinations C(K,d) from the flat binomial table
        auto binom_h = [&](int n, int k) -> u64 {
            if (k < 0 || n < 0 || k > n) return 0ull;
            return plan.binom[(size_t)n * (plan.binom_maxK + 1) + k];
        };
        u64 total = binom_h(plan.K, plan.d);
        if (total == 0 || plan.num_gamma == 0) return WEIGHT_NONE;

        // Hybrid dispatch: small levels are dominated by GPU launch latency -> CPU.
        u64 work = total > (1ull << 40) ? total : total * (u64)plan.num_gamma;
        if (work < gpu_min_work()) return cpu_backend()->enumerate(plan);

        std::lock_guard<std::mutex> lock(mtx);

        // Work split: cap threads, give each a contiguous chunk (identical to Metal).
        const u64 MAX_THREADS = 1u << 20;
        u64 num_threads = total < MAX_THREADS ? total : MAX_THREADS;
        u64 chunk = (total + num_threads - 1) / num_threads;
        num_threads = (total + chunk - 1) / chunk;

        // The cached upload must cover ALL generators: plan.num_gamma is only the active
        // prefix this level, and a deeper level may activate more.
        const int ng_total = plan.num_gamma_total > 0 ? plan.num_gamma_total : plan.num_gamma;
        size_t g_bytes_ = (size_t)ng_total * plan.K * plan.stride * sizeof(u64);
        size_t c_bytes_ = (size_t)plan.kcheck * plan.stride * sizeof(u64);
        size_t b_bytes_ = (size_t)(plan.binom_maxN + 1) * (plan.binom_maxK + 1) * sizeof(u64);

        if (plan.buffers_key != buf_key) {   // new solve -> re-upload constant buffers
            upload(plan.gamma, g_bytes_, g_buf);
            upload(plan.check, c_bytes_, c_buf);
            upload(plan.binom, b_bytes_, b_buf);
            buf_key = plan.buffers_key;
        }

        Params P;
        P.n = plan.n; P.stride = plan.stride; P.K = plan.K; P.d = plan.d;
        P.num_gamma = plan.num_gamma; P.kcheck = plan.kcheck;
        P.binom_maxK = plan.binom_maxK;
        P.total = total;
        P.chunk = chunk;
        P.current_best = (uint32_t)plan.current_best;

        // Result buffer initialised to the current best so a "no smaller logical found"
        // run returns >= plan.current_best.
        uint32_t init = (uint32_t)plan.current_best;
        cudaMemcpy(r_buf, &init, sizeof(unsigned int), cudaMemcpyHostToDevice);

        // Launch ceil(num_threads/256) blocks of 256 threads; the in-kernel `start < total`
        // guard handles the trailing threads of the last block.
        const unsigned int threads = 256;
        unsigned int blocks = (unsigned int)((num_threads + threads - 1) / threads);
        if (!launch(plan.stride, pos_bucket(plan.d), blocks, threads,
                    g_buf, c_buf, b_buf, P, r_buf))
            return cpu_backend()->enumerate(plan);

        uint32_t out = init;
        cudaMemcpy(&out, r_buf, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        return (int)out;
    }
};

} // namespace

Backend* cuda_backend() {
    static CudaBackend inst;
    return &inst;
}

} // namespace distfind
