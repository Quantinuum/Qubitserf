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
// applies the logical detector (kcheck==0 => always logical, else logical iff the parity
// of (check_row AND codeword) summed over words is odd for some check row), popcounts the
// full codeword, tracks a per-thread minimum, and finally folds it into a single global
// atomicMin. There are NO divergent SIMD reductions (the Metal version was buggy with
// simd_min) -- just a relaxed per-thread atomicMin, which is correct.

#include "qminweight/backend.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace qminweight {

namespace {

constexpr int MAX_WORDS = 16;  // codeword <= 1024 bits on the GPU path (else CPU fallback)
constexpr int MAX_D     = 32;  // combination weight <= 32 on the GPU path (else CPU fallback)

// A weight level with fewer than this many candidate codewords (combinations * gammas)
// is cheaper on the multicore CPU than paying GPU launch latency for. Tunable via env.
static u64 gpu_min_work() {
    static u64 v = [] {
        const char* e = std::getenv("QMINWEIGHT_GPU_MIN_WORK");
        return e ? (u64)std::strtoull(e, nullptr, 10) : (u64)(1u << 20);
    }();
    return v;
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
// Mirrors the host-side lambda and the Metal `binom` helper exactly.
__device__ __forceinline__ unsigned long long
binom(const unsigned long long* B, unsigned int maxK, int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0ull;
    return B[(unsigned int)n * (maxK + 1u) + (unsigned int)k];
}

// One thread per contiguous chunk of combination indices. Mirrors bz_enumerate in MSL.
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

    // Guard the work (rather than returning early) so the trailing atomicMin always runs,
    // mirroring the fix in the Metal kernel (no early return before the atomic).
    if (start < total) {
        unsigned long long end = start + P.chunk;
        if (end > total) end = total;

        int K = (int)P.K, d = (int)P.d;
        unsigned int stride = P.stride;

        // --- combinatorial-number-system unrank of `start` into the first combination ---
        int pos[32];  // d <= MAX_D
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

        unsigned long long cw[16];  // codeword scratch, stride <= MAX_WORDS words
        for (unsigned long long it = start; it < end; ++it) {
            for (unsigned int g = 0; g < P.num_gamma; ++g) {
                const unsigned long long* G =
                    gamma + (unsigned long long)g * (unsigned long long)K * (unsigned long long)stride;
                for (unsigned int w = 0; w < stride; ++w) cw[w] = 0ull;
                for (int i = 0; i < d; ++i) {
                    const unsigned long long* row =
                        G + (unsigned long long)pos[i] * (unsigned long long)stride;
                    for (unsigned int w = 0; w < stride; ++w) cw[w] ^= row[w];
                }
                bool logical = (P.kcheck == 0u);
                for (unsigned int c = 0; c < P.kcheck && !logical; ++c) {
                    unsigned long long acc = 0ull;
                    const unsigned long long* cr =
                        chk + (unsigned long long)c * (unsigned long long)stride;
                    for (unsigned int w = 0; w < stride; ++w) acc ^= (cr[w] & cw[w]);
                    if (__popcll(acc) & 1) logical = true;
                }
                if (logical) {
                    unsigned int wt = 0;
                    for (unsigned int w = 0; w < stride; ++w) wt += (unsigned int)__popcll(cw[w]);
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

    CudaBackend() {
        int count = 0;
        cudaError_t e = cudaGetDeviceCount(&count);
        ok = (e == cudaSuccess && count > 0);
    }

    ~CudaBackend() override {
        if (g_buf) cudaFree(g_buf);
        if (c_buf) cudaFree(c_buf);
        if (b_buf) cudaFree(b_buf);
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
        if (total == 0) return WEIGHT_NONE;

        // Hybrid dispatch: small levels are dominated by GPU launch latency -> CPU.
        u64 work = total > (1ull << 40) ? total : total * (u64)plan.num_gamma;
        if (work < gpu_min_work()) return cpu_backend()->enumerate(plan);

        std::lock_guard<std::mutex> lock(mtx);

        // Work split: cap threads, give each a contiguous chunk (identical to Metal).
        const u64 MAX_THREADS = 1u << 20;
        u64 num_threads = total < MAX_THREADS ? total : MAX_THREADS;
        u64 chunk = (total + num_threads - 1) / num_threads;
        num_threads = (total + chunk - 1) / chunk;

        size_t g_bytes_ = (size_t)plan.num_gamma * plan.K * plan.stride * sizeof(u64);
        size_t c_bytes_ = (size_t)plan.kcheck * plan.stride * sizeof(u64);
        size_t b_bytes_ = (size_t)(plan.binom_maxN + 1) * (plan.binom_maxK + 1) * sizeof(u64);

        if (plan.buffers_key != buf_key) {   // new solve -> re-upload constant buffers
            upload(plan.gamma, g_bytes_, g_buf);
            upload(plan.check, c_bytes_, c_buf);
            upload(plan.binom, b_bytes_, b_buf);
            buf_key = plan.buffers_key;
        }
        unsigned long long* gbuf = g_buf, * cbuf = c_buf, * bbuf = b_buf;

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
        unsigned int* rbuf = nullptr;
        cudaMalloc((void**)&rbuf, sizeof(unsigned int));
        cudaMemcpy(rbuf, &init, sizeof(unsigned int), cudaMemcpyHostToDevice);

        // Launch ceil(num_threads/256) blocks of 256 threads; the in-kernel `start < total`
        // guard handles the trailing threads of the last block.
        const unsigned int threads = 256;
        unsigned int blocks = (unsigned int)((num_threads + threads - 1) / threads);
        bz_enumerate<<<blocks, threads>>>(gbuf, cbuf, bbuf, P, rbuf);

        uint32_t out = init;
        cudaMemcpy(&out, rbuf, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaFree(rbuf);

        return (int)out;
    }
};

} // namespace

Backend* cuda_backend() {
    static CudaBackend inst;
    return &inst;
}

} // namespace qminweight
