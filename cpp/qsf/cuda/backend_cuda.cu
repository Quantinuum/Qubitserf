// CUDA GPU backend for the shared Brouwer-Zimmermann exponential enumeration -- BOTH
// modes of qsf::Backend, merged from the two pre-unification per-engine CUDA files:
//
//   * enumerate (min-weight, the distfind engine): the templated <STRIDE, POSN>
//     bz_enumerate kernel -- per-thread minimum of the full weight of a logical
//     codeword, folded into a single global atomicMin.  A 1:1 port of the old
//     src/distfind/cuda/backend_cuda.cu.
//   * collect (low-weight collection, the codeaut engine): the templated <STRIDE, POSN>
//     bz_collect_kernel -- every codeword of weight in [1, keep] is emitted through an
//     atomicAdd slot counter; on capacity overflow the level falls back to the CPU
//     backend (never a truncated set).  A 1:1 port of the kernel in the old
//     src/codeaut/cuda/backend_cuda.cu.  The subset-size (sw) loop that used to live
//     here has MOVED to the caller
//     (src/codeaut/bz.cpp): collect() now handles a SINGLE level, with plan.K = m rows
//     (row-major at plan.gamma, num_gamma == 1), plan.d = the subset size, and the
//     caller's flat saturating u64 binomial table at plan.binom.
//
// Compilation: this translation unit is compiled by CMake ONLY when a CUDA toolkit is
// found (nvcc present); CMake then defines QSF_CUDA target-wide, which suppresses the
// null `cuda_backend()` stub in src/core/backend_cpu.cpp.  The strong definition below
// takes its place -- so this file must NOT be wrapped in #ifdef itself.
//
// Provenance / testing: both kernels are literal ports of the WORKING, VALIDATED
// pre-merge CUDA files (which were themselves ports of the validated Metal / CPU
// reference implementations).  They are "GPU-tested-by-construction": NO NVIDIA
// hardware was available at authoring time, so this file was written to preserve the
// old kernels' per-combination math exactly rather than being run.  The CPU backend
// (src/core/backend_cpu.cpp + qsf/enum_core.hpp) is the oracle: for every plan the
// enumerate() result and the collect() output SET (order irrelevant; the codeaut host
// sorts+dedups) must equal the CPU's.  Any change here MUST be kept in lockstep with
// the CPU backend and the Metal twin (src/core/metal/backend_metal.mm).
//
// Algorithm (both modes): one GPU thread owns a contiguous slice of combination
// indices.  The combinatorial-number-system (CNS) unranking of the slice's start index
// gives the thread its first weight-d combination of the K message rows; a lexicographic
// next-combination step walks the slice.  For each combination the thread XORs the d
// selected bit-packed rows (stride u64 words each) into a codeword and then runs the
// mode's epilogue (logical test + weight minimum, or weight-window emit).
//
// PERFORMANCE: the per-thread scratch arrays are the throughput bottleneck -- sized to
// the worst case they spill to local memory and collapse occupancy.  So both kernels
// are TEMPLATED on a compile-time stride (the codeword loops unroll and `cw[STRIDE]`
// stays in registers, strides 1..16) and on a small pos[]/comb[] length bucket
// (POSN in {8, 16, 32}).  The host dispatch instantiates the variant matching
// (plan.stride, pos_bucket(plan.d)); anything out of range falls back to the CPU.
//
// DEVICE-BUFFER CACHE (decision): ONE shared (gamma, check, binom) cache serves BOTH
// modes, keyed on plan.buffers_key.  This is correct because -- unlike the Metal
// backend, which transposes gamma -- both CUDA kernels read the generator rows
// ROW-major (min mode: G + pos[i]*STRIDE; collect mode: rows + comb[t]*STRIDE), so the
// cached upload has one layout; the check buffer is simply empty in collect mode
// (kcheck == 0).  buffers_key comes from qsf::next_buffers_key(), which is process-wide
// unique ACROSS engines, so a min-mode solve and a collect-mode solve can never alias
// each other's cache entry.  We must never key on host pointer identity: the host
// vectors are freed between solves and often reallocated at the same address with
// different contents (the stale-buffer bug).

#include "qsf/backend.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <string>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace qsf {

namespace {

constexpr int MAX_WORDS = 16;  // codeword <= 1024 bits on the GPU path (else CPU fallback)
constexpr int MAX_D     = 32;  // combination weight <= 32 on the GPU path (else CPU fallback)

// A weight level with fewer than this many candidate codewords (combinations * gammas)
// is cheaper on the multicore CPU than paying GPU launch latency for.  Tunable per mode
// via the engines' historical env knobs.
static u64 min_work_env(const char* name, u64 dflt) {
    const char* e = std::getenv(name);
    return e ? (u64)std::strtoull(e, nullptr, 10) : dflt;
}
static u64 gpu_min_work_enumerate() {
    static u64 v = min_work_env("DISTFIND_GPU_MIN_WORK", (u64)(1u << 18));
    return v;
}
static u64 gpu_min_work_collect() {
    static u64 v = min_work_env("CODEAUT_GPU_MIN_WORK", (u64)(1u << 18));
    return v;
}

// Collect-mode output capacity in codewords (each `stride` u64 words); overflow (rare:
// low-weight enumeration emits few hits) falls the level back to the CPU.  At the max
// stride of 16 the default 1<<20 codewords is 16 Mi * 8 B = 128 MiB.  Env-overridable.
static unsigned int gpu_capacity() {
    static unsigned int v = [] {
        const char* e = std::getenv("CODEAUT_GPU_CAPACITY");
        unsigned long long c = e ? std::strtoull(e, nullptr, 10) : (1ull << 20);
        if (c == 0ull) c = 1ull << 20;
        if (c > 0x7fffffffull) c = 0x7fffffffull;  // keep within unsigned-int slot space
        return (unsigned int)c;
    }();
    return v;
}

// Smallest pos[]/comb[] bucket that holds `d` (mirrors the Metal host helper).
static int pos_bucket(int d) {
    if (d <= 8)  return 8;
    if (d <= 16) return 16;
    return 32;
}

// ===================================================================================
// Min-weight mode (enumerate) -- literal port of the old distfind CUDA backend.
// ===================================================================================

// Scalar launch parameters, passed to the kernel by value.  `total` and `chunk` are
// 64-bit to match the host's u64 combination counts (C(K,d) can exceed 2^32).
struct MinParams {
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

// One thread per contiguous chunk of combination indices.  Templated on the compile-time
// stride (STRIDE) and pos[] length (POSN) so the codeword scratch stays in registers.
template<int STRIDE, int POSN>
__global__ void bz_enumerate(
    const unsigned long long* __restrict__ gamma,
    const unsigned long long* __restrict__ chk,
    const unsigned long long* __restrict__ binomB,
    MinParams P,
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

    // Per-thread fold into the single global result.  Relaxed semantics are fine: atomicMin
    // is itself atomic and we only ever take a minimum (commutative/associative).
    atomicMin(result, best);
}

// ---- compile-time dispatch over (stride, posn), min mode --------------------------
template<int STRIDE>
static void launch_stride_min(int posn, unsigned int blocks, unsigned int threads,
                              const unsigned long long* g, const unsigned long long* c,
                              const unsigned long long* b, const MinParams& P,
                              unsigned int* r) {
    switch (posn) {
        case 8:  bz_enumerate<STRIDE, 8><<<blocks, threads>>>(g, c, b, P, r);  break;
        case 16: bz_enumerate<STRIDE, 16><<<blocks, threads>>>(g, c, b, P, r); break;
        default: bz_enumerate<STRIDE, 32><<<blocks, threads>>>(g, c, b, P, r); break;
    }
}

static bool launch_min(int stride, int posn, unsigned int blocks, unsigned int threads,
                       const unsigned long long* g, const unsigned long long* c,
                       const unsigned long long* b, const MinParams& P, unsigned int* r) {
    switch (stride) {
        case 1:  launch_stride_min<1>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 2:  launch_stride_min<2>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 3:  launch_stride_min<3>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 4:  launch_stride_min<4>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 5:  launch_stride_min<5>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 6:  launch_stride_min<6>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 7:  launch_stride_min<7>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 8:  launch_stride_min<8>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 9:  launch_stride_min<9>(posn, blocks, threads, g, c, b, P, r);  return true;
        case 10: launch_stride_min<10>(posn, blocks, threads, g, c, b, P, r); return true;
        case 11: launch_stride_min<11>(posn, blocks, threads, g, c, b, P, r); return true;
        case 12: launch_stride_min<12>(posn, blocks, threads, g, c, b, P, r); return true;
        case 13: launch_stride_min<13>(posn, blocks, threads, g, c, b, P, r); return true;
        case 14: launch_stride_min<14>(posn, blocks, threads, g, c, b, P, r); return true;
        case 15: launch_stride_min<15>(posn, blocks, threads, g, c, b, P, r); return true;
        case 16: launch_stride_min<16>(posn, blocks, threads, g, c, b, P, r); return true;
        default: return false;  // out of range -> caller falls back to CPU
    }
}

// ===================================================================================
// Collect mode -- literal port of the old codeaut CUDA kernel (single level; the sw
// loop now lives in src/codeaut/bz.cpp).
// ===================================================================================

// Scalar launch parameters, passed to the kernel by value.  STRIDE/POSN are template
// parameters, not fields.  `total`/`chunk` are 64-bit (C(m,sw) can exceed 2^32).
struct CollectParams {
    int m;                  // number of generator rows (= plan.K)
    int sw;                 // subset size for this launch (= plan.d)
    int keep_weight;        // collect weights in [1, keep_weight] (signed, like CPU)
    int binom_maxK;         // columns-1 of the flat binomial table
    unsigned int capacity;  // out_buf capacity in codewords
    unsigned long long total;   // C(m, sw)
    unsigned long long chunk;   // combination indices per thread
};

// ---- device helpers: literal ports of the host code ------------------------------

// Flat binomial lookup: 0 outside [0..binom_maxK] x [.. n] (the k > binom_maxK guard
// matters -- the flat table only has binom_maxK+1 columns).
__device__ __forceinline__ unsigned long long
dev_binom(const uint64_t* B, int binom_maxK, int n, int k) {
    if (k < 0 || k > binom_maxK || k > n) return 0ull;
    return (unsigned long long)B[(unsigned long long)n * (unsigned long long)(binom_maxK + 1)
                                 + (unsigned long long)k];
}

// Port of comb_unrank -- unrank `idx` into the ascending k-subset out[].
__device__ __forceinline__ void
dev_comb_unrank(unsigned long long idx, int m, int k,
                const uint64_t* B, int binom_maxK, int* out) {
    int x = 0;
    for (int i = 0; i < k; ++i) {
        for (;;) {
            unsigned long long cnt = dev_binom(B, binom_maxK, m - 1 - x, k - 1 - i);
            if (idx < cnt) break;
            idx -= cnt;
            ++x;
        }
        out[i] = x;
        ++x;
    }
}

// Port of comb_next -- step to the next ascending k-subset.
__device__ __forceinline__ bool
dev_comb_next(int* comb, int m, int k) {
    int i = k - 1;
    while (i >= 0 && comb[i] == m - k + i) --i;
    if (i < 0) return false;
    ++comb[i];
    for (int j = i + 1; j < k; ++j) comb[j] = comb[j - 1] + 1;
    return true;
}

// One thread per contiguous slice of combination indices.  Templated on the
// compile-time codeword stride (STRIDE) and combination-buffer length (POSN) so the
// per-thread scratch stays in registers.
template <int STRIDE, int POSN>
__global__ void bz_collect_kernel(
    const uint64_t* __restrict__ rows,     // m * STRIDE, row-major
    const uint64_t* __restrict__ binomB,   // flat binomial table
    CollectParams P,
    uint64_t* __restrict__ out_buf,        // capacity * STRIDE
    unsigned int* __restrict__ out_count)
{
    unsigned long long total = P.total;
    unsigned long long start =
        (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x) * P.chunk;
    if (start >= total) return;
    unsigned long long end = start + P.chunk;
    if (end > total) end = total;

    const int m  = P.m;
    const int sw = P.sw;

    int comb[POSN];
    dev_comb_unrank(start, m, sw, binomB, P.binom_maxK, comb);

    uint64_t cw[STRIDE];
    for (unsigned long long it = start; it < end; ++it) {
        // cw = XOR of the sw selected rows
        #pragma unroll
        for (int w = 0; w < STRIDE; ++w) cw[w] = 0ull;
        for (int t = 0; t < sw; ++t) {
            const uint64_t* row = rows + (unsigned long long)comb[t] * STRIDE;
            #pragma unroll
            for (int w = 0; w < STRIDE; ++w) cw[w] ^= row[w];
        }

        // Hamming weight
        unsigned int wt = 0;
        #pragma unroll
        for (int w = 0; w < STRIDE; ++w)
            wt += (unsigned int)__popcll((unsigned long long)cw[w]);

        // collect when 0 < wt <= keep_weight (signed compare mirrors the CPU exactly,
        // so a nonpositive keep_weight collects nothing, just like the CPU)
        if (wt > 0u && (int)wt <= P.keep_weight) {
            unsigned int slot = atomicAdd(out_count, 1u);
            if (slot < P.capacity) {
                uint64_t* dst = out_buf + (unsigned long long)slot * STRIDE;
                #pragma unroll
                for (int w = 0; w < STRIDE; ++w) dst[w] = cw[w];
            }
        }

        // advance to the next combination; stop at the slice end or the last subset
        if (!dev_comb_next(comb, m, sw)) break;
    }
}

// One-time warmup kernel: pays CUDA context/JIT cost on the first real launch.
__global__ void bz_warmup_kernel() {}

// ---- compile-time dispatch over (stride, posn), collect mode ----------------------

template <int STRIDE>
static void launch_stride_collect(int posn, unsigned int blocks, unsigned int threads,
                                  const uint64_t* rows, const uint64_t* binomB,
                                  const CollectParams& P, uint64_t* out_buf,
                                  unsigned int* out_count) {
    switch (posn) {
        case 8:  bz_collect_kernel<STRIDE, 8>
                     <<<blocks, threads>>>(rows, binomB, P, out_buf, out_count); break;
        case 16: bz_collect_kernel<STRIDE, 16>
                     <<<blocks, threads>>>(rows, binomB, P, out_buf, out_count); break;
        default: bz_collect_kernel<STRIDE, 32>
                     <<<blocks, threads>>>(rows, binomB, P, out_buf, out_count); break;
    }
}

// Returns false (=> caller falls back to CPU) for an unsupported stride.
static bool launch_collect(int stride, int posn, unsigned int blocks, unsigned int threads,
                           const uint64_t* rows, const uint64_t* binomB,
                           const CollectParams& P, uint64_t* out_buf,
                           unsigned int* out_count) {
    switch (stride) {
        case 1:  launch_stride_collect<1>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 2:  launch_stride_collect<2>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 3:  launch_stride_collect<3>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 4:  launch_stride_collect<4>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 5:  launch_stride_collect<5>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 6:  launch_stride_collect<6>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 7:  launch_stride_collect<7>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 8:  launch_stride_collect<8>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 9:  launch_stride_collect<9>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 10: launch_stride_collect<10>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 11: launch_stride_collect<11>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 12: launch_stride_collect<12>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 13: launch_stride_collect<13>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 14: launch_stride_collect<14>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 15: launch_stride_collect<15>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 16: launch_stride_collect<16>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        default: return false;  // out of range -> caller falls back to CPU
    }
}

// ===================================================================================
// The backend.
// ===================================================================================

struct CudaBackend : Backend {
    bool ok = false;
    bool warmed = false;
    std::mutex mtx;

    // Shared device-buffer cache for (gamma, check, binom), keyed on the plan's unique
    // per-solve token (see the header comment: one row-major cache serves BOTH modes;
    // NEVER key on host pointer identity -- the stale-buffer bug).
    u64 buf_key = ~0ull;
    unsigned long long* g_buf = nullptr;
    unsigned long long* c_buf = nullptr;
    unsigned long long* b_buf = nullptr;
    unsigned int* r_buf = nullptr;       // persistent 1-word min-mode result buffer
    unsigned int* count_buf = nullptr;   // persistent 1-word collect-mode hit counter
    uint64_t* out_buf = nullptr;         // collect-mode output buffer (grow-only)
    size_t out_cap = 0;                  // in u64 words

    CudaBackend() {
        int count = 0;
        cudaError_t e = cudaGetDeviceCount(&count);
        ok = (e == cudaSuccess && count > 0);
        cudaGetLastError();  // clear any sticky error from the probe (e.g. no device)
        if (ok && cudaMalloc((void**)&r_buf, sizeof(unsigned int)) != cudaSuccess) {
            r_buf = nullptr; ok = false;
        }
        if (ok && cudaMalloc((void**)&count_buf, sizeof(unsigned int)) != cudaSuccess) {
            count_buf = nullptr; ok = false;
        }
    }

    ~CudaBackend() override {
        if (g_buf)     cudaFree(g_buf);
        if (c_buf)     cudaFree(c_buf);
        if (b_buf)     cudaFree(b_buf);
        if (r_buf)     cudaFree(r_buf);
        if (count_buf) cudaFree(count_buf);
        if (out_buf)   cudaFree(out_buf);
    }

    std::string name() const override { return "cuda"; }
    bool available() const override { return ok; }

    // (Re)upload `bytes` from host `ptr` into a device buffer.  A zero-length input
    // still gets a 1-element allocation so the kernels always receive a valid pointer.
    // Returns false on any CUDA failure (caller then falls back to the CPU).
    static bool upload(const u64* ptr, size_t bytes, unsigned long long*& slot) {
        if (slot) { cudaFree(slot); slot = nullptr; }
        size_t alloc = bytes ? bytes : sizeof(unsigned long long);
        if (cudaMalloc((void**)&slot, alloc) != cudaSuccess) { slot = nullptr; return false; }
        if (bytes && cudaMemcpy(slot, ptr, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
        return true;
    }

    // (Re)upload the per-solve constant buffers when the solve token changes.  Must be
    // called with mtx held.  On failure the cache key is left invalid so a later call
    // retries the upload instead of trusting garbage.
    bool ensure_solve_buffers(const EnumPlan& plan) {
        if (plan.buffers_key == buf_key) return true;
        buf_key = ~0ull;
        // The cached upload must cover ALL generators: plan.num_gamma is only the active
        // prefix this level, and a deeper level may activate more.
        const int ng_total = plan.num_gamma_total > 0 ? plan.num_gamma_total : plan.num_gamma;
        size_t g_bytes = (size_t)ng_total * plan.K * plan.stride * sizeof(u64);
        size_t c_bytes = (size_t)plan.kcheck * plan.stride * sizeof(u64);
        size_t b_bytes = (size_t)(plan.binom_maxN + 1) * (plan.binom_maxK + 1) * sizeof(u64);
        if (!upload(plan.gamma, g_bytes, g_buf) ||
            !upload(plan.check, c_bytes, c_buf) ||
            !upload(plan.binom, b_bytes, b_buf))
            return false;
        buf_key = plan.buffers_key;
        return true;
    }

    // Ensure the collect output buffer holds at least `needed_words` u64 words;
    // grow-only.  Returns false on allocation failure (caller falls back to the CPU).
    bool ensure_out(size_t needed_words) {
        if (out_buf && out_cap >= needed_words) return true;
        if (out_buf) { cudaFree(out_buf); out_buf = nullptr; out_cap = 0; }
        size_t alloc = needed_words ? needed_words : 1;
        if (cudaMalloc((void**)&out_buf, alloc * sizeof(uint64_t)) != cudaSuccess) {
            out_buf = nullptr; out_cap = 0;
            return false;
        }
        out_cap = alloc;
        return true;
    }

    void warmup_once() {
        if (warmed) return;
        warmed = true;
        cudaFree(0);                  // force context creation
        bz_warmup_kernel<<<1, 1>>>(); // force module load / JIT
        cudaDeviceSynchronize();
        cudaGetLastError();
    }

    // ---- min-weight mode (distfind) -----------------------------------------------

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
        if (work < gpu_min_work_enumerate()) return cpu_backend()->enumerate(plan);

        std::lock_guard<std::mutex> lock(mtx);

        // Work split: cap threads, give each a contiguous chunk (identical to Metal).
        const u64 MAX_THREADS = 1u << 20;
        u64 num_threads = total < MAX_THREADS ? total : MAX_THREADS;
        u64 chunk = (total + num_threads - 1) / num_threads;
        num_threads = (total + chunk - 1) / chunk;

        if (!ensure_solve_buffers(plan))
            return cpu_backend()->enumerate(plan);

        MinParams P;
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
        if (!launch_min(plan.stride, pos_bucket(plan.d), blocks, threads,
                        g_buf, c_buf, b_buf, P, r_buf))
            return cpu_backend()->enumerate(plan);

        uint32_t out = init;
        cudaMemcpy(&out, r_buf, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        return (int)out;
    }

    // ---- collect mode (codeaut), one level per call -------------------------------

    CollectResult collect(const EnumPlan& plan, int keep_weight) override {
        // Nonpositive keep collects nothing (the emit condition is 0 < wt <= keep with
        // signed semantics): return empty without touching the device, matching the CPU.
        if (keep_weight <= 0) return CollectResult{};

        // Hard fallbacks (cheap, no device touched).  num_gamma != 1 never happens with
        // the codeaut caller but would need a gamma loop the collect kernel lacks.
        if (!ok || plan.stride <= 0 || plan.stride > MAX_WORDS ||
            plan.d <= 0 || plan.d > MAX_D || plan.num_gamma != 1)
            return cpu_backend()->collect(plan, keep_weight);

        // Work for this level: C(K,d) from the host-side flat table (saturating u64,
        // guarded against the table's column count).
        auto binom_h = [&](int n, int k) -> u64 {
            if (k < 0 || n < 0 || k > plan.binom_maxK || k > n) return 0ull;
            return plan.binom[(size_t)n * (plan.binom_maxK + 1) + k];
        };
        u64 total = binom_h(plan.K, plan.d);
        if (total == 0) return CollectResult{};   // matches the CPU: nothing to enumerate

        // Tiny-work heuristic: launch overhead would dominate -> CPU.
        if (total < gpu_min_work_collect())
            return cpu_backend()->collect(plan, keep_weight);

        std::lock_guard<std::mutex> lock(mtx);
        warmup_once();

        const unsigned int capacity = gpu_capacity();

        // Allocate / upload; any CUDA failure => full CPU recompute of the level.
        if (!ensure_out((size_t)capacity * plan.stride) ||
            !ensure_solve_buffers(plan) ||
            cudaMemset(count_buf, 0, sizeof(unsigned int)) != cudaSuccess)
            return cpu_backend()->collect(plan, keep_weight);

        // Work split: cap threads, give each a contiguous chunk (identical to min mode).
        const u64 MAX_THREADS = 1ull << 20;
        u64 num_threads = total < MAX_THREADS ? total : MAX_THREADS;
        u64 chunk = (total + num_threads - 1) / num_threads;
        num_threads = (total + chunk - 1) / chunk;

        CollectParams P;
        P.m = plan.K;
        P.sw = plan.d;
        P.keep_weight = keep_weight;
        P.binom_maxK = plan.binom_maxK;
        P.capacity = capacity;
        P.total = total;
        P.chunk = chunk;

        const unsigned int threads = 256;
        const unsigned int blocks = (unsigned int)((num_threads + threads - 1) / threads);

        // The shared cache stores row-major u64 rows; reinterpret to the kernel's
        // spelled pointer type (same 64-bit object representation).
        if (!launch_collect(plan.stride, pos_bucket(plan.d), blocks, threads,
                            reinterpret_cast<const uint64_t*>(g_buf),
                            reinterpret_cast<const uint64_t*>(b_buf),
                            P, out_buf, count_buf))
            return cpu_backend()->collect(plan, keep_weight);

        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaGetLastError()      != cudaSuccess)
            return cpu_backend()->collect(plan, keep_weight);

        unsigned int cnt = 0;
        if (cudaMemcpy(&cnt, count_buf, sizeof(unsigned int),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return cpu_backend()->collect(plan, keep_weight);

        // Output-buffer overflow: the emitted hits past capacity were dropped, so we
        // cannot return a complete set -- recompute the level on the CPU oracle.
        if (cnt > capacity)
            return cpu_backend()->collect(plan, keep_weight);

        CollectResult res;
        if (cnt > 0) {
            res.hits.resize((size_t)cnt * plan.stride);
            if (cudaMemcpy(res.hits.data(), out_buf,
                           (size_t)cnt * plan.stride * sizeof(u64),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                return cpu_backend()->collect(plan, keep_weight);
        }
        return res;
    }
};

} // namespace

Backend* cuda_backend() {
    static CudaBackend inst;
    return &inst;
}

} // namespace qsf
