// =============================================================================
// CUDA GPU backend for the codeaut Brouwer-Zimmermann low-weight enumeration.
// =============================================================================
//
// WHAT IT COMPUTES
//   Given `m` bit-packed generator rows (each `stride` u64 words, n coordinates),
//   collect EVERY codeword that is the XOR of some size-(1..p) subset of the rows
//   and whose Hamming weight lies in [1, keep_weight].  Each qualifying codeword is
//   appended to `result.hits` as `stride` u64 words.  Duplicates are allowed: the
//   host (src/bz.cpp) deduplicates the collected hits afterwards.  `result.combos`
//   counts the subset-combinations evaluated; `result.overflow` reports a budget hit.
//
//   This is the GPU twin of the CPU ground truth in src/backend_cpu.cpp.  The GPU
//   output SET must equal the CPU output set for every plan (order is irrelevant;
//   the host sorts+dedups).  All per-combination math here is a literal port of the
//   CPU worker loop and of include/codeaut/combinatorics.hpp (comb_unrank/comb_next)
//   and include/codeaut/bits.hpp (popcount/xor), so it is "tested-by-construction".
//
// DESIGN (mirrors research/distance/qminweight/src/cuda/backend_cuda.cu)
//   * Per-(stride, subset-size) compile-time-templated kernels.  Templating on a
//     compile-time STRIDE (1..16) keeps the per-thread codeword scratch `uint64_t
//     cw[STRIDE]` in registers (the loops fully unroll); strides > 16 fall back to
//     the CPU.  A second template parameter POSN (8/16/32) sizes the per-thread
//     combination buffer `int comb[POSN]` so it, too, stays in registers; subset
//     sizes > 32 fall back to the CPU.
//   * Combinatorial-number-system (CNS) work decomposition: each GPU thread owns a
//     contiguous slice [start, end) of the C(m, sw) combination indices.  It unranks
//     `start` into its first ascending subset with a device `comb_unrank`, then steps
//     through the slice with a device `comb_next` -- identical to the CPU worker.
//   * EMIT-via-atomicAdd collection (the key difference from qminweight, which kept a
//     single minimum via atomicMin).  When a thread finds a codeword of weight in
//     [1, keep_weight] it does `slot = atomicAdd(out_count, 1)` and, if `slot <
//     CAPACITY`, writes the `stride` words to `out_buf + slot*stride`.  After the
//     launch the host reads `out_count`; if it EXCEEDS CAPACITY the device buffer
//     overflowed and we FALL BACK to the CPU backend for a correct, complete result
//     (never a truncated set).  CAPACITY is 1<<20 codewords (env-overridable via
//     CODEAUT_GPU_CAPACITY); at the max stride of 16 that is 16 Mi*8 B = 128 MiB.
//   * Host control flow mirrors the CPU backend exactly: loop sw = 1..min(p, m),
//     launch one kernel per sw, accumulate `combos`, and honour `budget` with the
//     same pre-check `budget > 0 && combos + C(m,sw) > budget => overflow, stop`.
//   * Device-buffer management: the BZEnumPlan carries NO per-solve key (unlike
//     qminweight's `buffers_key`) and the generator rows differ on every call, so the
//     rows/binomial buffers are re-uploaded each call -- but the device allocations
//     are persistent and grow-only (reused whenever already large enough), so steady
//     state does no cudaMalloc/cudaFree churn.  A one-time warmup pays the CUDA
//     context/JIT cost on the first enumerate.
//
// CPU-FALLBACK CONDITIONS (return cpu_backend()->enumerate(plan))
//   * no CUDA device (available() == false), or any CUDA runtime error mid-run;
//   * stride <= 0 or stride > 16 (codeword scratch too large to template);
//   * m <= 0 or p <= 0 (nothing to do; the CPU returns an empty result);
//   * effective max subset size min(p, m) > 32 (combination buffer too large);
//   * tiny work: sum over sw of C(m, sw) below CODEAUT_GPU_MIN_WORK (default 1<<18,
//     env-overridable) -- launch overhead would dominate;
//   * output-buffer overflow (out_count > CAPACITY).
//   In every fallback the CPU backend recomputes the WHOLE plan, so the returned set,
//   combos, and overflow flag are exactly correct.
//
// BUILD GUARD
//   The entire file is wrapped in `#ifdef CODEAUT_CUDA` ... `#endif`.  CMake compiles
//   it (and defines CODEAUT_CUDA) only when a CUDA toolkit is found; the CPU
//   translation unit defines a null `cuda_backend()` stub ONLY when CODEAUT_CUDA is
//   NOT defined, so there is never a duplicate-symbol clash.  Uses only the CUDA
//   runtime API (no thrust) and C++17.
//
// TESTED-BY-CONSTRUCTION DISCLAIMER
//   No NVIDIA hardware was available at authoring time, so this kernel was NOT run.
//   It was written to mirror the validated CPU backend's per-combination semantics
//   byte-for-byte (XOR of selected rows, weight in [1, keep_weight], CNS unrank/next).
//   Any change here must be kept in lockstep with src/backend_cpu.cpp, the reference.
// =============================================================================

#ifdef CODEAUT_CUDA

#include "backend.hpp"
#include "combinatorics.hpp"   // codeaut::BinomTable (host-side total/table build)

#include <cuda_runtime.h>

#include <vector>
#include <mutex>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace codeaut {

namespace {

// Codeword width cap for the GPU register path: codeword <= 16*64 = 1024 bits.
constexpr int MAX_WORDS = 16;
// Subset-size cap for the GPU register path (largest POSN bucket).
constexpr int MAX_SW = 32;

// Output capacity in codewords (each `stride` u64 words).  Overflow => CPU fallback.
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

// A work level (sum of C(m,sw)) below this is cheaper on the multicore CPU than the
// GPU launch latency.  Mirrors qminweight's QMINWEIGHT_GPU_MIN_WORK knob.
static unsigned long long gpu_min_work() {
    static unsigned long long v = [] {
        const char* e = std::getenv("CODEAUT_GPU_MIN_WORK");
        return e ? std::strtoull(e, nullptr, 10) : (unsigned long long)(1u << 18);
    }();
    return v;
}

// Smallest comb[] bucket that holds a subset of size `sw` (mirrors qminweight).
static int pos_bucket(int sw) {
    if (sw <= 8)  return 8;
    if (sw <= 16) return 16;
    return 32;
}

// Scalar launch parameters, passed to the kernel by value.  STRIDE/POSN are template
// parameters, not fields.  `total`/`chunk` are 64-bit (C(m,sw) can exceed 2^32).
struct Params {
    int m;                  // number of generator rows
    int sw;                 // subset size for this launch (1..min(p,m))
    int keep_weight;        // collect weights in [1, keep_weight] (signed, like CPU)
    int binom_maxK;         // columns-1 of the flat binomial table
    unsigned int capacity;  // out_buf capacity in codewords
    unsigned long long total;   // C(m, sw)
    unsigned long long chunk;   // combination indices per thread
};

// ---- device helpers: literal ports of the host code ------------------------------

// Flat binomial lookup mirroring BinomTable::get(): 0 outside [0..binom_maxK] x [.. n].
__device__ __forceinline__ unsigned long long
dev_binom(const uint64_t* B, int binom_maxK, int n, int k) {
    if (k < 0 || k > binom_maxK || k > n) return 0ull;
    return (unsigned long long)B[(unsigned long long)n * (unsigned long long)(binom_maxK + 1)
                                 + (unsigned long long)k];
}

// Port of combinatorics.hpp::comb_unrank -- unrank `idx` into ascending k-subset out[].
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

// Port of combinatorics.hpp::comb_next -- step to the next ascending k-subset.
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
    Params P,
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
        // so a nonsensical negative keep_weight collects nothing, just like the CPU)
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

// One-time warmup kernel: pays CUDA context/JIT cost on the first enumerate.
__global__ void bz_warmup_kernel() {}

// ---- compile-time dispatch over (stride, posn) -----------------------------------

template <int STRIDE>
static void launch_stride(int posn, unsigned int blocks, unsigned int threads,
                          const uint64_t* rows, const uint64_t* binomB,
                          const Params& P, uint64_t* out_buf, unsigned int* out_count) {
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
static bool launch(int stride, int posn, unsigned int blocks, unsigned int threads,
                   const uint64_t* rows, const uint64_t* binomB, const Params& P,
                   uint64_t* out_buf, unsigned int* out_count) {
    switch (stride) {
        case 1:  launch_stride<1>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 2:  launch_stride<2>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 3:  launch_stride<3>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 4:  launch_stride<4>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 5:  launch_stride<5>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 6:  launch_stride<6>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 7:  launch_stride<7>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 8:  launch_stride<8>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 9:  launch_stride<9>(posn, blocks, threads, rows, binomB, P, out_buf, out_count);  return true;
        case 10: launch_stride<10>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 11: launch_stride<11>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 12: launch_stride<12>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 13: launch_stride<13>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 14: launch_stride<14>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 15: launch_stride<15>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        case 16: launch_stride<16>(posn, blocks, threads, rows, binomB, P, out_buf, out_count); return true;
        default: return false;  // out of range -> caller falls back to CPU
    }
}

// ---- the backend -----------------------------------------------------------------

struct CudaBackend : Backend {
    bool ok_ = false;
    bool warmed_ = false;
    std::mutex mtx_;

    // Persistent, grow-only device buffers (re-uploaded each call; rows differ per call
    // and BZEnumPlan carries no per-solve key, so content caching is not possible).
    uint64_t*     rows_buf_  = nullptr;  size_t rows_cap_  = 0;  // in u64 words
    uint64_t*     binom_buf_ = nullptr;  size_t binom_cap_ = 0;  // in u64 words
    uint64_t*     out_buf_   = nullptr;  size_t out_cap_   = 0;  // in u64 words
    unsigned int* count_buf_ = nullptr;  // 1 word

    CudaBackend() {
        int count = 0;
        cudaError_t e = cudaGetDeviceCount(&count);
        ok_ = (e == cudaSuccess && count >= 1);
        cudaGetLastError();  // clear any sticky error from the probe (e.g. no device)
        if (ok_) {
            if (cudaMalloc((void**)&count_buf_, sizeof(unsigned int)) != cudaSuccess) {
                count_buf_ = nullptr;
                ok_ = false;
            }
        }
    }

    ~CudaBackend() override {
        if (rows_buf_)  cudaFree(rows_buf_);
        if (binom_buf_) cudaFree(binom_buf_);
        if (out_buf_)   cudaFree(out_buf_);
        if (count_buf_) cudaFree(count_buf_);
    }

    const char* name() const override { return "cuda"; }
    bool available() const override { return ok_; }

    // Ensure `*slot` holds at least `needed_words` u64 words; grow-only.  Returns false
    // on allocation failure (caller then falls back to the CPU).
    static bool ensure(uint64_t** slot, size_t* cap, size_t needed_words) {
        if (*slot && *cap >= needed_words) return true;
        if (*slot) { cudaFree(*slot); *slot = nullptr; *cap = 0; }
        size_t alloc = needed_words ? needed_words : 1;
        if (cudaMalloc((void**)slot, alloc * sizeof(uint64_t)) != cudaSuccess) {
            *slot = nullptr; *cap = 0;
            return false;
        }
        *cap = alloc;
        return true;
    }

    void warmup_once() {
        if (warmed_) return;
        warmed_ = true;
        cudaFree(0);                  // force context creation
        bz_warmup_kernel<<<1, 1>>>(); // force module load / JIT
        cudaDeviceSynchronize();
        cudaGetLastError();
    }

    BZEnumResult enumerate(const BZEnumPlan& plan) override {
        // ---- hard fallbacks (cheap, no device touched) ----
        if (!ok_ || plan.m <= 0 || plan.p <= 0 ||
            plan.stride <= 0 || plan.stride > MAX_WORDS)
            return cpu_backend()->enumerate(plan);

        const int pmax = std::min(plan.p, plan.m);
        if (pmax > MAX_SW)  // comb[] would exceed the largest POSN bucket
            return cpu_backend()->enumerate(plan);

        // Host binomial table built EXACTLY as the CPU backend does, so totals match.
        BinomTable B(plan.m, pmax);

        // ---- tiny-work heuristic: sum of C(m,sw) (saturating) vs CODEAUT_GPU_MIN_WORK ----
        unsigned long long work = 0ull;
        for (int sw = 1; sw <= pmax; ++sw) {
            int64_t t = B.get(plan.m, sw);
            if (t <= 0) continue;
            unsigned long long ut = (unsigned long long)t;
            if (work > (~0ull) - ut) { work = ~0ull; break; }  // saturate
            work += ut;
        }
        if (work < gpu_min_work())
            return cpu_backend()->enumerate(plan);

        std::lock_guard<std::mutex> lock(mtx_);
        warmup_once();

        const int stride = plan.stride;
        const int binom_maxK = pmax;
        const unsigned int capacity = gpu_capacity();

        // Build a flat uint64 binomial table for the device (values are non-negative,
        // saturated at 1<<62 by BinomTable, so the int64->uint64 copy is value-preserving).
        const size_t binom_words = (size_t)(plan.m + 1) * (size_t)(binom_maxK + 1);
        std::vector<uint64_t> binom_h(binom_words);
        for (size_t i = 0; i < binom_words; ++i)
            binom_h[i] = (uint64_t)B.c[i];

        const size_t rows_words = (size_t)plan.m * (size_t)stride;
        const size_t out_words  = (size_t)capacity * (size_t)stride;

        // Allocate / upload constant inputs; any CUDA failure => full CPU recompute.
        if (!ensure(&rows_buf_,  &rows_cap_,  rows_words)  ||
            !ensure(&binom_buf_, &binom_cap_, binom_words) ||
            !ensure(&out_buf_,   &out_cap_,   out_words))
            return cpu_backend()->enumerate(plan);

        if (cudaMemcpy(rows_buf_, plan.rows, rows_words * sizeof(uint64_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(binom_buf_, binom_h.data(), binom_words * sizeof(uint64_t),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return cpu_backend()->enumerate(plan);

        BZEnumResult res;

        // ---- loop subset sizes sw = 1..min(p,m), one kernel launch per sw ----
        for (int sw = 1; sw <= pmax; ++sw) {
            const int64_t total = B.get(plan.m, sw);
            if (total <= 0) continue;
            // budget pre-check, byte-for-byte the CPU backend's behaviour
            if (plan.budget > 0 && res.combos + total > plan.budget) {
                res.overflow = true;
                break;
            }

            if (cudaMemset(count_buf_, 0, sizeof(unsigned int)) != cudaSuccess)
                return cpu_backend()->enumerate(plan);

            // Work split: cap threads, give each a contiguous chunk (as in qminweight).
            const unsigned long long ut = (unsigned long long)total;
            const unsigned long long MAX_THREADS = 1ull << 20;
            unsigned long long num_threads = ut < MAX_THREADS ? ut : MAX_THREADS;
            unsigned long long chunk = (ut + num_threads - 1) / num_threads;
            num_threads = (ut + chunk - 1) / chunk;

            Params P;
            P.m = plan.m;
            P.sw = sw;
            P.keep_weight = plan.keep_weight;
            P.binom_maxK = binom_maxK;
            P.capacity = capacity;
            P.total = ut;
            P.chunk = chunk;

            const unsigned int threads = 256;
            const unsigned int blocks =
                (unsigned int)((num_threads + threads - 1) / threads);

            if (!launch(stride, pos_bucket(sw), blocks, threads,
                        rows_buf_, binom_buf_, P, out_buf_, count_buf_))
                return cpu_backend()->enumerate(plan);

            if (cudaDeviceSynchronize() != cudaSuccess ||
                cudaGetLastError()      != cudaSuccess)
                return cpu_backend()->enumerate(plan);

            unsigned int cnt = 0;
            if (cudaMemcpy(&cnt, count_buf_, sizeof(unsigned int),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                return cpu_backend()->enumerate(plan);

            // Output-buffer overflow: the emitted hits past CAPACITY were dropped, so we
            // cannot return a complete set -- recompute the WHOLE plan on the CPU.
            if (cnt > capacity)
                return cpu_backend()->enumerate(plan);

            if (cnt > 0) {
                const size_t base = res.hits.size();
                res.hits.resize(base + (size_t)cnt * (size_t)stride);
                if (cudaMemcpy(res.hits.data() + base, out_buf_,
                               (size_t)cnt * (size_t)stride * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost) != cudaSuccess)
                    return cpu_backend()->enumerate(plan);
            }

            res.combos += total;
        }

        return res;
    }
};

}  // namespace

Backend* cuda_backend() {
    static CudaBackend inst;
    return &inst;
}

}  // namespace codeaut

#endif  // CODEAUT_CUDA
