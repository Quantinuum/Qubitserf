// Metal (Apple GPU) backend for the codeaut Brouwer-Zimmermann low-weight codeword
// enumeration.  It implements the same `codeaut::Backend` contract as the CPU backend
// (include/codeaut/backend.hpp): given `m` bit-packed generator rows (each `stride`
// little-endian u64 words, row-major `rows[j*stride + w]`), it COLLECTS every codeword
// that is the XOR of some size-(1..p) subset of the rows with Hamming weight in
// [1, keep_weight], appending each as `stride` u64 words to `result.hits`.  Duplicates are
// fine -- the host (src/bz.cpp) deduplicates.  `result.combos` and `result.overflow`
// (budget) are produced byte-for-byte identically to src/backend_cpu.cpp.
//
// DESIGN (mirrors research/distance/qminweight/src/metal/backend_metal.mm):
//   * The Metal Shading Language kernel is generated at runtime as a string by
//     make_kernel_src(stride, posn): `stride` is baked in as a compile-time literal STRIDE
//     (the per-codeword XOR/popcount loops fully unroll and `cw[STRIDE]` stays in
//     registers) and `posn` bounds the small, dynamically-indexed combination array
//     `pos[POSN]`.  Kernels are compiled lazily via newLibraryWithSource: and cached by
//     the key (stride << 8) | posn.  No offline `metal` toolchain is assumed.
//   * MTLDevice / MTLCommandQueue / shared-storage buffers, a power-of-two threadgroup
//     size, and a one-time warmup dispatch (to absorb first-dispatch driver latency).
//   * ARC (-fobjc-arc, set by CMake); @autoreleasepool around the Obj-C work.
//
// THE KEY DIFFERENCE FROM qminweight: qminweight finds the single MINIMUM-weight logical
// codeword via an atomic-min + threadgroup reduction.  THIS backend instead EMITS every
// qualifying codeword.  When a thread's XOR has weight in [1, keep_weight] it does
//     uint slot = atomic_fetch_add_explicit(out_count, 1, memory_order_relaxed);
//     if (slot < capacity) { write the STRIDE ulong words of the codeword to out_buf; }
// After the dispatch the host reads out_count; if out_count > capacity the on-GPU result
// is TRUNCATED, so we discard it and FALL BACK to cpu_backend()->enumerate(plan) for a
// complete result (we never return a truncated set).
//
// HOST DRIVER: exactly like backend_cpu.cpp, we loop subset sizes sw = 1..min(p, m),
// dispatch one kernel per sw, accumulate `combos`, and respect `budget` identically: if
// budget > 0 and combos + C(m, sw) > budget we set overflow and stop *before* that sw.
// Each MSL thread owns a contiguous chunk of combination indices: it unranks its start
// index through the combinatorial number system (using an uploaded saturating binomial
// table, identical to codeaut::BinomTable / comb_unrank), then steps with a device
// comb_next, XORs the sw selected rows, popcounts, and emits if the weight is in range.
//
// CPU FALLBACK CONDITIONS (each returns cpu_backend()->enumerate(plan), the oracle):
//   * MTLCreateSystemDefaultDevice() returned nil (no Metal device) / setup failed;
//   * stride exceeds the supported maximum (MAX_WORDS) or stride <= 0;
//   * the largest subset size min(p, m) exceeds the supported maximum (MAX_D);
//   * m <= 0 || p <= 0 (CPU returns an empty result for this; we delegate to it);
//   * pipeline compilation fails for a needed (stride, posn) variant;
//   * the total work is tiny (CODEAUT_GPU_MIN_WORK, default 1<<18, env-overridable) --
//     GPU dispatch latency would dominate;
//   * the output buffer overflowed for any sw (out_count > capacity).
//
// TESTED-BY-CONSTRUCTION: this file was written on a Linux host with NO Apple toolchain,
// so it could not be compiled or run here.  Its output semantics are matched line-for-line
// against src/backend_cpu.cpp (the validated oracle): same sw loop, same combinatorial
// unrank/step, same XOR + popcount + (0 < wt <= keep_weight) emit test, same combos and
// budget/overflow accounting; any uncertainty (no device, unsupported sizes, compile
// failure, tiny work, output overflow) degrades to the CPU backend rather than risk a
// wrong or truncated answer.

#ifdef CODEAUT_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "backend.hpp"
#include "combinatorics.hpp"

#include <vector>
#include <map>
#include <mutex>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

namespace codeaut {

namespace {

// Codewords of up to MAX_WORDS*64 bits go on the GPU path; wider rows fall back to CPU.
constexpr int MAX_WORDS = 16;   // <= 1024-bit codewords on the GPU path
// Subset sizes up to MAX_D go on the GPU path (bounds the per-thread pos[] scratch).
constexpr int MAX_D     = 32;
// Output capacity in codewords; out_buf holds capacity*stride u64 words (shared storage).
// Low-weight enumeration emits few hits in practice, so overflow (-> CPU fallback) is rare.
constexpr uint32_t CAPACITY = 1u << 20;

// Subset sizes whose total combination work is below this are cheaper on the multicore
// CPU than paying GPU dispatch latency.  Env-overridable; default 1<<18.
static uint64_t gpu_min_work() {
    static uint64_t v = [] {
        const char* e = std::getenv("CODEAUT_GPU_MIN_WORK");
        return e ? (uint64_t)std::strtoull(e, nullptr, 10) : (uint64_t)(1u << 18);
    }();
    return v;
}

// Smallest pos[] bucket that holds `d` -- keeping it tight minimises per-thread scratch
// (the lever for GPU occupancy); coarse buckets keep the compiled-variant count low.
static int pos_bucket(int d) {
    if (d <= 8)  return 8;
    if (d <= 16) return 16;
    return 32;
}

// MSL source for a kernel specialised to a compile-time `stride` (STRIDE) and pos[] length
// (POSN).  STRIDE as a literal lets the codeword loops unroll into registers; POSN bounds
// the small, dynamically-indexed combination array.
static std::string make_kernel_src(int stride, int posn) {
    static const char* kTemplate = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint STRIDE = %du;
constant constexpr int  POSN   = %d;

struct Params {
    uint m;            // number of generator rows
    uint n;            // number of coordinates (unused in-kernel; kept for parity)
    uint stride;       // words per row (== STRIDE literal; kept for parity)
    uint d;            // subset size for this dispatch (1..p, <= m)
    uint keep_weight;  // collect XORs with Hamming weight in [1, keep_weight]
    uint binom_maxK;   // column stride of the flat binomial table
    uint total_lo;     // low 32 bits of total = C(m, d)
    uint total_hi;     // high 32 bits of total
    uint chunk;        // combination indices per thread
    uint capacity;     // out_buf capacity in codewords
};

// Saturating binomial table lookup: B[n*(maxK+1)+k], matching codeaut::BinomTable layout.
inline ulong binomf(device const ulong* B, uint maxK, int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0ul;
    return B[(uint)n * (maxK + 1u) + (uint)k];
}

kernel void bz_collect(
    device const ulong*   rows      [[buffer(0)]],
    device const ulong*   binomB    [[buffer(1)]],
    constant Params&      P         [[buffer(2)]],
    device atomic_uint*   out_count [[buffer(3)]],
    device ulong*         out_buf   [[buffer(4)]],
    uint                  gid       [[thread_position_in_grid]])
{
    ulong total = (((ulong)P.total_hi) << 32) | (ulong)P.total_lo;
    ulong start = (ulong)gid * (ulong)P.chunk;
    if (start >= total) return;
    ulong end = start + (ulong)P.chunk;
    if (end > total) end = total;

    int m = (int)P.m;
    int d = (int)P.d;

    // Unrank `start` into the ascending d-subset pos[0..d-1] of {0..m-1} (combinatorial
    // number system) -- identical to codeaut::comb_unrank with B.get(m-1-x, d-1-i).
    int pos[POSN];
    {
        ulong r = start;
        int x = 0;
        for (int i = 0; i < d; ++i) {
            for (;;) {
                ulong cnt = binomf(binomB, P.binom_maxK, m - 1 - x, d - 1 - i);
                if (r < cnt) { pos[i] = x; ++x; break; }
                r -= cnt; ++x;
            }
        }
    }

    ulong cw[STRIDE];
    for (ulong it = start; it < end; ++it) {
        // cw = XOR of the d selected rows
        for (uint w = 0; w < STRIDE; ++w) cw[w] = 0ul;
        for (int i = 0; i < d; ++i) {
            device const ulong* row = rows + (ulong)pos[i] * STRIDE;
            for (uint w = 0; w < STRIDE; ++w) cw[w] ^= row[w];
        }
        // Hamming weight; emit when 0 < wt <= keep_weight (matches backend_cpu.cpp).
        uint wt = 0;
        for (uint w = 0; w < STRIDE; ++w) wt += (uint)popcount(cw[w]);
        if (wt > 0u && wt <= P.keep_weight) {
            uint slot = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
            if (slot < P.capacity) {
                device ulong* dst = out_buf + (ulong)slot * STRIDE;
                for (uint w = 0; w < STRIDE; ++w) dst[w] = cw[w];
            }
        }
        // advance to the next ascending d-subset (codeaut::comb_next over {0..m-1})
        int j = d - 1;
        while (j >= 0 && pos[j] == m - d + j) --j;
        if (j < 0) break;
        ++pos[j];
        for (int t = j + 1; t < d; ++t) pos[t] = pos[t - 1] + 1;
    }
}
)MSL";
    char buf[8192];
    std::snprintf(buf, sizeof(buf), kTemplate, stride, posn);
    return std::string(buf);
}

// Host mirror of the MSL Params struct -- field order/types MUST match exactly.
struct Params {
    uint32_t m;
    uint32_t n;
    uint32_t stride;
    uint32_t d;
    uint32_t keep_weight;
    uint32_t binom_maxK;
    uint32_t total_lo;
    uint32_t total_hi;
    uint32_t chunk;
    uint32_t capacity;
};

struct MetalBackend : Backend {
    id<MTLDevice>       dev   = nil;
    id<MTLCommandQueue> queue = nil;
    bool ok = false;
    std::mutex mtx;

    // Lazily-compiled kernel variants keyed by (stride << 8) | posn.
    std::map<uint32_t, id<MTLComputePipelineState>> pipelines;

    MetalBackend() {
        @autoreleasepool {
            dev = MTLCreateSystemDefaultDevice();   // nil if no Metal device -> ok stays false
            if (!dev) return;
            queue = [dev newCommandQueue];
            if (!queue) return;
            if (!pipeline_for(1, 8)) return;        // probe the runtime-compile path
            ok = true;
            warmup();                               // absorb one-time first-dispatch latency
        }
    }

    const char* name() const override { return "metal"; }
    bool available() const override { return ok; }

    // Get (compiling + caching on first use) the pipeline specialised to (stride, posn).
    id<MTLComputePipelineState> pipeline_for(int stride, int posn) {
        uint32_t key = ((uint32_t)stride << 8) | (uint32_t)posn;
        auto it = pipelines.find(key);
        if (it != pipelines.end()) return it->second;
        @autoreleasepool {
            NSError* err = nil;
            std::string src = make_kernel_src(stride, posn);
            id<MTLLibrary> lib =
                [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                  options:nil
                                    error:&err];
            if (!lib) { NSLog(@"codeaut metal compile error: %@", err); return nil; }
            id<MTLFunction> fn = [lib newFunctionWithName:@"bz_collect"];
            if (!fn) return nil;
            id<MTLComputePipelineState> pso =
                [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { NSLog(@"codeaut metal pipeline error: %@", err); return nil; }
            pipelines[key] = pso;
            return pso;
        }
    }

    // Tiny no-op dispatch so the first real solve does not eat first-dispatch latency.
    void warmup() {
        @autoreleasepool {
            id<MTLComputePipelineState> pso = pipeline_for(1, 8);
            if (!pso) return;
            id<MTLBuffer> rows  = [dev newBufferWithLength:sizeof(uint64_t)
                                                   options:MTLResourceStorageModeShared];
            // binom must be valid so the in-kernel unrank terminates: with m=d=1,maxK=0 it
            // reads index 0, which must hold C(0,0)=1 (buffer contents are otherwise undefined).
            id<MTLBuffer> binom = [dev newBufferWithLength:sizeof(uint64_t)
                                                   options:MTLResourceStorageModeShared];
            *(uint64_t*)[binom contents] = 1ull;
            id<MTLBuffer> cnt   = [dev newBufferWithLength:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
            *(uint32_t*)[cnt contents] = 0u;
            id<MTLBuffer> obuf  = [dev newBufferWithLength:sizeof(uint64_t)
                                                   options:MTLResourceStorageModeShared];
            Params P; std::memset(&P, 0, sizeof(P));
            P.m = 1; P.d = 1; P.chunk = 1; P.capacity = 0;  // capacity 0 => never writes out_buf
            P.total_lo = 1;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:rows offset:0 atIndex:0];
            [enc setBuffer:binom offset:0 atIndex:1];
            [enc setBytes:&P length:sizeof(P) atIndex:2];
            [enc setBuffer:cnt offset:0 atIndex:3];
            [enc setBuffer:obuf offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }

    BZEnumResult enumerate(const BZEnumPlan& plan) override {
        // --- cheap pre-checks: anything we can't (or shouldn't) do on the GPU -> CPU ---
        if (!ok) return cpu_backend()->enumerate(plan);
        if (plan.m <= 0 || plan.p <= 0) return cpu_backend()->enumerate(plan);  // CPU: empty res
        if (plan.stride <= 0 || plan.stride > MAX_WORDS)
            return cpu_backend()->enumerate(plan);

        const int m   = plan.m;
        const int p   = plan.p;
        const int maxsw = std::min(p, m);
        if (maxsw > MAX_D) return cpu_backend()->enumerate(plan);

        // Binomial table (matches the CPU backend's BinomTable exactly, saturation included).
        const int kmax = std::min(p, m);
        BinomTable B(m, kmax);

        // Total combination work the GPU loop would actually process, honouring `budget`
        // exactly as the sw loop below does (this is the CPU-vs-GPU decision metric only).
        {
            const int64_t CAP = (int64_t)1 << 62;
            int64_t combos = 0, work = 0;
            for (int sw = 1; sw <= maxsw; ++sw) {
                int64_t total = B.get(m, sw);
                if (total <= 0) continue;
                if (plan.budget > 0 && combos + total > plan.budget) break;  // budget cutoff
                combos += total;
                work = (work > CAP - total) ? CAP : work + total;
            }
            if ((uint64_t)work < gpu_min_work())
                return cpu_backend()->enumerate(plan);
        }

        const int stride = plan.stride;

        std::lock_guard<std::mutex> lock(mtx);

        BZEnumResult res;
        bool fallback = false;

        @autoreleasepool {
            // Constant-across-sw uploads: generator rows and the flat binomial table.
            id<MTLBuffer> rowbuf =
                [dev newBufferWithBytes:plan.rows
                                 length:(size_t)m * stride * sizeof(uint64_t)
                                options:MTLResourceStorageModeShared];
            // Flatten BinomTable (int64, layout B.c[i*(kmax+1)+j]) into u64, same layout.
            std::vector<uint64_t> binflat(B.c.size());
            for (size_t i = 0; i < B.c.size(); ++i) binflat[i] = (uint64_t)B.c[i];
            id<MTLBuffer> binbuf =
                [dev newBufferWithBytes:binflat.data()
                                 length:binflat.size() * sizeof(uint64_t)
                                options:MTLResourceStorageModeShared];

            id<MTLBuffer> cntbuf =
                [dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> outbuf =
                [dev newBufferWithLength:(size_t)CAPACITY * stride * sizeof(uint64_t)
                                 options:MTLResourceStorageModeShared];
            if (!rowbuf || !binbuf || !cntbuf || !outbuf) { fallback = true; }

            for (int sw = 1; !fallback && sw <= maxsw; ++sw) {
                int64_t total = B.get(m, sw);
                if (total <= 0) continue;
                if (plan.budget > 0 && res.combos + total > plan.budget) {
                    res.overflow = true; break;
                }

                id<MTLComputePipelineState> pso = pipeline_for(stride, pos_bucket(sw));
                if (!pso) { fallback = true; break; }

                // Work split: cap thread count, give each thread a contiguous chunk.
                const uint64_t MAX_THREADS = 1u << 20;
                uint64_t utotal = (uint64_t)total;
                uint64_t num_threads = utotal < MAX_THREADS ? utotal : MAX_THREADS;
                if (num_threads == 0) num_threads = 1;
                uint64_t chunk = (utotal + num_threads - 1) / num_threads;
                if (chunk == 0) chunk = 1;
                num_threads = (utotal + chunk - 1) / chunk;

                Params P; std::memset(&P, 0, sizeof(P));
                P.m = (uint32_t)m;
                P.n = (uint32_t)plan.n;
                P.stride = (uint32_t)stride;
                P.d = (uint32_t)sw;
                P.keep_weight = (uint32_t)plan.keep_weight;
                P.binom_maxK = (uint32_t)kmax;
                P.total_lo = (uint32_t)(utotal & 0xffffffffu);
                P.total_hi = (uint32_t)(utotal >> 32);
                P.chunk = (uint32_t)chunk;
                P.capacity = CAPACITY;

                *(uint32_t*)[cntbuf contents] = 0u;   // reset hit counter for this sw

                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:rowbuf offset:0 atIndex:0];
                [enc setBuffer:binbuf offset:0 atIndex:1];
                [enc setBytes:&P length:sizeof(P) atIndex:2];
                [enc setBuffer:cntbuf offset:0 atIndex:3];
                [enc setBuffer:outbuf offset:0 atIndex:4];

                NSUInteger tpt = 256;
                if (tpt > pso.maxTotalThreadsPerThreadgroup)
                    tpt = pso.maxTotalThreadsPerThreadgroup;
                NSUInteger pw = 1; while (pw * 2 <= tpt) pw *= 2; tpt = pw;  // power of two
                if (tpt < 1) tpt = 1;
                NSUInteger groups = (NSUInteger)((num_threads + tpt - 1) / tpt);
                if (groups < 1) groups = 1;
                [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];

                uint32_t cnt = *(uint32_t*)[cntbuf contents];
                if (cnt > CAPACITY) { fallback = true; break; }  // truncated -> CPU oracle

                if (cnt) {
                    const uint64_t* src = (const uint64_t*)[outbuf contents];
                    res.hits.insert(res.hits.end(), src, src + (size_t)cnt * stride);
                }
                res.combos += total;
            }
        }

        if (fallback) return cpu_backend()->enumerate(plan);
        return res;
    }
};

}  // namespace

Backend* metal_backend() {
    static MetalBackend inst;
    return &inst;
}

}  // namespace codeaut

#endif  // CODEAUT_METAL
