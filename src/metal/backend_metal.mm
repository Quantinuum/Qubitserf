// Metal GPU backend for the Brouwer-Zimmermann exponential enumeration.
//
// One GPU thread owns a contiguous slice of combination indices (combinatorial-number-
// system unranking gives it its start; advance steps through the slice). For each
// weight-d combination it XORs the selected rows of every information-set generator,
// tests the logical detector, popcounts, and folds the minimum into a global atomic.
//
// PERFORMANCE: the per-thread scratch arrays (`cw` for the codeword, `pos` for the
// combination) are the throughput bottleneck. Sized to the worst case (MAX_WORDS=16,
// MAX_D=32) they spill into thread-local (device-backed) memory and collapse GPU
// occupancy, leaving the device ~98% idle on the dominant weight levels. So we compile
// a small kernel VARIANT per (stride, d-bucket): `stride` becomes a compile-time literal
// (the codeword loops fully unroll and `cw[stride]` lives in registers) and `pos` is
// sized to the smallest bucket that fits `d`. Variants are compiled lazily and cached.
// This is a ~3x win on the deep levels with no change to the math (CPU stays the oracle).
//
// Shaders are compiled at runtime (newLibraryWithSource:) because the offline `metal`
// toolchain is not assumed to be installed.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "qubitserf/backend.hpp"
#include <vector>
#include <map>
#include <mutex>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>

namespace qubitserf {

namespace {

constexpr int MAX_WORDS = 16;  // codeword <= 1024 bits on the GPU path (else CPU fallback)
constexpr int MAX_D     = 32;  // combination weight <= 32 on the GPU path (else CPU fallback)

// A weight level with fewer than this many candidate codewords (combinations * gammas)
// is cheaper to run on the multicore CPU than to pay GPU dispatch latency for. Tunable.
// Lower than the historical 1<<20: the variant kernel makes the GPU pay off much sooner.
static u64 gpu_min_work() {
    static u64 v = [] {
        const char* e = std::getenv("QUBITSERF_GPU_MIN_WORK");
        return e ? (u64)std::strtoull(e, nullptr, 10) : (u64)(1u << 18);
    }();
    return v;
}

// Smallest pos[] bucket that holds `d`. Keeping this tight minimises per-thread
// scratch footprint (the lever for occupancy); coarse buckets keep the variant count low.
static int pos_bucket(int d) {
    if (d <= 8)  return 8;
    if (d <= 16) return 16;
    return 32;
}

// MSL source for a kernel specialised to a compile-time stride and pos[] length. `stride`
// as a literal lets the codeword loops unroll into registers; `posn` bounds the (small,
// dynamically-indexed) combination array.
static std::string make_kernel_src(int stride, int posn) {
    static const char* kTemplate = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint STRIDE = %du;
constant constexpr int  POSN   = %d;

struct Params {
    uint n, stride, K, d, num_gamma, kcheck, binom_maxK;
    uint total_lo, total_hi;
    uint chunk;
    uint current_best;
};

inline ulong binomf(device const ulong* B, uint maxK, int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0ul;
    return B[(uint)n * (maxK + 1u) + (uint)k];
}

kernel void bz_enumerate(
    device const ulong*   gamma   [[buffer(0)]],
    device const ulong*   chk     [[buffer(1)]],
    device const ulong*   binomB  [[buffer(2)]],
    constant Params&      P       [[buffer(3)]],
    device atomic_uint*   result  [[buffer(4)]],
    threadgroup uint*     tg      [[threadgroup(0)]],
    uint                  gid     [[thread_position_in_grid]],
    uint                  lid     [[thread_position_in_threadgroup]],
    uint                  tgsize  [[threads_per_threadgroup]])
{
    ulong total = (((ulong)P.total_hi) << 32) | (ulong)P.total_lo;
    ulong start = (ulong)gid * (ulong)P.chunk;
    uint best = P.current_best;

    // Guard the work (rather than returning early) so control flow stays uniform.
    if (start < total) {
        ulong end = start + (ulong)P.chunk;
        if (end > total) end = total;

        int K = (int)P.K, d = (int)P.d;

        int pos[POSN];
        {
            ulong r = start; int x = 0;
            for (int i = 0; i < d; ++i) {
                for (;;) {
                    ulong cnt = binomf(binomB, P.binom_maxK, K - 1 - x, d - 1 - i);
                    if (r < cnt) { pos[i] = x; ++x; break; }
                    r -= cnt; ++x;
                }
            }
        }

        ulong cw[STRIDE];
        for (ulong it = start; it < end; ++it) {
            for (uint g = 0; g < P.num_gamma; ++g) {
                device const ulong* G = gamma + (ulong)g * (ulong)K * STRIDE;
                for (uint w = 0; w < STRIDE; ++w) cw[w] = 0ul;
                for (int i = 0; i < d; ++i) {
                    device const ulong* row = G + (ulong)pos[i] * STRIDE;
                    for (uint w = 0; w < STRIDE; ++w) cw[w] ^= row[w];
                }
                bool logical = (P.kcheck == 0u);
                for (uint c = 0; c < P.kcheck && !logical; ++c) {
                    ulong acc = 0ul;
                    device const ulong* cr = chk + (ulong)c * STRIDE;
                    for (uint w = 0; w < STRIDE; ++w) acc ^= (cr[w] & cw[w]);
                    if (popcount(acc) & 1ul) logical = true;
                }
                if (logical) {
                    uint wt = 0;
                    for (uint w = 0; w < STRIDE; ++w) wt += (uint)popcount(cw[w]);
                    if (wt < best) best = wt;
                }
            }
            // advance combination
            int j = d - 1;
            while (j >= 0 && pos[j] == K - d + j) --j;
            if (j < 0) break;
            ++pos[j];
            for (int t = j + 1; t < d; ++t) pos[t] = pos[t - 1] + 1;
        }
    }

    // Threadgroup tree reduction (tgsize is a power of two), then ONE atomic per group.
    // Uniform threadgroups + neutral `current_best` for out-of-range threads keep the
    // reduction correct on the boundary without divergent SIMD ops.
    tg[lid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (lid < s) tg[lid] = min(tg[lid], tg[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0)
        atomic_fetch_min_explicit(result, tg[0], memory_order_relaxed);
}
)MSL";
    char buf[4096];
    std::snprintf(buf, sizeof(buf), kTemplate, stride, posn);
    return std::string(buf);
}

struct Params {
    uint32_t n, stride, K, d, num_gamma, kcheck, binom_maxK;
    uint32_t total_lo, total_hi;
    uint32_t chunk;
    uint32_t current_best;
};

struct MetalBackend : Backend {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> rbuf = nil;       // persistent 1-word result buffer (shared storage)
    bool ok = false;
    std::mutex mtx;

    // Lazily-compiled kernel variants, keyed by (stride << 8 | posn).
    std::map<uint32_t, id<MTLComputePipelineState>> pipelines;

    // Device-buffer cache for (gamma, check, binom). These are constant across weight
    // levels WITHIN a solve, so we key on the plan's unique per-solve token. We must NOT
    // key on host pointer identity: the host vectors are freed between solves and often
    // reallocated at the same address with different contents (stale-buffer bug).
    u64 buf_key = ~0ull;
    id<MTLBuffer> g_buf = nil, c_buf = nil, b_buf = nil;

    MetalBackend() {
        @autoreleasepool {
            dev = MTLCreateSystemDefaultDevice();
            if (!dev) return;
            queue = [dev newCommandQueue];
            rbuf = [dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            ok = (queue != nil && rbuf != nil);
            if (ok && !pipeline_for(1, 8)) { ok = false; return; }  // probe compile path
            if (ok) warmup();   // absorb one-time driver/pipeline first-dispatch latency
        }
    }

    std::string name() const override { return "metal"; }
    bool available() const override { return ok; }

    // Get (compiling+caching on first use) the pipeline specialised to (stride, posn).
    id<MTLComputePipelineState> pipeline_for(int stride, int posn) {
        uint32_t key = ((uint32_t)stride << 8) | (uint32_t)posn;
        auto it = pipelines.find(key);
        if (it != pipelines.end()) return it->second;
        @autoreleasepool {
            NSError* err = nil;
            std::string src = make_kernel_src(stride, posn);
            id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                                   options:nil error:&err];
            if (!lib) { NSLog(@"qubitserf metal compile error: %@", err); return nil; }
            id<MTLFunction> fn = [lib newFunctionWithName:@"bz_enumerate"];
            id<MTLComputePipelineState> pso =
                [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { NSLog(@"qubitserf metal pipeline error: %@", err); return nil; }
            pipelines[key] = pso;
            return pso;
        }
    }

    // Tiny no-op dispatch so the first real solve does not eat first-dispatch latency.
    void warmup() {
        @autoreleasepool {
            id<MTLComputePipelineState> pso = pipeline_for(1, 8);
            if (!pso) return;
            id<MTLBuffer> z = [dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
            Params P; std::memset(&P, 0, sizeof(P));
            P.stride = 1; P.K = 1; P.d = 1; P.chunk = 1; P.current_best = WEIGHT_NONE;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:z offset:0 atIndex:0];
            [enc setBuffer:z offset:0 atIndex:1];
            [enc setBuffer:z offset:0 atIndex:2];
            [enc setBytes:&P length:sizeof(P) atIndex:3];
            [enc setBuffer:rbuf offset:0 atIndex:4];
            [enc setThreadgroupMemoryLength:64 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }

    id<MTLBuffer> upload(const u64* ptr, size_t bytes) {
        return bytes ? [dev newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared]
                     : [dev newBufferWithLength:1 options:MTLResourceStorageModeShared];
    }

    int enumerate(const EnumPlan& plan) override {
        if (plan.stride > MAX_WORDS || plan.d > MAX_D || !ok)
            return cpu_backend()->enumerate(plan);

        // number of combinations C(K,d) from the flat binomial table
        auto binom = [&](int n, int k) -> u64 {
            if (k < 0 || n < 0 || k > n) return 0ull;
            return plan.binom[(size_t)n * (plan.binom_maxK + 1) + k];
        };
        u64 total = binom(plan.K, plan.d);
        if (total == 0) return WEIGHT_NONE;

        // Hybrid dispatch: small levels are dominated by GPU launch latency -> CPU.
        u64 work = total > (1ull << 40) ? total : total * (u64)plan.num_gamma;
        if (work < gpu_min_work()) return cpu_backend()->enumerate(plan);

        std::lock_guard<std::mutex> lock(mtx);

        id<MTLComputePipelineState> pso = pipeline_for(plan.stride, pos_bucket(plan.d));
        if (!pso) return cpu_backend()->enumerate(plan);

        @autoreleasepool {
            // pick a work split: cap threads, give each a contiguous chunk
            const u64 MAX_THREADS = 1u << 20;
            u64 num_threads = total < MAX_THREADS ? total : MAX_THREADS;
            u64 chunk = (total + num_threads - 1) / num_threads;
            num_threads = (total + chunk - 1) / chunk;

            size_t g_bytes_ = (size_t)plan.num_gamma * plan.K * plan.stride * sizeof(u64);
            size_t c_bytes_ = (size_t)plan.kcheck * plan.stride * sizeof(u64);
            size_t b_bytes_ = (size_t)(plan.binom_maxN + 1) * (plan.binom_maxK + 1) * sizeof(u64);

            if (plan.buffers_key != buf_key) {   // new solve -> re-upload constant buffers
                g_buf = upload(plan.gamma, g_bytes_);
                c_buf = upload(plan.check, c_bytes_);
                b_buf = upload(plan.binom, b_bytes_);
                buf_key = plan.buffers_key;
            }
            id<MTLBuffer> gbuf = g_buf, cbuf = c_buf, bbuf = b_buf;

            Params P;
            P.n = plan.n; P.stride = plan.stride; P.K = plan.K; P.d = plan.d;
            P.num_gamma = plan.num_gamma; P.kcheck = plan.kcheck;
            P.binom_maxK = plan.binom_maxK;
            P.total_lo = (uint32_t)(total & 0xffffffffu);
            P.total_hi = (uint32_t)(total >> 32);
            P.chunk = (uint32_t)chunk;
            P.current_best = (uint32_t)plan.current_best;

            *(uint32_t*)[rbuf contents] = (uint32_t)plan.current_best;

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:gbuf offset:0 atIndex:0];
            [enc setBuffer:cbuf offset:0 atIndex:1];
            [enc setBuffer:bbuf offset:0 atIndex:2];
            [enc setBytes:&P length:sizeof(P) atIndex:3];
            [enc setBuffer:rbuf offset:0 atIndex:4];

            NSUInteger tpt = 256;
            if (tpt > pso.maxTotalThreadsPerThreadgroup) tpt = pso.maxTotalThreadsPerThreadgroup;
            NSUInteger pw = 1; while (pw * 2 <= tpt) pw *= 2; tpt = pw; // power of two
            NSUInteger groups = (NSUInteger)((num_threads + tpt - 1) / tpt);
            [enc setThreadgroupMemoryLength:tpt * sizeof(uint32_t) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            return (int)(*(uint32_t*)[rbuf contents]);
        }
    }
};

} // namespace

Backend* metal_backend() {
    static MetalBackend inst;
    return &inst;
}

} // namespace qubitserf
