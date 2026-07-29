// Metal GPU backend for the shared Brouwer-Zimmermann exponential enumeration -- BOTH
// modes of qsf::Backend over one kernel template:
//
//   * enumerate (min-weight, the distfind engine): per-thread minimum of the full weight
//     of a logical codeword, folded through a threadgroup tree reduction into one global
//     atomic_min.
//   * collect (low-weight collection, the codeaut engine): every codeword of weight in
//     [1, keep] is appended to an output buffer via an atomic slot counter; on capacity
//     overflow the level falls back to the CPU backend (never a truncated set).
//
// One GPU thread owns a contiguous slice of combination indices (combinatorial-number-
// system unranking gives it its start; advance steps through the slice).
//
// PERFORMANCE: three stacked ideas, each preserving the exact math (CPU stays the oracle):
//
//  1. Kernel VARIANTS per (stride, d, tgcache, mode): `stride` and the weight level `d`
//     are compile-time literals (codeword and unrank/advance loops unroll, pos[] stays in
//     registers), `tgcache` selects threadgroup-cached vs device reads, and `mode` bakes
//     in the min-reduce vs collect epilogue. Variants are compiled lazily (~50ms each,
//     once per level per process) and cached.
//  2. TWO-LEVEL enumeration with an incremental codeword. The weight-d combination is
//     split into d-1 OUTER positions (advanced rarely, with the codeword updated by XOR
//     in place) and one INNER `last` index swept in a tight, SIMD-uniform loop, unrolled
//     x4 so independent first-word popcounts hide latency. The weight test runs FIRST with
//     a per-word early exit against the mode's bound (`best` for min-weight -- a codeword
//     with weight >= best can never improve it -- and the constant keep+1 for collect);
//     the logical check / the emit runs only for the rare lighter candidate.
//  3. Threadgroup staging of the generator + check matrices WHEN THEY FIT: gamma is stored
//     TRANSPOSED (word-major, so the unrolled word-0 reads of consecutive rows are
//     contiguous -> no bank conflicts). Codes whose matrices exceed the threadgroup-memory
//     budget (e.g. n=1024, K~1000) use the device-read variant instead -- never a partial
//     or out-of-bounds staging.
//
// The host uploads gamma TRANSPOSED per generator: word w of row r of generator g lives at
// gammaT[(g*STRIDE + w)*K + r]. plan.num_gamma may be a per-level ACTIVE PREFIX of the
// uploaded plan.num_gamma_total generators (the BZ driver skips generators whose lower-
// bound contribution is zero at this level); the buffer cache always holds the full set,
// keyed on plan.buffers_key (which qsf::next_buffers_key() keeps unique across BOTH
// engines -- never on host pointer identity).
//
// Shaders are compiled at runtime (newLibraryWithSource:) because the offline `metal`
// toolchain is not assumed to be installed.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "qsf/backend.hpp"
#include <vector>
#include <map>
#include <mutex>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>

namespace qsf {

namespace {

constexpr int MAX_WORDS = 16;  // codeword <= 1024 bits on the GPU path (else CPU fallback)
constexpr int MAX_D     = 32;  // combination weight <= 32 on the GPU path (else CPU fallback)

// A weight level with fewer than this many candidate codewords (combinations * gammas)
// is cheaper to run on the multicore CPU than to pay GPU dispatch latency for. Tunable
// per mode via the engines' historical env knobs.
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

// Collect-mode output capacity in codewords; overflow (rare: low-weight enumeration emits
// few hits) falls the level back to the CPU. Env-overridable.
static u32 collect_capacity() {
    static u32 v = [] {
        const char* e = std::getenv("CODEAUT_GPU_CAPACITY");
        unsigned long long c = e ? std::strtoull(e, nullptr, 10) : (1ull << 20);
        if (c == 0ull) c = 1ull << 20;
        if (c > 0x7fffffffull) c = 0x7fffffffull;
        return (u32)c;
    }();
    return v;
}

// Threads per threadgroup (power of two). 64 measured best on Apple Silicon for the
// two-level kernel; DISTFIND_TPT overrides for tuning.
static unsigned long default_tpt() {
    static unsigned long v = [] {
        const char* e = std::getenv("DISTFIND_TPT");
        return e ? std::strtoul(e, nullptr, 10) : 64ul;
    }();
    return v;
}

// MSL source for a kernel specialised to a compile-time stride, WEIGHT LEVEL d, read
// path, and mode. Baking d in as a literal fully unrolls the unrank/advance loops and
// promotes the pos[] array to registers (dynamic indexing would spill it to slow
// thread-local memory -- and the outer advance runs every ~(K-d)/(d-1) elements, so it
// is hot). One variant per level compiles lazily in ~50ms, trivially amortised.
// `tgcache` = 1 stages gammaT+check into threadgroup memory (the host guarantees they
// fit); 0 reads device memory directly. `collect` = 1 emits every light nonzero codeword
// through an atomic slot counter instead of folding an atomic minimum.
static std::string make_kernel_src(int stride, int d, int tgcache, int collect) {
    static const char* kTemplate = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint STRIDE = %du;
constant constexpr int  D      = %d;   // weight level, a literal: loops unroll, pos[] in registers
constant constexpr int  M      = D - 1;

#if TGC
#define GSPACE threadgroup
#else
#define GSPACE device
#endif

struct Params {
    uint n, stride, K, d, num_gamma, kcheck, binom_maxK;
    uint total_lo, total_hi;
    uint chunk;
    uint current_best;   // min mode: pruning bound; collect mode: keep_weight + 1
    uint capacity;       // collect mode: out_buf capacity in codewords
};

inline ulong binomf(device const ulong* B, uint maxK, int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0ul;
    return B[(uint)n * (maxK + 1u) + (uint)k];
}

kernel void bz_enumerate(
    device const ulong*   gammaT  [[buffer(0)]],   // transposed: [(g*STRIDE+w)*K + r]
    device const ulong*   chk     [[buffer(1)]],   // row-major: [c*STRIDE + w]
    device const ulong*   binomB  [[buffer(2)]],
    constant Params&      P       [[buffer(3)]],
#if COLLECT
    device atomic_uint*   out_cnt [[buffer(4)]],
    device ulong*         out_buf [[buffer(5)]],
#else
    device atomic_uint*   result  [[buffer(4)]],
    threadgroup uint*     tg      [[threadgroup(0)]],
#endif
    threadgroup ulong*    gcache  [[threadgroup(1)]],
    uint                  gid     [[thread_position_in_grid]],
    uint                  lid     [[thread_position_in_threadgroup]],
    uint                  tgsize  [[threads_per_threadgroup]])
{
    ulong total = (((ulong)P.total_hi) << 32) | (ulong)P.total_lo;
    ulong start = (ulong)gid * (ulong)P.chunk;
    uint best = P.current_best;

#if TGC
    // Stage gammaT (active prefix) + check into threadgroup memory once per group; the
    // host only selects this variant when they fit the threadgroup-memory budget.
    uint gcount = P.num_gamma * P.K * STRIDE;
    uint ccount = P.kcheck * STRIDE;
    for (uint i = lid; i < gcount; i += tgsize) gcache[i] = gammaT[i];
    for (uint i = lid; i < ccount; i += tgsize) gcache[gcount + i] = chk[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const ulong* Gall  = gcache;
    threadgroup const ulong* chk_p = gcache + gcount;
#else
    device const ulong* Gall  = gammaT;
    device const ulong* chk_p = chk;
#endif

    // Guard the work (rather than returning early) so control flow stays uniform.
    if (start < total) {
        ulong end = start + (ulong)P.chunk;
        if (end > total) end = total;

        int K = (int)P.K;

        // TWO-LEVEL enumeration. Split the weight-d combination into m=d-1 OUTER positions
        // pos[0..m-1] and one INNER index `last` in (pos[m-1], K). The outer positions
        // advance rarely (once per inner run); the inner loop is the hot path and is
        // UNIFORM across a SIMD group -- no per-step divergent carry and no pos[] writes.
        // cw_base = XOR of the outer rows stays in registers; for each inner `last` the
        // codeword is cw_base ^ row[last], fused into the early-exit popcount and only
        // materialised for the rare light candidate.
        int pos[D];
        for (uint g = 0; g < P.num_gamma; ++g) {
            // Transposed (word-major) generator g: row r word w is Gp[w*K + r].
            GSPACE const ulong* Gp  = Gall + (ulong)g * STRIDE * (ulong)K;
            GSPACE const ulong* Gw0 = Gp;                  // word-0 plane (the hot reads)

            // Unrank `start` into the weight-d combination pos[0..d-1].
            {
                ulong r = start; int x = 0;
                for (int i = 0; i < D; ++i) {
                    for (;;) {
                        ulong cnt = binomf(binomB, P.binom_maxK, K - 1 - x, D - 1 - i);
                        if (r < cnt) { pos[i] = x; ++x; break; }
                        r -= cnt; ++x;
                    }
                }
            }
            // cw_base = XOR of the M outer rows pos[0..M-1] (empty XOR = 0 when D==1).
            ulong cwb[STRIDE];
            for (uint w = 0; w < STRIDE; ++w) cwb[w] = 0ul;
            for (int i = 0; i < M; ++i)
                for (uint w = 0; w < STRIDE; ++w) cwb[w] ^= Gp[w * (uint)K + (uint)pos[i]];

            int last = pos[D - 1];          // inner start (may be mid-run for a chunk offset)
            ulong cnt = 0, need = end - start;
            while (cnt < need) {
                // Hot inner loop: uniform sweep of `last`, cw = cwb ^ row[last]. Unrolled x4
                // so the (independent) first-word popcounts of four consecutive `last` issue
                // together and hide read/popcount latency; the light candidate that needs the
                // full weight + the mode epilogue is rare and drops to the per-element slow
                // path. `hi` caps the run at the chunk end (need) or the inner range end (K).
                int last_start = last;
                int hi = K;
                if ((ulong)(K - last) > need - cnt) hi = last + (int)(need - cnt);
                ulong cwb0 = cwb[0];
                int lst4 = hi - 3;
                for (; last < lst4; last += 4) {
                    uint a0 = (uint)popcount(cwb0 ^ Gw0[last]);      // contiguous word-0 reads
                    uint a1 = (uint)popcount(cwb0 ^ Gw0[last + 1]);
                    uint a2 = (uint)popcount(cwb0 ^ Gw0[last + 2]);
                    uint a3 = (uint)popcount(cwb0 ^ Gw0[last + 3]);
                    if (a0 < best || a1 < best || a2 < best || a3 < best) {
                        uint aa[4] = {a0, a1, a2, a3};
                        for (int u = 0; u < 4; ++u) {
                            if (aa[u] >= best) continue;
                            uint r = (uint)(last + u);
                            uint wt = aa[u]; bool light = true;
                            for (uint w = 1; w < STRIDE && light; ++w) {
                                wt += (uint)popcount(cwb[w] ^ Gp[w * (uint)K + r]);
                                light = wt < best;
                            }
                            if (!light) continue;
#if COLLECT
                            if (wt > 0u) {
                                uint slot = atomic_fetch_add_explicit(out_cnt, 1u,
                                                                      memory_order_relaxed);
                                if (slot < P.capacity) {
                                    device ulong* dst = out_buf + (ulong)slot * STRIDE;
                                    for (uint w = 0; w < STRIDE; ++w)
                                        dst[w] = cwb[w] ^ Gp[w * (uint)K + r];
                                }
                            }
#else
                            bool logical = (P.kcheck == 0u);
                            for (uint c = 0; c < P.kcheck && !logical; ++c) {
                                ulong acc = 0ul;
                                GSPACE const ulong* cr = chk_p + (ulong)c * STRIDE;
                                for (uint w = 0; w < STRIDE; ++w)
                                    acc ^= (cr[w] & (cwb[w] ^ Gp[w * (uint)K + r]));
                                if (popcount(acc) & 1ul) logical = true;
                            }
                            if (logical) best = wt;
#endif
                        }
                    }
                }
                for (; last < hi; ++last) {   // tail (< 4 elements)
                    uint r = (uint)last;
                    uint wt = (uint)popcount(cwb0 ^ Gw0[last]);
                    bool light = wt < best;
                    for (uint w = 1; w < STRIDE && light; ++w) {
                        wt += (uint)popcount(cwb[w] ^ Gp[w * (uint)K + r]);
                        light = wt < best;
                    }
                    if (light) {
#if COLLECT
                        if (wt > 0u) {
                            uint slot = atomic_fetch_add_explicit(out_cnt, 1u,
                                                                  memory_order_relaxed);
                            if (slot < P.capacity) {
                                device ulong* dst = out_buf + (ulong)slot * STRIDE;
                                for (uint w = 0; w < STRIDE; ++w)
                                    dst[w] = cwb[w] ^ Gp[w * (uint)K + r];
                            }
                        }
#else
                        bool logical = (P.kcheck == 0u);
                        for (uint c = 0; c < P.kcheck && !logical; ++c) {
                            ulong acc = 0ul;
                            GSPACE const ulong* cr = chk_p + (ulong)c * STRIDE;
                            for (uint w = 0; w < STRIDE; ++w)
                                acc ^= (cr[w] & (cwb[w] ^ Gp[w * (uint)K + r]));
                            if (popcount(acc) & 1ul) logical = true;
                        }
                        if (logical) best = wt;
#endif
                    }
                }
                cnt += (ulong)(hi - last_start);   // combos processed in this inner run
                if (cnt >= need) break;            // chunk end reached (hi was the cap)
                // Advance the M outer positions to the next combination, updating cwb in place.
                int j = M - 1;
                while (j >= 0 && pos[j] == K - D + j) --j;
                if (j < 0) break;
                for (int t = j; t < M; ++t)
                    for (uint w = 0; w < STRIDE; ++w) cwb[w] ^= Gp[w * (uint)K + (uint)pos[t]];
                ++pos[j];
                for (int t = j + 1; t < M; ++t) pos[t] = pos[t - 1] + 1;
                for (int t = j; t < M; ++t)
                    for (uint w = 0; w < STRIDE; ++w) cwb[w] ^= Gp[w * (uint)K + (uint)pos[t]];
                last = pos[M - 1] + 1;      // inner restart for the fresh outer combination
            }
        }
    }

#if !COLLECT
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
#endif
}
)MSL";
    int need = std::snprintf(nullptr, 0, kTemplate, stride, d);
    std::string out((size_t)need + 1, '\0');
    std::snprintf(&out[0], out.size(), kTemplate, stride, d);
    out.resize((size_t)need);
    char def[64];
    std::snprintf(def, sizeof(def), "#define TGC %d\n#define COLLECT %d\n",
                  tgcache ? 1 : 0, collect ? 1 : 0);
    return std::string(def) + out;
}

struct Params {
    uint32_t n, stride, K, d, num_gamma, kcheck, binom_maxK;
    uint32_t total_lo, total_hi;
    uint32_t chunk;
    uint32_t current_best;
    uint32_t capacity;
};

struct MetalBackend : Backend {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> rbuf = nil;       // persistent 1-word result/count buffer (shared storage)
    id<MTLBuffer> obuf = nil;       // collect-mode output buffer (grow-only)
    size_t obuf_words = 0;
    bool ok = false;
    std::mutex mtx;

    // Lazily-compiled kernel variants, keyed by (stride, d, tgcache, collect).
    std::map<uint32_t, id<MTLComputePipelineState>> pipelines;

    // Device-buffer cache for (gammaT, check, binom). These are constant across weight
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
            if (ok && !pipeline_for(1, 8, 1, 0)) { ok = false; return; }  // probe compile path
            if (ok) warmup();   // absorb one-time driver/pipeline first-dispatch latency
        }
    }

    std::string name() const override { return "metal"; }
    bool available() const override { return ok; }

    // Get (compiling+caching on first use) the pipeline for (stride, d, tgc, collect).
    id<MTLComputePipelineState> pipeline_for(int stride, int d, int tgc, int collect) {
        uint32_t key = ((uint32_t)stride << 16) | ((uint32_t)d << 2)
                     | ((uint32_t)tgc << 1) | (uint32_t)collect;
        auto it = pipelines.find(key);
        if (it != pipelines.end()) return it->second;
        @autoreleasepool {
            NSError* err = nil;
            std::string src = make_kernel_src(stride, d, tgc, collect);
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
            id<MTLComputePipelineState> pso = pipeline_for(1, 8, 1, 0);
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
            [enc setThreadgroupMemoryLength:8 atIndex:1];
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

    // (Re)upload the per-solve constant buffers when the solve token changes. Must be
    // called with mtx held. Returns the total combination count helpers need.
    void ensure_solve_buffers(const EnumPlan& plan) {
        if (plan.buffers_key == buf_key) return;
        const int ng_total = plan.num_gamma_total > 0 ? plan.num_gamma_total : plan.num_gamma;
        size_t g_bytes_up = (size_t)ng_total * plan.K * plan.stride * sizeof(u64);
        size_t c_bytes_ = (size_t)plan.kcheck * plan.stride * sizeof(u64);
        size_t b_bytes_ = (size_t)(plan.binom_maxN + 1) * (plan.binom_maxK + 1) * sizeof(u64);
        // Transpose gamma per generator to word-major: word w of row r of
        // generator g at gt[(g*stride + w)*K + r].
        std::vector<u64> gt((size_t)ng_total * plan.K * plan.stride);
        for (int g = 0; g < ng_total; ++g)
            for (int r = 0; r < plan.K; ++r)
                for (int w = 0; w < plan.stride; ++w)
                    gt[((size_t)g * plan.stride + w) * plan.K + r] =
                        plan.gamma[((size_t)g * plan.K + r) * plan.stride + w];
        g_buf = upload(gt.data(), g_bytes_up);
        c_buf = upload(plan.check, c_bytes_);
        b_buf = upload(plan.binom, b_bytes_);
        buf_key = plan.buffers_key;
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
        if (total == 0 || plan.num_gamma == 0) return WEIGHT_NONE;

        // Hybrid dispatch: small levels are dominated by GPU launch latency -> CPU.
        u64 work = total > (1ull << 40) ? total : total * (u64)plan.num_gamma;
        if (work < gpu_min_work_enumerate()) return cpu_backend()->enumerate(plan);

        std::lock_guard<std::mutex> lock(mtx);

        size_t g_bytes_act = (size_t)plan.num_gamma * plan.K * plan.stride * sizeof(u64);
        size_t c_bytes_ = (size_t)plan.kcheck * plan.stride * sizeof(u64);

        NSUInteger tpt = (NSUInteger)default_tpt();
        // Threadgroup staging only when generators + check + the reduction scratch fit the
        // device's threadgroup-memory budget; large codes take the device-read variant.
        NSUInteger maxTG = [dev maxThreadgroupMemoryLength];
        bool tgc = (g_bytes_act + c_bytes_ + (size_t)tpt * sizeof(uint32_t) + 64) <= (size_t)maxTG;

        id<MTLComputePipelineState> pso = pipeline_for(plan.stride, plan.d, tgc ? 1 : 0, 0);
        if (!pso) return cpu_backend()->enumerate(plan);

        @autoreleasepool {
            // pick a work split: cap threads, give each a contiguous chunk
            const u64 MAX_THREADS = 1u << 20;
            u64 num_threads = total < MAX_THREADS ? total : MAX_THREADS;
            u64 chunk = (total + num_threads - 1) / num_threads;
            num_threads = (total + chunk - 1) / chunk;

            ensure_solve_buffers(plan);
            id<MTLBuffer> gbuf = g_buf, cbuf = c_buf, bbuf = b_buf;

            Params P;
            P.n = plan.n; P.stride = plan.stride; P.K = plan.K; P.d = plan.d;
            P.num_gamma = plan.num_gamma; P.kcheck = plan.kcheck;
            P.binom_maxK = plan.binom_maxK;
            P.total_lo = (uint32_t)(total & 0xffffffffu);
            P.total_hi = (uint32_t)(total >> 32);
            P.chunk = (uint32_t)chunk;
            P.current_best = (uint32_t)plan.current_best;
            P.capacity = 0;

            *(uint32_t*)[rbuf contents] = (uint32_t)plan.current_best;

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:gbuf offset:0 atIndex:0];
            [enc setBuffer:cbuf offset:0 atIndex:1];
            [enc setBuffer:bbuf offset:0 atIndex:2];
            [enc setBytes:&P length:sizeof(P) atIndex:3];
            [enc setBuffer:rbuf offset:0 atIndex:4];

            if (tpt > pso.maxTotalThreadsPerThreadgroup) tpt = pso.maxTotalThreadsPerThreadgroup;
            NSUInteger pw = 1; while (pw * 2 <= tpt) pw *= 2; tpt = pw; // power of two
            NSUInteger groups = (NSUInteger)((num_threads + tpt - 1) / tpt);
            [enc setThreadgroupMemoryLength:tpt * sizeof(uint32_t) atIndex:0];
            [enc setThreadgroupMemoryLength:(tgc ? g_bytes_act + c_bytes_ : 8) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            return (int)(*(uint32_t*)[rbuf contents]);
        }
    }

    CollectResult collect(const EnumPlan& plan, int keep_weight) override {
        if (plan.stride > MAX_WORDS || plan.d > MAX_D || !ok)
            return cpu_backend()->collect(plan, keep_weight);

        auto binom = [&](int n, int k) -> u64 {
            if (k < 0 || n < 0 || k > n) return 0ull;
            return plan.binom[(size_t)n * (plan.binom_maxK + 1) + k];
        };
        u64 total = binom(plan.K, plan.d);
        if (total == 0 || plan.num_gamma == 0 || keep_weight <= 0)
            return cpu_backend()->collect(plan, keep_weight);  // trivial: CPU returns it exactly

        u64 work = total > (1ull << 40) ? total : total * (u64)plan.num_gamma;
        if (work < gpu_min_work_collect()) return cpu_backend()->collect(plan, keep_weight);

        std::lock_guard<std::mutex> lock(mtx);

        size_t g_bytes_act = (size_t)plan.num_gamma * plan.K * plan.stride * sizeof(u64);

        NSUInteger tpt = (NSUInteger)default_tpt();
        NSUInteger maxTG = [dev maxThreadgroupMemoryLength];
        bool tgc = (g_bytes_act + 64) <= (size_t)maxTG;

        id<MTLComputePipelineState> pso = pipeline_for(plan.stride, plan.d, tgc ? 1 : 0, 1);
        if (!pso) return cpu_backend()->collect(plan, keep_weight);

        const u32 capacity = collect_capacity();

        @autoreleasepool {
            const u64 MAX_THREADS = 1u << 20;
            u64 num_threads = total < MAX_THREADS ? total : MAX_THREADS;
            u64 chunk = (total + num_threads - 1) / num_threads;
            num_threads = (total + chunk - 1) / chunk;

            ensure_solve_buffers(plan);
            id<MTLBuffer> gbuf = g_buf, cbuf = c_buf, bbuf = b_buf;

            // Grow-only output buffer sized for the configured capacity at this stride.
            size_t need_words = (size_t)capacity * plan.stride;
            if (!obuf || obuf_words < need_words) {
                obuf = [dev newBufferWithLength:need_words * sizeof(u64)
                                        options:MTLResourceStorageModeShared];
                obuf_words = obuf ? need_words : 0;
            }
            if (!obuf) return cpu_backend()->collect(plan, keep_weight);

            Params P;
            P.n = plan.n; P.stride = plan.stride; P.K = plan.K; P.d = plan.d;
            P.num_gamma = plan.num_gamma; P.kcheck = 0;
            P.binom_maxK = plan.binom_maxK;
            P.total_lo = (uint32_t)(total & 0xffffffffu);
            P.total_hi = (uint32_t)(total >> 32);
            P.chunk = (uint32_t)chunk;
            P.current_best = (uint32_t)keep_weight + 1u;   // the weight-first exit bound
            P.capacity = capacity;

            *(uint32_t*)[rbuf contents] = 0u;   // hit counter

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:gbuf offset:0 atIndex:0];
            [enc setBuffer:cbuf offset:0 atIndex:1];
            [enc setBuffer:bbuf offset:0 atIndex:2];
            [enc setBytes:&P length:sizeof(P) atIndex:3];
            [enc setBuffer:rbuf offset:0 atIndex:4];
            [enc setBuffer:obuf offset:0 atIndex:5];

            if (tpt > pso.maxTotalThreadsPerThreadgroup) tpt = pso.maxTotalThreadsPerThreadgroup;
            NSUInteger pw = 1; while (pw * 2 <= tpt) pw *= 2; tpt = pw; // power of two
            NSUInteger groups = (NSUInteger)((num_threads + tpt - 1) / tpt);
            [enc setThreadgroupMemoryLength:(tgc ? g_bytes_act : 8) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            uint32_t cnt = *(uint32_t*)[rbuf contents];
            if (cnt > capacity)   // truncated output -> recompute the level on the CPU oracle
                return cpu_backend()->collect(plan, keep_weight);

            CollectResult res;
            if (cnt) {
                const u64* src = (const u64*)[obuf contents];
                res.hits.assign(src, src + (size_t)cnt * plan.stride);
            }
            return res;
        }
    }
};

} // namespace

Backend* metal_backend() {
    static MetalBackend inst;
    return &inst;
}

} // namespace qsf
