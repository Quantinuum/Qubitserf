// The shared CPU enumeration core: ONE two-level, transposed, x4-unrolled kernel serving
// both engines through a compile-time Sink policy. This is the CPU port of the Metal
// two-level kernel (src/core/metal/backend_metal.mm): the exact same math, so the CPU
// stays the GPU's correctness oracle (guarded by src/distfind/tests/cpu_enum_ref.cpp).
//
//  * TWO-LEVEL enumeration: the weight-d combination splits into m = d-1 OUTER positions
//    pos[0..m-1] (advanced rarely, with the codeword base cw maintained by XOR-in-place)
//    and one INNER index `last` swept in a tight loop where the codeword is cw ^ row[last].
//  * WEIGHT-FIRST early exit: the codeword weight is computed first with a per-word exit
//    against the sink's bound (a codeword at least that heavy can never be accepted);
//    the sink's accept logic runs only for the rare candidate lighter than the bound.
//  * `Gt` holds the active generators TRANSPOSED (word-major: word w of row r of generator
//    g at Gt[(g*stride + w)*K + r]) so the inner sweep's word-0 reads of consecutive rows
//    are contiguous -- the x4-unrolled hot loop then auto-vectorizes (NEON XOR + popcount).
//
// Sink contract (methods must be cheap and inlinable):
//   int  bound() const;
//       Only candidates with full weight < bound() reach hit(). May shrink over time
//       (min-tracking) or stay constant (threshold collection); the kernel re-reads it
//       after every hit() call.
//   void hit(const EnumPlan& p, const u64* Gp, const u64* cwb, int r, int wt);
//       Called for a candidate codeword cwb ^ column(r) of full weight wt < bound().
//       Gp is the current TRANSPOSED generator: word w of row r' at Gp[w*K + r'].
//
// Two sinks cover both engines:
//   MinLogicalSink -- distfind: run the logical-detector test, fold the minimum weight.
//   CollectSink    -- codeaut: materialize and append every nonzero codeword (<= keep).
#pragma once
#include <thread>
#include <vector>
#include "qsf/backend.hpp"

namespace qsf {

// Lightweight BinomTable view over the plan's flat binomial buffer so unranking needs
// no copy.
struct BinomView {
    const u64* c; int maxK;
    inline u64 binom(int n, int k) const {
        if (k < 0 || n < 0 || k > n) return 0ull;
        return c[(size_t)n * (maxK + 1) + k];
    }
};

inline void unrank_flat(const BinomView& bt, int K, int d, u64 r, int* pos) {
    int x = 0;
    for (int i = 0; i < d; ++i) {
        for (;;) {
            u64 cnt = bt.binom(K - 1 - x, d - 1 - i);
            if (r < cnt) { pos[i] = x; ++x; break; }
            r -= cnt; ++x;
        }
    }
}

// Transposed (word-major) copy of the ACTIVE generators, built once per level and shared
// read-only across the worker threads: word w of row r of generator g at
// Gt[(g*stride + w)*K + r]. A few KB; makes the inner sweep's word-0 reads contiguous.
// O(num_gamma*K*stride) -- negligible beside the C(K,d) enumeration.
inline std::vector<u64> transpose_gammas(const EnumPlan& plan) {
    const int K = plan.K, stride = plan.stride, ng = plan.num_gamma;
    std::vector<u64> Gt((size_t)ng * stride * K);
    for (int g = 0; g < ng; ++g)
        for (int r = 0; r < K; ++r)
            for (int w = 0; w < stride; ++w)
                Gt[((size_t)g * stride + w) * K + r] =
                    plan.gamma[((size_t)g * K + r) * stride + w];
    return Gt;
}

// ---- the sinks --------------------------------------------------------------------

// distfind: test the logical detector, keep the minimum logical weight (prunes as it goes).
struct MinLogicalSink {
    int best = WEIGHT_NONE;
    inline int bound() const { return best; }
    inline void hit(const EnumPlan& p, const u64* Gp, const u64* cwb, int r, int wt) {
        bool logical = (p.kcheck == 0);
        for (int c = 0; c < p.kcheck && !logical; ++c) {
            u64 acc = 0;
            const u64* cr = p.check + (size_t)c * p.stride;
            for (int w = 0; w < p.stride; ++w)
                acc ^= cr[w] & (cwb[w] ^ Gp[(size_t)w * p.K + r]);
            if (popcount64(acc) & 1) logical = true;
        }
        if (logical) best = wt;
    }
};

// codeaut: materialize and append every nonzero codeword of weight <= keep (dups fine;
// the caller dedups).
struct CollectSink {
    int keep = 0;
    std::vector<u64> hits;
    inline int bound() const { return keep + 1; }
    inline void hit(const EnumPlan& p, const u64* Gp, const u64* cwb, int r, int wt) {
        if (wt == 0) return;   // XOR of a dependent subset: not a collectible codeword
        const size_t base = hits.size();
        hits.resize(base + p.stride);
        for (int w = 0; w < p.stride; ++w)
            hits[base + w] = cwb[w] ^ Gp[(size_t)w * p.K + r];
    }
};

// ---- the kernel -------------------------------------------------------------------

// Enumerate the combinations [start, end) of the K message rows for every active
// generator, feeding every candidate lighter than sink.bound() to sink.hit().
template <class Sink>
void enumerate_chunk(const EnumPlan& p, const u64* Gt, u64 start, u64 end, Sink& sink) {
    BinomView bt{p.binom, p.binom_maxK};
    const int K = p.K, stride = p.stride;
    const int D = p.d, M = D - 1;          // D==1 => M==0 (empty outer, cw base = 0)
    int bound = sink.bound();

    std::vector<int> pos(D);
    std::vector<u64> cw(stride);           // codeword base = XOR of the M outer rows

    for (int g = 0; g < p.num_gamma; ++g) {
        const u64* Gp  = Gt + (size_t)g * stride * K;  // transposed generator g
        const u64* Gw0 = Gp;                           // word-0 plane (the hot reads)

        // Unrank `start` into the weight-d combination pos[0..D-1].
        unrank_flat(bt, K, D, start, pos.data());

        // cw = XOR of the M outer rows pos[0..M-1] (empty XOR = 0 when D==1).
        for (int w = 0; w < stride; ++w) cw[w] = 0;
        for (int i = 0; i < M; ++i)
            for (int w = 0; w < stride; ++w) cw[w] ^= Gp[(size_t)w * K + pos[i]];

        int last = pos[D - 1];             // inner start (may be mid-run for a chunk offset)
        u64 cnt = 0, need = end - start;
        while (cnt < need) {
            // Hot inner loop: uniform sweep of `last`, codeword = cw ^ row[last]. `hi` caps
            // the run at the chunk end (need) or the inner range end (K).
            int last_start = last;
            int hi = K;
            if ((u64)(K - last) > need - cnt) hi = last + (int)(need - cnt);
            const u64 cw0 = cw[0];

            // x4-unrolled fast path: four independent word-0 popcounts issue together and
            // hide read/popcount latency; only a candidate lighter than the bound in word 0
            // drops to the per-element slow path (full weight + the sink's accept logic).
            int lst4 = hi - 3;
            for (; last < lst4; last += 4) {
                int a0 = popcount64(cw0 ^ Gw0[last]);      // contiguous word-0 reads
                int a1 = popcount64(cw0 ^ Gw0[last + 1]);
                int a2 = popcount64(cw0 ^ Gw0[last + 2]);
                int a3 = popcount64(cw0 ^ Gw0[last + 3]);
                if (a0 < bound || a1 < bound || a2 < bound || a3 < bound) {
                    int aa[4] = {a0, a1, a2, a3};
                    for (int u = 0; u < 4; ++u) {
                        if (aa[u] >= bound) continue;
                        int r = last + u;
                        int wt = aa[u]; bool light = true;
                        for (int w = 1; w < stride && light; ++w) {
                            wt += popcount64(cw[w] ^ Gp[(size_t)w * K + r]);
                            light = wt < bound;
                        }
                        if (!light) continue;
                        sink.hit(p, Gp, cw.data(), r, wt);
                        bound = sink.bound();
                    }
                }
            }
            for (; last < hi; ++last) {     // tail (< 4 elements)
                int r = last;
                int wt = popcount64(cw0 ^ Gw0[last]);
                bool light = wt < bound;
                for (int w = 1; w < stride && light; ++w) {
                    wt += popcount64(cw[w] ^ Gp[(size_t)w * K + r]);
                    light = wt < bound;
                }
                if (!light) continue;
                sink.hit(p, Gp, cw.data(), r, wt);
                bound = sink.bound();
            }

            cnt += (u64)(hi - last_start); // combos processed in this inner run
            if (cnt >= need) break;        // chunk end reached (hi was the cap)
            // Advance the M outer positions to the next combination, updating cw in place.
            int j = M - 1;
            while (j >= 0 && pos[j] == K - D + j) --j;
            if (j < 0) break;
            for (int t = j; t < M; ++t)
                for (int w = 0; w < stride; ++w) cw[w] ^= Gp[(size_t)w * K + pos[t]];
            ++pos[j];
            for (int t = j + 1; t < M; ++t) pos[t] = pos[t - 1] + 1;
            for (int t = j; t < M; ++t)
                for (int w = 0; w < stride; ++w) cw[w] ^= Gp[(size_t)w * K + pos[t]];
            last = pos[M - 1] + 1;         // inner restart for the fresh outer combination
        }
    }
}

// ---- the threaded level driver ----------------------------------------------------

// Run one weight level across sinks.size() threads (1 => inline on the calling thread),
// splitting [0, C(K,d)) proportionally. Each thread owns sinks[t]; the caller merges.
template <class Sink>
void enumerate_level(const EnumPlan& plan, const u64* Gt, u64 total,
                     std::vector<Sink>& sinks) {
    const int T = (int)sinks.size();
    if (total == 0 || T == 0) return;
    if (T == 1) { enumerate_chunk(plan, Gt, 0, total, sinks[0]); return; }

    std::vector<std::thread> pool;
    pool.reserve(T);
    for (int t = 0; t < T; ++t) {
        u64 s = (u64)((__uint128_t)total * t / T);
        u64 e = (u64)((__uint128_t)total * (t + 1) / T);
        pool.emplace_back([&, t, s, e]() { enumerate_chunk(plan, Gt, s, e, sinks[t]); });
    }
    for (auto& th : pool) th.join();
}

// Thread count the CPU backend should use for a level of `total` combinations:
// the configured/hardware count, capped by the work, and 1 for tiny problems.
int cpu_level_threads(u64 total);

} // namespace qsf
