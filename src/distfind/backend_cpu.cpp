#include "distfind/backend.hpp"
#include "distfind/combinatorics.hpp"
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>

namespace distfind {

static std::atomic<int> g_cpu_threads{0};
void set_cpu_threads(int n) { g_cpu_threads.store(n); }

namespace {

// Reconstruct a lightweight BinomTable view over the plan's flat binomial buffer so we
// can reuse unrank_comb without copying.
struct BinomView {
    const u64* c; int maxK;
    inline u64 binom(int n, int k) const {
        if (k < 0 || n < 0 || k > n) return 0ull;
        return c[(size_t)n * (maxK + 1) + k];
    }
};

inline void unrank(const BinomView& bt, int K, int d, u64 r, int* pos) {
    int x = 0;
    for (int i = 0; i < d; ++i) {
        for (;;) {
            u64 cnt = bt.binom(K - 1 - x, d - 1 - i);
            if (r < cnt) { pos[i] = x; ++x; break; }
            r -= cnt; ++x;
        }
    }
}

// Enumerate the combinations [start, end) of the K message rows for every active
// generator, returning the smallest full weight of a LOGICAL codeword strictly lighter
// than the plan's current_best (else current_best). This is the CPU port of the Metal
// two-level kernel (src/metal/backend_metal.mm): the exact same math, so the CPU stays the
// GPU's correctness oracle (guarded by src/tests/cpu_enum_ref.cpp).
//
//  * TWO-LEVEL enumeration: the weight-d combination splits into m = d-1 OUTER positions
//    pos[0..m-1] (advanced rarely, with the codeword base cw maintained by XOR-in-place)
//    and one INNER index `last` swept in a tight loop where the codeword is cw ^ row[last].
//  * WEIGHT-FIRST early exit: the codeword weight is computed first with a per-word exit
//    against `best` (a codeword with weight >= best can never improve the bound); the
//    expensive logical-detector check runs only for the rare candidate lighter than best.
//  * `Gt` holds the active generators TRANSPOSED (word-major: word w of row r of generator
//    g at Gt[(g*stride + w)*K + r]) so the inner sweep's word-0 reads of consecutive rows
//    are contiguous -- the x4-unrolled hot loop then auto-vectorizes (NEON XOR + popcount).
int worker(const EnumPlan& p, const u64* Gt, u64 start, u64 end) {
    BinomView bt{p.binom, p.binom_maxK};
    const int K = p.K, stride = p.stride;
    const int D = p.d, M = D - 1;          // D==1 => M==0 (empty outer, cw base = 0)
    int best = p.current_best;

    std::vector<int> pos(D);
    std::vector<u64> cw(stride);           // codeword base = XOR of the M outer rows

    for (int g = 0; g < p.num_gamma; ++g) {
        const u64* Gp  = Gt + (size_t)g * stride * K;  // transposed generator g
        const u64* Gw0 = Gp;                           // word-0 plane (the hot reads)

        // Unrank `start` into the weight-d combination pos[0..D-1].
        unrank(bt, K, D, start, pos.data());

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
            // hide read/popcount latency; only a candidate lighter than best in word 0 drops
            // to the per-element slow path (full weight + logical check).
            int lst4 = hi - 3;
            for (; last < lst4; last += 4) {
                int a0 = popcount64(cw0 ^ Gw0[last]);      // contiguous word-0 reads
                int a1 = popcount64(cw0 ^ Gw0[last + 1]);
                int a2 = popcount64(cw0 ^ Gw0[last + 2]);
                int a3 = popcount64(cw0 ^ Gw0[last + 3]);
                if (a0 < best || a1 < best || a2 < best || a3 < best) {
                    int aa[4] = {a0, a1, a2, a3};
                    for (int u = 0; u < 4; ++u) {
                        if (aa[u] >= best) continue;
                        int r = last + u;
                        int wt = aa[u]; bool light = true;
                        for (int w = 1; w < stride && light; ++w) {
                            wt += popcount64(cw[w] ^ Gp[(size_t)w * K + r]);
                            light = wt < best;
                        }
                        if (!light) continue;
                        bool logical = (p.kcheck == 0);
                        for (int c = 0; c < p.kcheck && !logical; ++c) {
                            u64 acc = 0;
                            const u64* cr = p.check + (size_t)c * stride;
                            for (int w = 0; w < stride; ++w)
                                acc ^= cr[w] & (cw[w] ^ Gp[(size_t)w * K + r]);
                            if (popcount64(acc) & 1) logical = true;
                        }
                        if (logical) best = wt;
                    }
                }
            }
            for (; last < hi; ++last) {     // tail (< 4 elements)
                int r = last;
                int wt = popcount64(cw0 ^ Gw0[last]);
                bool light = wt < best;
                for (int w = 1; w < stride && light; ++w) {
                    wt += popcount64(cw[w] ^ Gp[(size_t)w * K + r]);
                    light = wt < best;
                }
                if (!light) continue;
                bool logical = (p.kcheck == 0);
                for (int c = 0; c < p.kcheck && !logical; ++c) {
                    u64 acc = 0;
                    const u64* cr = p.check + (size_t)c * stride;
                    for (int w = 0; w < stride; ++w)
                        acc ^= cr[w] & (cw[w] ^ Gp[(size_t)w * K + r]);
                    if (popcount64(acc) & 1) logical = true;
                }
                if (logical) best = wt;
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
    return best;
}

struct CpuBackend : Backend {
    std::string name() const override { return "cpu"; }
    bool available() const override { return true; }

    int enumerate(const EnumPlan& plan) override {
        BinomView bt{plan.binom, plan.binom_maxK};
        u64 total = bt.binom(plan.K, plan.d);
        if (total == 0) return WEIGHT_NONE;

        // Transposed (word-major) copy of the ACTIVE generators, built once per level and
        // shared read-only across the worker threads: word w of row r of generator g at
        // Gt[(g*stride + w)*K + r]. A few KB; makes the inner sweep's word-0 reads
        // contiguous. O(num_gamma*K*stride) -- negligible beside the C(K,d) enumeration.
        const int K = plan.K, stride = plan.stride, ng = plan.num_gamma;
        std::vector<u64> Gt((size_t)ng * stride * K);
        for (int g = 0; g < ng; ++g)
            for (int r = 0; r < K; ++r)
                for (int w = 0; w < stride; ++w)
                    Gt[((size_t)g * stride + w) * K + r] =
                        plan.gamma[((size_t)g * K + r) * stride + w];

        int T = g_cpu_threads.load();
        if (T <= 0) T = std::max(1u, std::thread::hardware_concurrency());
        // Don't spawn more threads than work, and avoid threads for tiny problems.
        if (total < 4096) T = 1;
        if ((u64)T > total) T = (int)total;

        if (T == 1) return worker(plan, Gt.data(), 0, total);

        std::vector<int> results(T, WEIGHT_NONE);
        std::vector<std::thread> pool;
        pool.reserve(T);
        for (int t = 0; t < T; ++t) {
            u64 s = (u64)((__uint128_t)total * t / T);
            u64 e = (u64)((__uint128_t)total * (t + 1) / T);
            pool.emplace_back([&, t, s, e]() { results[t] = worker(plan, Gt.data(), s, e); });
        }
        for (auto& th : pool) th.join();
        return *std::min_element(results.begin(), results.end());
    }
};

} // namespace

Backend* cpu_backend() {
    static CpuBackend inst;
    return &inst;
}

// Overridden by the Metal / CUDA translation units when those are compiled in.
__attribute__((weak)) Backend* metal_backend() { return nullptr; }
__attribute__((weak)) Backend* cuda_backend() { return nullptr; }

Backend* select_backend(const std::string& name) {
    auto ok = [](Backend* b) { return b && b->available(); };
    if (name == "cpu")  return cpu_backend();
    if (name == "metal") { Backend* b = metal_backend(); return ok(b) ? b : cpu_backend(); }
    if (name == "cuda")  { Backend* b = cuda_backend();  return ok(b) ? b : cpu_backend(); }
    // gpu / auto: pick the machine-specific accelerator automatically.
    if (Backend* b = cuda_backend();  ok(b)) return b;
    if (Backend* b = metal_backend(); ok(b)) return b;
    return cpu_backend();
}

} // namespace distfind
