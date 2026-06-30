#include "qubitserf/bz.hpp"
#include "qubitserf/combinatorics.hpp"
#include "qubitserf/progress.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <thread>

namespace qubitserf {

// Random-information-set upper bound (the QDistRnd heuristic), used only to SEED the
// exact BZ bound: a tighter starting upper bound lets the proof terminate many weight
// levels earlier. Any logical codeword's weight is a valid upper bound, so this never
// affects correctness -- only speed. For each random column order we put code_gen in
// systematic form; every resulting row is a codeword, and the lightest logical row bounds
// the distance.
//
// Each trial costs an O(n*K^2) RREF. A tight seed is what keeps the BZ enumeration small,
// so we run a generous, FIXED number of trials -- but the trials are independent, so we
// run them in parallel across cores to keep the wall-time cost low.
static int random_is_seed(const DistProblem& prob, int init_best, int trials, unsigned seed) {
    const int n = prob.n;
    const int K = prob.code_gen.rows;
    const int stride = prob.code_gen.stride;
    if (K == 0) return init_best;

    int nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;
    if (nthreads > trials) nthreads = trials;

    auto run = [&](int lo, int hi, unsigned tseed) -> int {
        std::mt19937 rng(tseed);
        std::vector<int> cols(n);
        int best = init_best;
        for (int it = lo; it < hi; ++it) {
            std::iota(cols.begin(), cols.end(), 0);
            std::shuffle(cols.begin(), cols.end(), rng);
            GF2Mat M = prob.code_gen;
            restricted_rref(M, cols);
            for (int r = 0; r < M.rows; ++r) {
                int w = M.row_weight(r);
                if (w == 0 || w >= best) continue;
                bool logical = prob.count_all;
                for (int c = 0; c < prob.check.rows && !logical; ++c)
                    if (vec_dot(prob.check.row(c), M.row(r), stride)) logical = true;
                if (logical) best = w;
            }
        }
        return best;
    };

    if (nthreads == 1) return run(0, trials, seed);

    std::vector<int> results(nthreads, init_best);
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) {
        int lo = (int)((long)trials * t / nthreads);
        int hi = (int)((long)trials * (t + 1) / nthreads);
        pool.emplace_back([&, t, lo, hi]() {
            results[t] = run(lo, hi, seed + 0x9e3779b9u * (unsigned)(t + 1));
        });
    }
    for (auto& th : pool) th.join();
    int best = init_best;
    for (int r : results) best = std::min(best, r);
    return best;
}

BZResult bz_distance(const DistProblem& prob, const BZOptions& opt) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    BZResult res;
    Backend* backend = select_backend(opt.backend);
    res.backend = backend->name();

    const int n = prob.n;
    const int K = prob.code_gen.rows;
    const int stride = words_for(n);

    // No logicals / empty code => distance undefined (report -1).
    if (K == 0 || (!prob.count_all && prob.check.rows == 0)) {
        res.distance = -1;
        res.seconds = std::chrono::duration<double>(clk::now() - t0).count();
        return res;
    }

    // Build and flat-pack the information-set generators.
    std::vector<Gamma> seq = gamma_sequence(prob.code_gen);
    const int num_gamma = (int)seq.size();
    std::vector<u64> gamma((size_t)num_gamma * K * stride, 0ull);
    std::vector<int> ranks(num_gamma);
    for (int g = 0; g < num_gamma; ++g) {
        ranks[g] = seq[g].rank;
        const GF2Mat& M = seq[g].g;
        for (int r = 0; r < K; ++r)
            for (int w = 0; w < stride; ++w)
                gamma[((size_t)g * K + r) * stride + w] = M.row(r)[w];
    }

    const int kcheck = prob.count_all ? 0 : prob.check.rows;
    std::vector<u64> check((size_t)kcheck * stride, 0ull);
    for (int c = 0; c < kcheck; ++c)
        for (int w = 0; w < stride; ++w)
            check[(size_t)c * stride + w] = prob.check.row(c)[w];

    BinomTable bt(K, K);

    static std::atomic<u64> solve_counter{0};
    EnumPlan plan;
    plan.n = n; plan.stride = stride; plan.K = K;
    plan.num_gamma = num_gamma; plan.gamma = gamma.data();
    plan.kcheck = kcheck; plan.check = check.data();
    plan.binom = bt.c.data(); plan.binom_maxN = bt.maxN; plan.binom_maxK = bt.maxK;
    plan.buffers_key = ++solve_counter;   // unique per solve -> GPU caches stay coherent

    int inner = WEIGHT_NONE;                       // upper bound (best found)
    if (prob.seed_upper > 0) inner = prob.seed_upper;
    // Tighten the starting bound with (parallel) random information sets.
    inner = random_is_seed(prob, inner, /*trials=*/256, 0x9e3779b9u);
    int outer = 0;                                 // lower bound
    const int maxw = opt.max_weight > 0 ? std::min(K, opt.max_weight) : K;

    static const bool profile = std::getenv("QUBITSERF_PROFILE") != nullptr;

    int level = 0;
    for (int d = 1; d <= maxw; ++d) {
        plan.d = d;
        plan.current_best = inner;
        auto te0 = clk::now();
        int found = backend->enumerate(plan);
        if (profile) {
            double ms = std::chrono::duration<double, std::milli>(clk::now() - te0).count();
            BinomTable& b = bt;
            std::fprintf(stderr,
                "[prof %s] d=%d K=%d ng=%d stride=%d total=%llu work=%llu -> %.3f ms\n",
                res.backend.c_str(), d, K, num_gamma, stride,
                (unsigned long long)b.binom(K, d),
                (unsigned long long)b.binom(K, d) * (unsigned long long)num_gamma, ms);
        }
        if (found < inner) inner = found;
        ++level;

        outer = 0;
        for (int g = 0; g < num_gamma; ++g)
            outer += std::max(0, (d + 1) - (K - ranks[g]));
        if (prob.even) outer += (outer & 1);       // distance is even => round up

        if (inner <= outer) {
            res.proven = true;
            if (opt.verbose)
                verbose_final(inner, std::chrono::duration<double>(clk::now() - t0).count());
            break;
        }

        if (opt.verbose)
            verbose_bound(std::max(0, outer - 1),
                          std::chrono::duration<double>(clk::now() - te0).count());
    }

    res.distance = (inner >= WEIGHT_NONE) ? -1 : inner;
    res.levels = level;
    res.lower_bound = outer;
    res.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    return res;
}

// ---- interleaved CSS min over two subproblems -----------------------------------
//
// Running Z to completion then X starves X of any bound if Z stalls on a hard weight
// level. Instead we advance BOTH subproblems one weight level at a time, so their lower
// bounds rise in step. Once one side's distance is proven (or once a side's lower bound
// reaches the best distance found so far, so it cannot lower the min) that side is capped.
// The result is the exact min(dZ, dX). Verbose lines are tagged "Z"/"X".
namespace {

// One BZ subproblem's state, stepped one weight level at a time (the body of the
// bz_distance loop). Held in place (its EnumPlan points into its own vectors), so a
// BzSide must not be copied or moved after setup().
struct BzSide {
    bool has_logicals = false;
    int n = 0, K = 0, stride = 0, maxw = 0, num_gamma = 0, kcheck = 0;
    bool even = false;
    std::vector<u64> gamma, check;
    std::vector<int> ranks;
    BinomTable bt;
    EnumPlan plan{};
    Backend* backend = nullptr;
    const char* tag = "";

    int inner = WEIGHT_NONE;   // upper bound (best logical weight found)
    int outer = 0;             // lower bound (rises with the weight level)
    int level = 0;
    bool proven = false;       // inner <= outer reached
    bool resolved = false;     // proven OR capped (cannot affect the min)

    BzSide() = default;
    BzSide(const BzSide&) = delete;
    BzSide& operator=(const BzSide&) = delete;

    void setup(const DistProblem& prob, Backend* be, const BZOptions& opt, const char* t) {
        backend = be; tag = t;
        n = prob.n; K = prob.code_gen.rows; stride = words_for(n);
        even = prob.even;
        has_logicals = !(K == 0 || (!prob.count_all && prob.check.rows == 0));
        if (!has_logicals) { resolved = true; return; }

        std::vector<Gamma> seq = gamma_sequence(prob.code_gen);
        num_gamma = (int)seq.size();
        gamma.assign((size_t)num_gamma * K * stride, 0ull);
        ranks.resize(num_gamma);
        for (int g = 0; g < num_gamma; ++g) {
            ranks[g] = seq[g].rank;
            const GF2Mat& M = seq[g].g;
            for (int r = 0; r < K; ++r)
                for (int w = 0; w < stride; ++w)
                    gamma[((size_t)g * K + r) * stride + w] = M.row(r)[w];
        }
        kcheck = prob.count_all ? 0 : prob.check.rows;
        check.assign((size_t)kcheck * stride, 0ull);
        for (int c = 0; c < kcheck; ++c)
            for (int w = 0; w < stride; ++w)
                check[(size_t)c * stride + w] = prob.check.row(c)[w];

        bt.build(K, K);
        static std::atomic<u64> solve_counter{0};
        plan.n = n; plan.stride = stride; plan.K = K;
        plan.num_gamma = num_gamma; plan.gamma = gamma.data();
        plan.kcheck = kcheck; plan.check = check.data();
        plan.binom = bt.c.data(); plan.binom_maxN = bt.maxN; plan.binom_maxK = bt.maxK;
        plan.buffers_key = ++solve_counter;

        inner = prob.seed_upper > 0 ? prob.seed_upper : WEIGHT_NONE;
        inner = random_is_seed(prob, inner, /*trials=*/256, 0x9e3779b9u);
        maxw = opt.max_weight > 0 ? std::min(K, opt.max_weight) : K;
    }

    // Enumerate weight level d; update inner (upper) and outer (lower).
    void step(int d) {
        plan.d = d; plan.current_best = inner;
        int found = backend->enumerate(plan);
        if (found < inner) inner = found;
        ++level;
        outer = 0;
        for (int g = 0; g < num_gamma; ++g)
            outer += std::max(0, (d + 1) - (K - ranks[g]));
        if (even) outer += (outer & 1);
    }
};

} // namespace

namespace {

// Assemble a single side's BZResult from its stepper state.
BZResult assemble_bz_side(const BzSide& S, double seconds, const std::string& backend) {
    BZResult r;
    r.backend = backend;
    r.distance = (S.has_logicals && S.inner < WEIGHT_NONE) ? S.inner : -1;
    r.lower_bound = S.has_logicals ? S.outer : 0;
    r.levels = S.level;
    r.proven = S.proven;
    r.seconds = seconds;
    return r;
}

// Step Z and X one weight level at a time. With cap_to_min, a side stops as soon as its
// proven lower bound reaches the best distance found so far, so it cannot lower the min
// (so e.g. once dZ=2 is found, X stops the moment it has ruled out X<2 -- no wasted proof
// of X>2). WITHOUT cap_to_min (the --zx case), there is NO cross-side cap: each side runs
// to its own full proof, so finding Z never stops X's distance from being found -- the
// interleaving only keeps both lower bounds advancing together. Verbose lines are tagged.
void bz_interleave_core(BzSide& Z, BzSide& X, const BZOptions& opt, bool cap_to_min,
                        std::chrono::steady_clock::time_point t0) {
    using clk = std::chrono::steady_clock;
    BzSide* sides[2] = {&Z, &X};
    int best = WEIGHT_NONE;
    if (cap_to_min) {
        if (Z.has_logicals) best = std::min(best, Z.inner);
        if (X.has_logicals) best = std::min(best, X.inner);
    }
    const int maxw = std::max(Z.has_logicals ? Z.maxw : 0, X.has_logicals ? X.maxw : 0);

    for (int d = 1; d <= maxw; ++d) {
        for (BzSide* S : sides) {
            if (S->resolved) continue;
            if (d > S->maxw) continue;                                       // exhausted (unproven)
            if (cap_to_min && S->outer >= best) { S->resolved = true; continue; }  // can't beat min
            auto te0 = clk::now();
            S->step(d);
            if (cap_to_min) best = std::min(best, S->inner);
            if (S->inner <= S->outer) {
                S->proven = true; S->resolved = true;
                if (opt.verbose)
                    verbose_final(S->inner,
                                  std::chrono::duration<double>(clk::now() - t0).count(), S->tag);
            } else {
                if (opt.verbose)
                    verbose_bound(std::max(0, S->outer - 1),
                                  std::chrono::duration<double>(clk::now() - te0).count(), S->tag);
                if (cap_to_min && S->outer >= best) S->resolved = true;      // capped after this level
            }
        }
        if (Z.resolved && X.resolved) break;
    }
}

} // namespace

BZResult bz_min_interleaved(const DistProblem& pz, const DistProblem& px,
                            const BZOptions& opt) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    Backend* be = select_backend(opt.backend);
    BzSide Z, X;
    Z.setup(pz, be, opt, "Z");
    X.setup(px, be, opt, "X");
    bz_interleave_core(Z, X, opt, /*cap_to_min=*/true, t0);

    auto v = [](const BzSide& S) { return S.has_logicals ? S.inner : WEIGHT_NONE; };
    int dist = std::min(v(Z), v(X));
    BZResult r;
    r.backend = be->name();
    r.distance = (dist >= WEIGHT_NONE) ? -1 : dist;
    int lb = std::min(Z.has_logicals ? Z.outer : WEIGHT_NONE,
                      X.has_logicals ? X.outer : WEIGHT_NONE);
    r.lower_bound = (lb >= WEIGHT_NONE) ? 0 : lb;
    r.levels = std::max(Z.level, X.level);
    // The min is proven once both sides are resolved (proven, or capped so they cannot
    // lower the min): the side achieving `best` is then exact and the other is >= best.
    r.proven = Z.resolved && X.resolved && (dist < WEIGHT_NONE);
    r.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    return r;
}

std::pair<BZResult, BZResult> bz_zx_interleaved(const DistProblem& pz, const DistProblem& px,
                                                const BZOptions& opt) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    Backend* be = select_backend(opt.backend);
    BzSide Z, X;
    Z.setup(pz, be, opt, "Z");
    X.setup(px, be, opt, "X");
    bz_interleave_core(Z, X, opt, /*cap_to_min=*/false, t0);   // no cross-side cap: both full
    double secs = std::chrono::duration<double>(clk::now() - t0).count();
    return {assemble_bz_side(Z, secs, be->name()), assemble_bz_side(X, secs, be->name())};
}

BZResult bz_css_distance(const GF2Mat& Hx, const GF2Mat& Hz, const BZOptions& opt) {
    return bz_min_interleaved(css_problem(Hx, Hz, 'Z'), css_problem(Hx, Hz, 'X'), opt);
}

} // namespace qubitserf
