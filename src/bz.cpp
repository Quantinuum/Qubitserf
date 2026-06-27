#include "qminweight/bz.hpp"
#include "qminweight/combinatorics.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <thread>

namespace qminweight {

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

    static const bool profile = std::getenv("QMINWEIGHT_PROFILE") != nullptr;

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

        if (opt.verbose)
            std::fprintf(stderr, "[bz %s] d=%d upper=%d lower=%d\n",
                         res.backend.c_str(), d, inner, outer);

        if (inner <= outer) { res.proven = true; break; }
    }

    res.distance = (inner >= WEIGHT_NONE) ? -1 : inner;
    res.levels = level;
    res.lower_bound = outer;
    res.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    return res;
}

BZResult bz_css_distance(const GF2Mat& Hx, const GF2Mat& Hz, const BZOptions& opt) {
    BZResult z = bz_distance(css_problem(Hx, Hz, 'Z'), opt);
    BZResult x = bz_distance(css_problem(Hx, Hz, 'X'), opt);
    BZResult r = z;
    // combine: distance = min of the two valid distances
    auto val = [](int d) { return d < 0 ? WEIGHT_NONE : d; };
    if (val(x.distance) < val(z.distance)) r = x;
    r.distance = std::min(val(z.distance), val(x.distance));
    if (r.distance >= WEIGHT_NONE) r.distance = -1;
    r.seconds = z.seconds + x.seconds;
    r.proven = z.proven && x.proven;
    return r;
}

} // namespace qminweight
