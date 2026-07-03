// Oracle-independence guard for the CPU BZ enumerator.
//
// The CPU backend is the correctness oracle for the GPU (src/tests/backend_compare.cpp).
// Since the CPU enumerate() was restructured to run the SAME two-level algorithm as the
// Metal/CUDA kernels (transposed generators, weight-first early exit, inner-index sweep),
// it can no longer be trusted purely by construction. This test pins it against a
// straightforward INDEPENDENT reference -- the old, obviously-correct pattern (unrank the
// combination, walk it with next_comb, and for each combination x generator call the shared
// eval_combo, which does the logical-detector check THEN the weight) -- compiled right here.
//
// It generates a spread of randomized small EnumPlans (varying K, stride, weight level d,
// generator count, detector count, and current_best including WEIGHT_NONE / no cap) and one
// multithreaded batch big enough to exercise the [start,end) chunk-boundary splitting. Every
// plan's new enumerate() result must equal the reference exactly.
#include "qubitserf/backend.hpp"
#include "qubitserf/combinatorics.hpp"
#include <random>
#include <vector>
#include <cstdio>
#include <algorithm>

using namespace qubitserf;

// Straightforward reference: unrank 0, walk all C(K,d) combinations with next_comb, and for
// each (combination, generator) run the shared eval_combo (logical-check-THEN-weight), taking
// the min against current_best. Deliberately the simplest possible full enumeration.
static int reference_enumerate(const EnumPlan& p, const BinomTable& bt) {
    u64 total = bt.binom(p.K, p.d);
    if (total == 0) return WEIGHT_NONE;
    std::vector<int> pos(p.d);
    std::vector<u64> scratch(p.stride);
    unrank_comb(bt, p.K, p.d, 0, pos.data());
    int best = p.current_best;
    for (u64 it = 0; it < total; ++it) {
        for (int g = 0; g < p.num_gamma; ++g) {
            int w = eval_combo(p, g, pos.data(), scratch.data());
            if (w < best) best = w;
        }
        if (!next_comb(p.K, p.d, pos.data())) break;
    }
    return best;
}

static int failures = 0;
static int ntests = 0;

// Build a plan from freshly-randomized generator/check data and assert new == reference.
static void run_case(std::mt19937_64& rng, int K, int stride, int d, int ng, int kcheck,
                     int current_best) {
    d = std::max(1, std::min(d, K));
    ng = std::max(1, ng);
    kcheck = std::max(0, kcheck);

    std::vector<u64> gamma((size_t)ng * K * stride);
    for (auto& v : gamma) v = rng();
    std::vector<u64> check((size_t)kcheck * stride);
    for (auto& v : check) v = rng();
    BinomTable bt(K, K);

    EnumPlan p;
    p.n = stride * 64; p.stride = stride; p.K = K; p.d = d;
    p.num_gamma = ng; p.num_gamma_total = ng;
    p.gamma = gamma.data();
    p.kcheck = kcheck; p.check = check.data();
    p.binom = bt.c.data(); p.binom_maxN = bt.maxN; p.binom_maxK = bt.maxK;
    p.current_best = current_best;
    p.buffers_key = (u64)(++ntests);

    int got  = cpu_backend()->enumerate(p);
    int want = reference_enumerate(p, bt);
    if (got != want) {
        ++failures;
        std::printf("  [FAIL] K=%d stride=%d d=%d ng=%d kcheck=%d cur=%d  got=%d want=%d\n",
                    K, stride, d, ng, kcheck, current_best, got, want);
    }
}

int main() {
    std::mt19937_64 rng(0xC0FFEEULL);
    auto rint = [&](int lo, int hi) { return lo + (int)(rng() % (u64)(hi - lo + 1)); };
    auto rbest = [&](int stride) {
        // A spread of caps: WEIGHT_NONE (no cap), a tight cap, and a mid-range cap.
        int roll = rint(0, 3);
        if (roll == 0) return (int)WEIGHT_NONE;
        return rint(0, stride * 64);
    };

    // ---- batch 1: single-threaded coverage (K in 4..14, stride 1..3, d in 1..5) --------
    for (int i = 0; i < 50; ++i) {
        int K = rint(4, 14);
        int stride = rint(1, 3);
        int d = rint(1, 5);
        int ng = rint(1, 3);
        int kcheck = rint(0, 3);
        run_case(rng, K, stride, d, ng, kcheck, rbest(stride));
    }

    // ---- explicit edge cases -----------------------------------------------------------
    run_case(rng, 8,  1, 1, 1, 0, WEIGHT_NONE);   // d==1  => M==0 (empty outer base)
    run_case(rng, 6,  2, 1, 2, 2, WEIGHT_NONE);   // d==1, multi-gamma
    run_case(rng, 5,  1, 5, 1, 1, WEIGHT_NONE);   // d==K  => single combination
    run_case(rng, 7,  2, 4, 2, 0, WEIGHT_NONE);   // kcheck==0 => every codeword logical
    run_case(rng, 10, 3, 3, 1, 2, 0);             // current_best==0 => everything capped
    run_case(rng, 12, 1, 4, 3, 1, 1);             // very tight cap

    // ---- batch 2: multithreaded chunk-boundary stress (total >= 4096 so enumerate splits
    //      into per-thread [start,end) ranges; mid-run inner starts exercise the boundary
    //      math). K deliberately raised so the totals cross the threading threshold. --------
    set_cpu_threads(16);
    for (int i = 0; i < 20; ++i) {
        int K = rint(15, 22);
        int stride = rint(1, 3);
        int d = rint(4, 6);
        int ng = rint(1, 3);
        int kcheck = rint(0, 3);
        // Ensure the total actually crosses the multithread threshold; else it runs on one
        // thread and adds no boundary coverage (still correct, just less interesting).
        BinomTable probe(K, K);
        if (probe.binom(K, d) < 4096) d = std::min(K, d + 2);
        run_case(rng, K, stride, d, ng, kcheck, rbest(stride));
    }
    // A couple of pinned large-total cases with explicit thread counts.
    for (int T : {2, 3, 8, 16}) {
        set_cpu_threads(T);
        run_case(rng, 18, 2, 5, 2, 2, WEIGHT_NONE);  // C(18,5)=8568
        run_case(rng, 16, 3, 6, 1, 1, 40);           // C(16,6)=8008, with a cap
    }
    set_cpu_threads(0);

    std::printf("cpu_enum_ref: %d plans, %d failures\n", ntests, failures);
    std::printf("%s\n", failures ? "CPU ENUM REF FAILED" : "CPU ENUM REF: ALL PLANS AGREE");
    return failures ? 1 : 0;
}
