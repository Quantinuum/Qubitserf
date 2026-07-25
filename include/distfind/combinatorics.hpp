// Combinatorial number system: rank/unrank of fixed-weight subsets, the primitive
// that lets every GPU thread compute its own independent slice of work.
//
// Ordering: lexicographic on strictly-increasing position tuples (p0<p1<...<p_{d-1})
// drawn from [0, K). unrank(r) and advance() are mutually consistent so a thread can
// unrank its start index once and then step with advance().
#pragma once
#include <vector>
#include "distfind/bits.hpp"

namespace distfind {

constexpr u64 BINOM_INF = ~0ull; // saturating "infinity" for overflowing binomials

// Binomial coefficients C[n][k] for n<=maxN, k<=maxK, saturated at BINOM_INF.
struct BinomTable {
    int maxN = 0, maxK = 0;
    std::vector<u64> c; // (maxN+1)*(maxK+1)

    BinomTable() = default;
    BinomTable(int N, int K) { build(N, K); }

    void build(int N, int K) {
        maxN = N; maxK = K;
        c.assign((size_t)(N + 1) * (K + 1), 0ull);
        for (int n = 0; n <= N; ++n) {
            at(n, 0) = 1;
            int kmax = n < K ? n : K;
            for (int k = 1; k <= kmax; ++k) {
                u64 a = at(n - 1, k - 1);
                u64 b = (k <= K) ? at(n - 1, k) : 0ull;
                u64 s = (a == BINOM_INF || b == BINOM_INF || a > BINOM_INF - b)
                            ? BINOM_INF : a + b;
                at(n, k) = s;
            }
        }
    }
    inline u64&       at(int n, int k)       { return c[(size_t)n * (maxK + 1) + k]; }
    inline u64        at(int n, int k) const { return c[(size_t)n * (maxK + 1) + k]; }
    inline u64 binom(int n, int k) const {
        if (k < 0 || n < 0 || k > n) return 0ull;
        if (n > maxN || k > maxK) return BINOM_INF;
        return at(n, k);
    }
};

// Number of weight-d subsets of [0,K), saturated.
inline u64 num_combinations(const BinomTable& bt, int K, int d) { return bt.binom(K, d); }

// Unrank: fill pos[0..d-1] with the r-th (lexicographic) increasing d-subset of [0,K).
inline void unrank_comb(const BinomTable& bt, int K, int d, u64 r, int* pos) {
    int x = 0;
    for (int i = 0; i < d; ++i) {
        for (;;) {
            u64 cnt = bt.binom(K - 1 - x, d - 1 - i);
            if (r < cnt) { pos[i] = x; ++x; break; }
            r -= cnt; ++x;
        }
    }
}

// Advance pos[] to the lexicographic-next d-subset of [0,K). Returns false at end.
inline bool next_comb(int K, int d, int* pos) {
    int j = d - 1;
    while (j >= 0 && pos[j] == K - d + j) --j;
    if (j < 0) return false;
    ++pos[j];
    for (int t = j + 1; t < d; ++t) pos[t] = pos[t - 1] + 1;
    return true;
}

} // namespace distfind
