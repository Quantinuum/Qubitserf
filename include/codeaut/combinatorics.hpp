// Combination enumeration in the combinatorial number system (CNS): unrank a global index
// into the corresponding k-subset of {0..m-1} (lexicographic order), and step to the next
// subset.  Used to split the C(m,k) enumeration across threads (CPU) and GPU lanes.
#ifndef CODEAUT_COMBINATORICS_HPP
#define CODEAUT_COMBINATORICS_HPP

#include <cstdint>
#include <vector>

namespace codeaut {

// Binomial table C[i][j] for 0<=i<=m, 0<=j<=kmax (saturating at INT64_MAX to avoid overflow).
struct BinomTable {
    int m, kmax;
    std::vector<int64_t> c;   // (m+1)*(kmax+1)
    BinomTable(int m_, int kmax_) : m(m_), kmax(kmax_), c((size_t)(m_ + 1) * (kmax_ + 1), 0) {
        const int64_t CAP = (int64_t)1 << 62;
        for (int i = 0; i <= m; ++i) {
            at(i, 0) = 1;
            for (int j = 1; j <= kmax && j <= i; ++j) {
                int64_t a = at(i - 1, j - 1), b = (j <= i - 1) ? at(i - 1, j) : 0;
                int64_t s = a + b;
                at(i, j) = (s < 0 || s > CAP) ? CAP : s;
            }
        }
    }
    int64_t& at(int i, int j) { return c[(size_t)i * (kmax + 1) + j]; }
    int64_t get(int i, int j) const {
        if (j < 0 || j > kmax || j > i) return 0;
        return c[(size_t)i * (kmax + 1) + j];
    }
};

// Unrank `idx` (0-based, lexicographic) into the k-subset `out[0..k-1]` of {0..m-1}, ascending.
inline void comb_unrank(int64_t idx, int m, int k, const BinomTable& B, int* out) {
    int x = 0;
    for (int i = 0; i < k; ++i) {
        // choose the smallest next element so that the remaining count still covers idx
        while (true) {
            int64_t cnt = B.get(m - 1 - x, k - 1 - i);
            if (idx < cnt) break;
            idx -= cnt;
            ++x;
        }
        out[i] = x;
        ++x;
    }
}

// Advance `comb[0..k-1]` (an ascending k-subset of {0..m-1}) to the next one in lexicographic
// order; returns false if `comb` was the last subset.
inline bool comb_next(int* comb, int m, int k) {
    int i = k - 1;
    while (i >= 0 && comb[i] == m - k + i) --i;
    if (i < 0) return false;
    ++comb[i];
    for (int j = i + 1; j < k; ++j) comb[j] = comb[j - 1] + 1;
    return true;
}

}  // namespace codeaut
#endif
