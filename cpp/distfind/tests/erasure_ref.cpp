#include "distfind/erasure_bound.hpp"
#include <cstdio>
#include <random>

using namespace distfind;

static bool oracle(const GF2Mat& H, const GF2Mat& L, unsigned region,
                   unsigned candidates, int extra, unsigned fixed = 0) {
    for (unsigned x = 1; x < (1u << H.cols); ++x) {
        if ((x & fixed) != fixed) continue;
        unsigned outside = x & ~(region | fixed);
        if ((outside & ~candidates) || __builtin_popcount(outside) > extra) continue;
        bool closed = true, logical = false;
        for (int j = 0; j < H.rows; ++j)
            if (__builtin_parity(x & (unsigned)H.row(j)[0])) { closed = false; break; }
        if (!closed) continue;
        for (int j = 0; j < L.rows; ++j)
            logical |= __builtin_parity(x & (unsigned)L.row(j)[0]) != 0;
        if (logical) return false;
    }
    return true;
}

int main() {
    std::mt19937 rng(6301430);
    int checked = 0, inconclusive = 0;
    for (int trial = 0; trial < 4300; ++trial) {
        bool wide = trial >= 4000;
        int n = wide ? 10 : 1 + rng() % 10;
        GF2Mat H(wide ? 65 + 64 * (trial % 5) : rng() % 9, n),
               L(wide ? 64 : rng() % 5, n);
        for (int j = 0; j < H.rows; ++j)
            for (int q = 0; q < n; ++q) H.set(j, q, rng() % 3 == 0);
        for (int j = 0; j < L.rows; ++j)
            for (int q = 0; q < n; ++q) L.set(j, q, rng() & 1);
        if (wide) {
            for (int j = 0; j < H.rows; ++j) {
                H.set(j, 8, H.get(j, 0) ^ H.get(j, 1));
                H.set(j, 9, H.get(j, 2) ^ H.get(j, 3));
            }
            for (int j = 0; j < L.rows; ++j) L.set(j, 8, L.get(j, 0) ^ L.get(j, 1));
            L.set(63, 9, 1 ^ L.get(63, 2) ^ L.get(63, 3));
        }
        ErasureData data(H, L);
        ErasureBound bound(data, true);
        for (int repeat = 0; repeat < 4; ++repeat) {
            unsigned region = rng() & ((1u << n) - 1);
            unsigned candidates = rng() & ((1u << n) - 1) & ~region;
            std::vector<int> R, C;
            for (int q = 0; q < n; ++q) {
                if ((region >> q) & 1) R.push_back(q);
                if ((candidates >> q) & 1) C.push_back(q);
            }
            for (int extra = 0; extra <= 2; ++extra) {
                bool got = bound.excludes(R, C, extra);
                bool want = oracle(H, L, region, candidates, extra);
                ++checked;
                if (got && !want) {
                    std::fprintf(stderr, "FALSE ERASURE PROOF trial=%d repeat=%d extra=%d\n",
                                 trial, repeat, extra);
                    return 1;
                }
                inconclusive += !got && want;
                unsigned fixed = rng() & ((1u << n) - 1) & ~(region | candidates);
                std::vector<int> F;
                for (int q = 0; q < n; ++q) if ((fixed >> q) & 1) F.push_back(q);
                got = bound.excludes(R, extra, [&](int q) { return (candidates >> q) & 1; }, F);
                want = oracle(H, L, region, candidates, extra, fixed);
                ++checked;
                if (got && !want) {
                    std::fprintf(stderr, "FALSE AFFINE PROOF trial=%d repeat=%d extra=%d\n",
                                 trial, repeat, extra);
                    return 1;
                }
                inconclusive += !got && want;
            }
        }
    }
    std::printf("%d erasure oracle cases passed (%d conservative outcomes)\n",
                checked, inconclusive);
}
