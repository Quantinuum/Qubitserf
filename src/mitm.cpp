// Meet-in-the-Middle (MITM) minimum-distance algorithm — deterministic, exact.
//
// Computes the minimum Hamming weight over nonzero c in rowspan(code_gen) such that c
// is a non-trivial logical, i.e.
//     count_all ? (c != 0) : (check * c^T != 0).
// This is exactly what bz_distance computes; the result matches it on every code.
//
// Reduction (coordinate-split, syndrome matching)
// -----------------------------------------------
// The code equals ker(H_code) with H_code = nullspace(code_gen) (a valid parity check:
//   c in code  <=>  H_code * c^T = 0).
// Partition coordinates into Left = [0, nL) and Right = [nL, n), nL = n/2. A codeword of
// weight d splits into a left part (support SL, weight wL) and a right part (support SR,
// weight wR = d - wL). Because the supports are disjoint,
//   H_code * c^T = (H_code|SL * 1) XOR (H_code|SR * 1) = synL XOR synR,
//   check   * c^T = logL XOR logR,
// so c is a codeword  <=>  synL == synR, and c is a non-trivial logical  <=>
//   count_all ? (d >= 1)  :  (logL XOR logR != 0).
// For each total weight d we hash left parts of weight wL keyed by synL, then probe with
// right parts of weight wR, matching synL == synR and demanding the logical mismatch.
// The first d with a hit is the exact distance.

#include "qminweight/mitm.hpp"
#include "qminweight/css.hpp"
#include "qminweight/gf2.hpp"
#include "qminweight/combinatorics.hpp"
#include "qminweight/progress.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace qminweight {

namespace {

// A packed bit-vector key: r-bit syndrome stored in ceil(r/64) words. Hashable.
struct PackedKey {
    std::vector<u64> w;
    bool operator==(const PackedKey& o) const { return w == o.w; }
};

struct PackedKeyHash {
    size_t operator()(const PackedKey& k) const {
        // FNV-1a over the words.
        u64 h = 1469598103934665603ull;
        for (u64 x : k.w) { h ^= x; h *= 1099511628257ull; }
        return (size_t)h;
    }
};

// Column-major projection of a GF(2) matrix M (rows x cols): for each coordinate j we
// pre-pack M's j-th column as a bit-vector of ceil(rows/64) words, so the syndrome of a
// support set is just the XOR of the selected columns. (M * e^T == XOR_{j in e} col_j.)
struct ColumnView {
    int rows = 0;          // syndrome length
    int kw = 0;            // words per column = words_for(rows)
    std::vector<u64> cols; // n columns, each kw words; cols[j*kw .. j*kw+kw)

    void build(const GF2Mat& M, int n) {
        rows = M.rows;
        kw = words_for(rows);
        cols.assign((size_t)n * kw, 0ull);
        for (int j = 0; j < n; ++j) {
            u64* c = &cols[(size_t)j * kw];
            for (int i = 0; i < rows; ++i)
                if (M.get(i, j)) c[i >> 6] |= 1ull << (i & 63);
        }
    }
    inline const u64* col(int j) const { return &cols[(size_t)j * kw]; }
};

inline void xor_into(u64* dst, const u64* src, int kw) {
    for (int i = 0; i < kw; ++i) dst[i] ^= src[i];
}
inline bool nonzero(const u64* v, int kw) {
    for (int i = 0; i < kw; ++i) if (v[i]) return true;
    return false;
}

// One enumerated partial vector: its syndrome (key) and logical fingerprint (value).
struct Part {
    PackedKey syn;       // synL / synR (rows = syndrome length)
    std::vector<u64> log; // logL / logR (kcheck bits; empty when count_all)
};

// Enumerate all weight-w subsets of the coordinate list `idx`, producing the syndrome
// (via `synV`) and logical fingerprint (via `logV`) of each. With w == 0 we emit the
// single empty support (zero syndrome, zero logical), which lets a full-weight codeword
// live entirely on one side.
void enumerate_side(const std::vector<int>& idx, int w,
                    const ColumnView& synV, const ColumnView& logV,
                    bool count_all, std::vector<Part>& out) {
    out.clear();
    const int m = (int)idx.size();
    if (w < 0 || w > m) return;
    const int sw = synV.kw;
    const int lw = count_all ? 0 : logV.kw;

    if (w == 0) {
        Part p;
        p.syn.w.assign(sw, 0ull);
        if (lw) p.log.assign(lw, 0ull);
        out.push_back(std::move(p));
        return;
    }

    std::vector<int> pos(w);
    for (int i = 0; i < w; ++i) pos[i] = i; // first combination 0,1,...,w-1
    std::vector<u64> syn(sw), log(lw);
    do {
        std::fill(syn.begin(), syn.end(), 0ull);
        if (lw) std::fill(log.begin(), log.end(), 0ull);
        for (int i = 0; i < w; ++i) {
            int coord = idx[pos[i]];
            xor_into(syn.data(), synV.col(coord), sw);
            if (lw) xor_into(log.data(), logV.col(coord), lw);
        }
        Part p;
        p.syn.w = syn;
        if (lw) p.log = log;
        out.push_back(std::move(p));
    } while (next_comb(m, w, pos.data()));
}

// Does total weight d admit a non-trivial logical codeword? Splits d = wL + wR over ALL
// left weights wL in [0, d] (Left and Right are FIXED, distinct coordinate sets, so the
// left weight may exceed the right weight — we cannot cap at floor(d/2)); hashes the left
// parts and probes the right parts. Multithreads the (independent) right-side probing.
bool has_codeword_of_weight(int d, const std::vector<int>& left,
                            const std::vector<int>& right, const ColumnView& synV,
                            const ColumnView& logV, bool count_all, int threads) {
    const int sw = synV.kw;
    const int lw = count_all ? 0 : logV.kw;

    for (int wL = 0; wL <= d; ++wL) {
        int wR = d - wL;
        if (wL > (int)left.size() || wR > (int)right.size()) continue;

        // Build the left hash map: synL -> list of logL fingerprints.
        std::vector<Part> leftParts;
        enumerate_side(left, wL, synV, logV, count_all, leftParts);
        if (leftParts.empty()) continue;

        std::unordered_map<PackedKey, std::vector<std::vector<u64>>, PackedKeyHash> table;
        table.reserve(leftParts.size() * 2 + 1);
        for (auto& p : leftParts) {
            auto& bucket = table[p.syn];
            if (count_all) {
                // Logical is implicit (d >= 1 => nonzero codeword). One sentinel entry
                // per syndrome is enough: any right hit with the same syndrome wins.
                if (bucket.empty()) bucket.emplace_back();
            } else {
                bucket.push_back(std::move(p.log));
            }
        }

        // Enumerate the right parts once, then probe in parallel.
        std::vector<Part> rightParts;
        enumerate_side(right, wR, synV, logV, count_all, rightParts);
        if (rightParts.empty()) continue;

        const int nR = (int)rightParts.size();
        const int nthreads = std::max(1, std::min(threads, nR));

        std::atomic<bool> found{false};
        auto worker = [&](int t) {
            for (int i = t; i < nR && !found.load(std::memory_order_relaxed); i += nthreads) {
                const Part& rp = rightParts[i];
                auto it = table.find(rp.syn);
                if (it == table.end()) continue;
                if (count_all) {
                    // syndrome match with d >= 1 => a nonzero codeword of weight d.
                    found.store(true, std::memory_order_relaxed);
                    return;
                }
                // Need logL XOR logR != 0 for at least one stored left fingerprint.
                std::vector<u64> tmp(lw);
                for (const auto& logL : it->second) {
                    for (int x = 0; x < lw; ++x) tmp[x] = logL[x] ^ rp.log[x];
                    if (nonzero(tmp.data(), lw)) {
                        found.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        };

        if (nthreads <= 1) {
            worker(0);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(nthreads);
            for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker, t);
            for (auto& th : pool) th.join();
        }
        if (found.load()) return true;
    }
    return false;
}

} // namespace

BZResult mitm_distance(const DistProblem& prob, const BZOptions& opt) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    BZResult res;
    res.backend = opt.backend.empty() ? std::string("cpu") : opt.backend;

    const int n = prob.n;
    const int K = prob.code_gen.rows;

    // No logicals / empty code => distance undefined (report -1), matching bz.cpp.
    if (K == 0 || (!prob.count_all && prob.check.rows == 0)) {
        res.distance = -1;
        res.seconds = std::chrono::duration<double>(clk::now() - t0).count();
        return res;
    }

    // Parity check of the code: c in code <=> H_code * c^T == 0.
    GF2Mat Hcode = nullspace(prob.code_gen);

    ColumnView synV; synV.build(Hcode, n);
    ColumnView logV;
    if (!prob.count_all) logV.build(prob.check, n);

    // Coordinate split.
    const int nL = n / 2;
    std::vector<int> left, right;
    left.reserve(nL); right.reserve(n - nL);
    for (int j = 0; j < nL; ++j) left.push_back(j);
    for (int j = nL; j < n; ++j) right.push_back(j);

    // Upper bound / cap on the search weight.
    int cap = prob.code_gen.cols;            // == n; weight cannot exceed n
    if (opt.max_weight > 0) cap = std::min(cap, opt.max_weight);
    int target = (prob.seed_upper > 0) ? std::min(prob.seed_upper, cap) : cap;

    int threads = opt.threads > 0 ? opt.threads
                                  : (int)std::max(1u, std::thread::hardware_concurrency());

    int found = -1;
    int level = 0;
    for (int d = 1; d <= target; ++d) {
        ++level;
        auto te0 = clk::now();
        if (has_codeword_of_weight(d, left, right, synV, logV, prob.count_all, threads)) {
            found = d;
            if (opt.verbose)
                verbose_final(d, std::chrono::duration<double>(clk::now() - t0).count());
            break;
        }
        if (opt.verbose)
            verbose_bound(d, std::chrono::duration<double>(clk::now() - te0).count());
    }

    // If we only stopped because we hit the seeded upper bound without an explicit hit,
    // the seed itself certifies a logical of that weight exists.
    if (found < 0 && prob.seed_upper > 0 && target == prob.seed_upper) {
        found = prob.seed_upper;
        if (opt.verbose)
            verbose_final(found, std::chrono::duration<double>(clk::now() - t0).count());
    }

    res.distance = found;
    res.levels = level;
    res.lower_bound = (found >= 0) ? found : 0;
    res.proven = (found >= 0);
    res.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    return res;
}

} // namespace qminweight
