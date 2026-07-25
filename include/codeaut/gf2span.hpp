// Bit-packed GF(2) linear algebra for the automorphism-span backend.
//
// A GF(2) row vector is stored as ``Row`` = a little-endian array of 64-bit words
// (bit ``b`` lives in word ``b>>6`` at offset ``b&63``).  All the hot operations --
// XOR of rows, AND-popcount parity (the GF(2) dot product), leading-bit scan -- are
// single machine instructions over each 64-bit word, so a length-``L`` dot product
// costs ``ceil(L/64)`` word ops instead of ``L`` bit ops.  ``Basis`` is an incremental
// GF(2) row-space rank: each inserted row is reduced against the current pivots
// (keyed by its leading set bit) and kept iff it raises the rank.  Merging two bases
// (insert one's rows into the other) realises a parallel rank reduction.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gf2 {

using Row = std::vector<uint64_t>;

inline int words_for(int bits) { return (bits + 63) / 64; }

inline void set_bit(Row& r, int b) { r[(size_t)(b >> 6)] |= (uint64_t)1 << (b & 63); }

inline bool get_bit(const Row& r, int b) {
    return (r[(size_t)(b >> 6)] >> (b & 63)) & 1u;
}

// Highest set bit index, or -1 if the row is zero.
inline int high_bit(const Row& r) {
    for (int w = (int)r.size() - 1; w >= 0; --w) {
        uint64_t x = r[(size_t)w];
        if (x) return w * 64 + 63 - __builtin_clzll(x);
    }
    return -1;
}

inline void xor_into(Row& a, const Row& b) {
    const size_t W = a.size();
    for (size_t i = 0; i < W; ++i) a[i] ^= b[i];
}

// GF(2) dot product = parity of the AND.
inline uint64_t dot(const Row& a, const Row& b) {
    uint64_t acc = 0;
    const size_t W = a.size();
    for (size_t i = 0; i < W; ++i) acc ^= a[i] & b[i];
    return (uint64_t)(__builtin_popcountll(acc) & 1u);
}

// Pack a row-major ``rows x cols`` uint8 GF(2) matrix into one ``Row`` per matrix row.
inline std::vector<Row> pack_matrix(const uint8_t* M, int rows, int cols) {
    const int W = words_for(cols);
    std::vector<Row> out((size_t)rows, Row((size_t)W, 0));
    for (int r = 0; r < rows; ++r) {
        const uint8_t* src = M + (size_t)r * cols;
        Row& dst = out[(size_t)r];
        for (int c = 0; c < cols; ++c)
            if (src[c] & 1u) set_bit(dst, c);
    }
    return out;
}

// Gather: ``out[i] = src[perm[i]]`` -- applies the permutation matrix ``P_perm`` to a
// length-``n`` bit vector (used for both ``Lx[:, perm]`` and the fold's ``D = P_perm``).
inline Row gather(const Row& src, const int32_t* perm, int n) {
    Row out((size_t)words_for(n), 0);
    for (int i = 0; i < n; ++i)
        if (get_bit(src, perm[i])) set_bit(out, i);
    return out;
}

// Incremental GF(2) row-space basis (rank), pivots keyed by leading bit.
struct Basis {
    int W;
    std::vector<Row> rows;                 // reduced basis rows
    std::unordered_map<int, int> pivot;    // leading-bit -> index into rows

    explicit Basis(int W_) : W(W_) {}

    // Reduce ``v`` against the basis; if nonzero, store it and return true (rank++).
    bool add(Row v) {
        for (;;) {
            int h = high_bit(v);
            if (h < 0) return false;
            auto it = pivot.find(h);
            if (it == pivot.end()) {
                pivot.emplace(h, (int)rows.size());
                rows.push_back(std::move(v));
                return true;
            }
            xor_into(v, rows[(size_t)it->second]);
        }
    }

    void merge(const Basis& other) {
        for (const Row& r : other.rows) add(r);
    }

    // Merge ``other`` but stop as soon as the rank reaches ``target`` (avoids reducing many
    // more rows to zero once the basis is already full).  Returns true if full.
    bool merge_until(const Basis& other, int target) {
        if ((int)rows.size() >= target) return true;
        for (const Row& r : other.rows) {
            add(r);
            if ((int)rows.size() >= target) return true;
        }
        return false;
    }

    int rank() const { return (int)rows.size(); }
};

}  // namespace gf2
