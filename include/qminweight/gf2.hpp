// Dense bit-packed GF(2) matrix and the linear algebra the distance algorithms need.
#pragma once
#include <vector>
#include "qminweight/bits.hpp"

namespace qminweight {

// Row-major bit-packed matrix over GF(2). Each row occupies `stride` u64 words.
struct GF2Mat {
    int rows = 0, cols = 0, stride = 0;
    std::vector<u64> w;

    GF2Mat() = default;
    GF2Mat(int r, int c) : rows(r), cols(c), stride(words_for(c)),
                           w((size_t)r * words_for(c), 0ull) {}

    inline u64*       row(int i)       { return w.data() + (size_t)i * stride; }
    inline const u64* row(int i) const { return w.data() + (size_t)i * stride; }

    inline int get(int i, int j) const { return (int)((row(i)[j >> 6] >> (j & 63)) & 1ull); }
    inline void set(int i, int j, int v) {
        u64* r = row(i); u64 m = 1ull << (j & 63);
        if (v) r[j >> 6] |= m; else r[j >> 6] &= ~m;
    }
    inline void flip(int i, int j) { row(i)[j >> 6] ^= 1ull << (j & 63); }
    inline int row_weight(int i) const { return vec_weight(row(i), stride); }
    inline void add_row(int src, int dst) { vec_xor(row(dst), row(src), stride); }
    void swap_rows(int a, int b);
    void append_row(const u64* src);     // src has `stride` words (cols must match)
    GF2Mat clone() const { return *this; }
};

// Build from a dense row-major array of 0/1 bytes (rows*cols).
GF2Mat from_dense(const u8* data, int rows, int cols);

// In-place reduced row echelon form. Returns rank. If `pivot_cols` given, it is
// filled with the pivot column index of each rank row (in row order).
int rref(GF2Mat& m, std::vector<int>* pivot_cols = nullptr);

// Reduce the submatrix on the given `cols` (in order) to RREF using full-width row
// ops. Returns the list of pivot columns (a subset of `cols`). Mirrors Qubitserf's
// restricted_row_echelon: used to peel off disjoint information sets.
std::vector<int> restricted_rref(GF2Mat& m, const std::vector<int>& cols);

// Remove all-zero rows (compacting). Returns new row count.
int drop_zero_rows(GF2Mat& m);

// Basis of the right null space {x : m * x^T = 0}, as rows (cols-rank of them).
GF2Mat nullspace(const GF2Mat& m);

// Is `vec` (stride words, cols bits) in the row span of `m`? `m` is reduced to RREF
// internally if not already; pass a pre-RREF'd matrix + its pivots for speed.
bool in_span(const GF2Mat& rref_m, const std::vector<int>& pivot_cols, const u64* vec);

GF2Mat transpose(const GF2Mat& m);

} // namespace qminweight
