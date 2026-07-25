#include "distfind/gf2.hpp"
#include <algorithm>
#include <utility>

namespace distfind {

void GF2Mat::swap_rows(int a, int b) {
    if (a == b) return;
    for (int i = 0; i < stride; ++i) std::swap(row(a)[i], row(b)[i]);
}

void GF2Mat::append_row(const u64* src) {
    w.insert(w.end(), src, src + stride);
    rows += 1;
}

GF2Mat from_dense(const u8* data, int rows, int cols) {
    GF2Mat m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (data[(size_t)i * cols + j] & 1) m.set(i, j, 1);
    return m;
}

int rref(GF2Mat& m, std::vector<int>* pivot_cols) {
    if (pivot_cols) pivot_cols->clear();
    int rank = 0;
    for (int col = 0; col < m.cols && rank < m.rows; ++col) {
        int sel = -1;
        for (int r = rank; r < m.rows; ++r)
            if (m.get(r, col)) { sel = r; break; }
        if (sel < 0) continue;
        m.swap_rows(rank, sel);
        for (int r = 0; r < m.rows; ++r)
            if (r != rank && m.get(r, col)) m.add_row(rank, r);
        if (pivot_cols) pivot_cols->push_back(col);
        ++rank;
    }
    return rank;
}

std::vector<int> restricted_rref(GF2Mat& m, const std::vector<int>& cols) {
    std::vector<int> pivots;
    int rank = 0;
    for (int ci = 0; ci < (int)cols.size() && rank < m.rows; ++ci) {
        int col = cols[ci];
        int sel = -1;
        for (int r = rank; r < m.rows; ++r)
            if (m.get(r, col)) { sel = r; break; }
        if (sel < 0) continue;
        m.swap_rows(rank, sel);
        for (int r = 0; r < m.rows; ++r)
            if (r != rank && m.get(r, col)) m.add_row(rank, r);
        pivots.push_back(col);
        ++rank;
    }
    return pivots;
}

int drop_zero_rows(GF2Mat& m) {
    int out = 0;
    for (int r = 0; r < m.rows; ++r) {
        if (m.row_weight(r) == 0) continue;
        if (out != r) for (int i = 0; i < m.stride; ++i) m.row(out)[i] = m.row(r)[i];
        ++out;
    }
    m.w.resize((size_t)out * m.stride);
    m.rows = out;
    return out;
}

GF2Mat nullspace(const GF2Mat& in) {
    GF2Mat m = in;
    std::vector<int> pivots;
    rref(m, &pivots);
    int rank = (int)pivots.size();
    std::vector<char> is_pivot(m.cols, 0);
    for (int p : pivots) is_pivot[p] = 1;

    // pivot column -> its row index
    std::vector<int> pivot_row(m.cols, -1);
    for (int r = 0; r < rank; ++r) pivot_row[pivots[r]] = r;

    GF2Mat ns(m.cols - rank, m.cols);
    int out = 0;
    for (int free_col = 0; free_col < m.cols; ++free_col) {
        if (is_pivot[free_col]) continue;
        ns.set(out, free_col, 1);
        for (int r = 0; r < rank; ++r)
            if (m.get(r, free_col)) ns.set(out, pivots[r], 1);
        ++out;
    }
    return ns;
}

bool in_span(const GF2Mat& rref_m, const std::vector<int>& pivot_cols, const u64* vec) {
    // copy vec, eliminate using pivots
    std::vector<u64> tmp(vec, vec + rref_m.stride);
    int rank = (int)pivot_cols.size();
    for (int r = 0; r < rank; ++r) {
        int col = pivot_cols[r];
        if ((tmp[col >> 6] >> (col & 63)) & 1ull)
            vec_xor(tmp.data(), rref_m.row(r), rref_m.stride);
    }
    for (int i = 0; i < rref_m.stride; ++i) if (tmp[i]) return false;
    return true;
}

GF2Mat transpose(const GF2Mat& m) {
    GF2Mat t(m.cols, m.rows);
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            if (m.get(i, j)) t.set(j, i, 1);
    return t;
}

} // namespace distfind
