#include "qminweight/css.hpp"
#include "qminweight/combinatorics.hpp"
#include <algorithm>
#include <functional>
#include <vector>

namespace qminweight {

namespace {

int lowest_set_col(const u64* v, int stride) {
    for (int i = 0; i < stride; ++i)
        if (v[i]) return i * 64 + __builtin_ctzll(v[i]);
    return -1;
}

// Incrementally-maintained reducing basis (triangular, by pivot column).
struct ReduceBasis {
    GF2Mat rows;
    std::vector<int> pivots;
    int stride;
    explicit ReduceBasis(int cols) : rows(0, cols), stride(words_for(cols)) {}

    bool reduce(u64* v) const { // returns true if nonzero after reduction
        for (int i = 0; i < rows.rows; ++i) {
            int p = pivots[i];
            if ((v[p >> 6] >> (p & 63)) & 1ull) vec_xor(v, rows.row(i), stride);
        }
        for (int i = 0; i < stride; ++i) if (v[i]) return true;
        return false;
    }
    void add(const u64* reduced) {
        rows.append_row(reduced);
        pivots.push_back(lowest_set_col(reduced, stride));
    }
    // Try to insert v (a copy is reduced); if independent, store `keep` in `out` and
    // extend the basis. Returns true if v was independent.
    bool insert_if_independent(const u64* v, const u64* keep, GF2Mat* out) {
        std::vector<u64> tmp(v, v + stride);
        if (!reduce(tmp.data())) return false;
        add(tmp.data());
        if (out) out->append_row(keep);
        return true;
    }
};

bool all_rows_even(const GF2Mat& m) {
    for (int i = 0; i < m.rows; ++i) if (m.row_weight(i) & 1) return false;
    return true;
}

int min_row_weight(const GF2Mat& m) {
    int best = 1 << 30;
    for (int i = 0; i < m.rows; ++i) best = std::min(best, m.row_weight(i));
    return best;
}

// GF(2) commutation matrix E = A * B^T (a_rows x b_rows), E[i][j] = <A_i, B_j>.
// A and B must share the same number of columns (same physical qubits).
GF2Mat gauge_commute(const GF2Mat& A, const GF2Mat& B) {
    GF2Mat E(A.rows, B.rows);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < B.rows; ++j)
            if (vec_dot(A.row(i), B.row(j), A.stride)) E.set(i, j, 1);
    return E;
}

// Product C * B over GF(2): out row i = XOR of B's rows j where C[i][j] = 1.
// C is p x q with q <= B.rows; out is p x B.cols.
GF2Mat rows_combine(const GF2Mat& C, const GF2Mat& B) {
    GF2Mat out(C.rows, B.cols);
    for (int i = 0; i < C.rows; ++i)
        for (int j = 0; j < B.rows; ++j)
            if (C.get(i, j)) vec_xor(out.row(i), B.row(j), B.stride);
    return out;
}

DistProblem build(const GF2Mat& Hself, const GF2Mat& Hother) {
    DistProblem p;
    p.n = Hself.cols;
    p.code_gen = nullspace(Hself);                 // basis of ker(Hself)
    p.check    = quotient_basis(Hself, Hother);    // logical detector
    GF2Mat code_logicals = quotient_basis(Hother, Hself); // valid logical codewords
    p.even = all_rows_even(p.code_gen);
    p.seed_upper = code_logicals.rows ? min_row_weight(code_logicals) : 0;
    return p;
}

} // namespace

// Basis of ker(other) modulo rowspan(self): rows of ker(other) independent of self.
// (Defined outside the anonymous namespace so the symplectic builder can reuse it; it
// still references the file-local ReduceBasis above, which has internal linkage.)
GF2Mat quotient_basis(const GF2Mat& self, const GF2Mat& other) {
    GF2Mat ko = nullspace(other);
    ReduceBasis basis(self.cols);
    for (int i = 0; i < self.rows; ++i) {       // seed with rowspan(self)
        std::vector<u64> tmp(self.row(i), self.row(i) + self.stride);
        if (basis.reduce(tmp.data())) basis.add(tmp.data());
    }
    GF2Mat out(0, self.cols);
    for (int i = 0; i < ko.rows; ++i)
        basis.insert_if_independent(ko.row(i), ko.row(i), &out);
    return out;
}

DistProblem css_problem(const GF2Mat& Hx, const GF2Mat& Hz, char which) {
    return which == 'X' ? build(Hz, Hx) : build(Hx, Hz);
}

DistProblem classical_problem(const GF2Mat& H) {
    DistProblem p;
    p.n = H.cols;
    p.code_gen = nullspace(H);
    p.check = GF2Mat(0, H.cols);     // empty check => every nonzero codeword counts
    p.count_all = true;
    p.even = all_rows_even(p.code_gen);
    p.seed_upper = p.code_gen.rows ? min_row_weight(p.code_gen) : 0;
    return p;
}

std::pair<GF2Mat, GF2Mat> css_center(const GF2Mat& Gx, const GF2Mat& Gz) {
    GF2Mat E = gauge_commute(Gx, Gz);          // ax x az : <Gx_i, Gz_j>
    // Sx: c in leftnull(E) = nullspace(E^T) -> v = c*Gx commutes with all Gz.
    GF2Mat Sx = rows_combine(nullspace(transpose(E)), Gx);
    rref(Sx);
    drop_zero_rows(Sx);
    // Sz: c in nullspace(E) -> v = c*Gz commutes with all Gx.
    GF2Mat Sz = rows_combine(nullspace(E), Gz);
    rref(Sz);
    drop_zero_rows(Sz);
    return {Sx, Sz};
}

DistProblem subsystem_problem(const GF2Mat& Gx, const GF2Mat& Gz, char which) {
    std::pair<GF2Mat, GF2Mat> center = css_center(Gx, Gz);
    // 'Z': normalizer = ker(Sx), quotient by gauge Gz; 'X': ker(Sz), quotient by Gx.
    return which == 'X' ? build(center.second, Gx) : build(center.first, Gz);
}

bool in_rowspace(const GF2Mat& G, const uint8_t* vec) {
    GF2Mat R = G;
    std::vector<int> pivots;
    rref(R, &pivots);
    GF2Mat v(1, G.cols);
    for (int j = 0; j < G.cols; ++j) if (vec[j] & 1) v.set(0, j, 1);
    return in_span(R, pivots, v.row(0));
}

DistProblem coset_problem(const GF2Mat& G, const uint8_t* vec) {
    DistProblem p;
    const int n = G.cols;
    p.n = n;
    // code_gen = basis of rowspace([G ; vec]).
    GF2Mat M(G.rows + 1, n);
    for (int i = 0; i < G.rows; ++i) vec_xor(M.row(i), G.row(i), G.stride);
    for (int j = 0; j < n; ++j) if (vec[j] & 1) M.set(G.rows, j, 1);
    rref(M);
    drop_zero_rows(M);
    p.code_gen = M;
    // check = a single functional phi in nullspace(G) with phi*vec^T = 1. phi vanishes on
    // rowspace(G); it exists iff vec is not in rowspace(G). (If vec is in rowspace(G) the
    // check stays empty -- callers short-circuit that case to weight 0 via in_rowspace.)
    GF2Mat N = nullspace(G);
    GF2Mat vpacked(1, n);
    for (int j = 0; j < n; ++j) if (vec[j] & 1) vpacked.set(0, j, 1);
    GF2Mat check(0, n);
    for (int i = 0; i < N.rows; ++i)
        if (vec_dot(N.row(i), vpacked.row(0), N.stride)) { check.append_row(N.row(i)); break; }
    p.check = check;
    p.even = all_rows_even(p.code_gen);
    p.seed_upper = vec_weight(vpacked.row(0), vpacked.stride);
    return p;
}

// ---- greedy information-set sequence (Qubitserf-style peel) ----------------------
static std::vector<Gamma> gamma_sequence_greedy(const GF2Mat& code_gen) {
    std::vector<Gamma> seq;
    const int n = code_gen.cols;
    std::vector<int> active(n);
    for (int i = 0; i < n; ++i) active[i] = i;
    GF2Mat working = code_gen;

    while (!active.empty()) {
        std::vector<int> pivots = restricted_rref(working, active);
        seq.push_back({working, (int)pivots.size()});

        std::vector<char> pivot_set(n, 0);
        for (int c : pivots) pivot_set[c] = 1;
        std::vector<char> zero_col(n, 1);
        for (int r = 0; r < working.rows; ++r)
            for (int c : active)
                if (working.get(r, c)) zero_col[c] = 0;

        std::vector<int> next;
        next.reserve(active.size());
        for (int c : active)
            if (!pivot_set[c] && !zero_col[c]) next.push_back(c);
        if (next.size() == active.size()) break;
        active.swap(next);
    }
    return seq;
}

// ---- matroid-partition information sets (Edmonds matroid union) -------------------
//
// The columns of code_gen form a linear matroid over GF(2); an "information set" is an
// independent column set. The Brouwer-Zimmermann lower bound is tightest when the columns
// are packed into the *minimum* number of independent sets (so that as many as possible
// are full bases of rank K) -- this is exactly Edmonds' matroid partitioning, the
// refinement Magma uses. Any partition that covers all columns yields a SOUND bound
// (restricted_rref recovers each set's true rank), so this only ever affects speed.
namespace mp {

// Symmetric difference of two sorted index lists (GF(2) origin bookkeeping).
static std::vector<int> symdiff(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> r;
    r.reserve(a.size() + b.size());
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) r.push_back(a[i++]);
        else if (b[j] < a[i]) r.push_back(b[j++]);
        else { ++i; ++j; }
    }
    while (i < a.size()) r.push_back(a[i++]);
    while (j < b.size()) r.push_back(b[j++]);
    return r;
}

// Express column `z` over an independent class (column indices `cls`). Returns {in_span,
// circuit}: if z lies in the span, `circuit` is the set of class columns whose XOR equals
// z (any one may be exchanged out for z); otherwise z is independent of the class.
struct Expr { bool in_span; std::vector<int> circuit; };

static Expr express(const std::vector<int>& cls, const std::vector<u64>& colvec,
                    int kw, int z) {
    std::vector<std::vector<u64>> basis;
    std::vector<int> bpivot;
    std::vector<std::vector<int>> borigin;
    auto lowbit = [&](const std::vector<u64>& v) -> int {
        for (int w = 0; w < kw; ++w) if (v[w]) return w * 64 + __builtin_ctzll(v[w]);
        return -1;
    };
    for (int idx : cls) {
        std::vector<u64> v(colvec.begin() + (size_t)idx * kw,
                           colvec.begin() + (size_t)idx * kw + kw);
        std::vector<int> orig{idx};
        for (size_t b = 0; b < basis.size(); ++b) {
            int p = bpivot[b];
            if ((v[p >> 6] >> (p & 63)) & 1ull) {
                for (int w = 0; w < kw; ++w) v[w] ^= basis[b][w];
                orig = symdiff(orig, borigin[b]);
            }
        }
        int lb = lowbit(v);
        if (lb >= 0) { basis.push_back(v); bpivot.push_back(lb); borigin.push_back(orig); }
    }
    std::vector<u64> v(colvec.begin() + (size_t)z * kw, colvec.begin() + (size_t)z * kw + kw);
    std::vector<int> orig;
    for (size_t b = 0; b < basis.size(); ++b) {
        int p = bpivot[b];
        if ((v[p >> 6] >> (p & 63)) & 1ull) {
            for (int w = 0; w < kw; ++w) v[w] ^= basis[b][w];
            orig = symdiff(orig, borigin[b]);
        }
    }
    bool zero = true;
    for (int w = 0; w < kw; ++w) if (v[w]) { zero = false; break; }
    return {zero, zero ? orig : std::vector<int>{}};
}

} // namespace mp

static bool matroid_partition(const GF2Mat& code_gen, std::vector<std::vector<int>>& out) {
    const int n = code_gen.cols, K = code_gen.rows, kw = words_for(K);
    // pack columns as K-bit vectors
    std::vector<u64> colvec((size_t)n * kw, 0ull);
    std::vector<int> nonzero, zeros;
    for (int j = 0; j < n; ++j) {
        u64* c = &colvec[(size_t)j * kw];
        bool nz = false;
        for (int i = 0; i < K; ++i)
            if (code_gen.get(i, j)) { c[i >> 6] |= 1ull << (i & 63); nz = true; }
        (nz ? nonzero : zeros).push_back(j);
    }

    std::vector<std::vector<int>> cls;
    std::vector<char> visited(n, 0);

    // DFS augmenting placement of column z, never returning it to `forbidden`.
    std::function<bool(int, int)> place = [&](int z, int forbidden) -> bool {
        for (int c = 0; c < (int)cls.size(); ++c) {
            if (c == forbidden) continue;
            if (!mp::express(cls[c], colvec, kw, z).in_span) { cls[c].push_back(z); return true; }
        }
        for (int c = 0; c < (int)cls.size(); ++c) {
            if (c == forbidden) continue;
            mp::Expr e = mp::express(cls[c], colvec, kw, z);
            if (!e.in_span) continue;            // (shouldn't happen: handled above)
            for (int y : e.circuit) {
                if (visited[y]) continue;
                visited[y] = 1;
                auto& vec = cls[c];
                vec.erase(std::find(vec.begin(), vec.end(), y));
                vec.push_back(z);
                if (place(y, c)) return true;
                vec.erase(std::find(vec.begin(), vec.end(), z));
                vec.push_back(y);
            }
        }
        return false;
    };

    for (int z : nonzero) {
        std::fill(visited.begin(), visited.end(), 0);
        if (!place(z, -1)) cls.push_back({z});
    }
    if (cls.empty()) cls.push_back({});
    for (int j : zeros) cls[0].push_back(j);

    // sanity: every column covered exactly once
    std::vector<char> seen(n, 0);
    int total = 0;
    for (auto& c : cls) for (int j : c) { if (seen[j]) return false; seen[j] = 1; ++total; }
    if (total != n) return false;
    out.swap(cls);
    return true;
}

static int count_full(const std::vector<Gamma>& seq, int K) {
    int f = 0;
    for (const auto& g : seq) if (g.rank == K) ++f;
    return f;
}

std::vector<Gamma> gamma_sequence(const GF2Mat& code_gen) {
    const int K = code_gen.rows, n = code_gen.cols;
    std::vector<Gamma> greedy = gamma_sequence_greedy(code_gen);
    if (K == 0) return greedy;

    int nnz = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < K; ++i)
            if (code_gen.get(i, j)) { ++nnz; break; }
    const int max_full = nnz / K;              // can't pack more disjoint bases than this
    const int greedy_full = count_full(greedy, K);

    // Greedy already optimal, or the code is large enough that the (setup-time) matroid
    // partition would not pay for itself -> keep greedy.
    if (greedy_full >= max_full || n > 200) return greedy;

    std::vector<std::vector<int>> part;
    if (!matroid_partition(code_gen, part)) return greedy;
    std::sort(part.begin(), part.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) { return a.size() > b.size(); });

    std::vector<Gamma> seq;
    for (auto& cols : part) {
        if (cols.empty()) continue;
        GF2Mat working = code_gen;
        std::vector<int> pivots = restricted_rref(working, cols);
        seq.push_back({working, (int)pivots.size()});
    }
    if (seq.empty() || count_full(seq, K) <= greedy_full) return greedy;
    return seq;
}

} // namespace qminweight
