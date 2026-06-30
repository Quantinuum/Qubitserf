#include "qubitserf/stab.hpp"
#include "qubitserf/bz.hpp"
#include "qubitserf/css.hpp"
#include <algorithm>
#include <cstdio>

namespace qubitserf {

namespace {

// GF(2) commutation matrix E = A * B^T (a_rows x b_rows), E[i][j] = <A_i, B_j> (dot).
// A and B must share the same number of columns.
GF2Mat dot_matrix(const GF2Mat& A, const GF2Mat& B) {
    GF2Mat E(A.rows, B.rows);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < B.rows; ++j)
            if (vec_dot(A.row(i), B.row(j), A.stride)) E.set(i, j, 1);
    return E;
}

// Product C * B over GF(2): out row i = XOR of B's rows j where C[i][j] = 1.
GF2Mat rows_combine(const GF2Mat& C, const GF2Mat& B) {
    GF2Mat out(C.rows, B.cols);
    for (int i = 0; i < C.rows; ++i)
        for (int j = 0; j < B.rows; ++j)
            if (C.get(i, j)) vec_xor(out.row(i), B.row(j), B.stride);
    return out;
}

} // namespace

GF2Mat symplectic_swap(const GF2Mat& m) {
    const int cols = m.cols;
    const int n = cols / 2;        // qubits
    GF2Mat out(m.rows, cols);
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < n; ++j) {
            if (m.get(i, j))     out.set(i, n + j, 1);   // z -> x half
            if (m.get(i, n + j)) out.set(i, j, 1);       // x -> z half
        }
    return out;
}

int symplectic_row_weight(const u64* row, int qubits, int stride) {
    (void)stride;
    int w = 0;
    for (int j = 0; j < qubits; ++j) {
        int z = (int)((row[j >> 6] >> (j & 63)) & 1ull);
        int xj = qubits + j;
        int x = (int)((row[xj >> 6] >> (xj & 63)) & 1ull);
        if (z | x) ++w;
    }
    return w;
}

bool is_css_symplectic(const GF2Mat& S) {
    const int n = S.cols / 2;
    for (int i = 0; i < S.rows; ++i) {
        bool has_z = false, has_x = false;
        for (int j = 0; j < n; ++j) {
            if (S.get(i, j))     has_z = true;
            if (S.get(i, n + j)) has_x = true;
        }
        if (has_z && has_x) return false;     // a row mixing Z and X => non-CSS
    }
    return true;
}

std::pair<GF2Mat, GF2Mat> split_css(const GF2Mat& S) {
    const int n = S.cols / 2;
    GF2Mat Hx(0, n), Hz(0, n);
    GF2Mat zrow(1, n), xrow(1, n);
    for (int i = 0; i < S.rows; ++i) {
        bool has_z = false, has_x = false;
        for (int j = 0; j < n; ++j) { zrow.set(0, j, 0); xrow.set(0, j, 0); }
        for (int j = 0; j < n; ++j) {
            if (S.get(i, j))     { zrow.set(0, j, 1); has_z = true; }
            if (S.get(i, n + j)) { xrow.set(0, j, 1); has_x = true; }
        }
        if (has_x) Hx.append_row(xrow.row(0));
        else if (has_z) Hz.append_row(zrow.row(0));
        // all-identity rows contribute nothing
    }
    return {Hx, Hz};
}

GF2Mat symplectic_center(const GF2Mat& G) {
    // v = c.G lies in the center iff <v, g_i> = 0 for every generator g_i, i.e.
    //   c . (G . swap(G)^T) = 0  (the symplectic Gram matrix is symmetric, so the left
    // null space equals the null space). center = nullspace(Gram) . G, RREF'd.
    GF2Mat Gram = dot_matrix(G, symplectic_swap(G));   // m x m, Gram[i][j] = <g_i, g_j>
    GF2Mat C = rows_combine(nullspace(Gram), G);
    rref(C);
    drop_zero_rows(C);
    return C;
}

DistProblem symplectic_problem(const GF2Mat& Snorm, const GF2Mat& Gtriv) {
    DistProblem p;
    p.symplectic = true;
    p.n = Snorm.cols;            // 2*qubits columns
    p.qubits = Snorm.cols / 2;

    // code to search = centralizer of the normalizing stabilizer = C(Snorm).
    p.code_gen = nullspace(symplectic_swap(Snorm));

    // logical detector: rows L with  c in rowspace(Gtriv)  <=>  <L, c> = 0 for all L,
    // valid for c in C(Snorm). C(Gtriv) works (C(Gtriv)^perp = rowspace(Gtriv)), but its
    // rows lying in rowspace(Snorm) pair to 0 with every c in C(Snorm), so we drop them:
    // L = C(Gtriv) mod rowspace(Snorm) = quotient_basis(Snorm, swap(Gtriv)). This trims the
    // detector from n+k to ~2k rows (plain code), shrinking the MITM logical fingerprint.
    // Stored pre-swapped so the MITM's ordinary product check*c^T equals <L, c>.
    GF2Mat L = quotient_basis(Snorm, symplectic_swap(Gtriv));
    p.check = symplectic_swap(L);

    p.even = false;              // symplectic weight has no general parity guarantee
    p.count_all = false;

    // seed upper bound: smallest symplectic weight among code_gen rows that are genuine
    // nontrivial operators (not in rowspace(Gtriv)). Each such row is a real logical, so
    // its weight is a sound upper bound on the distance.
    int best = 0;
    std::vector<uint8_t> bytes(p.n);
    for (int i = 0; i < p.code_gen.rows; ++i) {
        for (int j = 0; j < p.n; ++j) bytes[j] = (uint8_t)p.code_gen.get(i, j);
        if (in_rowspace(Gtriv, bytes.data())) continue;       // trivial: skip
        int w = symplectic_row_weight(p.code_gen.row(i), p.qubits, p.code_gen.stride);
        if (best == 0 || w < best) best = w;
    }
    p.seed_upper = best;
    return p;
}

DistProblem symplectic_coset_problem(const GF2Mat& G, const uint8_t* op) {
    // Reuse the Hamming coset reduction (code = rowspace([G; op]); check = one functional
    // phi in nullspace(G) with phi.op = 1), then switch the cost metric to symplectic.
    DistProblem p = coset_problem(G, op);
    p.symplectic = true;
    p.qubits = G.cols / 2;
    // seed: symplectic weight of op itself (a valid coset member / upper bound).
    GF2Mat oprow(1, G.cols);
    for (int j = 0; j < G.cols; ++j) if (op[j] & 1) oprow.set(0, j, 1);
    p.seed_upper = symplectic_row_weight(oprow.row(0), p.qubits, oprow.stride);
    return p;
}

DistProblem isometry_extend(const DistProblem& sp) {
    // phi : (a | b) -> (a | b | a^b) over q qubits. Columns 0..q-1 = z(=a), q..2q-1 = x(=b),
    // 2q..3q-1 = a^b. wt_H(phi(v)) = 2 * wt_s(v); phi is linear & injective on the first 2q
    // coordinates, so rowspace and the (first-2q-only) detector carry over unchanged.
    const int q = sp.qubits;
    const int n2 = sp.n;          // 2q columns of the symplectic problem
    DistProblem e;
    e.symplectic = false;        // now a plain binary Hamming-weight problem
    e.qubits = 0;
    e.n = 3 * q;
    e.even = true;               // every phi-image has even Hamming weight
    e.count_all = sp.count_all;
    e.seed_upper = sp.seed_upper > 0 ? 2 * sp.seed_upper : 0;

    e.code_gen = GF2Mat(sp.code_gen.rows, e.n);
    for (int i = 0; i < sp.code_gen.rows; ++i)
        for (int j = 0; j < q; ++j) {
            int a = sp.code_gen.get(i, j);
            int b = sp.code_gen.get(i, q + j);
            if (a)      e.code_gen.set(i, j, 1);
            if (b)      e.code_gen.set(i, q + j, 1);
            if (a ^ b)  e.code_gen.set(i, 2 * q + j, 1);
        }

    // Detector acts on the first 2q coordinates only (the third block is dependent), so the
    // pre-swapped symplectic detector rows transfer verbatim, zero-padded to width 3q.
    e.check = GF2Mat(sp.check.rows, e.n);
    for (int i = 0; i < sp.check.rows; ++i)
        for (int j = 0; j < n2; ++j)
            if (sp.check.get(i, j)) e.check.set(i, j, 1);
    return e;
}

BZResult symplectic_bz_distance(const DistProblem& sp, const BZOptions& opt) {
    DistProblem e = isometry_extend(sp);
    if (opt.verbose)
        std::fprintf(stderr,
            "qubitserf: non-CSS BZ via the (a|b)->(a|b|a^b) isometry; the bounds below "
            "are Hamming weights of the length-%d code (= 2 x symplectic weight), so the "
            "symplectic distance is half the reported value.\n", e.n);
    BZResult r = bz_distance(e, opt);
    // Hamming distance / lower bound are exactly twice the symplectic ones (always even).
    if (r.distance > 0)    r.distance /= 2;
    if (r.lower_bound > 0) r.lower_bound /= 2;
    return r;
}

} // namespace qubitserf
