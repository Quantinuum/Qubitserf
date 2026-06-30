// General (non-CSS) stabilizer / subsystem codes in the symplectic representation.
//
// A stabilizer code is given by a binary matrix S of shape m x 2n whose rows are the
// stabilizer generators in [z | x] column order: row r is the Pauli with Z-support
// S[r, 0..n) and X-support S[r, n..2n). The symplectic (commutation) product of two rows
// a=(a_z|a_x), b=(b_z|b_x) is  <a,b> = a_z.b_x + a_x.b_z (mod 2); they commute iff 0.
// The Pauli weight of (z|x) is the SYMPLECTIC weight = #{ qubits j : z_j=1 OR x_j=1 }.
//
// Distance of a stabilizer code = min symplectic weight of a Pauli in the normalizer
// (centralizer of S) that is not itself a stabilizer:  e in C(S) \ rowspace(S).
//
// Key identities used throughout (swap = exchange the [z] and [x] halves):
//   * <a,b> = swap(a) . b           (symplectic product as an ordinary GF(2) dot product)
//   * C(S)  = { e : <e,s>=0 for all s } = nullspace(swap(S))
//   * C(S)^perp_symplectic = rowspace(S)     (so C(C(S)) = rowspace(S))
// The last identity is what lets the meet-in-the-middle logical detector use the ordinary
// GF(2) product: storing the detector rows pre-swapped turns <detector, c> into a dot.
#pragma once
#include <cstdint>
#include <utility>
#include "qminweight/bz.hpp"
#include "qminweight/css.hpp"
#include "qminweight/gf2.hpp"

namespace qminweight {

// Exchange the [z | x] halves of every row: swap(z|x) = (x|z). `m` must have an even
// number of columns (== 2*qubits).
GF2Mat symplectic_swap(const GF2Mat& m);

// Symplectic weight of a single 2n-bit row in [z | x] order over `qubits` qubits.
int symplectic_row_weight(const u64* row, int qubits, int stride);

// True iff every row of S (m x 2n, [z|x] order) is pure-X or pure-Z, i.e. the code is CSS.
bool is_css_symplectic(const GF2Mat& S);

// Split a CSS symplectic matrix S (every row pure X or pure Z) into (Hx, Hz), each with
// `qubits` columns: pure-X rows -> Hx (their x-half), pure-Z rows -> Hz (their z-half).
// PRECONDITION: is_css_symplectic(S). All-identity rows are dropped.
std::pair<GF2Mat, GF2Mat> split_css(const GF2Mat& S);

// Stabilizer center of the gauge group with generators G (m x 2n): the elements of
// rowspace(G) that commute (symplectically) with every generator of G. For an ordinary
// stabilizer code (G already abelian) this returns rowspace(G).
GF2Mat symplectic_center(const GF2Mat& G);

// Unified symplectic distance subproblem (cost = symplectic weight):
//   search e in C(Snorm) \ rowspace(Gtriv).
// Plain stabilizer code: Snorm = Gtriv = S.
// Subsystem dressed distance: Snorm = symplectic_center(G), Gtriv = G.
// Both inputs are m x 2n in [z|x] order and must share the same number of columns.
DistProblem symplectic_problem(const GF2Mat& Snorm, const GF2Mat& Gtriv);

// Symplectic coset (operator-weight) subproblem: minimum symplectic weight over the coset
// op + rowspace(G), where `op` is a length-2n [z|x] Pauli and G is m x 2n. PRECONDITION:
// op is NOT in rowspace(G) (callers short-circuit weight 0 via in_rowspace first).
DistProblem symplectic_coset_problem(const GF2Mat& G, const uint8_t* op);

// Map a SYMPLECTIC DistProblem (cost = symplectic weight over q qubits, 2q columns in
// [z|x] order) to an equivalent HAMMING DistProblem via the weight-doubling isometry
//   phi : (a | b)  ->  (a | b | a^b)   in F2^{3q},   wt_H(phi(v)) = 2 * wt_s(v).
// phi is F2-linear and injective, so the search space and the "not a stabilizer" detector
// transfer faithfully and the resulting binary code's minimum Hamming weight is exactly
// twice the symplectic distance. Every phi-image has EVEN Hamming weight (each qubit adds
// 0 or 2), so the returned problem sets even=true -- giving Brouwer-Zimmermann its
// even-distance speedup for free. This is what makes BZ sound for non-CSS codes: the
// symplectic-weight problem becomes an ordinary binary Hamming-weight problem.
DistProblem isometry_extend(const DistProblem& sp);

// Brouwer-Zimmermann on a symplectic problem, via `isometry_extend`: run the existing
// binary BZ on the length-3q code, then halve (Hamming distance = 2 * symplectic). Works
// for plain stabilizer distance, subsystem dressed distance, and operator-weight cosets.
BZResult symplectic_bz_distance(const DistProblem& sp, const BZOptions& opt);

} // namespace qminweight
