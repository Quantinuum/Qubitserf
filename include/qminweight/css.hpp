// Reduce a CSS code distance problem to: "minimum Hamming weight of a vector in the
// row span of `code_gen` that is a non-trivial logical (check * c != 0)."
#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include "qminweight/gf2.hpp"

namespace qminweight {

// A single (Z or X) distance subproblem.
//
// Two cost metrics are supported:
//   * Hamming (default, symplectic=false): the columns are physical bits and the cost
//     of a codeword is its Hamming weight. This is the CSS / classical case.
//   * Symplectic (symplectic=true): the `cols` are 2*qubits columns in [z | x] order
//     and the cost of a codeword is its SYMPLECTIC weight -- the number of qubits j with
//     z_j=1 OR x_j=1. Used for general (non-CSS) stabilizer / subsystem codes. The
//     logical detector `check` is stored pre-swapped so that the ordinary GF(2) product
//     check*c^T equals the symplectic product <check_row, c> (see stab.hpp).
struct DistProblem {
    int n = 0;            // block length in COLUMNS (== 2*qubits when symplectic)
    GF2Mat code_gen;      // K x n : basis of the code space to search (e.g. ker(Hx))
    GF2Mat check;         // k x n : logical detector; c is a logical iff check*c^T != 0
    bool even = false;    // every codeword has even weight (distance is even)
    bool count_all = false; // classical: every nonzero codeword counts (check unused)
    int seed_upper = 0;   // a cheap valid upper bound on the distance (0 = none)
    bool symplectic = false; // cost is symplectic weight over `qubits` qubits, not Hamming
    int qubits = 0;       // number of qubits (only meaningful when symplectic; == n/2)
};

// One information-set generator matrix + its rank within that set.
struct Gamma {
    GF2Mat g;   // K x n, systematic on its information columns
    int rank;   // r_i : number of pivot columns of this information set
};

// Build the Z-distance subproblem (code = ker(Hx), check = X-logicals) when which='Z',
// or the X-distance subproblem (code = ker(Hz), check = Z-logicals) when which='X'.
DistProblem css_problem(const GF2Mat& Hx, const GF2Mat& Hz, char which);

// Build a classical linear-code distance problem from a parity-check matrix H:
// code = ker(H), and every nonzero codeword counts (check has one always-violated row).
DistProblem classical_problem(const GF2Mat& H);

// ---- subsystem / operator-weight support ----------------------------------------

// Stabilizer center of the CSS gauge group <Gx (X-type), Gz (Z-type)>:
//   Sx = { v in rowspace(Gx) : Gz*v^T = 0 }  (X-gauge commuting with every Z-gauge gen)
//   Sz = { v in rowspace(Gz) : Gx*v^T = 0 }
// Each returned matrix is RREF'd with all-zero rows dropped. For an ordinary stabilizer
// code (Gx=Hx, Gz=Hz) this returns (rowspace(Hx), rowspace(Hz)).
std::pair<GF2Mat, GF2Mat> css_center(const GF2Mat& Gx, const GF2Mat& Gz);

// DRESSED subsystem distance subproblem from gauge generators Gx, Gz.
//   which='Z': search e in ker(Sx) that is NOT in rowspace(Gz)  (Z-type dressed operator)
//   which='X': search e in ker(Sz) that is NOT in rowspace(Gx)
// The kernel/normalizer constraint uses the stabilizer center (Sx/Sz); the
// triviality/quotient uses the GAUGE group (Gz/Gx) -- this is what makes it the dressed
// (not bare) distance. Equals css_problem when the gauge group is the stabilizer group.
DistProblem subsystem_problem(const GF2Mat& Gx, const GF2Mat& Gz, char which);

// Is `vec` (length G.cols 0/1 bytes) in the row span of G?
bool in_rowspace(const GF2Mat& G, const uint8_t* vec);

// Basis of ker(other) modulo rowspace(self): the rows of a basis of nullspace(other) that
// are linearly independent of rowspace(self). This is the logical-operator detector the
// CSS DistProblem builder uses; the symplectic builder reuses it to trim the detector.
GF2Mat quotient_basis(const GF2Mat& self, const GF2Mat& other);

// Coset distance subproblem: minimum Hamming weight over the coset  vec + rowspace(G).
// Reduces to a DistProblem (code = basis of rowspace([G; vec]); check = a single
// functional phi in nullspace(G) with phi*vec^T = 1). PRECONDITION: vec is NOT in
// rowspace(G) (callers short-circuit weight 0 via in_rowspace first); if it is, the
// returned check is empty.
DistProblem coset_problem(const GF2Mat& G, const uint8_t* vec);

// Brouwer-Zimmermann sequence of disjoint information sets for code_gen.
std::vector<Gamma> gamma_sequence(const GF2Mat& code_gen);

} // namespace qminweight
