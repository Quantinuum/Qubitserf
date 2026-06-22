// Reduce a CSS code distance problem to: "minimum Hamming weight of a vector in the
// row span of `code_gen` that is a non-trivial logical (check * c != 0)."
#pragma once
#include <vector>
#include "qminweight/gf2.hpp"

namespace qminweight {

// A single (Z or X) distance subproblem.
struct DistProblem {
    int n = 0;            // block length
    GF2Mat code_gen;      // K x n : basis of the code space to search (e.g. ker(Hx))
    GF2Mat check;         // k x n : logical detector; c is a logical iff check*c^T != 0
    bool even = false;    // every codeword has even weight (distance is even)
    bool count_all = false; // classical: every nonzero codeword counts (check unused)
    int seed_upper = 0;   // a cheap valid upper bound on the distance (0 = none)
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

// Brouwer-Zimmermann sequence of disjoint information sets for code_gen.
std::vector<Gamma> gamma_sequence(const GF2Mat& code_gen);

} // namespace qminweight
