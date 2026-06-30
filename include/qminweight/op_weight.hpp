// Operator weight: minimum weight of a Pauli operator modulo the stabilizer/gauge group.
//
// For a CSS group <Gx (X-type), Gz (Z-type)> the Z-part and X-part are independent
// minimum-weight coset leaders:
//   z_weight = min weight over  z_op + rowspace(Gz)
//   x_weight = min weight over  x_op + rowspace(Gx)
// Each coset reduces to an ordinary DistProblem (see css.hpp::coset_problem), so the
// existing BZ / MITM solvers answer it exactly -- no bespoke search. An operator part
// that already lies in the group's rowspace has weight 0 (it is equivalent to identity).
//
// NB: this is the CORRECT definition. Quantinuum's qubitserf matches MITM syndromes
// against [Hz ; X-logicals], which is only a valid parity-check of rowspace(Gz) when the
// Z-generators are self-orthogonal; for non-self-orthogonal codes (surface, toric,
// bivariate-bicycle) it returns wrong (nonzero) answers for stabilizer operators.
#pragma once
#include <cstdint>
#include <string>
#include "qminweight/bz.hpp"   // BZOptions
#include "qminweight/gf2.hpp"

namespace qminweight {

struct OpWeight {
    int z_weight = -1;   // min weight of the Z-part modulo rowspace(Gz)
    int x_weight = -1;   // min weight of the X-part modulo rowspace(Gx)
    double seconds = 0.0;
    bool proven = true;
    std::string backend;
};

// Operator weight modulo the CSS group <Gx, Gz>. z_op / x_op are length-n 0/1 vectors
// (Z-support and X-support; a Y sets both). Pass stabilizer generators for a stabilizer
// code, gauge generators for a subsystem code. method: "bz" or "mitm" ("cc" falls back to
// "bz" with a one-line stderr note, since the coset parity-check is dense).
OpWeight css_operator_weight(const GF2Mat& Gx, const GF2Mat& Gz,
                             const uint8_t* z_op, const uint8_t* x_op, int n,
                             const std::string& method, const BZOptions& opt);

} // namespace qminweight
