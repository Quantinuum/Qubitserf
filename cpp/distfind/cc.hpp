// Connected-cluster (Pryadko) exact minimum-distance algorithm for sparse CSS codes.
//
// The minimum-weight logical operator has CONNECTED support on the Tanner graph (any
// disconnected piece would itself be a lighter logical or stabilizer). So we only grow
// connected error clusters: from a seed qubit, repeatedly fix the lowest-index unsatisfied
// check and branch over the qubits in it, until the syndrome closes (codeword) with a
// non-trivial logical. Weights are tried in increasing order, so the first hit is exact.
//
// Unlike Brouwer-Zimmermann this exploits sparsity, so it certifies LDPC / topological /
// bivariate-bicycle codes (e.g. the gross code) where BZ's lower bound is too weak. It is
// inherently a sequential, data-dependent search; we parallelize over independent seeds.
#pragma once
#include "distfind/bz.hpp"   // BZResult, BZOptions
#include "distfind/gf2.hpp"

namespace distfind {

// CSS distance via connected cluster. which = 'Z', 'X', or 'M' (= min). Uses the ORIGINAL
// sparse Hx/Hz (do not pass a row-reduced version -- sparsity is the whole point).
BZResult cc_css_distance(const GF2Mat& Hx, const GF2Mat& Hz, char which, const BZOptions& opt);

// Subsystem (dressed) distance via connected cluster. The stabilizer center Sx/Sz is the
// sparse parity-check (kept sparse so CC keeps its advantage on topological subsystem
// codes); detZ/detX are the dressed-logical detectors (= subsystem_problem(...).check for
// 'Z'/'X'). which = 'Z', 'X', or 'M' (= min).
BZResult cc_subsystem_distance(const GF2Mat& Sx, const GF2Mat& Sz,
                               const GF2Mat& detZ, const GF2Mat& detX,
                               char which, const BZOptions& opt);

} // namespace distfind
