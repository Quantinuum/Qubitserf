// Meet-in-the-Middle distance algorithm (deterministic). Same result type as BZ.
#pragma once
#include "qminweight/bz.hpp"

namespace qminweight {

// Exact minimum distance of a subproblem via meet-in-the-middle. The GPU is used for
// the (embarrassingly parallel) half-weight enumeration; the collision/match step is
// host side.
BZResult mitm_distance(const DistProblem& prob, const BZOptions& opt);

} // namespace qminweight
