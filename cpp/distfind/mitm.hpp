// Meet-in-the-Middle distance algorithm (deterministic). Same result type as BZ.
#pragma once
#include "distfind/bz.hpp"

namespace distfind {

// Exact minimum distance of a subproblem via meet-in-the-middle. The GPU is used for
// the (embarrassingly parallel) half-weight enumeration; the collision/match step is
// host side.
BZResult mitm_distance(const DistProblem& prob, const BZOptions& opt);

// Exact min over two subproblems (Z- and X-distance), INTERLEAVED weight level by weight
// level so both lower bounds advance together and a stalled side no longer starves the
// other. Verbose progress is tagged "Z"/"X". Used for which='M'.
BZResult mitm_min_interleaved(const DistProblem& pz, const DistProblem& px,
                              const BZOptions& opt);

// Both distances, INTERLEAVED but UNCAPPED: each side runs to its own first hit (so finding
// one never stops the other), while both bounds advance together. Returns {Z, X}. For --zx.
std::pair<BZResult, BZResult> mitm_zx_interleaved(const DistProblem& pz, const DistProblem& px,
                                                  const BZOptions& opt);

} // namespace distfind
