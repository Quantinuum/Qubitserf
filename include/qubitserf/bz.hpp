// Brouwer-Zimmermann distance driver: orchestrates the weight loop, the converging
// upper/lower bounds and the early stop, dispatching the exponential enumeration to a
// pluggable backend.
#pragma once
#include <string>
#include <utility>
#include "qubitserf/css.hpp"
#include "qubitserf/backend.hpp"

namespace qubitserf {

struct BZResult {
    int distance = -1;
    int levels = 0;          // how many weight levels were enumerated
    int lower_bound = 0;     // final lower bound (== distance on success)
    std::string backend;
    double seconds = 0.0;
    bool proven = false;     // true if upper<=lower was reached (exact)
};

struct BZOptions {
    std::string backend = "auto";
    int threads = 0;          // 0 => hardware concurrency (CPU backend)
    int max_weight = 0;       // 0 => up to K; safety cap on the enumeration weight
    bool verbose = false;
};

// Exact minimum distance of a single subproblem (code span minus trivial/stabilizer).
BZResult bz_distance(const DistProblem& prob, const BZOptions& opt);

// Exact min over two subproblems (e.g. Z- and X-distance), INTERLEAVED weight level by
// weight level so both lower bounds advance together -- a side that stalls on a hard level
// no longer starves the other of any bound, and once one side determines the min the other
// is capped. Verbose progress is tagged "Z"/"X". Used for which='M'.
BZResult bz_min_interleaved(const DistProblem& pz, const DistProblem& px,
                            const BZOptions& opt);

// Both distances, INTERLEAVED but UNCAPPED: each side runs to its own full proof (so
// finding one never stops the other being found), while both lower bounds still advance
// together level by level. Returns {Z-result, X-result}. Used for --zx.
std::pair<BZResult, BZResult> bz_zx_interleaved(const DistProblem& pz, const DistProblem& px,
                                                const BZOptions& opt);

// CSS distance = interleaved min(Z-distance, X-distance).
BZResult bz_css_distance(const GF2Mat& Hx, const GF2Mat& Hz, const BZOptions& opt);

} // namespace qubitserf
