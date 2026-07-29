// Forwarding header: the combinatorial number system now lives in the shared core
// (include/qsf/combinatorics.hpp), used by both the distfind and codeaut engines.
#pragma once
#include "qsf/combinatorics.hpp"
#include "distfind/bits.hpp"

namespace distfind {
using qsf::BINOM_INF;
using qsf::BinomTable;
using qsf::num_combinations;
using qsf::unrank_comb;
using qsf::next_comb;
} // namespace distfind
