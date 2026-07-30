// Forwarding header: the bit-packed GF(2) primitives now live in the shared core
// (include/qsf/bits.hpp), used by both the distfind and codeaut engines.
#pragma once
#include "qsf/bits.hpp"

namespace distfind {
using qsf::u64;
using qsf::u32;
using qsf::u8;
using qsf::words_for;
using qsf::popcount64;
using qsf::vec_weight;
using qsf::vec_xor;
using qsf::vec_dot;
} // namespace distfind
