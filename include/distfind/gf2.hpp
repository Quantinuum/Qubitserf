// Forwarding header: the dense bit-packed GF(2) matrix now lives in the shared core
// (include/qsf/gf2.hpp), used by both the distfind and codeaut engines.
#pragma once
#include "qsf/gf2.hpp"
#include "distfind/bits.hpp"

namespace distfind {
using qsf::GF2Mat;
using qsf::from_dense;
using qsf::rref;
using qsf::restricted_rref;
using qsf::drop_zero_rows;
using qsf::nullspace;
using qsf::in_span;
using qsf::transpose;
} // namespace distfind
