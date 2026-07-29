// Forwarding header: the enumeration plan and backend interface now live in the shared
// core (include/qsf/backend.hpp), used by both the distfind and codeaut engines. The
// distfind BZ driver uses the min-weight mode (Backend::enumerate); codeaut's low-weight
// collection uses Backend::collect on the same plan type.
#pragma once
#include "qsf/backend.hpp"
#include "distfind/bits.hpp"

namespace distfind {
using qsf::WEIGHT_NONE;
using qsf::EnumPlan;
using qsf::Backend;
using qsf::cpu_backend;
using qsf::metal_backend;
using qsf::cuda_backend;
using qsf::select_backend;
using qsf::set_cpu_threads;
using qsf::eval_combo;
} // namespace distfind
