#include "qminweight/op_weight.hpp"
#include "qminweight/css.hpp"
#include "qminweight/bz.hpp"
#include "qminweight/mitm.hpp"
#include <cstdio>

namespace qminweight {

OpWeight css_operator_weight(const GF2Mat& Gx, const GF2Mat& Gz,
                             const uint8_t* z_op, const uint8_t* x_op, int n,
                             const std::string& method, const BZOptions& opt) {
    OpWeight w;
    std::string m = method;
    if (m == "cc") {
        std::fprintf(stderr,
            "qminweight: operator weight has no connected-cluster form (the coset "
            "parity-check is dense); using BZ instead.\n");
        m = "bz";
    }

    // Min weight over  vec + rowspace(G). 0 when vec is already in rowspace(G).
    auto solve = [&](const GF2Mat& G, const uint8_t* vec) -> int {
        if (in_rowspace(G, vec)) return 0;
        DistProblem prob = coset_problem(G, vec);
        BZResult r = (m == "mitm") ? mitm_distance(prob, opt) : bz_distance(prob, opt);
        w.seconds += r.seconds;
        w.proven = w.proven && r.proven;
        if (w.backend.empty()) w.backend = r.backend;
        return r.distance;
    };

    w.z_weight = solve(Gz, z_op);   // Z-part modulo the Z-type generators
    w.x_weight = solve(Gx, x_op);   // X-part modulo the X-type generators
    if (w.backend.empty()) w.backend = opt.backend;
    return w;
}

} // namespace qminweight
