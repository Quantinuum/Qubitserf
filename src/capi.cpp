#include "qubitserf/capi.h"
#include "qubitserf/bz.hpp"
#include "qubitserf/mitm.hpp"
#include "qubitserf/cc.hpp"
#include "qubitserf/css.hpp"
#include "qubitserf/op_weight.hpp"
#include "qubitserf/stab.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

using namespace qubitserf;

namespace {
void fill(QubitserfResult* out, const BZResult& r) {
    out->distance = r.distance;
    out->lower_bound = r.lower_bound;
    out->proven = r.proven ? 1 : 0;
    out->levels = r.levels;
    out->seconds = r.seconds;
    std::strncpy(out->backend, r.backend.c_str(), sizeof(out->backend) - 1);
    out->backend[sizeof(out->backend) - 1] = '\0';
}
}

extern "C" {

int qubitserf_css_distance(
    const uint8_t* Hx, int hx_rows, int hx_cols,
    const uint8_t* Hz, int hz_rows, int hz_cols,
    const char* method, char which, const char* backend,
    int threads, int max_weight, int verbose,
    QubitserfResult* out) {
    if (!out || !Hx || !Hz) return 1;
    GF2Mat mHx = from_dense(Hx, hx_rows, hx_cols);
    GF2Mat mHz = from_dense(Hz, hz_rows, hz_cols);
    set_cpu_threads(threads);

    BZOptions opt;
    opt.backend = backend ? backend : "auto";
    opt.threads = threads; opt.max_weight = max_weight; opt.verbose = verbose != 0;
    std::string m = method ? method : "bz";

    // Connected cluster needs the original sparse Hx/Hz and handles Z/X/min itself.
    if (m == "cc") {
        char w = (which == 'Z' || which == 'X') ? which : 'M';
        BZResult r = cc_css_distance(mHx, mHz, w, opt);
        fill(out, r);
        return 0;
    }

    auto solve = [&](char w) -> BZResult {
        DistProblem prob = css_problem(mHx, mHz, w);
        if (m == "mitm") return mitm_distance(prob, opt);
        return bz_distance(prob, opt);
    };

    BZResult r;
    if (which == 'Z') r = solve('Z');
    else if (which == 'X') r = solve('X');
    else { // min over Z and X
        BZResult z = solve('Z'), x = solve('X');
        auto v = [](int d) { return d < 0 ? (1 << 30) : d; };
        r = (v(x.distance) < v(z.distance)) ? x : z;
        r.distance = std::min(v(z.distance), v(x.distance));
        if (r.distance >= (1 << 30)) r.distance = -1;
        r.lower_bound = std::min(z.lower_bound, x.lower_bound);
        r.seconds = z.seconds + x.seconds;
        r.proven = z.proven && x.proven;
    }
    fill(out, r);
    return 0;
}

int qubitserf_classical_distance(
    const uint8_t* H, int rows, int cols,
    const char* method, const char* backend,
    int threads, int max_weight, int verbose,
    QubitserfResult* out) {
    if (!out || !H) return 1;
    GF2Mat mH = from_dense(H, rows, cols);
    set_cpu_threads(threads);
    BZOptions opt;
    opt.backend = backend ? backend : "auto";
    opt.threads = threads; opt.max_weight = max_weight; opt.verbose = verbose != 0;
    std::string m = method ? method : "bz";
    DistProblem prob = classical_problem(mH);
    BZResult r = (m == "mitm") ? mitm_distance(prob, opt) : bz_distance(prob, opt);
    fill(out, r);
    return 0;
}

int qubitserf_operator_weight(
    const uint8_t* Gx, int gx_rows, int gx_cols,
    const uint8_t* Gz, int gz_rows, int gz_cols,
    const uint8_t* z_op, const uint8_t* x_op, int n,
    const char* method, const char* backend,
    int threads, int max_weight, int verbose,
    QubitserfOpResult* out) {
    if (!out || !Gx || !Gz || !z_op || !x_op) return 1;
    GF2Mat mGx = from_dense(Gx, gx_rows, gx_cols);
    GF2Mat mGz = from_dense(Gz, gz_rows, gz_cols);
    set_cpu_threads(threads);

    BZOptions opt;
    opt.backend = backend ? backend : "auto";
    opt.threads = threads; opt.max_weight = max_weight; opt.verbose = verbose != 0;
    std::string m = method ? method : "bz";

    OpWeight w = css_operator_weight(mGx, mGz, z_op, x_op, n, m, opt);
    out->z_weight = w.z_weight;
    out->x_weight = w.x_weight;
    out->proven = w.proven ? 1 : 0;
    out->seconds = w.seconds;
    std::strncpy(out->backend, w.backend.c_str(), sizeof(out->backend) - 1);
    out->backend[sizeof(out->backend) - 1] = '\0';
    return 0;
}

int qubitserf_subsystem_distance(
    const uint8_t* Gx, int gx_rows, int gx_cols,
    const uint8_t* Gz, int gz_rows, int gz_cols,
    const char* method, char which, const char* backend,
    int threads, int max_weight, int verbose,
    QubitserfResult* out) {
    if (!out || !Gx || !Gz) return 1;
    GF2Mat mGx = from_dense(Gx, gx_rows, gx_cols);
    GF2Mat mGz = from_dense(Gz, gz_rows, gz_cols);
    set_cpu_threads(threads);

    BZOptions opt;
    opt.backend = backend ? backend : "auto";
    opt.threads = threads; opt.max_weight = max_weight; opt.verbose = verbose != 0;
    std::string m = method ? method : "bz";

    // Connected cluster keeps the sparse stabilizer center as its parity-check.
    if (m == "cc") {
        std::pair<GF2Mat, GF2Mat> center = css_center(mGx, mGz);
        DistProblem pZ = subsystem_problem(mGx, mGz, 'Z');  // .check = dressed Z-detector
        DistProblem pX = subsystem_problem(mGx, mGz, 'X');  // .check = dressed X-detector
        char w = (which == 'Z' || which == 'X') ? which : 'M';
        BZResult r = cc_subsystem_distance(center.first, center.second,
                                           pZ.check, pX.check, w, opt);
        fill(out, r);
        return 0;
    }

    auto solve = [&](char w) -> BZResult {
        DistProblem prob = subsystem_problem(mGx, mGz, w);
        if (m == "mitm") return mitm_distance(prob, opt);
        return bz_distance(prob, opt);
    };

    BZResult r;
    if (which == 'Z') r = solve('Z');
    else if (which == 'X') r = solve('X');
    else { // min over Z and X
        BZResult z = solve('Z'), x = solve('X');
        auto v = [](int d) { return d < 0 ? (1 << 30) : d; };
        r = (v(x.distance) < v(z.distance)) ? x : z;
        r.distance = std::min(v(z.distance), v(x.distance));
        if (r.distance >= (1 << 30)) r.distance = -1;
        r.lower_bound = std::min(z.lower_bound, x.lower_bound);
        r.seconds = z.seconds + x.seconds;
        r.proven = z.proven && x.proven;
    }
    fill(out, r);
    return 0;
}

namespace {

// Shared driver for the two symplectic distance entry points. `gauge` is the input matrix
// (stabilizers for a stabilizer code, gauge generators for a subsystem code); `subsystem`
// selects the stabilizer-center reduction. CSS input is split and routed to the existing
// CSS Hx/Hz solvers; non-CSS input uses the symplectic MITM (bz/cc fall back to mitm).
int symplectic_distance_driver(const GF2Mat& gauge, bool subsystem,
                               const char* method, char which, const BZOptions& opt,
                               QubitserfResult* out) {
    if (gauge.cols % 2 != 0) return 2;   // not a [z|x] matrix
    std::string m = method ? method : "bz";

    // CSS fast path: route to the dedicated Hx/Hz solvers (unchanged behaviour).
    if (is_css_symplectic(gauge)) {
        std::pair<GF2Mat, GF2Mat> hxhz = split_css(gauge);
        const GF2Mat& Hx = hxhz.first;
        const GF2Mat& Hz = hxhz.second;
        auto solve = [&](char w) -> BZResult {
            DistProblem prob = subsystem ? subsystem_problem(Hx, Hz, w)
                                         : css_problem(Hx, Hz, w);
            if (m == "cc") {
                if (subsystem) {
                    std::pair<GF2Mat, GF2Mat> ctr = css_center(Hx, Hz);
                    DistProblem pZ = subsystem_problem(Hx, Hz, 'Z');
                    DistProblem pX = subsystem_problem(Hx, Hz, 'X');
                    return cc_subsystem_distance(ctr.first, ctr.second, pZ.check, pX.check,
                                                 w, opt);
                }
                return cc_css_distance(Hx, Hz, w, opt);
            }
            if (m == "mitm") return mitm_distance(prob, opt);
            return bz_distance(prob, opt);
        };
        BZResult r;
        if (which == 'Z') r = solve('Z');
        else if (which == 'X') r = solve('X');
        else {
            BZResult z = solve('Z'), x = solve('X');
            auto v = [](int d) { return d < 0 ? (1 << 30) : d; };
            r = (v(x.distance) < v(z.distance)) ? x : z;
            r.distance = std::min(v(z.distance), v(x.distance));
            if (r.distance >= (1 << 30)) r.distance = -1;
            r.lower_bound = std::min(z.lower_bound, x.lower_bound);
            r.seconds = z.seconds + x.seconds;
            r.proven = z.proven && x.proven;
        }
        fill(out, r);
        return 0;
    }

    // Genuinely non-CSS. BZ generalizes via the weight-doubling isometry
    // (a|b)->(a|b|a^b) (symplectic distance = 1/2 the Hamming distance of the length-3n
    // binary code), so bz is sound here. cc has no non-CSS form -> falls back to mitm.
    GF2Mat Snorm = subsystem ? symplectic_center(gauge) : gauge;
    DistProblem prob = symplectic_problem(Snorm, gauge);
    BZResult r;
    if (m == "mitm") {
        r = mitm_distance(prob, opt);
    } else if (m == "cc") {
        std::fprintf(stderr,
            "qubitserf: connected-cluster has no non-CSS generalization; using the "
            "symplectic meet-in-the-middle search instead.\n");
        r = mitm_distance(prob, opt);
    } else {  // bz (default): isometry to a length-3n binary code, then classical BZ.
        r = symplectic_bz_distance(prob, opt);
    }
    fill(out, r);
    return 0;
}

} // namespace

int qubitserf_stabilizer_distance(
    const uint8_t* S, int rows, int cols,
    const char* method, char which, const char* backend,
    int threads, int max_weight, int verbose,
    QubitserfResult* out) {
    if (!out || !S) return 1;
    GF2Mat mS = from_dense(S, rows, cols);
    set_cpu_threads(threads);
    BZOptions opt;
    opt.backend = backend ? backend : "auto";
    opt.threads = threads; opt.max_weight = max_weight; opt.verbose = verbose != 0;
    return symplectic_distance_driver(mS, /*subsystem=*/false, method, which, opt, out);
}

int qubitserf_subsystem_stabilizer_distance(
    const uint8_t* G, int rows, int cols,
    const char* method, char which, const char* backend,
    int threads, int max_weight, int verbose,
    QubitserfResult* out) {
    if (!out || !G) return 1;
    GF2Mat mG = from_dense(G, rows, cols);
    set_cpu_threads(threads);
    BZOptions opt;
    opt.backend = backend ? backend : "auto";
    opt.threads = threads; opt.max_weight = max_weight; opt.verbose = verbose != 0;
    return symplectic_distance_driver(mG, /*subsystem=*/true, method, which, opt, out);
}

int qubitserf_stabilizer_operator_weight(
    const uint8_t* G, int rows, int cols,
    const uint8_t* op, int op_len,
    const char* method, const char* backend,
    int threads, int max_weight, int verbose,
    QubitserfResult* out) {
    if (!out || !G || !op) return 1;
    if (cols % 2 != 0 || op_len != cols) return 2;
    GF2Mat mG = from_dense(G, rows, cols);
    set_cpu_threads(threads);
    BZOptions opt;
    opt.backend = backend ? backend : "auto";
    opt.threads = threads; opt.max_weight = max_weight; opt.verbose = verbose != 0;
    std::string m = method ? method : "bz";

    BZResult r;
    if (in_rowspace(mG, op)) {
        // op is in the group -> equivalent to identity -> weight 0 (proven).
        r.distance = 0; r.lower_bound = 0; r.proven = true;
        r.backend = opt.backend.empty() ? std::string("cpu") : opt.backend;
    } else {
        DistProblem prob = symplectic_coset_problem(mG, op);
        if (m == "mitm") {
            r = mitm_distance(prob, opt);
        } else if (m == "cc") {
            std::fprintf(stderr,
                "qubitserf: connected-cluster has no operator-weight form; using the "
                "symplectic meet-in-the-middle search instead.\n");
            r = mitm_distance(prob, opt);
        } else {  // bz (default): same weight-doubling isometry as the distance path.
            r = symplectic_bz_distance(prob, opt);
        }
    }
    fill(out, r);
    return 0;
}

int qubitserf_backend_available(const char* name) {
    std::string requested = name ? name : "auto";
    Backend* b = select_backend(requested);
    // select_backend falls back to cpu; report whether the requested public backend
    // is actually usable.
    if (requested == "auto") return 1;
    if (requested == "gpu")
        return b && b->available() && b->name() != std::string("cpu") ? 1 : 0;
    return b && b->name() == requested && b->available() ? 1 : 0;
}

const char* qubitserf_version(void) { return "0.1.0"; }

} // extern "C"
