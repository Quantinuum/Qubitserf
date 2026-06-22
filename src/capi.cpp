#include "qminweight/capi.h"
#include "qminweight/bz.hpp"
#include "qminweight/mitm.hpp"
#include "qminweight/cc.hpp"
#include <cstring>
#include <string>

using namespace qminweight;

namespace {
void fill(QMinWeightResult* out, const BZResult& r) {
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

int qminweight_css_distance(
    const uint8_t* Hx, int hx_rows, int hx_cols,
    const uint8_t* Hz, int hz_rows, int hz_cols,
    const char* method, char which, const char* backend,
    int threads, int max_weight, int verbose,
    QMinWeightResult* out) {
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

int qminweight_classical_distance(
    const uint8_t* H, int rows, int cols,
    const char* method, const char* backend,
    int threads, int max_weight, int verbose,
    QMinWeightResult* out) {
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

int qminweight_backend_available(const char* name) {
    std::string requested = name ? name : "auto";
    Backend* b = select_backend(requested);
    // select_backend falls back to cpu; report whether the requested public backend
    // is actually usable.
    if (requested == "auto") return 1;
    if (requested == "gpu")
        return b && b->available() && b->name() != std::string("cpu") ? 1 : 0;
    return b && b->name() == requested && b->available() ? 1 : 0;
}

const char* qminweight_version(void) { return "0.1.0"; }

} // extern "C"
