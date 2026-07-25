// Smoke test for operator weight (coset leader) and subsystem (dressed) distance.
//
// Guards in particular the original distfind's operator-weight bug: a single stabilizer row
// fed as the operator MUST have weight 0, including on a NON-self-orthogonal code (surface(3)),
// where the original distfind's [Hz; X-logical] syndrome match returns a wrong nonzero answer.
#include "distfind/op_weight.hpp"
#include "distfind/css.hpp"
#include "distfind/bz.hpp"
#include "distfind/mitm.hpp"
#include <cstdio>
#include <string>
#include <vector>

using namespace distfind;

static GF2Mat mat(std::vector<std::vector<int>> rows) {
    int r = (int)rows.size(), c = r ? (int)rows[0].size() : 0;
    std::vector<u8> d((size_t)r * c);
    for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) d[(size_t)i * c + j] = (u8)rows[i][j];
    return from_dense(d.data(), r, c);
}

static int failures = 0;
static void check(const std::string& tag, int got, int want) {
    bool ok = got == want;
    if (!ok) ++failures;
    std::printf("  [%s] %-34s got=%d want=%d\n", ok ? "PASS" : "FAIL", tag.c_str(), got, want);
}

// Dressed subsystem distance via the library subproblem (method = bz or mitm).
static int subsystem_dist(const GF2Mat& Gx, const GF2Mat& Gz, char which,
                          const std::string& method, const BZOptions& opt) {
    auto one = [&](char w) {
        DistProblem p = subsystem_problem(Gx, Gz, w);
        return (method == "mitm" ? mitm_distance(p, opt) : bz_distance(p, opt)).distance;
    };
    if (which == 'Z') return one('Z');
    if (which == 'X') return one('X');
    int z = one('Z'), x = one('X');
    auto v = [](int d) { return d < 0 ? (1 << 30) : d; };
    int m = std::min(v(z), v(x));
    return m >= (1 << 30) ? -1 : m;
}

int main() {
    BZOptions opt; opt.backend = "cpu";

    // ---- Steane [[7,1,3]] (self-orthogonal: Hx = Hz = Hamming[7,4,3]) ----
    auto Hsteane = mat({{1,0,1,0,1,0,1},{0,1,1,0,0,1,1},{0,0,0,1,1,1,1}});

    // logical Z = ZZZZZZZ : z_op all ones, x_op zero -> z_weight 3, x_weight 0.
    {
        std::vector<uint8_t> zall(7, 1), xz(7, 0);
        OpWeight w = css_operator_weight(Hsteane, Hsteane, zall.data(), xz.data(), 7, "bz", opt);
        check("steane op ZZZZZZZ z", w.z_weight, 3);
        check("steane op ZZZZZZZ x", w.x_weight, 0);
    }
    // a Z-stabilizer (row 0 of Hz) as operator -> weight 0.
    {
        std::vector<uint8_t> zrow = {1,0,1,0,1,0,1}, xz(7, 0);
        OpWeight w = css_operator_weight(Hsteane, Hsteane, zrow.data(), xz.data(), 7, "bz", opt);
        check("steane stabilizer-op", w.z_weight, 0);
    }
    // mitm agrees on the logical.
    {
        std::vector<uint8_t> zall(7, 1), xz(7, 0);
        OpWeight w = css_operator_weight(Hsteane, Hsteane, zall.data(), xz.data(), 7, "mitm", opt);
        check("steane op ZZZZZZZ z (mitm)", w.z_weight, 3);
    }

    // ---- surface(3) [[13,1,3]] = HGP of rep[3,1,3] (Hz NOT self-orthogonal) ----
    auto surfHx = mat({
        {1,0,0,1,0,0,0,0,0,1,0,0,0},
        {0,1,0,0,1,0,0,0,0,1,1,0,0},
        {0,0,1,0,0,1,0,0,0,0,1,0,0},
        {0,0,0,1,0,0,1,0,0,0,0,1,0},
        {0,0,0,0,1,0,0,1,0,0,0,1,1},
        {0,0,0,0,0,1,0,0,1,0,0,0,1}});
    auto surfHz = mat({
        {1,1,0,0,0,0,0,0,0,1,0,0,0},
        {0,1,1,0,0,0,0,0,0,0,1,0,0},
        {0,0,0,1,1,0,0,0,0,1,0,1,0},
        {0,0,0,0,1,1,0,0,0,0,1,0,1},
        {0,0,0,0,0,0,1,1,0,0,0,1,0},
        {0,0,0,0,0,0,0,1,1,0,0,0,1}});
    // REGRESSION GUARD: a single Z-stabilizer row as operator must be weight 0.
    {
        std::vector<uint8_t> zrow = {1,1,0,0,0,0,0,0,0,1,0,0,0}, xz(13, 0);
        OpWeight w = css_operator_weight(surfHx, surfHz, zrow.data(), xz.data(), 13, "bz", opt);
        check("surface(3) stabilizer-op -> 0", w.z_weight, 0);
        OpWeight wm = css_operator_weight(surfHx, surfHz, zrow.data(), xz.data(), 13, "mitm", opt);
        check("surface(3) stabilizer-op (mitm)", wm.z_weight, 0);
    }
    // an X-stabilizer row as operator (x part) -> 0 too.
    {
        std::vector<uint8_t> xrow = {1,0,0,1,0,0,0,0,0,1,0,0,0}, zz(13, 0);
        OpWeight w = css_operator_weight(surfHx, surfHz, zz.data(), xrow.data(), 13, "bz", opt);
        check("surface(3) X-stabilizer-op -> 0", w.x_weight, 0);
    }

    // ---- Bacon-Shor d=3 subsystem code: dressed distance == 3 ----
    auto baconGx = mat({
        {1,1,0,0,0,0,0,0,0},
        {0,1,1,0,0,0,0,0,0},
        {0,0,0,1,1,0,0,0,0},
        {0,0,0,0,1,1,0,0,0},
        {0,0,0,0,0,0,1,1,0},
        {0,0,0,0,0,0,0,1,1}});
    auto baconGz = mat({
        {1,0,0,1,0,0,0,0,0},
        {0,0,0,1,0,0,1,0,0},
        {0,1,0,0,1,0,0,0,0},
        {0,0,0,0,1,0,0,1,0},
        {0,0,1,0,0,1,0,0,0},
        {0,0,0,0,0,1,0,0,1}});
    check("bacon-shor d=3 dressed (bz)",   subsystem_dist(baconGx, baconGz, 'M', "bz", opt), 3);
    check("bacon-shor d=3 dressed (mitm)", subsystem_dist(baconGx, baconGz, 'M', "mitm", opt), 3);

    // center invariants: Sx*Gz^T == 0 and Sz*Gx^T == 0; centers nonempty.
    {
        std::pair<GF2Mat, GF2Mat> c = css_center(baconGx, baconGz);
        GF2Mat& Sx = c.first; GF2Mat& Sz = c.second;
        int bad = 0;
        for (int i = 0; i < Sx.rows; ++i)
            for (int j = 0; j < baconGz.rows; ++j)
                if (vec_dot(Sx.row(i), baconGz.row(j), Sx.stride)) ++bad;
        for (int i = 0; i < Sz.rows; ++i)
            for (int j = 0; j < baconGx.rows; ++j)
                if (vec_dot(Sz.row(i), baconGx.row(j), Sz.stride)) ++bad;
        check("bacon-shor center commutes", bad, 0);
    }

    // ---- Steane as a subsystem code with gauge == stabilizers -> css_distance == 3 ----
    check("steane subsystem(gauge=stab)", subsystem_dist(Hsteane, Hsteane, 'M', "bz", opt), 3);
    check("steane subsystem Z",           subsystem_dist(Hsteane, Hsteane, 'Z', "bz", opt), 3);

    std::printf("%s\n", failures ? "OP SMOKE TEST FAILED" : "ALL OP SMOKE TESTS PASSED");
    return failures ? 1 : 0;
}
