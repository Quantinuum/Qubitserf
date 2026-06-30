// Standalone test: Meet-in-the-Middle distance must equal Brouwer-Zimmermann distance on
// the SAME DistProblem for a spread of classical and CSS quantum codes.
//
// Build & run (no CMake; backend_cpu.cpp supplies weak metal/cuda stubs so it links):
//   xcrun clang++ -std=c++17 -O2 -Iinclude \
//     src/gf2.cpp src/css.cpp src/bz.cpp src/backend_cpu.cpp src/mitm.cpp \
//     src/tests/mitm_smoke.cpp -o /tmp/mitm_smoke && /tmp/mitm_smoke
#include "qubitserf/bz.hpp"
#include "qubitserf/mitm.hpp"
#include "qubitserf/css.hpp"
#include <cstdio>
#include <string>
#include <vector>

using namespace qubitserf;

static GF2Mat mat(std::vector<std::vector<int>> rows) {
    int r = (int)rows.size(), c = r ? (int)rows[0].size() : 0;
    std::vector<u8> d((size_t)r * c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) d[(size_t)i * c + j] = (u8)rows[i][j];
    return from_dense(d.data(), r, c);
}

static int failures = 0;

// Compare mitm_distance to bz_distance on a single DistProblem. Also check the known
// distance when one is supplied (want >= 0).
static void cmp(const std::string& tag, const DistProblem& p, int want = -1) {
    BZOptions opt; opt.backend = "cpu";
    int bz = bz_distance(p, opt).distance;
    int mm = mitm_distance(p, opt).distance;
    bool ok = (mm == bz);
    if (want >= 0 && bz != want) ok = false; // sanity: BZ itself must match the known d
    if (!ok) ++failures;
    if (want >= 0)
        std::printf("  [%s] %-26s mitm=%d bz=%d (want %d)\n",
                    ok ? "PASS" : "FAIL", tag.c_str(), mm, bz, want);
    else
        std::printf("  [%s] %-26s mitm=%d bz=%d\n",
                    ok ? "PASS" : "FAIL", tag.c_str(), mm, bz);
}

// Build the parity check of a length-n repetition code: rows {i, i+1} for i in [0, n-1).
static GF2Mat rep_check(int n) {
    std::vector<std::vector<int>> rows;
    for (int i = 0; i + 1 < n; ++i) {
        std::vector<int> r(n, 0);
        r[i] = 1; r[i + 1] = 1;
        rows.push_back(r);
    }
    return mat(rows);
}

int main() {
    // Hamming [7,4,3] parity check (also the Steane Hx = Hz).
    auto Hsteane = mat({{1,0,1,0,1,0,1},{0,1,1,0,0,1,1},{0,0,0,1,1,1,1}});

    // --- Classical codes (count_all): repetition n = 3..8 and Hamming[7,4,3]. ---
    for (int n = 3; n <= 8; ++n)
        cmp("rep[" + std::to_string(n) + "]", classical_problem(rep_check(n)), n);
    cmp("hamming[7,4,3]", classical_problem(Hsteane), 3);

    // Extended Hamming [8,4,4] (even-weight subcode): append an overall-parity row.
    auto Hext = mat({{1,0,1,0,1,0,1,0},{0,1,1,0,0,1,1,0},{0,0,0,1,1,1,1,0},{1,1,1,1,1,1,1,1}});
    cmp("ext-hamming[8,4,4]", classical_problem(Hext), 4);

    // --- CSS quantum codes: compare BOTH the Z and X subproblems. ---
    // Steane [[7,1,3]].
    cmp("steane Z", css_problem(Hsteane, Hsteane, 'Z'), 3);
    cmp("steane X", css_problem(Hsteane, Hsteane, 'X'), 3);

    // Shor [[9,1,3]].
    auto ShorHz = mat({
        {1,1,0, 0,0,0, 0,0,0},{0,1,1, 0,0,0, 0,0,0},
        {0,0,0, 1,1,0, 0,0,0},{0,0,0, 0,1,1, 0,0,0},
        {0,0,0, 0,0,0, 1,1,0},{0,0,0, 0,0,0, 0,1,1}});
    auto ShorHx = mat({
        {1,1,1, 1,1,1, 0,0,0},
        {0,0,0, 1,1,1, 1,1,1}});
    cmp("shor Z", css_problem(ShorHx, ShorHz, 'Z'), 3);
    cmp("shor X", css_problem(ShorHx, ShorHz, 'X'), 3);

    // Distance-2 toric-like surface patch (2x2 rotated surface, [[4,1,2]]).
    // Hx (X stabilizers) and Hz (Z stabilizers) on a 4-qubit ring/plaquette.
    auto SurfHx = mat({{1,1,1,1}});
    auto SurfHz = mat({{1,1,1,1}});
    cmp("surf2 Z", css_problem(SurfHx, SurfHz, 'Z'), 2);
    cmp("surf2 X", css_problem(SurfHx, SurfHz, 'X'), 2);

    // [[4,2,2]] CSS code as a small multi-logical check.
    auto C422_Hx = mat({{1,1,1,1}});
    auto C422_Hz = mat({{1,1,1,1}});
    cmp("[[4,2,2]] Z", css_problem(C422_Hx, C422_Hz, 'Z'), 2);
    cmp("[[4,2,2]] X", css_problem(C422_Hx, C422_Hz, 'X'), 2);

    // A larger toric-like patch: distance-3 rotated surface code [[9,1,3]] (Hx/Hz weight-4
    // plaquettes + weight-2 boundary). Use the standard 3x3 surface-code checks.
    auto RotHx = mat({
        {1,1,0, 0,0,0, 0,0,0},
        {0,1,1, 0,1,1, 0,0,0},
        {0,0,0, 1,1,0, 1,1,0},
        {0,0,0, 0,0,0, 0,1,1}});
    auto RotHz = mat({
        {1,0,0, 1,0,0, 0,0,0},
        {0,1,1, 0,1,1, 0,0,0},
        {0,0,0, 1,1,0, 1,1,0},
        {0,0,0, 0,0,1, 0,0,1}});
    cmp("rotsurf3 Z", css_problem(RotHx, RotHz, 'Z'));
    cmp("rotsurf3 X", css_problem(RotHx, RotHz, 'X'));

    std::printf("%s\n", failures ? "MITM SMOKE TEST FAILED" : "ALL MITM TESTS PASSED");
    return failures ? 1 : 0;
}
