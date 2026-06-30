// Standalone CPU smoke test for the BZ pipeline on codes with known distance.
#include "qubitserf/bz.hpp"
#include <cstdio>
#include <vector>
#include <string>

using namespace qubitserf;

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
    std::printf("  [%s] %-22s got d=%d want d=%d\n", ok ? "PASS" : "FAIL", tag.c_str(), got, want);
}

int main() {
    BZOptions opt; opt.backend = "cpu";

    // Hamming [7,4,3] parity check (also the Steane Hx=Hz).
    auto Hsteane = mat({{1,0,1,0,1,0,1},{0,1,1,0,0,1,1},{0,0,0,1,1,1,1}});

    // Classical codes.
    check("rep[3,1,3]", bz_distance(classical_problem(mat({{1,1,0},{0,1,1}})), opt).distance, 3);
    check("rep[5,1,5]", bz_distance(classical_problem(mat({{1,1,0,0,0},{0,1,1,0,0},{0,0,1,1,0},{0,0,0,1,1}})), opt).distance, 5);
    check("hamming[7,4,3]", bz_distance(classical_problem(Hsteane), opt).distance, 3);

    // CSS quantum codes.
    check("steane[[7,1,3]]", bz_css_distance(Hsteane, Hsteane, opt).distance, 3);

    // Shor [[9,1,3]]: Hz = bit-flip (Z) checks within triples, Hx = phase checks across triples.
    auto ShorHz = mat({
        {1,1,0, 0,0,0, 0,0,0},{0,1,1, 0,0,0, 0,0,0},
        {0,0,0, 1,1,0, 0,0,0},{0,0,0, 0,1,1, 0,0,0},
        {0,0,0, 0,0,0, 1,1,0},{0,0,0, 0,0,0, 0,1,1}});
    auto ShorHx = mat({
        {1,1,1, 1,1,1, 0,0,0},
        {0,0,0, 1,1,1, 1,1,1}});
    check("shor[[9,1,3]]", bz_css_distance(ShorHx, ShorHz, opt).distance, 3);

    std::printf("%s\n", failures ? "SMOKE TEST FAILED" : "ALL SMOKE TESTS PASSED");
    return failures ? 1 : 0;
}
