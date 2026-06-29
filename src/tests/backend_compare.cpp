// Cross-check every available backend against the CPU reference on known codes.
#include "qminweight/bz.hpp"
#include <cstdio>
#include <vector>
#include <string>

using namespace qminweight;

static GF2Mat mat(std::vector<std::vector<int>> rows) {
    int r = (int)rows.size(), c = r ? (int)rows[0].size() : 0;
    std::vector<u8> d((size_t)r * c);
    for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) d[(size_t)i * c + j] = (u8)rows[i][j];
    return from_dense(d.data(), r, c);
}

// Generator of the first-order Reed-Muller code RM(1, m): row 0 is the all-ones
// constant monomial, rows 1..m are the coordinate functions x_0..x_{m-1} evaluated
// over all 2^m points. RM(1,m) is self-orthogonal for m >= 3, so Hx = Hz = G_RM(1,m)
// is a valid CSS code [[2^m, 2^m - 2(m+1), 4]] -- a DENSE (non-QLDPC) code whose exact
// distance (4) BZ certifies quickly. We use it to exercise the >256-qubit BZ path:
// m=9 -> n=512 (stride 8), m=10 -> n=1024 (stride 16, the GPU's max native width).
static GF2Mat rm1(int m) {
    const int n = 1 << m;
    std::vector<u8> d((size_t)(m + 1) * n);
    for (int j = 0; j < n; ++j) d[j] = 1;                       // constant monomial
    for (int b = 0; b < m; ++b)
        for (int i = 0; i < n; ++i)
            d[(size_t)(b + 1) * n + i] = (u8)((i >> b) & 1);    // coordinate x_b
    return from_dense(d.data(), m + 1, n);
}

static int failures = 0;

static void run(const std::string& tag, const GF2Mat& Hx, const GF2Mat& Hz, int want) {
    for (const char* be : {"cpu", "gpu"}) {
        Backend* b = select_backend(be);
        if (std::string(be) == "gpu" && b->name() == "cpu") {
            std::printf("  [SKIP] %-18s gpu unavailable\n", tag.c_str());
            continue;
        }
        BZOptions opt; opt.backend = be;
        auto r = bz_css_distance(Hx, Hz, opt);
        bool ok = r.distance == want;
        if (!ok) ++failures;
        std::printf("  [%s] %-18s backend=%-6s d=%d (want %d) %.4fs\n",
                    ok ? "PASS" : "FAIL", tag.c_str(), r.backend.c_str(),
                    r.distance, want, r.seconds);
    }
}

int main() {
    auto Hsteane = mat({{1,0,1,0,1,0,1},{0,1,1,0,0,1,1},{0,0,0,1,1,1,1}});
    auto ShorHz = mat({
        {1,1,0, 0,0,0, 0,0,0},{0,1,1, 0,0,0, 0,0,0},
        {0,0,0, 1,1,0, 0,0,0},{0,0,0, 0,1,1, 0,0,0},
        {0,0,0, 0,0,0, 1,1,0},{0,0,0, 0,0,0, 0,1,1}});
    auto ShorHx = mat({{1,1,1, 1,1,1, 0,0,0},{0,0,0, 1,1,1, 1,1,1}});

    run("steane[[7,1,3]]", Hsteane, Hsteane, 3);
    run("shor[[9,1,3]]", ShorHx, ShorHz, 3);

    // Large-n CSS codes (> 256 qubits): the GPU runs its native kernel up to stride 16
    // (n <= 1024) and falls back to the CPU above that. Both must agree with d = 4.
    auto rm1_9  = rm1(9);   // QRM(1,9)  [[512, 492, 4]]   stride 8
    auto rm1_10 = rm1(10);  // QRM(1,10) [[1024, 1002, 4]] stride 16 (GPU max native width)
    run("qrm(1,9)[[512]]",  rm1_9,  rm1_9,  4);
    run("qrm(1,10)[[1024]]", rm1_10, rm1_10, 4);

    std::printf("%s\n", failures ? "BACKEND COMPARE FAILED" : "ALL BACKENDS AGREE");
    return failures ? 1 : 0;
}
