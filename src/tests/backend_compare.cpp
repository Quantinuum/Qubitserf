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

    std::printf("%s\n", failures ? "BACKEND COMPARE FAILED" : "ALL BACKENDS AGREE");
    return failures ? 1 : 0;
}
