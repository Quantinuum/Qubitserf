// CPU backend for the Brouwer-Zimmermann low-weight enumeration, plus the backend selector.
// Enumerates, for sizes 1..p, every size-`sw` subset of the m generator rows, XORs them, and
// collects the result when its Hamming weight is <= keep_weight.  The C(m,sw) enumeration for
// each `sw` is split across threads via the combinatorial number system (each worker unranks
// its start combination and steps with comb_next), mirroring the GPU lane decomposition.

#include "backend.hpp"
#include "bits.hpp"
#include "combinatorics.hpp"

#include <thread>
#include <algorithm>

namespace codeaut {

namespace {

void worker(const BZEnumPlan& plan, const BinomTable& B, int sw,
            int64_t start, int64_t end, std::vector<uint64_t>& local) {
    const int stride = plan.stride;
    std::vector<int> comb(sw);
    std::vector<uint64_t> acc(stride);
    comb_unrank(start, plan.m, sw, B, comb.data());
    for (int64_t idx = start; idx < end; ++idx) {
        // acc = XOR of the sw selected rows
        for (int w = 0; w < stride; ++w) acc[w] = 0;
        for (int t = 0; t < sw; ++t) xor_into(acc.data(), plan.rows + (size_t)comb[t] * stride, stride);
        int wt = popcount_row(acc.data(), stride);
        if (wt <= plan.keep_weight && wt > 0) {
            local.insert(local.end(), acc.begin(), acc.end());
        }
        if (!comb_next(comb.data(), plan.m, sw)) break;
    }
}

struct CpuBackend : Backend {
    const char* name() const override { return "cpu"; }
    bool available() const override { return true; }

    BZEnumResult enumerate(const BZEnumPlan& plan) override {
        BZEnumResult res;
        if (plan.m <= 0 || plan.p <= 0) return res;
        BinomTable B(plan.m, std::min(plan.p, plan.m));

        unsigned hw = std::thread::hardware_concurrency();
        int nthreads = plan.threads > 0 ? plan.threads : (hw ? (int)hw : 4);

        for (int sw = 1; sw <= plan.p && sw <= plan.m; ++sw) {
            int64_t total = B.get(plan.m, sw);
            if (total <= 0) continue;
            if (plan.budget > 0 && res.combos + total > plan.budget) { res.overflow = true; break; }

            int T = (int)std::min<int64_t>(nthreads, total);
            if (T < 1) T = 1;
            std::vector<std::vector<uint64_t>> locals(T);
            std::vector<std::thread> pool;
            int64_t chunk = (total + T - 1) / T;
            for (int t = 0; t < T; ++t) {
                int64_t s = (int64_t)t * chunk;
                int64_t e = std::min(s + chunk, total);
                if (s >= e) break;
                pool.emplace_back(worker, std::cref(plan), std::cref(B), sw, s, e, std::ref(locals[t]));
            }
            for (auto& th : pool) th.join();
            for (auto& L : locals) res.hits.insert(res.hits.end(), L.begin(), L.end());
            res.combos += total;
        }
        return res;
    }
};

CpuBackend g_cpu;

}  // namespace

Backend* cpu_backend() { return &g_cpu; }

#ifndef CODEAUT_CUDA
Backend* cuda_backend() { return nullptr; }
#endif
#ifndef CODEAUT_METAL
Backend* metal_backend() { return nullptr; }
#endif

Backend* select_backend(int which) {
    auto ok = [](Backend* b) { return b && b->available(); };
    if (which == 0) return cpu_backend();
    if (which == 1) {  // gpu requested: cuda then metal, else cpu
        if (Backend* b = cuda_backend(); ok(b)) return b;
        if (Backend* b = metal_backend(); ok(b)) return b;
        return cpu_backend();
    }
    // auto: prefer an available GPU, else cpu
    if (Backend* b = cuda_backend(); ok(b)) return b;
    if (Backend* b = metal_backend(); ok(b)) return b;
    return cpu_backend();
}

}  // namespace codeaut
