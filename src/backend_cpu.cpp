#include "qubitserf/backend.hpp"
#include "qubitserf/combinatorics.hpp"
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>

namespace qubitserf {

static std::atomic<int> g_cpu_threads{0};
void set_cpu_threads(int n) { g_cpu_threads.store(n); }

namespace {

// Reconstruct a lightweight BinomTable view over the plan's flat binomial buffer so we
// can reuse unrank_comb without copying.
struct BinomView {
    const u64* c; int maxK;
    inline u64 binom(int n, int k) const {
        if (k < 0 || n < 0 || k > n) return 0ull;
        return c[(size_t)n * (maxK + 1) + k];
    }
};

inline void unrank(const BinomView& bt, int K, int d, u64 r, int* pos) {
    int x = 0;
    for (int i = 0; i < d; ++i) {
        for (;;) {
            u64 cnt = bt.binom(K - 1 - x, d - 1 - i);
            if (r < cnt) { pos[i] = x; ++x; break; }
            r -= cnt; ++x;
        }
    }
}

int worker(const EnumPlan p, u64 start, u64 end) {
    BinomView bt{p.binom, p.binom_maxK};
    std::vector<int> pos(p.d);
    std::vector<u64> scratch(p.stride);
    unrank(bt, p.K, p.d, start, pos.data());
    int best = p.current_best;
    for (u64 it = start; it < end; ++it) {
        for (int g = 0; g < p.num_gamma; ++g) {
            int w = eval_combo(p, g, pos.data(), scratch.data());
            if (w < best) best = w;
        }
        if (!next_comb(p.K, p.d, pos.data())) break;
    }
    return best;
}

struct CpuBackend : Backend {
    std::string name() const override { return "cpu"; }
    bool available() const override { return true; }

    int enumerate(const EnumPlan& plan) override {
        BinomView bt{plan.binom, plan.binom_maxK};
        u64 total = bt.binom(plan.K, plan.d);
        if (total == 0) return WEIGHT_NONE;

        int T = g_cpu_threads.load();
        if (T <= 0) T = std::max(1u, std::thread::hardware_concurrency());
        // Don't spawn more threads than work, and avoid threads for tiny problems.
        if (total < 4096) T = 1;
        if ((u64)T > total) T = (int)total;

        if (T == 1) return worker(plan, 0, total);

        std::vector<int> results(T, WEIGHT_NONE);
        std::vector<std::thread> pool;
        pool.reserve(T);
        for (int t = 0; t < T; ++t) {
            u64 s = (u64)((__uint128_t)total * t / T);
            u64 e = (u64)((__uint128_t)total * (t + 1) / T);
            pool.emplace_back([&, t, s, e]() { results[t] = worker(plan, s, e); });
        }
        for (auto& th : pool) th.join();
        return *std::min_element(results.begin(), results.end());
    }
};

} // namespace

Backend* cpu_backend() {
    static CpuBackend inst;
    return &inst;
}

// Overridden by the Metal / CUDA translation units when those are compiled in.
__attribute__((weak)) Backend* metal_backend() { return nullptr; }
__attribute__((weak)) Backend* cuda_backend() { return nullptr; }

Backend* select_backend(const std::string& name) {
    auto ok = [](Backend* b) { return b && b->available(); };
    if (name == "cpu")  return cpu_backend();
    if (name == "metal") { Backend* b = metal_backend(); return ok(b) ? b : cpu_backend(); }
    if (name == "cuda")  { Backend* b = cuda_backend();  return ok(b) ? b : cpu_backend(); }
    // gpu / auto: pick the machine-specific accelerator automatically.
    if (Backend* b = cuda_backend();  ok(b)) return b;
    if (Backend* b = metal_backend(); ok(b)) return b;
    return cpu_backend();
}

} // namespace qubitserf
