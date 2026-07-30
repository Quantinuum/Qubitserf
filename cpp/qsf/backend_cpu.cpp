// Unified CPU backend: both enumeration modes (min-weight and collect) are thin
// instantiations of the shared two-level kernel in qsf/enum_core.hpp, plus the backend
// selector shared by every engine.
#include "qsf/enum_core.hpp"
#include <algorithm>
#include <atomic>

namespace qsf {

static std::atomic<int> g_cpu_threads{0};
void set_cpu_threads(int n) { g_cpu_threads.store(n); }

u64 next_buffers_key() {
    static std::atomic<u64> counter{0};
    return ++counter;
}

int cpu_level_threads(u64 total) {
    int T = g_cpu_threads.load();
    if (T <= 0) T = std::max(1u, std::thread::hardware_concurrency());
    // Don't spawn more threads than work, and avoid threads for tiny problems.
    if (total < 4096) T = 1;
    if ((u64)T > total) T = (int)total;
    return T;
}

namespace {

struct CpuBackend : Backend {
    std::string name() const override { return "cpu"; }
    bool available() const override { return true; }

    int enumerate(const EnumPlan& plan) override {
        BinomView bt{plan.binom, plan.binom_maxK};
        u64 total = bt.binom(plan.K, plan.d);
        if (total == 0) return WEIGHT_NONE;

        std::vector<u64> Gt = transpose_gammas(plan);
        int T = cpu_level_threads(total);
        std::vector<MinLogicalSink> sinks(T);
        for (auto& s : sinks) s.best = plan.current_best;
        enumerate_level(plan, Gt.data(), total, sinks);

        int best = plan.current_best;
        for (const auto& s : sinks) best = std::min(best, s.best);
        return best;
    }

    CollectResult collect(const EnumPlan& plan, int keep_weight) override {
        CollectResult res;
        BinomView bt{plan.binom, plan.binom_maxK};
        u64 total = bt.binom(plan.K, plan.d);
        if (total == 0) return res;

        std::vector<u64> Gt = transpose_gammas(plan);
        int T = cpu_level_threads(total);
        std::vector<CollectSink> sinks(T);
        for (auto& s : sinks) s.keep = keep_weight;
        enumerate_level(plan, Gt.data(), total, sinks);

        size_t words = 0;
        for (const auto& s : sinks) words += s.hits.size();
        res.hits.reserve(words);
        for (auto& s : sinks) res.hits.insert(res.hits.end(), s.hits.begin(), s.hits.end());
        return res;
    }
};

} // namespace

Backend* cpu_backend() {
    static CpuBackend inst;
    return &inst;
}

// Overridden by the Metal / CUDA translation units when those are compiled in.
#ifndef QSF_METAL
Backend* metal_backend() { return nullptr; }
#endif
#ifndef QSF_CUDA
Backend* cuda_backend() { return nullptr; }
#endif

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

} // namespace qsf
