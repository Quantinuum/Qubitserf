#include "qminweight/cc.hpp"
#include "qminweight/css.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace qminweight {

namespace {

// Sparse, column-oriented view of a check matrix H and logical detector L.
struct CCData {
    int n = 0, mh = 0, kl = 0, sw = 0, lw = 0;
    std::vector<u64> HT;              // n columns of H, sw words each (qubit -> checks)
    std::vector<u64> LT;              // n columns of L, lw words each (qubit -> logicals)
    std::vector<std::vector<int>> HE; // mh checks -> list of qubits in that check
};

CCData build_cc(const GF2Mat& H, const GF2Mat& L) {
    CCData d;
    d.n = H.cols; d.mh = H.rows; d.kl = L.rows;
    d.sw = words_for(d.mh); d.lw = words_for(d.kl);
    d.HT.assign((size_t)d.n * d.sw, 0ull);
    d.LT.assign((size_t)d.n * d.lw, 0ull);
    for (int q = 0; q < d.n; ++q) {
        u64* ht = &d.HT[(size_t)q * d.sw];
        for (int j = 0; j < d.mh; ++j) if (H.get(j, q)) ht[j >> 6] |= 1ull << (j & 63);
        u64* lt = &d.LT[(size_t)q * d.lw];
        for (int c = 0; c < d.kl; ++c) if (L.get(c, q)) lt[c >> 6] |= 1ull << (c & 63);
    }
    d.HE.assign(d.mh, {});
    for (int j = 0; j < d.mh; ++j)
        for (int q = 0; q < d.n; ++q) if (H.get(j, q)) d.HE[j].push_back(q);
    return d;
}

// Per-thread DFS state (syndrome / logical maintained by XOR + backtrack).
struct CCSearch {
    const CCData* D;
    int target;
    std::vector<u64> syn, lg;
    std::vector<char> incl;
    std::atomic<bool>* hit;

    void xeq(std::vector<u64>& v, const u64* col, int w) { for (int i = 0; i < w; ++i) v[i] ^= col[i]; }
    bool syn_zero() const { for (u64 x : syn) if (x) return false; return true; }
    bool lg_zero()  const { for (u64 x : lg)  if (x) return false; return true; }
    int syn_min() const {
        for (int i = 0; i < D->sw; ++i) if (syn[i]) return i * 64 + __builtin_ctzll(syn[i]);
        return -1;
    }

    bool dfs(int w, int e0) {
        if (w == target) return syn_zero() && !lg_zero();
        int j = syn_min();
        if (j < 0) return false;            // closed cluster of weight < target: dead end
        if (hit->load(std::memory_order_relaxed)) return false;
        for (int k : D->HE[j]) {
            if (k <= e0 || incl[k]) continue;
            xeq(syn, &D->HT[(size_t)k * D->sw], D->sw);
            xeq(lg,  &D->LT[(size_t)k * D->lw], D->lw);
            incl[k] = 1;
            bool r = dfs(w + 1, e0);
            incl[k] = 0;
            xeq(syn, &D->HT[(size_t)k * D->sw], D->sw);
            xeq(lg,  &D->LT[(size_t)k * D->lw], D->lw);
            if (r) return true;
        }
        return false;
    }

    bool search_seed(int e0) {
        std::fill(syn.begin(), syn.end(), 0ull);
        std::fill(lg.begin(),  lg.end(),  0ull);
        std::fill(incl.begin(), incl.end(), 0);
        xeq(syn, &D->HT[(size_t)e0 * D->sw], D->sw);
        xeq(lg,  &D->LT[(size_t)e0 * D->lw], D->lw);
        incl[e0] = 1;
        return dfs(1, e0);
    }
};

// Min weight of e in ker(H) with L*e != 0, searched up to maxw. Returns WEIGHT_NONE if none.
// With verbose, reports progress on stderr in the repo's "[cc <label>] ..." style: a header,
// a ~5s in-level heartbeat (seeds dispatched / total) so a long weight level is never silent,
// and a per-level line giving the converging lower bound ("d>N") or the hit ("FOUND").
int cc_search(const CCData& D, int maxw, int nthreads, bool verbose, const char* label) {
    using clk = std::chrono::steady_clock;
    if (D.kl == 0) return -1;               // no logicals
    int T = std::max(1, std::min(nthreads, D.n));
    if (verbose)
        std::fprintf(stderr, "[cc %s] n=%d checks=%d logicals=%d threads=%d maxw=%d\n",
                     label, D.n, D.mh, D.kl, T, maxw);
    for (int d = 1; d <= maxw; ++d) {
        auto td0 = clk::now();
        std::atomic<int> next{0};
        std::atomic<bool> hit{false};
        auto worker = [&]() {
            CCSearch s; s.D = &D; s.target = d; s.hit = &hit;
            s.syn.assign(D.sw, 0ull); s.lg.assign(D.lw, 0ull); s.incl.assign(D.n, 0);
            for (;;) {
                if (hit.load(std::memory_order_relaxed)) break;
                int e0 = next.fetch_add(1);
                if (e0 >= D.n) break;
                if (s.search_seed(e0)) { hit.store(true); break; }
            }
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < T; ++t) pool.emplace_back(worker);

        // Heartbeat: while the pool grinds on weight d, a monitor thread prints how many seed
        // qubits have been dispatched, every ~5s, so the user can see liveness on a slow level.
        std::atomic<bool> level_done{false};
        std::thread monitor;
        if (verbose) {
            monitor = std::thread([&]() {
                while (!level_done.load(std::memory_order_relaxed)) {
                    for (int i = 0; i < 50 && !level_done.load(std::memory_order_relaxed); ++i)
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if (level_done.load(std::memory_order_relaxed)) break;
                    int seeds = std::min(next.load(std::memory_order_relaxed), D.n);
                    double el = std::chrono::duration<double>(clk::now() - td0).count();
                    std::fprintf(stderr, "[cc %s] d=%d  seeds %d/%d  %.0fs\n",
                                 label, d, seeds, D.n, el);
                }
            });
        }

        for (auto& th : pool) th.join();
        level_done.store(true);
        if (monitor.joinable()) monitor.join();

        double el = std::chrono::duration<double>(clk::now() - td0).count();
        if (hit.load()) {
            if (verbose)
                std::fprintf(stderr, "[cc %s] d=%d  FOUND weight-%d logical  (%.2fs)\n",
                             label, d, d, el);
            return d;
        }
        if (verbose)
            std::fprintf(stderr, "[cc %s] no weight-%d logical -> d>%d  (%.2fs)\n",
                         label, d, d, el);
    }
    return WEIGHT_NONE;
}

BZResult cc_one(const GF2Mat& H, const GF2Mat& L, const BZOptions& opt, const char* label) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    BZResult res; res.backend = "cc";
    CCData D = build_cc(H, L);
    int nthreads = opt.threads > 0 ? opt.threads : (int)std::thread::hardware_concurrency();
    int maxw = opt.max_weight > 0 ? std::min(opt.max_weight, D.n) : D.n;
    int d = cc_search(D, maxw, nthreads, opt.verbose, label);
    res.distance = (d >= WEIGHT_NONE) ? -1 : d;
    res.lower_bound = res.distance;          // exact when found
    res.proven = (d < WEIGHT_NONE);
    res.levels = (d >= WEIGHT_NONE) ? maxw : d;
    res.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    return res;
}

} // namespace

BZResult cc_css_distance(const GF2Mat& Hx, const GF2Mat& Hz, char which, const BZOptions& opt) {
    if (which == 'X') {
        DistProblem p = css_problem(Hx, Hz, 'X');   // p.check = Z-logicals
        return cc_one(Hz, p.check, opt, "dX");
    }
    if (which == 'Z') {
        DistProblem p = css_problem(Hx, Hz, 'Z');   // p.check = X-logicals
        return cc_one(Hx, p.check, opt, "dZ");
    }
    BZResult z = cc_css_distance(Hx, Hz, 'Z', opt);
    BZResult x = cc_css_distance(Hx, Hz, 'X', opt);
    auto v = [](int d) { return d < 0 ? WEIGHT_NONE : d; };
    BZResult r = (v(x.distance) < v(z.distance)) ? x : z;
    r.distance = std::min(v(z.distance), v(x.distance));
    if (r.distance >= WEIGHT_NONE) r.distance = -1;
    r.lower_bound = r.distance;
    r.seconds = z.seconds + x.seconds;
    r.proven = z.proven && x.proven;
    return r;
}

} // namespace qminweight
