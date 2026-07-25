#include "distfind/cc.hpp"
#include "distfind/css.hpp"
#include "distfind/progress.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace distfind {

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

// Does a weight-d e in ker(H) with L*e != 0 exist? Parallel search over all seed qubits.
bool cc_level(const CCData& D, int d, int T) {
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
    for (auto& th : pool) th.join();
    return hit.load();
}

// Min weight of e in ker(H) with L*e != 0, searched up to maxw. Returns WEIGHT_NONE if none.
// With verbose, reports progress on stderr in the original distfind style: a per-level "Distance
// bound: >N" (the level just ruled out) with its elapsed time, then a final "Distance: =N".
int cc_search(const CCData& D, int maxw, int nthreads, bool verbose, const char* tag) {
    using clk = std::chrono::steady_clock;
    if (D.kl == 0) return -1;               // no logicals
    int T = std::max(1, std::min(nthreads, D.n));
    auto t_all = clk::now();
    for (int d = 1; d <= maxw; ++d) {
        auto td0 = clk::now();
        if (cc_level(D, d, T)) {
            if (verbose)
                verbose_final(d, std::chrono::duration<double>(clk::now() - t_all).count(), tag);
            return d;
        }
        if (verbose)
            verbose_bound(d, std::chrono::duration<double>(clk::now() - td0).count(), tag);
    }
    return WEIGHT_NONE;
}

BZResult cc_one(const GF2Mat& H, const GF2Mat& L, const BZOptions& opt, const char* tag) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    BZResult res; res.backend = "cc";
    CCData D = build_cc(H, L);
    int nthreads = opt.threads > 0 ? opt.threads : (int)std::thread::hardware_concurrency();
    int maxw = opt.max_weight > 0 ? std::min(opt.max_weight, D.n) : D.n;
    int d = cc_search(D, maxw, nthreads, opt.verbose, tag);
    res.distance = (d >= WEIGHT_NONE) ? -1 : d;
    res.lower_bound = res.distance;          // exact when found
    res.proven = (d < WEIGHT_NONE);
    res.levels = (d >= WEIGHT_NONE) ? maxw : d;
    res.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    return res;
}

// CSS min distance with the Z- and X-subproblems INTERLEAVED weight level by weight
// level, so both lower bounds advance in step. (Running one to completion first would
// starve the other of any bound if the first gets stuck on a hard level.) Each side's
// progress is tagged "Z"/"X" so the two streams stay distinguishable.
BZResult cc_css_min(const GF2Mat& Hx, const GF2Mat& Hz, const BZOptions& opt) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    DistProblem pz = css_problem(Hx, Hz, 'Z');   // Z-distance: e in ker(Hx), nontrivial vs X-logicals
    DistProblem px = css_problem(Hx, Hz, 'X');   // X-distance: e in ker(Hz), nontrivial vs Z-logicals
    CCData DZ = build_cc(Hx, pz.check);
    CCData DX = build_cc(Hz, px.check);

    int nthreads = opt.threads > 0 ? opt.threads : (int)std::thread::hardware_concurrency();
    int TZ = std::max(1, std::min(nthreads, DZ.n));
    int TX = std::max(1, std::min(nthreads, DX.n));
    int maxw = opt.max_weight > 0
                   ? std::min(opt.max_weight, std::max(DZ.n, DX.n))
                   : std::max(DZ.n, DX.n);

    bool zno = (DZ.kl == 0), xno = (DX.kl == 0);  // a side with no logicals has no distance
    int dz = 0, dx = 0;                           // found distance (0 = not yet found)
    int rz = 0, rx = 0;                           // weights ruled out: that distance > r
    int best = WEIGHT_NONE;                        // min found distance so far (upper bound on min)
    int levels = 0;

    // A side is worth searching at weight d only if it has logicals, hasn't been found, d is
    // within its qubit count, and d can still beat the best-so-far (rz < best, i.e. d <= best).
    auto z_active = [&](int d) { return !zno && dz == 0 && d <= DZ.n && rz < best; };
    auto x_active = [&](int d) { return !xno && dx == 0 && d <= DX.n && rx < best; };

    for (int d = 1; d <= maxw; ++d) {
        if (!z_active(d) && !x_active(d)) break;
        levels = d;
        if (z_active(d)) {
            auto td0 = clk::now();
            if (cc_level(DZ, d, TZ)) {
                dz = d; best = std::min(best, d);
                if (opt.verbose)
                    verbose_final(d, std::chrono::duration<double>(clk::now() - t0).count(), "Z");
            } else {
                rz = d;
                if (opt.verbose)
                    verbose_bound(d, std::chrono::duration<double>(clk::now() - td0).count(), "Z");
            }
        }
        if (x_active(d)) {
            auto td0 = clk::now();
            if (cc_level(DX, d, TX)) {
                dx = d; best = std::min(best, d);
                if (opt.verbose)
                    verbose_final(d, std::chrono::duration<double>(clk::now() - t0).count(), "X");
            } else {
                rx = d;
                if (opt.verbose)
                    verbose_bound(d, std::chrono::duration<double>(clk::now() - td0).count(), "X");
            }
        }
    }

    // A side is resolved if it has no logicals, was found, or was ruled out through `best`
    // (so its distance exceeds the min and can't change the answer).
    bool zres = zno || dz > 0 || (best < WEIGHT_NONE && rz >= best);
    bool xres = xno || dx > 0 || (best < WEIGHT_NONE && rx >= best);

    auto v = [](int d) { return d <= 0 ? WEIGHT_NONE : d; };
    int dist = std::min(v(dz), v(dx));
    BZResult res; res.backend = "cc";
    res.distance = (dist >= WEIGHT_NONE) ? -1 : dist;
    res.lower_bound = res.distance;
    res.levels = levels;
    res.proven = zres && xres;
    res.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    return res;
}

} // namespace

BZResult cc_css_distance(const GF2Mat& Hx, const GF2Mat& Hz, char which, const BZOptions& opt) {
    if (which == 'X') {
        DistProblem p = css_problem(Hx, Hz, 'X');   // p.check = Z-logicals
        return cc_one(Hz, p.check, opt, "X");
    }
    if (which == 'Z') {
        DistProblem p = css_problem(Hx, Hz, 'Z');   // p.check = X-logicals
        return cc_one(Hx, p.check, opt, "Z");
    }
    return cc_css_min(Hx, Hz, opt);
}

BZResult cc_subsystem_distance(const GF2Mat& Sx, const GF2Mat& Sz,
                               const GF2Mat& detZ, const GF2Mat& detX,
                               char which, const BZOptions& opt) {
    // Z-distance: e in ker(Sx), e nontrivial vs the dressed-logical detector detZ.
    if (which == 'Z') return cc_one(Sx, detZ, opt, "Z");
    if (which == 'X') return cc_one(Sz, detX, opt, "X");
    // 'M': sequential Z then X with a min combine (each CC search is exact when it returns).
    BZResult z = cc_one(Sx, detZ, opt, "Z");
    BZResult x = cc_one(Sz, detX, opt, "X");
    auto v = [](int d) { return d < 0 ? (1 << 30) : d; };
    BZResult r = (v(x.distance) < v(z.distance)) ? x : z;
    r.distance = std::min(v(z.distance), v(x.distance));
    if (r.distance >= (1 << 30)) r.distance = -1;
    r.lower_bound = r.distance;
    r.levels = std::max(z.levels, x.levels);
    r.proven = z.proven && x.proven;
    r.seconds = z.seconds + x.seconds;
    r.backend = "cc";
    return r;
}

} // namespace distfind
