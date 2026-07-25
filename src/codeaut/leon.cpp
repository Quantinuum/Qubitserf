// Leon's algorithm for the permutation automorphism group of a binary linear code.
//
// This is the method MAGMA uses (J. S. Leon, "Computing automorphism groups of
// error-correcting codes", IEEE Trans. Inf. Theory 28 (1982) 496-511; refined with the
// Plesken-Souvignier backtrack and Leon's ordered-partition refinement).  For a binary
// linear code ``C = rowspace(G)`` it computes
//
//     Aut(C) = { pi in Sym(n) : pi . C = C }   (column permutations fixing the code)
//
// -- the SAME group as Sage's ``LinearCode.permutation_automorphism_group`` and MAGMA's
// ``AutomorphismGroup`` -- via the reduction at the heart of Leon's method:
//
//   1. Take an Aut(C)-invariant set of codewords that SPANS C: the minimum-weight
//      codewords, adding the next weight classes until they span (each weight class is
//      Aut-invariant since Aut preserves weight).  A coordinate permutation preserves
//      every used weight-class set  <=>  it preserves their span = C; so
//
//          Aut(C) = Aut( coordinate <-> codeword incidence, codewords coloured by weight ).
//
//      (Unlike the parity-check / Tanner-graph getaround, which preserves the row multiset
//      of one particular matrix and therefore UNDER-reports, this is exact.)
//
//   2. Compute that coloured-incidence automorphism group by partition backtracking:
//      individualization + equitable refinement (1-dim Weisfeiler-Leman on the incidence,
//      i.e. Leon's per-coordinate "signature"), pruned by discovered automorphisms /
//      orbits and by a first-leaf refinement invariant.  This is the same individualization
//      -refinement framework as nauty / Traces / bliss, specialised to the code incidence
//      and bit-packed over ``gf2span.hpp`` Rows.
//
// Flat ``extern "C"`` ABI for ctypes (see ``src/lib/native/leon.py``).  Build: ``make`` here.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gf2span.hpp"

using gf2::Row;
using gf2::words_for;

namespace {

// =====================================================================================
//  GF(2) row reduction (for the codeword-spanning basis), bit-packed.
// =====================================================================================

// In-place reduce ``rows`` to a row-echelon basis (leading-bit pivots); drop zeros.
std::vector<Row> row_basis(std::vector<Row> rows, int /*nbits*/) {
    std::vector<Row> basis;
    std::vector<int> piv;  // leading bit of basis[i]
    for (Row v : rows) {
        for (;;) {
            int h = gf2::high_bit(v);
            if (h < 0) break;
            size_t k = basis.size();
            bool reduced = false;
            for (size_t i = 0; i < basis.size(); ++i)
                if (piv[i] == h) { gf2::xor_into(v, basis[i]); reduced = true; break; }
            if (!reduced) { basis.push_back(std::move(v)); piv.push_back(h); break; }
            (void)k;
        }
    }
    return basis;
}

// Incremental rank in the k-bit MESSAGE space.  The generator-basis map F_2^k -> C is
// injective, so a set of codewords has exactly the rank of its coefficient messages.  The
// congruence selector uses this scalar form instead of n-bit heap-allocated Rows: one uint64
// XOR per pivot, no hashing or allocation in the 2^k streaming loop.
struct MessageBasis {
    std::vector<uint64_t> pivot;  // slot keyed by leading message bit; 0 = empty
    int rank_ = 0;

    explicit MessageBasis(int k) : pivot((size_t)k, 0) {}

    bool add(uint64_t v) {
        while (v) {
            const int h = 63 - __builtin_clzll(v);
            if (!pivot[(size_t)h]) {
                pivot[(size_t)h] = v;
                ++rank_;
                return true;
            }
            v ^= pivot[(size_t)h];
        }
        return false;
    }

    bool merge_until(const MessageBasis& other, int target) {
        if (rank_ >= target) return true;
        for (uint64_t v : other.pivot) if (v) {
            add(v);
            if (rank_ >= target) return true;
        }
        return false;
    }

    int rank() const { return rank_; }
};

// Rank of the row space of ``rows`` over GF(2).  (Unused by the two-pass enumeration; kept
// for parity with the production source so the only diff is inside enumerate_spanning_codewords.)
[[maybe_unused]] int gf2_rank(const std::vector<Row>& rows, int nbits) {
    return (int)row_basis(rows, nbits).size();
}

// =====================================================================================
//  Codeword enumeration: the minimal collection of smallest-weight codeword classes
//  whose union spans C (the Aut-invariant structure of Leon's reduction).
// =====================================================================================

struct Codewords {
    int n = 0;
    int dim = 0;                       // dim of C
    std::vector<Row> words;            // each a length-n support (a codeword)
    std::vector<int> weight_class;     // colour: 0,1,2,... by ascending used weight
    std::vector<int> weights;          // the actual weights, one per used class
    int selector = 0;                  // 0=minweight prefix, 1=weight-congruence class
    int modulus = 0;                   // selected wt % modulus == residue (0 for prefix)
    int residue = 0;
    bool ok = false;                   // false => dim too large (over max_dim)
};

// Two-pass, low-memory variant (Engineer / ENGINEER_PASS1).
//
// The original single pass materialised ALL 2^dim codeword supports in ``by_weight`` before
// pruning to the low-weight spanning classes -- peak RSS ~ 57 bytes * 2^dim (a heap ``Row``
// per word), even though only the KEPT (lowest-weight, spanning) words survive (often a few
// dozen).  This caused OOM past dim ~28 (14.7 GB at dim 28; ~120 GB at dim 32).
//
// Here peak memory scales with the KEPT words, not 2^dim:
//   * Pass 1 streams the 2^dim words by Gray code and computes only W* = the stop weight of
//     the old algorithm (smallest weight whose classes-up-to-it span C), storing NOTHING but a
//     k-row basis.  W* equals the max weight of the MINIMUM-WEIGHT basis of C, which we keep
//     online by leading-bit echelon exchange (lighter codeword wins each pivot; the evicted one
//     keeps reducing) -- a binary-matroid greedy whose weight multiset is unique, so W* is
//     independent of stream order.  Because the basis is seeded full (rank k) from the input
//     rows, a word heavier than the running max can never lower W*, so it is skipped (only the
//     unavoidable popcount is paid for it).
//   * Pass 2 re-enumerates and collects ONLY the words of weight <= W*.
// 2x the enumeration (popcount) work, but no per-word allocation and peak RSS = kept words.
// Downstream is byte-identical to the old path: same W*, same kept word set, same colours.
Codewords enumerate_minweight_codewords(const uint8_t* G, int m, int n, int max_dim) {
    Codewords out;
    out.n = n;
    const int Wn = words_for(n);
    std::vector<Row> rows = gf2::pack_matrix(G, m, n);
    std::vector<Row> B = row_basis(rows, n);              // k x n echelon basis of C (copy rows)
    const int k = (int)B.size();
    out.dim = k;
    if (k == 0) { out.ok = true; return out; }            // C = {0}: no codewords
    // The Gray-code counter below is uint64_t and enumerates ``1ULL << k`` messages.  Shifting
    // by 63 or more is either undefined or cannot represent the terminal value, so fail
    // explicitly instead of silently enumerating a truncated code (and returning a wrong
    // automorphism group).
    if (k > max_dim || k >= 63) { out.ok = false; return out; }

    auto weight_of = [&](const Row& r) {
        int w = 0;
        for (int wd = 0; wd < Wn; ++wd) w += __builtin_popcountll(r[(size_t)wd]);
        return w;
    };

    // ---- Pass 1: W* via an online minimum-weight echelon basis (memory = k rows). ----
    std::vector<Row> mwb_row((size_t)n, Row((size_t)Wn, 0));  // slot keyed by leading bit
    std::vector<int> mwb_wt((size_t)n, -1);                   // its weight (-1 == empty)
    int cur_max_wt = -1;                                      // max weight over occupied slots
    auto recompute_max = [&]() {
        int mx = -1;
        for (int p = 0; p < n; ++p) if (mwb_wt[(size_t)p] > mx) mx = mwb_wt[(size_t)p];
        cur_max_wt = mx;
    };
    auto lbx_insert = [&](Row v, int wv) {
        for (;;) {
            int h = gf2::high_bit(v);
            if (h < 0) return;                               // dependent: discard
            if (mwb_wt[(size_t)h] < 0) {                     // empty slot: take it
                mwb_row[(size_t)h] = std::move(v);
                mwb_wt[(size_t)h] = wv;
                return;
            }
            if (wv < mwb_wt[(size_t)h]) {                    // lighter wins; evict & continue
                std::swap(v, mwb_row[(size_t)h]);
                std::swap(wv, mwb_wt[(size_t)h]);
            }
            gf2::xor_into(v, mwb_row[(size_t)h]);            // reduce (clears bit h)
        }
    };

    // Seed with the raw input rows (the low-weight stabiliser generators -- they span C, so the
    // basis is full from the start, and their weights give a tight initial threshold).
    for (const Row& r : rows) lbx_insert(r, weight_of(r));
    recompute_max();

    {
        Row cur((size_t)Wn, 0);
        const uint64_t total = (uint64_t)1 << k;
        for (uint64_t i = 1; i < total; ++i) {
            int bit = __builtin_ctzll(i);
            gf2::xor_into(cur, B[(size_t)bit]);
            int w = weight_of(cur);
            if (w < cur_max_wt) { lbx_insert(cur, w); recompute_max(); }
        }
    }
    const int Wstar = cur_max_wt;                            // == old algorithm's stop weight

    // ---- Pass 2: collect only the codewords of weight <= W*. ----
    std::vector<std::vector<Row>> by_weight((size_t)Wstar + 1);
    {
        Row cur((size_t)Wn, 0);
        const uint64_t total = (uint64_t)1 << k;
        for (uint64_t i = 1; i < total; ++i) {
            int bit = __builtin_ctzll(i);
            gf2::xor_into(cur, B[(size_t)bit]);
            int w = weight_of(cur);
            if (w <= Wstar) by_weight[(size_t)w].push_back(cur);
        }
    }

    // Ascending weight classes (all <= W*; together they span C by construction of W*).
    int colour = 0;
    for (int w = 1; w <= Wstar; ++w) {
        if (by_weight[(size_t)w].empty()) continue;
        for (const Row& r : by_weight[(size_t)w]) {
            out.words.push_back(r);
            out.weight_class.push_back(colour);
        }
        out.weights.push_back(w);
        ++colour;
    }
    out.ok = true;
    return out;
}

// Support-minimal codewords (the cocircuits of the represented binary matroid) form an
// Aut(C)-invariant spanning set.  Moreover, every codeword is a disjoint sum of support-minimal
// words contained in its support.  Therefore filtering the legacy <=W* prefix to cocircuits
// preserves its span and its stopping weight, while never increasing the incidence graph.
//
// For a full-row-rank basis B and nonzero c with support S, the subcode supported inside S is
// the kernel of B restricted to the complementary coordinates.  It is exactly <c> iff that
// restriction has rank k-1, which is precisely support-minimality.  This avoids subset tries.
Codewords enumerate_cocircuit_codewords(const uint8_t* G, int m, int n, int max_dim) {
    Codewords prefix = enumerate_minweight_codewords(G, m, n, max_dim);
    if (!prefix.ok || prefix.dim == 0) {
        prefix.selector = 3;
        return prefix;
    }

    const int Wn = words_for(n);
    std::vector<Row> packed = gf2::pack_matrix(G, m, n);
    std::vector<Row> B = row_basis(packed, n);
    const int k = (int)B.size();
    std::vector<std::vector<Row>> by_class(prefix.weights.size());
    for (size_t index = 0; index < prefix.words.size(); ++index) {
        const Row& support = prefix.words[index];
        std::vector<Row> restricted = B;
        for (Row& row : restricted)
            for (int wd = 0; wd < Wn; ++wd)
                row[(size_t)wd] &= ~support[(size_t)wd];
        if ((int)row_basis(std::move(restricted), n).size() == k - 1)
            by_class[(size_t)prefix.weight_class[index]].push_back(support);
    }

    Codewords out;
    out.n = n;
    out.dim = k;
    out.selector = 3;
    out.ok = true;
    int colour = 0;
    for (size_t old_colour = 0; old_colour < by_class.size(); ++old_colour) {
        if (by_class[old_colour].empty()) continue;
        for (const Row& word : by_class[old_colour]) {
            out.words.push_back(word);
            out.weight_class.push_back(colour);
        }
        out.weights.push_back(prefix.weights[old_colour]);
        ++colour;
    }
    // The theorem says this is full rank.  Retaining the prefix on any implementation mismatch
    // makes the optimization fail safe rather than returning a graph for a non-spanning set.
    if (gf2_rank(out.words, n) != k) return prefix;
    return out;
}

// Complete weight-congruence-class selector.
//
// For S_{q,r} = {c in C \\ {0} : wt(c) == r (mod q)}, every code automorphism preserves
// S_{q,r}.  If S_{q,r} spans C, the converse also holds: every coordinate permutation
// preserving S_{q,r} preserves span(S_{q,r}) = C.  Therefore its coloured incidence has
// automorphism group exactly Aut(C).  We retain exact-weight colours inside the selected
// residue class; they are invariant too and make refinement at least as strong as using one
// coarse residue colour.
//
// Pass 1 streams all words once, retaining only a rank-k basis and a count for each exact
// weight.  Candidate residue classes are then ranked by merging those tiny bases.  The chosen
// candidate minimises (#codewords, #incidence edges, modulus, residue).  Pass 2 materialises
// exactly that complete class.  ``allow_prefix`` is the hybrid mode: use the congruence class
// only when it has strictly fewer word vertices than the legacy ascending-weight prefix.
Codewords enumerate_congruence_codewords(const uint8_t* G, int m, int n, int max_dim,
                                         int max_modulus, bool allow_prefix) {
    Codewords out;
    out.n = n;
    const int Wn = words_for(n);
    std::vector<Row> rows = gf2::pack_matrix(G, m, n);
    std::vector<Row> B = row_basis(rows, n);
    const int k = (int)B.size();
    out.dim = k;
    if (k == 0) { out.ok = true; return out; }
    if (k > max_dim || k >= 63) { out.ok = false; return out; }

    auto weight_of = [&](const Row& r) {
        int w = 0;
        for (int wd = 0; wd < Wn; ++wd) w += __builtin_popcountll(r[(size_t)wd]);
        return w;
    };

    std::vector<uint64_t> count((size_t)n + 1, 0);
    std::vector<MessageBasis> weight_basis;
    weight_basis.reserve((size_t)n + 1);
    for (int w = 0; w <= n; ++w) weight_basis.emplace_back(k);

    {
        Row cur((size_t)Wn, 0);
        uint64_t message = 0;
        const uint64_t total = (uint64_t)1 << k;
        for (uint64_t i = 1; i < total; ++i) {
            const int bit = __builtin_ctzll(i);
            gf2::xor_into(cur, B[(size_t)bit]);
            message ^= (uint64_t)1 << bit;
            const int w = weight_of(cur);
            ++count[(size_t)w];
            if (weight_basis[(size_t)w].rank() < k) weight_basis[(size_t)w].add(message);
        }
    }

    // Recover the legacy prefix footprint from the per-weight bases.  This also supplies the
    // exact fallback for hybrid mode without another enumeration pass.
    MessageBasis prefix_basis(k);
    uint64_t prefix_count = 0;
    int prefix_stop = -1;
    for (int w = 1; w <= n; ++w) {
        prefix_count += count[(size_t)w];
        if (prefix_basis.merge_until(weight_basis[(size_t)w], k)) {
            prefix_stop = w;
            break;
        }
    }

    const int qmax = std::max(2, std::min(max_modulus > 0 ? max_modulus : n + 1, n + 1));
    uint64_t best_count = std::numeric_limits<uint64_t>::max();
    uint64_t best_edges = std::numeric_limits<uint64_t>::max();
    int best_q = -1, best_r = -1;
    for (int q = 2; q <= qmax; ++q) {
        for (int r = 0; r < q; ++r) {
            uint64_t candidate_count = 0;
            uint64_t candidate_edges = 0;
            for (int w = 1; w <= n; ++w) if (w % q == r) {
                candidate_count += count[(size_t)w];
                candidate_edges += count[(size_t)w] * (uint64_t)w;
            }
            if (candidate_count < (uint64_t)k || candidate_count > best_count) continue;
            if (candidate_count == best_count && candidate_edges > best_edges) continue;

            MessageBasis candidate_basis(k);
            for (int w = 1; w <= n; ++w) if (w % q == r)
                if (candidate_basis.merge_until(weight_basis[(size_t)w], k)) break;
            if (candidate_basis.rank() != k) continue;

            const bool better = candidate_count < best_count ||
                (candidate_count == best_count &&
                 (candidate_edges < best_edges ||
                  (candidate_edges == best_edges &&
                   (best_q < 0 || q < best_q || (q == best_q && r < best_r)))));
            if (better) {
                best_count = candidate_count;
                best_edges = candidate_edges;
                best_q = q;
                best_r = r;
            }
        }
    }

    const bool use_prefix = best_q < 0 ||
        (allow_prefix && prefix_stop >= 0 && best_count >= prefix_count);
    out.selector = use_prefix ? 0 : 1;
    out.modulus = use_prefix ? 0 : best_q;
    out.residue = use_prefix ? 0 : best_r;

    std::vector<std::vector<Row>> by_weight((size_t)n + 1);
    {
        Row cur((size_t)Wn, 0);
        const uint64_t total = (uint64_t)1 << k;
        for (uint64_t i = 1; i < total; ++i) {
            const int bit = __builtin_ctzll(i);
            gf2::xor_into(cur, B[(size_t)bit]);
            const int w = weight_of(cur);
            const bool keep = use_prefix ? (w <= prefix_stop) : (w % best_q == best_r);
            if (keep) by_weight[(size_t)w].push_back(cur);
        }
    }

    int colour = 0;
    for (int w = 1; w <= n; ++w) {
        if (by_weight[(size_t)w].empty()) continue;
        for (const Row& r : by_weight[(size_t)w]) {
            out.words.push_back(r);
            out.weight_class.push_back(colour);
        }
        out.weights.push_back(w);
        ++colour;
    }
    out.ok = true;
    return out;
}

Codewords enumerate_spanning_codewords(const uint8_t* G, int m, int n, int max_dim,
                                       int selector, int max_modulus) {
    if (selector == 1)
        return enumerate_congruence_codewords(G, m, n, max_dim, max_modulus, false);
    if (selector == 2) {
        // Cost-aware hybrid.  First run the highly tuned legacy selector; for the common case
        // where it retains only a tiny low-weight class, do not pay the richer per-weight rank
        // scan at all.  Probe congruences when the prefix occupies >= 1/4 of C\{0} (cheap small-k
        // cases) or already has >=512 graph vertices (where shrinking the incidence can repay a
        // second two-pass scan).  The congruence routine still falls back unless it is strictly
        // smaller, so the graph itself never grows in auto mode.
        Codewords prefix = enumerate_minweight_codewords(G, m, n, max_dim);
        if (!prefix.ok || prefix.dim == 0 || prefix.dim >= 63) return prefix;
        const uint64_t total = ((uint64_t)1 << prefix.dim) - 1;
        const uint64_t kept = (uint64_t)prefix.words.size();
        if (kept < 512 && kept * 4 < total) return prefix;
        return enumerate_congruence_codewords(G, m, n, max_dim, max_modulus, true);
    }
    if (selector == 3)
        return enumerate_cocircuit_codewords(G, m, n, max_dim);
    return enumerate_minweight_codewords(G, m, n, max_dim);
}

// =====================================================================================
//  Coloured incidence graph: n coordinate vertices + one vertex per codeword.
// =====================================================================================

struct Graph {
    int N = 0;                 // total vertices (n coords + #codewords)
    int n = 0;                 // coordinate vertices are [0, n)
    int WN = 0;                // words_for(N)
    std::vector<Row> adj;      // adj[v] = bit-packed neighbour set over N
    std::vector<int> colour;   // initial colour per vertex
};

Graph build_graph(const Codewords& cw) {
    Graph g;
    const int n = cw.n;
    const int E = (int)cw.words.size();
    g.n = n;
    g.N = n + E;
    g.WN = words_for(g.N);
    g.adj.assign((size_t)g.N, Row((size_t)g.WN, 0));
    g.colour.assign((size_t)g.N, 0);
    const int Wn = words_for(std::max(1, n));
    for (int e = 0; e < E; ++e) {
        const int ev = n + e;
        g.colour[(size_t)ev] = 1 + cw.weight_class[(size_t)e];   // codewords: colour by weight
        const Row& sup = cw.words[(size_t)e];
        for (int p = 0; p < n; ++p) {
            if (p < Wn * 64 && gf2::get_bit(sup, p)) {
                gf2::set_bit(g.adj[(size_t)p], ev);              // coord <-> codeword
                gf2::set_bit(g.adj[(size_t)ev], p);
            }
        }
    }
    return g;
}

// =====================================================================================
//  Individualization-Refinement automorphism search (nauty/Leon style), bit-packed.
//
//  Ordered partition stored nauty-style in lab[]/ptn[]:
//    * lab[0..N-1] lists the vertices in partition order;
//    * a cell is a maximal run lab[s..e] with ptn[s..e-1] != 0 and ptn[e] == 0.
// =====================================================================================

struct UnionFind {
    std::vector<int> p;
    void init(int N) { p.resize((size_t)N); std::iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[(size_t)x] != x) { p[(size_t)x] = p[(size_t)p[(size_t)x]]; x = p[(size_t)x]; } return x; }
    void unite(int a, int b) { a = find(a); b = find(b); if (a != b) p[(size_t)a] = b; }
};

// ---- permutation helpers (full maps on [0, N)) --------------------------------------
using Perm = std::vector<int>;
inline Perm perm_id(int N) { Perm p((size_t)N); std::iota(p.begin(), p.end(), 0); return p; }
inline Perm perm_compose(const Perm& a, const Perm& b) {  // (a o b)[x] = a[b[x]]
    Perm r(b.size());
    for (size_t i = 0; i < b.size(); ++i) r[i] = a[(size_t)b[i]];
    return r;
}
inline Perm perm_inverse(const Perm& p) {
    Perm r(p.size());
    for (size_t i = 0; i < p.size(); ++i) r[(size_t)p[i]] = (int)i;
    return r;
}

// ---- incremental BSGS used as a generator FILTER (Schreier-Sims sift) ----------------
//
// Base = the first-leaf individualized points ``first_base`` (a base for Aut(graph): the
// leaf is discrete, so only the identity fixes them all).  ``strong`` (an alias of the
// solver's kept generators) plus per-level transversals form a partial BSGS.  We SIFT each
// newly verified automorphism ``γ``: if it sifts to the identity it is provably already in
// ``<strong>`` -> DROP it; otherwise the residue (which fixes a base prefix) is kept as a new
// strong generator and the affected transversal levels are rebuilt.  We do NOT close under
// Schreier generators (that blows up memory/time on highly symmetric codes -- and is
// unnecessary: sift-to-identity is always a sound "already in the group" certificate, so the
// kept residues still generate Aut(graph) exactly, just with a few redundant generators).
struct BSGS {
    int N = 0;
    std::vector<int> base;                                        // base points
    std::vector<std::unordered_map<int, Perm>> transversal;       // per level: point -> coset rep
    std::vector<Perm>& strong;                                    // alias of Solver::gens

    explicit BSGS(std::vector<Perm>& gens_ref) : strong(gens_ref) {}

    void init(const std::vector<int>& base_, int N_) {
        base = base_;
        N = N_;
        transversal.assign(base.size(), {});
        for (size_t i = 0; i < base.size(); ++i) transversal[i][base[i]] = perm_id(N);
    }

    // Orbit of base[j] under the strong generators that fix base[0..j-1], with coset reps.
    void rebuild_level(int j) {
        transversal[(size_t)j].clear();
        transversal[(size_t)j][base[(size_t)j]] = perm_id(N);
        std::vector<const Perm*> Sj;
        for (const Perm& s : strong) {
            bool fix = true;
            for (int t = 0; t < j; ++t)
                if (s[(size_t)base[(size_t)t]] != base[(size_t)t]) { fix = false; break; }
            if (fix) Sj.push_back(&s);
        }
        std::vector<int> frontier{base[(size_t)j]};
        while (!frontier.empty()) {
            std::vector<int> next;
            for (int x : frontier) {
                Perm ux = transversal[(size_t)j].at(x);   // copy: insertion may rehash
                for (const Perm* s : Sj) {
                    int y = (*s)[(size_t)x];
                    if (!transversal[(size_t)j].count(y)) {
                        transversal[(size_t)j][y] = perm_compose(*s, ux);  // maps base[j] -> y
                        next.push_back(y);
                    }
                }
            }
            frontier.swap(next);
        }
    }

    // Sift ``g``; return true iff it reduces to the identity (already in <strong>).  On
    // false, ``residue`` fixes ``base[0..level_fail-1]`` and moves ``base[level_fail]`` to a
    // point outside its current orbit.
    bool sift(const Perm& g, int& level_fail, Perm& residue) const {
        Perm h = g;
        for (size_t i = 0; i < base.size(); ++i) {
            int beta = h[(size_t)base[i]];
            auto it = transversal[i].find(beta);
            if (it == transversal[i].end()) {
                level_fail = (int)i;
                residue = std::move(h);
                return false;
            }
            h = perm_compose(perm_inverse(it->second), h);   // now h fixes base[0..i]
        }
        return true;   // fixes the whole base => identity
    }

    void add(const Perm& residue, int from_level) {
        strong.push_back(residue);
        for (int j = from_level; j < (int)base.size(); ++j) rebuild_level(j);
    }
};

struct Solver {
    const Graph& g;
    int N, WN;
    bool use_invariant;

    // ALL distinct verified automorphisms (full perms on N) -- the orbit-pruning pool.  The
    // incremental union-find pruning wants every discovered automorphism (the "redundant" ones
    // still connect orbits sooner), so this is kept whole and is what the search prunes with.
    std::vector<Perm> gens;
    std::unordered_set<std::string> seen_gen;
    // A SMALL strong generating set, distilled from ``gens`` by an incremental Schreier-Sims
    // sift filter -- used for the OUTPUT and the exact order (one generator per orbit, not one
    // per explored leaf).  Decoupled from pruning so the reduction never slows the search.
    std::vector<Perm> out_gens;
    BSGS bsgs;
    bool bsgs_ready = false;

    // First (reference) leaf and its inverse, plus the per-level refinement invariant and
    // individualized base along the first path.
    std::vector<int> first_leaf;       // lab[] at the first discrete leaf
    std::vector<int> first_inv;        // first_inv[vertex] = position
    bool have_first = false;
    std::vector<uint64_t> path_inv;    // path_inv[level] = node invariant on the first path
    std::vector<int> first_base;       // individualized vertices along the first path

    // Per-level orbit-pruning frames live implicitly in the recursion via a stack:
    std::vector<int> base;             // current individualized vertices (path prefix)

    // Coordinate-driven search: an automorphism of this coloured incidence graph is determined
    // by its action on the n coordinate vertices [0,n) -- each codeword vertex's image is forced
    // by its (colour, support).  We branch ONLY on coordinate cells (target_cell) and, at a
    // coordinate-discrete leaf, look every codeword up by its permuted support (make_auto_coord).
    // This bounds the search depth by n instead of n+#codewords (no factorial codeword branching).
    int E = 0;                                            // #codeword vertices = N - n
    int Wn = 0;                                           // words_for(n)
    std::vector<Row> supports;                            // supports[e] = n-bit support of word n+e
    std::vector<int> wclass;                              // wclass[e] = colour of word vertex n+e
    std::unordered_map<std::string, int> support_index;  // packed n-bit support -> e

    explicit Solver(const Graph& g_, bool inv)
        : g(g_), N(g_.N), WN(g_.WN), use_invariant(inv), bsgs(out_gens) {
        E = g.N - g.n;
        Wn = words_for(std::max(1, g.n));
        supports.assign((size_t)E, Row((size_t)Wn, 0));
        wclass.assign((size_t)E, 0);
        support_index.reserve((size_t)E * 2 + 1);
        for (int e = 0; e < E; ++e) {
            const Row& a = g.adj[(size_t)(g.n + e)];     // word n+e neighbours only coords in [0,n)
            for (int w = 0; w < Wn; ++w) supports[(size_t)e][(size_t)w] = a[(size_t)w];
            wclass[(size_t)e] = g.colour[(size_t)(g.n + e)];
            support_index.emplace(
                std::string((const char*)supports[(size_t)e].data(), (size_t)Wn * sizeof(uint64_t)),
                e);
        }
    }

    // popcount(adj[v] & mask)
    inline int deg(int v, const Row& mask) const {
        const Row& a = g.adj[(size_t)v];
        int c = 0;
        for (int w = 0; w < WN; ++w) c += __builtin_popcountll(a[(size_t)w] & mask[(size_t)w]);
        return c;
    }

    // Equitable refinement.  ``active`` holds cell-start indices to use as splitters; the
    // partition is refined in place and a sound, isomorphism-invariant node invariant is
    // accumulated into ``inv``.  Splitting a cell only reorders within its own range, so the
    // start indices of untouched cells stay valid; every newly created subcell is queued as a
    // future splitter, so the loop reaches the (unique) coarsest equitable partition.
    void refine(std::vector<int>& lab, std::vector<int>& ptn, std::vector<int>& active,
                uint64_t& inv) {
        Row mask((size_t)WN, 0);
        size_t qh = 0;
        while (qh < active.size()) {
            int ws = active[(size_t)qh++];
            int we = ws;
            while (ptn[(size_t)we] != 0) ++we;                  // splitter cell W = [ws, we]
            for (int w = 0; w < WN; ++w) mask[(size_t)w] = 0;
            for (int i = ws; i <= we; ++i) gf2::set_bit(mask, lab[(size_t)i]);
            inv = inv * 1000003ull + (uint64_t)((ws + 1) * 131 + (we - ws + 1));

            // Bipartite skip: the incidence graph has edges only between coordinate vertices
            // [0,n) and codeword vertices [n,N), and no cell ever mixes the two types (the two
            // are separated by colour, and refinement only subdivides cells).  So a splitter
            // made of codewords can only ever split COORDINATE cells (codeword cells have zero
            // edges to it -> uniform deg 0 -> never split, never touch ``inv``), and a
            // coordinate splitter only codeword cells.  Scan just the opposite-type region
            // instead of all N -- for codeword-heavy codes a codeword splitter then scans n
            // (e.g. 56) instead of N (e.g. 2317).  Behaviour-identical to scanning all N.
            const bool splitter_is_coord = (lab[(size_t)ws] < g.n);
            int vs = splitter_is_coord ? g.n : 0;
            const int v_end = splitter_is_coord ? N : g.n;
            while (vs < v_end) {
                int ve = vs;
                while (ptn[(size_t)ve] != 0) ++ve;               // cell V = [vs, ve]
                if (ve > vs) {                                   // |V| > 1: try to split
                    std::vector<std::pair<int, int>> kv((size_t)(ve - vs + 1));
                    bool uniform = true;
                    int c0 = deg(lab[(size_t)vs], mask);
                    for (int i = vs; i <= ve; ++i) {
                        int c = deg(lab[(size_t)i], mask);
                        kv[(size_t)(i - vs)] = {c, lab[(size_t)i]};
                        if (c != c0) uniform = false;
                    }
                    if (!uniform) {
                        std::stable_sort(kv.begin(), kv.end(),
                                         [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                                             return a.first < b.first;
                                         });
                        for (int i = vs; i <= ve; ++i) lab[(size_t)i] = kv[(size_t)(i - vs)].second;
                        // Boundaries between distinct counts; queue every subcell start.
                        for (int i = vs; i <= ve; ++i) {
                            bool last = (i == ve) || (kv[(size_t)(i - vs)].first != kv[(size_t)(i - vs + 1)].first);
                            ptn[(size_t)i] = last ? 0 : 1;
                        }
                        active.push_back(vs);
                        inv = inv * 1000003ull + (uint64_t)((vs + 1) * 31);
                        for (int i = vs; i < ve; ++i)
                            if (ptn[(size_t)i] == 0) {
                                active.push_back(i + 1);
                                inv = inv * 1000003ull + (uint64_t)((i + 2) * 31);
                            }
                    }
                }
                vs = ve + 1;
            }
        }
    }

    // First non-singleton COORDINATE cell of minimum size; returns its start (or -1 if all
    // coordinate vertices are singletons).  Coordinates always occupy positions [0,n) (colour 0,
    // sorted first, and refinement only reorders within a cell), so we scan only [0,n) and never
    // branch on codeword cells.
    int target_cell(const std::vector<int>& ptn) const {
        int best = -1, best_len = 0;
        int vs = 0;
        while (vs < g.n) {
            int ve = vs;
            while (ptn[(size_t)ve] != 0) ++ve;
            int len = ve - vs + 1;
            if (len > 1 && (best == -1 || len < best_len)) { best = vs; best_len = len; }
            vs = ve + 1;
        }
        return best;
    }

    // Discrete <=> every COORDINATE cell is a singleton (codeword cells may still be coarse; their
    // images are forced by support, so we treat coordinate-discreteness as a search leaf).
    bool is_discrete(const std::vector<int>& ptn) const {
        for (int i = 0; i < g.n - 1; ++i) if (ptn[(size_t)i] != 0) return false;
        return true;
    }

    // Verify gamma = leaf ∘ first_leaf^{-1} is a graph automorphism (colours + adjacency).
    bool make_and_check_auto(const std::vector<int>& leaf, std::vector<int>& gamma) const {
        gamma.assign((size_t)N, 0);
        for (int pos = 0; pos < N; ++pos) gamma[(size_t)first_leaf[(size_t)pos]] = leaf[(size_t)pos];
        for (int v = 0; v < N; ++v)
            if (g.colour[(size_t)gamma[(size_t)v]] != g.colour[(size_t)v]) return false;
        Row tmp((size_t)WN, 0);
        for (int v = 0; v < N; ++v) {
            for (int w = 0; w < WN; ++w) tmp[(size_t)w] = 0;
            const Row& a = g.adj[(size_t)v];
            for (int w = 0; w < WN; ++w) {
                uint64_t bits = a[(size_t)w];
                while (bits) {
                    int b = w * 64 + __builtin_ctzll(bits);
                    bits &= bits - 1;
                    gf2::set_bit(tmp, gamma[(size_t)b]);
                }
            }
            if (tmp != g.adj[(size_t)gamma[(size_t)v]]) return false;
        }
        return true;
    }

    // Build the COORDINATE permutation gc (length n) from a coordinate-discrete leaf:
    // gc[first_leaf[p]] = lab[p] for p in [0,n).  Then verify gc is a code automorphism: every
    // selected codeword's gc-image support must again be a selected codeword of the SAME weight
    // class.  This is Leon's exact criterion (the selected weight classes span C and are
    // Aut-invariant), and since distinct binary codewords have distinct supports a passing gc is
    // a genuine automorphism -- so no full-N adjacency recheck is needed.  Working in length-n
    // coordinate space (not length-N) is what keeps the pruning pool / BSGS / seen-set cheap on
    // codes with a huge automorphism group and many codeword vertices.
    bool make_auto_coord(const std::vector<int>& lab, std::vector<int>& gc) const {
        gc.assign((size_t)g.n, -1);
        for (int p = 0; p < g.n; ++p) gc[(size_t)first_leaf[(size_t)p]] = lab[(size_t)p];
        Row img((size_t)Wn, 0);
        for (int e = 0; e < E; ++e) {
            for (int w = 0; w < Wn; ++w) img[(size_t)w] = 0;
            const Row& sup = supports[(size_t)e];
            for (int w = 0; w < Wn; ++w) {
                uint64_t bits = sup[(size_t)w];
                while (bits) {
                    int b = w * 64 + __builtin_ctzll(bits);
                    bits &= bits - 1;
                    gf2::set_bit(img, gc[(size_t)b]);
                }
            }
            auto it = support_index.find(
                std::string((const char*)img.data(), (size_t)Wn * sizeof(uint64_t)));
            if (it == support_index.end()) return false;
            if (wclass[(size_t)it->second] != wclass[(size_t)e]) return false;
        }
        return true;
    }

    // Recursive backtracking search.  ``lab/ptn`` describe the current (equitable) node;
    // ``level`` = length of ``base``.  Returns nothing; fills ``gens``.
    void search(std::vector<int> lab, std::vector<int> ptn, int level,
                std::vector<int> live, size_t mark) {
        if (is_discrete(ptn)) {
            if (!have_first) {
                first_leaf = lab;
                first_inv.assign((size_t)N, 0);
                for (int pos = 0; pos < N; ++pos) first_inv[(size_t)lab[(size_t)pos]] = pos;
                have_first = true;
                return;
            }
            std::vector<int> gc;                                  // length-n coordinate perm
            if (make_auto_coord(lab, gc)) {
                std::string key((const char*)gc.data(), gc.size() * sizeof(int));
                if (seen_gen.insert(std::move(key)).second) {     // a new distinct automorphism
                    if (!bsgs_ready) { bsgs.init(first_base, g.n); bsgs_ready = true; }
                    int level_fail;
                    Perm residue;
                    if (!bsgs.sift(gc, level_fail, residue))       // not yet in <out_gens>:
                        bsgs.add(residue, level_fail);             // keep its residue (-> out_gens)
                    gens.push_back(std::move(gc));                 // always feed the pruning pool
                }
            }
            return;
        }

        int ts = target_cell(ptn);
        int te = ts;
        while (ptn[(size_t)te] != 0) ++te;
        std::vector<int> cell(lab.begin() + ts, lab.begin() + te + 1);
        std::sort(cell.begin(), cell.end());

        // Orbit pruning: union-find over the n coordinates, fed by discovered automorphisms that
        // fix the current base prefix (base[0..level-1]).  Skip a target vertex sharing an orbit
        // with one already tried at this node.  Generators act on the n coordinates only (codeword
        // images are derived) and the target cell is always a coordinate cell, so the union-find
        // runs over [0,n), not [0,N).
        //
        // ``live`` = indices into ``gens`` of the pool automorphisms that fix base[0..level-1]
        // (covering gens[0..mark)); it is threaded DOWN the path so we never rescan the whole pool.
        // A pool gen that fails to fix base[0..level-1] also fails for every descendant (a longer
        // prefix), so filtering once and passing the survivors down turns the old per-node
        // O(|pool|) rescan into O(sum of |live|).  On highly symmetric codes the pool grows into the
        // tens of thousands and this rescan dominated the whole solve (engineer profile: 28.2s of a
        // 31.8s run); it is now ~0.15s, with byte-identical pruning (same nodes, same orbits).
        // ``absorb_new_gens`` only filters gens discovered since ``mark`` (found in this subtree).
        UnionFind uf;
        uf.init(g.n);
        for (int idx : live)
            for (int v = 0; v < g.n; ++v) uf.unite(v, gens[(size_t)idx][(size_t)v]);
        size_t gens_seen = mark;
        auto absorb_new_gens = [&]() {
            for (; gens_seen < gens.size(); ++gens_seen) {
                const std::vector<int>& gm = gens[gens_seen];
                bool fixes_prefix = true;
                for (int i = 0; i < level; ++i)
                    if (gm[(size_t)base[(size_t)i]] != base[(size_t)i]) { fixes_prefix = false; break; }
                if (fixes_prefix) {
                    for (int v = 0; v < g.n; ++v) uf.unite(v, gm[(size_t)v]);
                    live.push_back((int)gens_seen);   // survivor: visible to later siblings & children
                }
            }
        };

        std::vector<char> tried_rep((size_t)g.n, 0);
        for (int u : cell) {
            absorb_new_gens();
            if (tried_rep[(size_t)uf.find(u)]) continue;   // same orbit as a tried rep
            tried_rep[(size_t)uf.find(u)] = 1;

            // Individualize u: move it to the front of the target cell as its own singleton.
            std::vector<int> clab = lab, cptn = ptn;
            int pos = ts;
            while (clab[(size_t)pos] != u) ++pos;
            for (int i = pos; i > ts; --i) clab[(size_t)i] = clab[(size_t)i - 1];
            clab[(size_t)ts] = u;
            cptn[(size_t)ts] = 0;  // singleton boundary after u

            std::vector<int> active{ts};
            uint64_t inv = 1469598103934665603ull;
            refine(clab, cptn, active, inv);

            base.push_back(u);
            bool prune = false;
            if (!have_first) {
                // First descent (always cell[0] at each level): record the reference path.
                if ((int)path_inv.size() == level) path_inv.push_back(inv);
                if ((int)first_base.size() == level) first_base.push_back(u);
            } else if (use_invariant && level < (int)path_inv.size() && inv != path_inv[(size_t)level]) {
                prune = true;  // node invariant differs from the reference path: no automorphism here
            }
            if (!prune) {
                // child_live = pool gens fixing base[0..level] = those in ``live`` that also fix u.
                std::vector<int> child_live;
                child_live.reserve(live.size());
                for (int idx : live)
                    if (gens[(size_t)idx][(size_t)u] == u) child_live.push_back(idx);
                search(std::move(clab), std::move(cptn), level + 1, std::move(child_live),
                       gens.size());
            }
            base.pop_back();
        }
    }

    void run() {
        std::vector<int> lab((size_t)N), ptn((size_t)N, 1);
        std::iota(lab.begin(), lab.end(), 0);
        // Initial ordered partition by colour (stable by vertex within a colour).
        std::stable_sort(lab.begin(), lab.end(),
                         [&](int a, int b) { return g.colour[(size_t)a] < g.colour[(size_t)b]; });
        std::vector<int> active;
        int s = 0;
        for (int i = 0; i < N; ++i) {
            bool boundary = (i == N - 1) || g.colour[(size_t)lab[(size_t)i]] != g.colour[(size_t)lab[(size_t)i + 1]];
            ptn[(size_t)i] = boundary ? 0 : 1;
            if (boundary) { active.push_back(s); s = i + 1; }
        }
        uint64_t inv = 1469598103934665603ull;
        refine(lab, ptn, active, inv);
        search(std::move(lab), std::move(ptn), 0, std::vector<int>{}, 0);
    }

    // Exact group order = product over first-path levels of the fundamental orbit length of
    // base[L] under the subgroup of generators fixing base[0..L-1].  Returned as int64
    // factors (Python multiplies them as a big integer).
    std::vector<int64_t> order_factors() const {
        std::vector<int64_t> factors;
        for (size_t L = 0; L < first_base.size(); ++L) {
            // generators fixing base[0..L-1]  (the reduced strong set generates the same group)
            std::vector<const std::vector<int>*> H;
            for (const auto& gm : out_gens) {
                bool ok = true;
                for (size_t i = 0; i < L; ++i)
                    if (gm[(size_t)first_base[i]] != first_base[i]) { ok = false; break; }
                if (ok) H.push_back(&gm);
            }
            // orbit of base[L] under H (BFS over the n coordinates; H are length-n coord perms)
            std::vector<char> seen((size_t)g.n, 0);
            std::vector<int> frontier{first_base[L]};
            seen[(size_t)first_base[L]] = 1;
            int64_t cnt = 1;
            while (!frontier.empty()) {
                std::vector<int> next;
                for (int x : frontier)
                    for (auto* gm : H) {
                        int y = (*gm)[(size_t)x];
                        if (!seen[(size_t)y]) { seen[(size_t)y] = 1; ++cnt; next.push_back(y); }
                    }
                frontier.swap(next);
            }
            factors.push_back(cnt);
        }
        return factors;
    }
};

// Opaque result handle handed back to Python.
struct LeonResult {
    int n = 0;
    int dim = 0;
    int num_codewords = 0;
    int64_t num_incidences = 0;
    int num_classes = 0;
    bool ok = false;
    std::vector<std::vector<int>> gens;   // restricted to coordinates [0, n)
    std::vector<int64_t> factors;         // order = product(factors)
    std::vector<int> weights;
    int selector = 0;
    int modulus = 0;
    int residue = 0;
    int64_t enumeration_ns = 0;
    int64_t search_ns = 0;
};

}  // namespace

// =====================================================================================
//  C ABI
// =====================================================================================
extern "C" {

// Run the full pipeline on G (m x n row-major uint8).  Returns an opaque handle (NULL on
// allocation failure).  ``use_invariant`` toggles the first-leaf refinement-invariant
// pruning (orbit pruning is always on).  If dim(C) > max_dim the handle reports ok=0.
void* qaut_leon_run_ex(const uint8_t* G, int32_t m, int32_t n, int32_t max_dim,
                       int32_t use_invariant, int32_t selector, int32_t max_modulus) {
    LeonResult* R = new LeonResult();
    R->n = n;
    const auto enum_start = std::chrono::steady_clock::now();
    Codewords cw = enumerate_spanning_codewords(G, m, n, max_dim, selector, max_modulus);
    const auto enum_end = std::chrono::steady_clock::now();
    R->enumeration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        enum_end - enum_start).count();
    R->dim = cw.dim;
    R->ok = cw.ok;
    if (!cw.ok) return R;                          // dim too large
    R->num_codewords = (int)cw.words.size();
    for (const Row& word : cw.words)
        for (uint64_t bits : word) R->num_incidences += __builtin_popcountll(bits);
    R->num_classes = (int)cw.weights.size();
    R->weights = cw.weights;
    R->selector = cw.selector;
    R->modulus = cw.modulus;
    R->residue = cw.residue;

    if (cw.dim == 0) {                             // C = {0}: Aut = full Sym(n)
        // Sym(n) generated by a transposition (0 1) and the n-cycle.
        if (n >= 2) {
            std::vector<int> t(n); std::iota(t.begin(), t.end(), 0); std::swap(t[0], t[1]);
            R->gens.push_back(t);
        }
        if (n >= 3) {
            std::vector<int> c(n); for (int i = 0; i < n; ++i) c[(size_t)i] = (i + 1) % n;
            R->gens.push_back(c);
        }
        for (int i = 2; i <= n; ++i) R->factors.push_back(i);  // |Sym(n)| = n!
        return R;
    }

    Graph g = build_graph(cw);
    Solver solver(g, use_invariant != 0);
    const auto search_start = std::chrono::steady_clock::now();
    solver.run();
    const auto search_end = std::chrono::steady_clock::now();
    R->search_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        search_end - search_start).count();

    // Restrict the reduced strong generators to coordinate vertices [0, n).
    for (const auto& gm : solver.out_gens) {
        std::vector<int> p(n);
        for (int i = 0; i < n; ++i) p[(size_t)i] = gm[(size_t)i];
        R->gens.push_back(std::move(p));
    }
    R->factors = solver.order_factors();
    return R;
}

// Backward-compatible ABI: the historical entry point keeps the exact legacy selector.
void* qaut_leon_run(const uint8_t* G, int32_t m, int32_t n, int32_t max_dim,
                    int32_t use_invariant) {
    return qaut_leon_run_ex(G, m, n, max_dim, use_invariant, 0, 0);
}

int32_t qaut_leon_ok(void* h) { return h && ((LeonResult*)h)->ok ? 1 : 0; }
int32_t qaut_leon_dim(void* h) { return h ? ((LeonResult*)h)->dim : -1; }
int32_t qaut_leon_n(void* h) { return h ? ((LeonResult*)h)->n : -1; }
int32_t qaut_leon_num_codewords(void* h) { return h ? ((LeonResult*)h)->num_codewords : -1; }
int64_t qaut_leon_num_incidences(void* h) { return h ? ((LeonResult*)h)->num_incidences : -1; }
int32_t qaut_leon_num_classes(void* h) { return h ? ((LeonResult*)h)->num_classes : -1; }
int32_t qaut_leon_num_gens(void* h) { return h ? (int32_t)((LeonResult*)h)->gens.size() : -1; }
int32_t qaut_leon_num_factors(void* h) { return h ? (int32_t)((LeonResult*)h)->factors.size() : -1; }
int32_t qaut_leon_selector(void* h) { return h ? ((LeonResult*)h)->selector : -1; }
int32_t qaut_leon_modulus(void* h) { return h ? ((LeonResult*)h)->modulus : -1; }
int32_t qaut_leon_residue(void* h) { return h ? ((LeonResult*)h)->residue : -1; }
int64_t qaut_leon_enumeration_ns(void* h) { return h ? ((LeonResult*)h)->enumeration_ns : -1; }
int64_t qaut_leon_search_ns(void* h) { return h ? ((LeonResult*)h)->search_ns : -1; }

// Copy the generators (num_gens x n int32) into ``buf``.
void qaut_leon_copy_gens(void* h, int32_t* buf) {
    LeonResult* R = (LeonResult*)h;
    int n = R->n;
    for (size_t gi = 0; gi < R->gens.size(); ++gi)
        for (int i = 0; i < n; ++i) buf[gi * (size_t)n + (size_t)i] = R->gens[gi][(size_t)i];
}

// Copy the order factors (num_factors int64) into ``buf``.
void qaut_leon_copy_factors(void* h, int64_t* buf) {
    LeonResult* R = (LeonResult*)h;
    for (size_t i = 0; i < R->factors.size(); ++i) buf[i] = R->factors[i];
}

// Copy the used weight classes (num_classes int32) into ``buf``.
void qaut_leon_copy_weights(void* h, int32_t* buf) {
    LeonResult* R = (LeonResult*)h;
    for (size_t i = 0; i < R->weights.size(); ++i) buf[i] = R->weights[i];
}

void qaut_leon_free(void* h) { delete (LeonResult*)h; }

}  // extern "C"
