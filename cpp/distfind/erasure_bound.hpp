#pragma once
#include "distfind/gf2.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace distfind {

struct ErasureData {
    int n, m, words;
    GF2Mat columns;
    std::vector<std::vector<int>> incident;
    std::vector<u64> labels, check_hashes;
    std::vector<std::vector<int>> row_columns;
    std::vector<u64> column_hashes, index_key;
    std::vector<int> index_head, index_next, pair_head;
    struct Pair { int a, b, next; };
    std::vector<Pair> pairs;

    ErasureData(const GF2Mat& H, const GF2Mat& L)
        : n(H.cols), m(H.rows), words(words_for(H.rows)), columns(transpose(H)),
          incident(n), labels(n, 0), check_hashes(m), row_columns(m), column_hashes(n, 0),
          index_next(n, -1) {
        if (L.cols != n || L.rows > 64) throw std::invalid_argument("erasure detector shape");
        for (int q = 0; q < n; ++q) {
            for (int j = 0; j < m; ++j) if (H.get(j, q)) {
                incident[q].push_back(j); row_columns[j].push_back(q);
            }
            for (int j = 0; j < L.rows; ++j) if (L.get(j, q)) labels[q] |= 1ull << j;
        }
        for (int j = 0; j < m; ++j) {
            u64 x = (u64)j + 0x9e3779b97f4a7c15ull;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
            check_hashes[j] = x ^ (x >> 31);
        }
        size_t capacity = 1;
        while (capacity < 2 * (size_t)n) capacity <<= 1;
        index_key.resize(capacity); index_head.assign(capacity, -1);
        for (int q = 0; q < n; ++q) {
            for (int j : incident[q]) column_hashes[q] ^= check_hashes[j];
            u64 h = column_hashes[q];
            size_t slot = h & (capacity - 1);
            while (index_head[slot] >= 0 && index_key[slot] != h)
                slot = (slot + 1) & (capacity - 1);
            index_key[slot] = h;
            index_next[q] = index_head[slot]; index_head[slot] = q;
        }
        capacity = 1;
        size_t pair_count = n ? (size_t)n * (n - 1) / 2 : 0;
        while (capacity < 2 * pair_count) capacity <<= 1;
        pair_head.assign(capacity, -1); pairs.reserve(pair_count);
        for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b) {
            size_t slot = (column_hashes[a] ^ column_hashes[b]) & (capacity - 1);
            pairs.push_back({a, b, pair_head[slot]});
            pair_head[slot] = (int)pairs.size() - 1;
        }
    }
};

// Prove that no logical is supported on an erased region plus <=2 further columns.
// Hash collisions can only make this return false (inconclusive), never a false proof.
// A fixed support asks for an affine completion instead. Regions above 128 columns
// are conservatively declined; this is an optional pruning bound, not a full solver.
class ErasureBound {
    const ErasureData& d;
    std::vector<int> count, sole, touched, queue, pivots;
    std::vector<unsigned char> state;
    std::vector<std::pair<int, int>> peeled;
    std::vector<u64> basis, basis_labels, word, phi, logical;
    std::vector<u64> table_key, table_label;
    std::vector<unsigned> table_stamp;
    std::vector<unsigned char> table_diverse;
    std::vector<int> affected;
    std::vector<unsigned> affected_stamp;
    std::vector<u64> changed_hash, changed_label;
    bool verify_projection;
    unsigned epoch = 0;

    void assign_pivot(int pivot, const u64* column, u64 label) {
        u64 h = 0, l = label;
        for (int w = 0; w < d.words; ++w)
            for (u64 bits = column[w]; bits; bits &= bits - 1) {
                int j = 64 * w + __builtin_ctzll(bits);
                if (j != pivot) { h ^= phi[j]; l ^= logical[j]; }
            }
        phi[pivot] = h;
        logical[pivot] = l;
    }

    std::pair<u64, u64> project(int q) const {
        u64 h = 0, l = d.labels[q];
        for (int j : d.incident[q]) { h ^= phi[j]; l ^= logical[j]; }
        return {h, l};
    }

public:
    explicit ErasureBound(const ErasureData& data, bool verify = false)
        : d(data), count(d.m), sole(d.m), state(d.n),
          word(d.words), phi(d.m), logical(d.m), affected_stamp(d.n, 0),
          changed_hash(d.n), changed_label(d.n), verify_projection(verify) {
        size_t capacity = 1;
        while (capacity < 2 * (size_t)d.n) capacity <<= 1;
        table_key.resize(capacity); table_label.resize(capacity);
        table_stamp.assign(capacity, 0);
        table_diverse.resize(capacity);
        touched.reserve(d.m); queue.reserve(d.m);
        peeled.reserve(d.n); pivots.reserve(d.n);
        basis.reserve((size_t)d.n * d.words); basis_labels.reserve(d.n);
        affected.reserve(d.n);
    }

    template<class Eligible>
    bool excludes(const std::vector<int>& region, int extra, Eligible eligible,
                  const std::vector<int>& fixed = {}) {
        if (extra < 0 || extra > 2) throw std::invalid_argument("erasure extra must be 0, 1, or 2");
        if (region.size() > 128) return false;
        std::fill(count.begin(), count.end(), 0);
        std::fill(sole.begin(), sole.end(), 0);
        std::fill(state.begin(), state.end(), 0);
        touched.clear(); queue.clear(); peeled.clear();
        pivots.clear(); basis.clear(); basis_labels.clear();
        for (int q : region) {
            if (q < 0 || q >= d.n || state[q]) throw std::invalid_argument("invalid erasure region");
            state[q] = 1;
            for (int j : d.incident[q]) {
                if (!count[j]) touched.push_back(j);
                ++count[j]; sole[j] ^= q;
            }
        }
        for (int q : fixed) {
            if (q < 0 || q >= d.n || state[q]) throw std::invalid_argument("invalid fixed support");
            state[q] = 3;
        }
        for (int j : touched) if (count[j] == 1) queue.push_back(j);
        for (size_t head = 0; head < queue.size(); ++head) {
            int pivot = queue[head];
            if (count[pivot] != 1) continue;
            int q = sole[pivot];
            state[q] = 2;
            peeled.emplace_back(q, pivot);
            for (int j : d.incident[q]) {
                --count[j]; sole[j] ^= q;
                if (count[j] == 1) queue.push_back(j);
            }
        }
        // Only the residual stopping set needs dense elimination.
        for (int q : region) if (state[q] == 1) {
            for (int w = 0; w < d.words; ++w)
                word[w] = d.columns.w[(size_t)q * d.words + w];
            u64 label = d.labels[q];
            for (size_t b = 0; b < pivots.size(); ++b)
                if ((word[pivots[b] >> 6] >> (pivots[b] & 63)) & 1) {
                    for (int w = 0; w < d.words; ++w) word[w] ^= basis[b * d.words + w];
                    label ^= basis_labels[b];
                }
            int pivot = -1;
            for (int w = 0; w < d.words; ++w) if (word[w]) {
                pivot = 64 * w + __builtin_ctzll(word[w]); break;
            }
            if (pivot < 0) {
                if (label) return false;
            } else {
                pivots.push_back(pivot);
                basis.insert(basis.end(), word.begin(), word.end());
                basis_labels.push_back(label);
            }
        }
        if (extra == 0 && fixed.empty()) return true;

        // Construct phi*H_region=0 and F*H_region=L_region. The first is a
        // consistency fingerprint; the second cleans every detector off the region.
        phi = d.check_hashes;
        std::fill(logical.begin(), logical.end(), 0);
        for (size_t b = pivots.size(); b-- > 0;)
            assign_pivot(pivots[b], &basis[b * d.words], basis_labels[b]);
        for (size_t i = peeled.size(); i-- > 0;) {
            auto [q, pivot] = peeled[i];
            u64 h = 0, l = d.labels[q];
            for (int j : d.incident[q]) if (j != pivot) { h ^= phi[j]; l ^= logical[j]; }
            phi[pivot] = h; logical[pivot] = l;
        }
        if (verify_projection) for (int q : region) {
            auto [h, l] = project(q);
            if (h || l) throw std::logic_error("invalid erasure projection");
        }
        u64 target_hash = 0, target_label = 0;
        for (int q : fixed) {
            auto [h, l] = project(q);
            target_hash ^= h; target_label ^= l;
        }
        if (!target_hash && target_label) return false;
        if (extra == 0) return true;
        if (++epoch == 0) {
            std::fill(table_stamp.begin(), table_stamp.end(), 0);
            std::fill(affected_stamp.begin(), affected_stamp.end(), 0);
            epoch = 1;
        }
        affected.clear();
        auto change_row = [&](int j) {
            u64 dh = phi[j] ^ d.check_hashes[j], dl = logical[j];
            if (!dh && !dl) return;
            for (int q : d.row_columns[j]) if (!state[q] && eligible(q)) {
                if (affected_stamp[q] != epoch) {
                    affected_stamp[q] = epoch;
                    changed_hash[q] = d.column_hashes[q];
                    changed_label[q] = d.labels[q];
                    affected.push_back(q);
                }
                changed_hash[q] ^= dh; changed_label[q] ^= dl;
            }
        };
        for (int j : pivots) change_row(j);
        for (auto [q, pivot] : peeled) change_row(pivot);
        auto unchanged = [&](int q) {
            return !state[q] && affected_stamp[q] != epoch && eligible(q);
        };
        size_t original = target_hash & (d.index_key.size() - 1);
        while (d.index_head[original] >= 0 && d.index_key[original] != target_hash)
            original = (original + 1) & (d.index_key.size() - 1);
        for (int q = d.index_head[original]; q >= 0; q = d.index_next[q])
            if (d.labels[q] != target_label && unchanged(q)) return false;
        if (extra == 2) {
            for (int p = d.pair_head[target_hash & (d.pair_head.size() - 1)]; p >= 0;
                 p = d.pairs[p].next) {
                auto pair = d.pairs[p];
                if ((d.column_hashes[pair.a] ^ d.column_hashes[pair.b]) == target_hash &&
                    (d.labels[pair.a] ^ d.labels[pair.b]) != target_label &&
                    unchanged(pair.a) && unchanged(pair.b)) return false;
            }
        }
        for (int q : affected) {
            u64 h = changed_hash[q], l = changed_label[q];
            if (h == target_hash && l != target_label) return false;
            if (extra == 1) continue;
            u64 wanted = h ^ target_hash, wanted_label = l ^ target_label;
            original = wanted & (d.index_key.size() - 1);
            while (d.index_head[original] >= 0 && d.index_key[original] != wanted)
                original = (original + 1) & (d.index_key.size() - 1);
            for (int other = d.index_head[original]; other >= 0; other = d.index_next[other])
                if (d.labels[other] != wanted_label && unchanged(other)) return false;
            size_t query = wanted & (table_key.size() - 1);
            while (table_stamp[query] == epoch && table_key[query] != wanted)
                query = (query + 1) & (table_key.size() - 1);
            if (table_stamp[query] == epoch &&
                (table_diverse[query] || table_label[query] != wanted_label)) return false;
            size_t slot = h & (table_key.size() - 1);
            while (table_stamp[slot] == epoch && table_key[slot] != h)
                slot = (slot + 1) & (table_key.size() - 1);
            if (table_stamp[slot] == epoch) {
                if (table_label[slot] != l) table_diverse[slot] = 1;
            } else {
                table_stamp[slot] = epoch;
                table_key[slot] = h; table_label[slot] = l;
                table_diverse[slot] = 0;
            }
        }
        return true;
    }

    bool excludes(const std::vector<int>& region, const std::vector<int>& candidates, int extra) {
        std::vector<unsigned char> allowed(d.n, 0);
        for (int q : candidates) {
            if (q < 0 || q >= d.n) throw std::invalid_argument("invalid erasure candidate");
            allowed[q] = 1;
        }
        return excludes(region, extra, [&](int q) { return allowed[q] != 0; });
    }
};

} // namespace distfind
