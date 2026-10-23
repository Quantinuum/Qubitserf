#include "distfind/cc.hpp"
#include "distfind/css.hpp"
#include "distfind/erasure_bound.hpp"
#include "distfind/progress.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <numeric>
#include <set>
#include <thread>
#include <type_traits>
#include <vector>

namespace distfind {

namespace {

// Sparse, column-oriented view of a check matrix H and logical detector L.
struct CCData {
    int n = 0, mh = 0, kl = 0, sw = 0, lw = 0;
    int max_degree = 0;
    bool even = false;
    bool small_rows = true;
    bool progress = false;
    int cyclic_half = 0;
    std::vector<u64> HT;              // n columns of H, sw words each (qubit -> checks)
    std::vector<u64> LT;              // n columns of L, lw words each (qubit -> logicals)
    std::vector<std::vector<int>> HE; // mh checks -> list of qubits in that check
    std::vector<std::vector<int>> QS; // qubit -> trivial generators containing it
    std::vector<std::vector<int>> SQ; // cut -> constrained qubits (may omit its pivot)
    std::vector<std::vector<std::pair<int, u64>>> QH;
    std::vector<int> initially_forbidden;
    std::shared_ptr<ErasureData> erasure;
    std::vector<u32> left_labels;
    std::vector<u8> left_distance;
    int left_radius = 0;
    std::vector<int> half_weight, seeds;
    std::vector<u64> hashes;
    struct Pair { int a, b, next; };
    std::vector<Pair> pairs;
    std::vector<int> buckets;
};

u64 mix64(u64 x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// A sufficient (not necessary) automorphism test. False negatives only cost time.
bool block_shift_preserves(const GF2Mat& H) {
    const int m = H.cols / 2;
    std::set<std::vector<int>> original, shifted;
    for (int r = 0; r < H.rows; ++r) {
        std::vector<int> a, b;
        for (int q = 0; q < H.cols; ++q) if (H.get(r, q)) {
            a.push_back(q);
            b.push_back((q / m) * m + (q + 1) % m);
        }
        std::sort(b.begin(), b.end());
        original.insert(a);
        shifted.insert(b);
    }
    return original == shifted;
}

struct PermutationSpan {
    GF2Mat basis;
    std::vector<int> pivots;
    std::vector<std::vector<int>> supports;
    std::set<std::vector<u64>> rows;

    explicit PermutationSpan(const GF2Mat& matrix) : basis(matrix), supports(matrix.rows) {
        rref(basis, &pivots);
        for (int row = 0; row < matrix.rows; ++row) {
            rows.emplace(matrix.row(row), matrix.row(row) + matrix.stride);
            for (int column = 0; column < matrix.cols; ++column)
                if (matrix.get(row, column)) supports[row].push_back(column);
        }
    }

    bool contains_image(const PermutationSpan& source, const std::vector<int>& permutation) const {
        std::vector<u64> image(basis.stride);
        for (const auto& support : source.supports) {
            std::fill(image.begin(), image.end(), 0);
            for (int column : support) {
                int mapped = permutation[column];
                image[mapped >> 6] |= 1ull << (mapped & 63);
            }
            if (rows.find(image) == rows.end() && !in_span(basis, pivots, image.data()))
                return false;
        }
        return true;
    }

    bool preserves(const std::vector<int>& permutation) const {
        return contains_image(*this, permutation);
    }
};

bool css_sectors_equivalent(const GF2Mat& checks_x, const GF2Mat& checks_z) {
    if (checks_x.cols != checks_z.cols || checks_x.cols > 2048) return false;
    PermutationSpan span_x(checks_x), span_z(checks_z);
    if (span_x.pivots.size() != span_z.pivots.size()) return false;
    std::vector<int> permutation(checks_x.cols);
    for (int candidate = 0; candidate < 3; ++candidate) {
        for (int column = 0; column < checks_x.cols; ++column) {
            if (candidate == 0) permutation[column] = checks_x.cols - 1 - column;
            else if (candidate == 1) permutation[column] = column;
            else permutation[column] = (column + checks_x.cols / 2) % checks_x.cols;
        }
        if (span_z.contains_image(span_x, permutation) &&
            span_x.contains_image(span_z, permutation)) return true;
    }
    return false;
}

std::vector<int> translation_seeds(const GF2Mat& checks, const GF2Mat& trivial) {
    std::vector<int> representatives(checks.cols), permutation(checks.cols);
    std::iota(representatives.begin(), representatives.end(), 0);
    if (checks.cols > 2048) return representatives;
    PermutationSpan check_span(checks), trivial_span(trivial);
    auto root = [&](int column) {
        while (representatives[column] != column) column = representatives[column];
        return column;
    };
    for (int block = 2; block <= checks.cols; ++block) {
        if (checks.cols % block) continue;
        for (int shift = 1; shift < block; ++shift) {
            if (block % shift) continue;
            bool merges = false;
            for (int column = 0; column < checks.cols; ++column) {
                permutation[column] = column / block * block + (column + shift) % block;
                merges = merges || root(column) != root(permutation[column]);
            }
            if (!merges || !check_span.preserves(permutation) ||
                !trivial_span.preserves(permutation)) continue;
            for (int column = 0; column < checks.cols; ++column) {
                int first = root(column), second = root(permutation[column]);
                representatives[std::max(first, second)] = std::min(first, second);
            }
        }
    }
    std::vector<int> seeds;
    for (int column = 0; column < checks.cols; ++column)
        if (root(column) == column) seeds.push_back(column);
    return seeds;
}

GF2Mat order_checks(const GF2Mat& H) {
    if (H.rows < 2 || H.cols == 0) return H;
    std::vector<std::vector<int>> rows(H.rows), columns(H.cols);
    for (int j = 0; j < H.rows; ++j)
        for (int q = 0; q < H.cols; ++q) if (H.get(j, q)) {
            rows[j].push_back(q); columns[q].push_back(j);
        }
    std::vector<int> order;
    std::vector<char> seen(H.rows, 0);
    if (H.cols) for (int j : columns[0]) { seen[j] = 1; order.push_back(j); }
    size_t head = 0;
    for (int start = -1; start < H.rows; ++start) {
        if (start >= 0 && !seen[start]) {
            seen[start] = 1; order.push_back(start);
        }
        while (head < order.size()) {
            int j = order[head++];
            for (int q : rows[j]) for (int neighbor : columns[q])
                if (!seen[neighbor]) { seen[neighbor] = 1; order.push_back(neighbor); }
        }
    }
    GF2Mat out(H.rows, H.cols);
    std::reverse(order.begin(), order.end());
    for (int j = 0; j < H.rows; ++j)
        std::copy(H.row(order[j]), H.row(order[j]) + H.stride, out.row(j));
    return out;
}

GF2Mat augment_sparse_checks(const GF2Mat& matrix) {
    if (matrix.rows < 2 || matrix.rows > 512 || matrix.cols > 2048) return matrix;
    std::vector<int> weights(matrix.rows);
    for (int row = 0; row < matrix.rows; ++row) {
        weights[row] = matrix.row_weight(row);
        if (weights[row] > 16) return matrix;
    }
    auto less = [](const std::vector<u64>& first, const std::vector<u64>& second) {
        return std::lexicographical_compare(first.rbegin(), first.rend(),
                                            second.rbegin(), second.rend());
    };
    std::set<std::vector<u64>, decltype(less)> rows(less);
    for (int row = 0; row < matrix.rows; ++row)
        rows.emplace(matrix.row(row), matrix.row(row) + matrix.stride);
    size_t original_count = rows.size(), capacity = std::min<size_t>(2048, 4 * matrix.rows);
    std::vector<u64> combined(matrix.stride);
    for (int first = 0; first < matrix.rows && rows.size() < capacity; ++first)
        for (int second = 0; second < first && rows.size() < capacity; ++second) {
            int weight = 0;
            for (int word = 0; word < matrix.stride; ++word) {
                combined[word] = matrix.row(first)[word] ^ matrix.row(second)[word];
                weight += __builtin_popcountll(combined[word]);
            }
            if (weight > 0 && weight <= std::max(weights[first], weights[second]))
                rows.insert(combined);
        }
    if (rows.size() == original_count) return matrix;
    GF2Mat augmented((int)rows.size(), matrix.cols);
    int row = 0;
    for (const auto& support : rows)
        std::copy(support.begin(), support.end(), augmented.row(row++));
    return augmented;
}

CCData build_cc(const GF2Mat& original, const GF2Mat& L, const GF2Mat* trivial = nullptr) {
    GF2Mat H = order_checks(augment_sparse_checks(original)), augmented_trivial;
    if (trivial) {
        augmented_trivial = augment_sparse_checks(*trivial);
        trivial = &augmented_trivial;
    }
    CCData d;
    d.n = H.cols; d.mh = H.rows; d.kl = L.rows;
    d.sw = words_for(d.mh); d.lw = words_for(d.kl);
    d.HT.assign((size_t)d.n * d.sw, 0ull);
    d.LT.assign((size_t)d.n * d.lw, 0ull);
    d.hashes.assign(d.n, 0);
    d.even = true;
    for (int q = 0; q < d.n; ++q) {
        int degree = 0;
        for (int j = 0; j < d.mh; ++j) if (H.get(j, q)) {
            d.HT[(size_t)q * d.sw + (j >> 6)] |= 1ull << (j & 63);
            d.hashes[q] ^= mix64(j);
            ++degree;
        }
        d.max_degree = std::max(d.max_degree, degree);
        d.even = d.even && (degree % 2 == 1);
        for (int c = 0; c < d.kl; ++c) if (L.get(c, q))
            d.LT[(size_t)q * d.lw + (c >> 6)] |= 1ull << (c & 63);
    }
    d.HE.assign(d.mh, {});
    for (int j = 0; j < d.mh; ++j)
        for (int q = 0; q < d.n; ++q) if (H.get(j, q)) d.HE[j].push_back(q);
    for (const auto& row : d.HE) d.small_rows = d.small_rows && row.size() <= 64;
    d.QH.resize(d.n);
    if (d.small_rows)
        for (int j = 0; j < d.mh; ++j)
            for (size_t i = 0; i < d.HE[j].size(); ++i)
                d.QH[d.HE[j][i]].emplace_back(j, 1ull << i);
    d.QS.resize(d.n);
    d.initially_forbidden.assign(d.n, 0);
    if (trivial) {
        d.SQ.resize(trivial->rows);
        for (int r = 0; r < trivial->rows; ++r) {
            int weight = trivial->row_weight(r);
            bool skip_first = weight > 0 && weight % 2 == 0;
            d.half_weight.push_back((weight - (skip_first ? 1 : 0)) / 2);
            for (int q = 0; q < d.n; ++q) if (trivial->get(r, q)) {
                // Equal-weight stabilizer moves may not improve sorted support.
                if (skip_first) { skip_first = false; continue; }
                d.QS[q].push_back(r);
                d.SQ[r].push_back(q);
                if (d.half_weight[r] == 0) ++d.initially_forbidden[q];
            }
        }
        if (d.n >= 2) {
            std::vector<int> first, second;
            for (int r = 0; r < trivial->rows; ++r) {
                if (trivial->get(r, 0)) first.push_back(r);
                if (trivial->get(r, 1)) second.push_back(r);
            }
            if (first.size() <= 8 && second.size() <= 8) {
                std::set<std::pair<int, int>> used;
                for (int a : first) for (int b : second) {
                    if (a == b || !used.emplace(std::min(a, b), std::max(a, b)).second) continue;
                    std::vector<int> support;
                    for (int q = 0; q < d.n; ++q)
                        if (trivial->get(a, q) != trivial->get(b, q)) support.push_back(q);
                    if (support.size() < 2 || support.size() > 10 ||
                        support[0] != 0 || support[1] != 1) continue;
                    int r = (int)d.half_weight.size(), skip = support.size() % 2 == 0 ? 1 : 0;
                    d.half_weight.push_back(((int)support.size() - skip) / 2);
                    d.SQ.emplace_back();
                    for (size_t i = skip; i < support.size(); ++i) {
                        int q = support[i];
                        d.QS[q].push_back(r); d.SQ[r].push_back(q);
                        if (d.half_weight[r] == 0) ++d.initially_forbidden[q];
                    }
                }
            }
        }
    }
    d.seeds.resize(d.n);
    std::iota(d.seeds.begin(), d.seeds.end(), 0);
    if (trivial) d.seeds = translation_seeds(H, *trivial);
    if (trivial && d.n >= 4 && d.n % 2 == 0 &&
        block_shift_preserves(H) && block_shift_preserves(*trivial)) {
        d.seeds = {0, d.n / 2};
        d.cyclic_half = d.n / 2;
    }
    if (d.cyclic_half && d.cyclic_half <= 512 && d.kl) {
        int half = d.cyclic_half;
        GF2Mat right(H.rows, half);
        for (int j = 0; j < H.rows; ++j)
            for (int q = 0; q < half; ++q) if (H.get(j, half + q)) right.set(j, q, 1);
        GF2Mat dependencies = nullspace(transpose(right));
        GF2Mat projected(dependencies.rows, half);
        for (int j = 0; j < dependencies.rows; ++j)
            for (int q = 0; q < half; ++q)
                if (vec_dot(dependencies.row(j), &d.HT[(size_t)q * d.sw], d.sw))
                    projected.set(j, q, 1);
        rref(projected); drop_zero_rows(projected);
        if (projected.rows <= 20) {
            d.left_labels.assign(half, 0);
            for (int q = 0; q < half; ++q)
                for (int j = 0; j < projected.rows; ++j)
                    if (projected.get(j, q)) d.left_labels[q] |= 1u << j;
            std::vector<u32> generators = d.left_labels;
            std::sort(generators.begin(), generators.end());
            generators.erase(std::unique(generators.begin(), generators.end()), generators.end());
            generators.erase(std::remove(generators.begin(), generators.end(), 0), generators.end());
            size_t states = size_t(1) << projected.rows;
            if (states * generators.size() <= 40000000) {
                d.left_distance.assign(states, 255);
                std::vector<u32> queue;
                queue.reserve(states); queue.push_back(0); d.left_distance[0] = 0;
                for (size_t head = 0; head < queue.size(); ++head) {
                    u32 label = queue[head];
                    for (u32 g : generators) if (d.left_distance[label ^ g] == 255) {
                        d.left_distance[label ^ g] = d.left_distance[label] + 1;
                        queue.push_back(label ^ g);
                    }
                }
                if (queue.size() != states) throw std::logic_error("incomplete projected syndrome table");
                d.left_radius = d.left_distance[queue.back()];
            }
        }
    }
    if (d.left_distance.empty() && d.small_rows && d.max_degree <= 4 &&
        L.rows > 0 && L.rows <= 64 && d.n <= 1024)
        d.erasure = std::make_shared<ErasureData>(H, L);

    // Bound preprocessing memory: at most ~40 MiB, independent of target weight.
    // Larger inputs retain the recursive exact search without a pair table.
    if (d.n >= 2 && d.n <= 2048) {
        size_t count = (size_t)d.n * (d.n - 1) / 2, capacity = 1;
        while (capacity < 2 * count) capacity <<= 1;
        d.buckets.assign(capacity, -1);
        d.pairs.reserve(count);
        for (int a = 0; a < d.n; ++a) for (int b = a + 1; b < d.n; ++b) {
            size_t bucket = (d.hashes[a] ^ d.hashes[b]) & (capacity - 1);
            d.pairs.push_back({a, b, d.buckets[bucket]});
            d.buckets[bucket] = (int)d.pairs.size() - 1;
        }
    }
    return d;
}

// Per-thread DFS state (syndrome / logical maintained by XOR + backtrack).
struct CCJob {
    int seed;
    int gap, second;
    std::vector<int> prefix;
};

template<int FixedSW = 0, int FixedLW = 0, bool AtMost = false>
struct CCSearch {
    const CCData* D;
    int target;
    std::conditional_t<FixedSW == 0, std::vector<u64>, std::array<u64, FixedSW>> syn;
    std::conditional_t<FixedLW == 0, std::vector<u64>, std::array<u64, FixedLW>> lg;
    std::vector<int> blocked, overlap;
    std::vector<int> forbidden, choice_counts;
    std::vector<u64> choices;
    std::vector<int> selected, packing_rows, region;
    std::unique_ptr<ErasureBound> erasure;
    std::vector<unsigned> packed;
    unsigned epoch = 0;
    u64 hash = 0;
    std::atomic<bool>* hit;
    int root_gap = 1, root_second = -1, selected_left = 0, left_limit = 0;
    u32 left_label = 0;

    int syndrome_words() const { return FixedSW ? FixedSW : D->sw; }
    int logical_words() const { return FixedLW ? FixedLW : D->lw; }
    void change_availability(int q, int delta) {
        for (auto [j, bit] : D->QH[q]) {
            choices[j] ^= bit;
            choice_counts[j] += delta;
        }
    }
    void set_blocked(int q, int value) {
        bool was = blocked[q] == 0 && forbidden[q] == 0;
        blocked[q] = value;
        bool now = blocked[q] == 0 && forbidden[q] == 0;
        if (was != now) change_availability(q, now ? 1 : -1);
    }
    void change_forbidden(int q, int delta) {
        bool was = blocked[q] == 0 && forbidden[q] == 0;
        forbidden[q] += delta;
        bool now = blocked[q] == 0 && forbidden[q] == 0;
        if (was != now) change_availability(q, now ? 1 : -1);
    }
    void toggle(int q, int delta) {
        if (delta > 0) selected.push_back(q);
        else selected.pop_back();
        for (int i = 0; i < syndrome_words(); ++i) syn[i] ^= D->HT[(size_t)q * syndrome_words() + i];
        for (int i = 0; i < logical_words(); ++i) lg[i] ^= D->LT[(size_t)q * logical_words() + i];
        hash ^= D->hashes[q];
        if (!D->left_distance.empty() && q < D->cyclic_half) {
            left_label ^= D->left_labels[q];
            selected_left += delta;
            for (int offset = 1; offset < root_gap; ++offset) {
                int a = q + offset, b = q - offset;
                if (a >= D->cyclic_half) a -= D->cyclic_half;
                if (b < 0) b += D->cyclic_half;
                if (a != q) change_forbidden(a, delta);
                if (b != q && b != a) change_forbidden(b, delta);
            }
        }
        for (int r : D->QS[q]) {
            bool was = overlap[r] >= D->half_weight[r];
            overlap[r] += delta;
            bool now = overlap[r] >= D->half_weight[r];
            if (was != now)
                for (int other : D->SQ[r]) change_forbidden(other, now ? 1 : -1);
        }
    }
    bool can_add(int q) const {
        // The monotone weight/lexicographic cuts are safe on partial supports.
        return forbidden[q] == 0;
    }
    bool left_impossible(int remaining) const {
        if (D->left_distance.empty()) return false;
        int slots = std::min(remaining, left_limit - selected_left);
        return slots < D->left_radius && slots < D->left_distance[left_label];
    }
    bool syn_zero() const { for (u64 x : syn) if (x) return false; return true; }
    bool lg_zero()  const { for (u64 x : lg)  if (x) return false; return true; }
    int syn_min() const {
        for (int i = 0; i < syndrome_words(); ++i) if (syn[i]) return i * 64 + __builtin_ctzll(syn[i]);
        return -1;
    }

    bool pair_tail() const {
        for (int p = D->buckets[hash & (D->buckets.size() - 1)]; p >= 0;
             p = D->pairs[p].next) {
            const auto& pair = D->pairs[p];
            if (blocked[pair.a] || blocked[pair.b]) continue;
            bool closes = true;
            for (int i = 0; i < syndrome_words(); ++i)
                if ((D->HT[(size_t)pair.a * syndrome_words() + i] ^
                     D->HT[(size_t)pair.b * syndrome_words() + i]) != syn[i]) {
                    closes = false; break;
                }
            if (!closes) continue;  // hashes are only an index, never evidence
            for (int i = 0; i < logical_words(); ++i)
                if (lg[i] ^ D->LT[(size_t)pair.a * logical_words() + i] ^
                            D->LT[(size_t)pair.b * logical_words() + i]) return true;
        }
        return false;
    }

    bool single_tail() const {
        int j = syn_min();
        if (j < 0) return false;
        for (int q : D->HE[j]) if (!blocked[q]) {
            bool closes = true;
            for (int i = 0; i < syndrome_words(); ++i)
                if (syn[i] != D->HT[(size_t)q * syndrome_words() + i]) { closes = false; break; }
            if (!closes) continue;
            for (int i = 0; i < logical_words(); ++i)
                if (lg[i] ^ D->LT[(size_t)q * logical_words() + i]) return true;
        }
        return false;
    }

    int choose_check(int remaining) {
        if (++epoch == 0) {
            std::fill(packed.begin(), packed.end(), 0);
            epoch = 1;
        }
        int best = -1, best_count = D->n + 1, packing = 0;
        packing_rows.clear();
        for (int word = 0; word < syndrome_words(); ++word) {
            u64 bits = syn[word];
            while (bits) {
                int j = 64 * word + __builtin_ctzll(bits);
                bits &= bits - 1;
                if (D->small_rows) {
                    int count = choice_counts[j];
                    if (!count) return -1;
                    if (count < best_count) { best = j; best_count = count; }
                    bool disjoint = true;
                    for (u64 mask = choices[j]; mask; mask &= mask - 1) {
                        int q = D->HE[j][__builtin_ctzll(mask)];
                        if (packed[q] == epoch) { disjoint = false; break; }
                    }
                    if (disjoint) {
                        if (++packing > remaining) return -1;
                        packing_rows.push_back(j);
                        for (u64 mask = choices[j]; mask; mask &= mask - 1)
                            packed[D->HE[j][__builtin_ctzll(mask)]] = epoch;
                    }
                    continue;
                }
                int count = 0;
                bool disjoint = true;
                for (int q : D->HE[j]) if (!blocked[q] && can_add(q)) {
                    ++count;
                    if (packed[q] == epoch) disjoint = false;
                }
                if (!count) return -1;
                if (count < best_count) { best = j; best_count = count; }
                // Disjoint eligible neighborhoods require distinct added qubits.
                if (disjoint) {
                    if (++packing > remaining) return -1;
                    packing_rows.push_back(j);
                    for (int q : D->HE[j])
                        if (!blocked[q] && can_add(q)) packed[q] = epoch;
                }
            }
        }
        return best;
    }

    bool erasure_excludes(int remaining) {
        int extra = remaining - (int)packing_rows.size();
        if (!erasure || extra < 0 || extra > 2 || 2 * selected.size() + 2 > (size_t)target)
            return false;
        region.clear();
        for (int j : packing_rows)
            for (u64 mask = choices[j]; mask; mask &= mask - 1)
                region.push_back(D->HE[j][__builtin_ctzll(mask)]);
        if (region.size() > 128) return false;
        // The disjoint odd-check packing charges one added qubit to each
        // neighborhood. At most `extra` additions can lie outside their union.
        return erasure->excludes(region, extra, [&](int q) {
            return !blocked[q] && can_add(q) && packed[q] != epoch;
        }, selected);
    }

    bool dfs(int w) {
        if (w == target) return syn_zero() && !lg_zero();
        if (hit->load(std::memory_order_relaxed)) return false;
        int odd = 0, remaining = target - w;
        if (left_impossible(remaining)) return false;
        for (u64 word : syn) odd += __builtin_popcountll(word);
        if (!odd) return AtMost && !lg_zero();
        if (odd > remaining * D->max_degree) return false;
        if (remaining == 2 && !D->buckets.empty()) {
            if constexpr (AtMost) if (!D->even && single_tail()) return true;
            return pair_tail();
        }
        int j = choose_check(remaining);
        if (j < 0) return false;
        if (erasure_excludes(remaining)) return false;
        bool found = false;
        for (int k : D->HE[j]) {
            if (blocked[k] || !can_add(k)) continue;
            toggle(k, 1);
            set_blocked(k, -1);
            found = dfs(w + 1);
            toggle(k, -1);
            // Later branches omit earlier alternatives: partition supports by
            // the first selected neighbor of this odd check (no duplicates).
            set_blocked(k, w + 1);
            if (found) break;
        }
        for (int k : D->HE[j]) if (blocked[k] == w + 1) set_blocked(k, 0);
        return found;
    }

    void init(const CCData& data, int d, std::atomic<bool>& found) {
        D = &data; target = d; hit = &found;
        if constexpr (FixedSW) syn.fill(0);
        else syn.assign(D->sw, 0);
        if constexpr (FixedLW) lg.fill(0);
        else lg.assign(D->lw, 0);
        blocked.assign(D->n, 0); overlap.assign(D->half_weight.size(), 0);
        packed.assign(D->n, 0);
        forbidden.assign(D->n, 0);
        choices.assign(D->mh, 0); choice_counts.assign(D->mh, 0);
        selected.reserve(d); packing_rows.reserve(d);
        region.reserve(D->n);
        if (D->erasure) erasure = std::make_unique<ErasureBound>(*D->erasure);
    }

    bool reset_seed(int e0, int gap = 1, int second = -1) {
        std::fill(syn.begin(), syn.end(), 0ull);
        std::fill(lg.begin(),  lg.end(),  0ull);
        selected.clear();
        root_gap = gap; root_second = second; selected_left = 0; left_label = 0;
        left_limit = D->cyclic_half / gap;
        std::fill(blocked.begin(), blocked.end(), 0);
        std::fill(overlap.begin(), overlap.end(), 0);
        std::fill(blocked.begin(), blocked.begin() + e0 + 1, -2);
        forbidden = D->initially_forbidden;
        if (D->small_rows) {
            std::fill(choices.begin(), choices.end(), 0);
            std::fill(choice_counts.begin(), choice_counts.end(), 0);
            for (int q = 0; q < D->n; ++q)
                if (!blocked[q] && !forbidden[q]) change_availability(q, 1);
        }
        // Starting inside a run of equal minimum gaps is not lexicographically
        // first. Keep the full periodic-cycle exception when it fits the budget.
        if (second >= 0 && (D->cyclic_half % gap || D->cyclic_half / gap > target))
            set_blocked(D->cyclic_half - gap, -2);
        hash = 0;
        if (!can_add(e0)) return false;
        toggle(e0, 1);
        if (second >= 0) {
            if (target < 2 || blocked[second] || !can_add(second)) return false;
            toggle(second, 1); set_blocked(second, -1);
        }
        return !left_impossible(target - (int)selected.size());
    }

    void split(int w, int seed, int depth, std::vector<int>& prefix,
               std::vector<CCJob>& jobs) {
        if constexpr (AtMost) {
            if (hit->load(std::memory_order_relaxed)) return;
            if (syn_zero() && !lg_zero()) { hit->store(true); return; }
        }
        if (left_impossible(target - w)) return;
        if (w == depth) { jobs.push_back({seed, root_gap, root_second, prefix}); return; }
        int j = choose_check(target - w);
        if (j < 0) return;
        for (int q : D->HE[j]) {
            if (blocked[q] || !can_add(q)) continue;
            toggle(q, 1); set_blocked(q, -1);
            prefix.push_back(q);
            split(w + 1, seed, depth, prefix, jobs);
            prefix.pop_back();
            toggle(q, -1); set_blocked(q, w + 1);
        }
        for (int q : D->HE[j]) if (blocked[q] == w + 1) set_blocked(q, 0);
    }

    bool search_job(const CCJob& job) {
        if (!reset_seed(job.seed, job.gap, job.second)) return false;
        for (int first : job.prefix) {
            int j = choose_check(target - (int)selected.size());
            if (j < 0 || !can_add(first)) return false;
            for (int q : D->HE[j]) if (q < first) set_blocked(q, -2);
            set_blocked(first, -1);
            toggle(first, 1);
        }
        return dfs((int)selected.size());
    }
};

// Exact-level mode follows smaller exclusions. AtMost instead retains a minimum
// logical anywhere below the cap, with no assumed or cached lower bound.
template<int SW, int LW, bool AtMost>
bool cc_level_impl(const CCData& D, int d, int T) {
    auto work_start = std::chrono::steady_clock::now();
    std::atomic<int> next{0};
    std::atomic<bool> hit{false};
    std::vector<CCJob> jobs;
    CCSearch<SW, LW, AtMost> splitter;
    splitter.init(D, d, hit);
    std::vector<int> prefix;
    size_t max_branch = 1, estimate = D.seeds.size();
    for (const auto& row : D.HE) max_branch = std::max(max_branch, row.size());
    int depth = 1;
    // Small prefix jobs balance the few symmetry-reduced roots without
    // materializing an exponential queue for dense Tanner graphs.
    while (depth < std::min(7, d - 2) && estimate <= 16384 / max_branch) {
        estimate *= max_branch;
        ++depth;
    }
    if (!D.left_distance.empty()) {
        auto root = [&](int seed, int gap, int second, size_t budget) {
            if (!splitter.reset_seed(seed, gap, second)) return;
            int weight = (int)splitter.selected.size(), split_depth = weight;
            size_t work = 1;
            while (split_depth < std::min(7, d - 2) && work <= budget / max_branch) {
                work *= max_branch;
                ++split_depth;
            }
            splitter.split(weight, seed, split_depth, prefix, jobs);
        };
        // A lex-first minimum logical with >=2 left coordinates starts at a
        // smallest circular gap. Smaller gaps anywhere would improve its shift.
        if (d >= 2)
            for (int gap = 1; gap <= D.cyclic_half / 2; ++gap)
                root(0, gap, gap, 2048);
        root(0, D.cyclic_half, -1, 512); // exactly one left coordinate
        root(D.cyclic_half, 1, -1, 512); // no left coordinates
    } else {
        for (int seed : D.seeds) {
            if (splitter.reset_seed(seed))
                splitter.split(1, seed, depth, prefix, jobs);
        }
    }
    if (!D.left_distance.empty()) {
        std::vector<std::vector<CCJob>> families(D.cyclic_half + 2);
        size_t longest = 0;
        for (auto& job : jobs) {
            int family = job.seed == 0 ? job.gap : D.cyclic_half + 1;
            families[family].push_back(std::move(job));
            longest = std::max(longest, families[family].size());
        }
        jobs.clear();
        for (size_t i = 0; i < longest; ++i)
            for (auto& family : families) if (i < family.size())
                jobs.push_back(std::move(family[i]));
    }
    if (D.progress)
        std::fprintf(stderr, "CC decision jobs: %zu; projected syndromes: %zu; cuts: %zu\n",
                     jobs.size(), D.left_distance.size(), D.half_weight.size());
    std::vector<int> family_total(D.cyclic_half + 2, 0);
    std::vector<std::atomic<int>> family_done(D.cyclic_half + 2);
    for (auto& count : family_done) count.store(0);
    if (D.progress && !D.left_distance.empty())
        for (const auto& job : jobs)
            ++family_total[job.seed == 0 ? job.gap : D.cyclic_half + 1];
    auto worker = [&]() {
        CCSearch<SW, LW, AtMost> s;
        s.init(D, d, hit);
        for (;;) {
            if (hit.load(std::memory_order_relaxed)) break;
            int job = next.fetch_add(1);
            if (job >= (int)jobs.size()) break;
            if (s.search_job(jobs[job])) { hit.store(true); break; }
            if (D.progress && !D.left_distance.empty()) {
                int family = jobs[job].seed == 0 ? jobs[job].gap : D.cyclic_half + 1;
                int done = family_done[family].fetch_add(1) + 1;
                if (done == family_total[family] ||
                    (family <= 4 && done % std::max(1, family_total[family] / 4) == 0))
                    std::fprintf(stderr, "CC root family progress: seed=%d gap=%d jobs=%d/%d elapsed=%.3fs\n",
                                 jobs[job].seed, jobs[job].gap, done, family_total[family],
                                 std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                               work_start).count());
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < std::min(T, (int)jobs.size()); ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    return hit.load();
}

template<bool AtMost = false>
bool cc_level(const CCData& D, int d, int T) {
    if (D.lw == 1) {
        if (D.sw == 1) return cc_level_impl<1, 1, AtMost>(D, d, T);
        if (D.sw == 2) return cc_level_impl<2, 1, AtMost>(D, d, T);
        if (D.sw == 3) return cc_level_impl<3, 1, AtMost>(D, d, T);
        if (D.sw == 4) return cc_level_impl<4, 1, AtMost>(D, d, T);
    }
    return cc_level_impl<0, 0, AtMost>(D, d, T);
}

// Min weight of e in ker(H) with L*e != 0, searched up to maxw. Returns WEIGHT_NONE if none.
// With verbose, reports progress on stderr in the original distfind style: a per-level "Distance
// bound: >N" (the level just ruled out) with its elapsed time, then a final "Distance: =N".
int cc_search(const CCData& D, int maxw, int nthreads, bool verbose, const char* tag,
              bool equivalent = false) {
    using clk = std::chrono::steady_clock;
    if (D.kl == 0) return -1;               // no logicals
    int T = std::max(1, std::min(nthreads, D.n));
    auto t_all = clk::now();
    for (int d = 1; d <= maxw; ++d) {
        if (D.even && d % 2) continue;
        auto td0 = clk::now();
        if (cc_level(D, d, T)) {
            if (verbose) {
                verbose_final(d, std::chrono::duration<double>(clk::now() - t_all).count(), tag);
                if (equivalent) verbose_final(d, 0.0, "X");
            }
            return d;
        }
        if (verbose) {
            verbose_bound(d, std::chrono::duration<double>(clk::now() - td0).count(), tag);
            if (equivalent) verbose_bound(d, 0.0, "X");
        }
    }
    return WEIGHT_NONE;
}

BZResult cc_one(const GF2Mat& H, const GF2Mat& L, const BZOptions& opt, const char* tag,
                const GF2Mat* trivial = nullptr, bool equivalent = false) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    BZResult res; res.backend = "cc";
    CCData D = build_cc(H, L, trivial);
    if (opt.verbose)
        std::fprintf(stderr, "%s-CC: %zu seeds, %zu syndrome pairs, even=%d\n",
                     tag, D.seeds.size(), D.pairs.size(), (int)D.even);
    int nthreads = opt.threads > 0 ? opt.threads : (int)std::thread::hardware_concurrency();
    int maxw = D.n;
    int d = cc_search(D, maxw, nthreads, opt.verbose, tag, equivalent);
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
    if (css_sectors_equivalent(Hx, Hz)) {
        if (opt.verbose) std::fprintf(stderr, "CC: verified equivalent X/Z sectors\n");
        BZResult result = cc_one(Hx, pz.check, opt, "Z", &Hz, true);
        result.seconds = std::chrono::duration<double>(clk::now() - t0).count();
        return result;
    }
    DistProblem px = css_problem(Hx, Hz, 'X');   // X-distance: e in ker(Hz), nontrivial vs Z-logicals
    CCData DZ = build_cc(Hx, pz.check, &Hz);
    CCData DX = build_cc(Hz, px.check, &Hx);

    int nthreads = opt.threads > 0 ? opt.threads : (int)std::thread::hardware_concurrency();
    int TZ = std::max(1, std::min(nthreads, DZ.n));
    int TX = std::max(1, std::min(nthreads, DX.n));
    int maxw = std::max(DZ.n, DX.n);

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
        if (z_active(d) && !(DZ.even && d % 2)) {
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
        if (x_active(d) && !(DX.even && d % 2)) {
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

int cc_css_distance_at_least(const GF2Mat& Hx, const GF2Mat& Hz, int lower,
                             char which, const BZOptions& opt) {
    auto prove = [&](char side) {
        auto start = std::chrono::steady_clock::now();
        DistProblem problem = css_problem(Hx, Hz, side);
        if (!problem.check.rows) return -1;
        if (lower <= 1) return 1;
        if (lower > Hx.cols) return 0;
        const GF2Mat& H = side == 'Z' ? Hx : Hz;
        const GF2Mat& S = side == 'Z' ? Hz : Hx;
        CCData data = build_cc(H, problem.check, &S);
        data.progress = opt.verbose;
        int cap = lower - 1;
        if (data.even && cap % 2) --cap;
        int threads = opt.threads > 0 ? opt.threads : (int)std::thread::hardware_concurrency();
        bool proved = cap == 0 || !cc_level<true>(data, cap, std::max(1, threads));
        if (opt.verbose && proved) {
            char tag[2] = {side, 0};
            verbose_bound(lower - 1,
                          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                          tag);
        }
        return proved ? 1 : 0;
    };
    if (which == 'Z' || which == 'X') return prove(which);
    int z = prove('Z');
    if (z != 1) return z;
    if (!css_sectors_equivalent(Hx, Hz)) return prove('X');
    if (opt.verbose) verbose_bound(lower - 1, 0.0, "X");
    return 1;
}

BZResult cc_css_distance(const GF2Mat& Hx, const GF2Mat& Hz, char which, const BZOptions& opt) {
    if (which == 'X') {
        DistProblem p = css_problem(Hx, Hz, 'X');   // p.check = Z-logicals
        return cc_one(Hz, p.check, opt, "X", &Hx);
    }
    if (which == 'Z') {
        DistProblem p = css_problem(Hx, Hz, 'Z');   // p.check = X-logicals
        return cc_one(Hx, p.check, opt, "Z", &Hz);
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
