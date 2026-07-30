// Brouwer-Zimmermann low-weight enumeration: the C ABI (codeaut_bz_*) declared in capi.h.
//
// Packs the (m x n) generator matrix into bit-packed rows and drives the SHARED
// enumeration core (qsf::Backend::collect -- the same two-level kernel and CPU/CUDA/Metal
// backends the distfind engine uses in min-weight mode): for each subset size sw = 1..p
// it collects every codeword of weight <= keep_weight that is the XOR of some size-sw
// subset of the rows, then DEDUPLICATES the collected codewords here (backends only emit
// hits, duplicates allowed). The Python `lowweight` layer loops the BZ information-set
// generators, calls this per generator, and merges/certifies.
//
// The subset-size loop lives HERE (once), not in the backends; the enumeration runs to
// completion for every subset size sw = 1..p (no caps -- kill the process if it takes
// too long).

#include "capi.h"
#include "qsf/backend.hpp"
#include "qsf/combinatorics.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <cstring>

using qsf::u64;

namespace {

// Historical saturation cap for combination counts (the old int64 BinomTable's CAP).
constexpr int64_t COMBO_CAP = (int64_t)1 << 62;

struct BZHandle {
    int n = 0, stride = 0;
    bool ok = false;
    int64_t combos = 0;
    int64_t count = 0;
    std::vector<uint64_t> rows;   // deduped, `stride` u64 words each
    std::string backend;
};

// Lexicographic comparison of two stride-word rows.
struct RowLess {
    int stride;
    const uint64_t* base;
    bool operator()(int64_t a, int64_t b) const {
        const uint64_t* pa = base + a * stride;
        const uint64_t* pb = base + b * stride;
        for (int i = 0; i < stride; ++i) {
            if (pa[i] != pb[i]) return pa[i] < pb[i];
        }
        return false;
    }
};

}  // namespace

extern "C" {

void* codeaut_bz_collect(const uint8_t* G, int32_t m, int32_t n,
                         int32_t p, int32_t keep_weight,
                         int32_t backend, int32_t threads) {
    BZHandle* H = new BZHandle();
    H->n = n;
    H->stride = qsf::words_for(n);
    if (m <= 0 || n <= 0 || p <= 0) { H->ok = true; return H; }

    // pack G (row-major uint8) into bit-packed rows
    const int stride = H->stride;
    std::vector<uint64_t> packed((size_t)m * stride, 0);
    qsf::pack_rows(G, m, n, packed.data());

    const int maxsw = std::min(p, m);
    qsf::BinomTable bt(m, maxsw);

    qsf::set_cpu_threads(threads > 0 ? threads : 0);
    qsf::Backend* be = qsf::select_backend(backend == 0 ? "cpu" : "auto");
    H->backend = be->name();

    qsf::EnumPlan plan;
    plan.n = n; plan.stride = stride; plan.K = m;
    plan.num_gamma = 1; plan.num_gamma_total = 1;
    plan.gamma = packed.data();
    plan.kcheck = 0; plan.check = nullptr;
    plan.binom = bt.c.data(); plan.binom_maxN = bt.maxN; plan.binom_maxK = bt.maxK;
    plan.buffers_key = qsf::next_buffers_key();   // rows constant across the sw levels of this call

    std::vector<uint64_t> hits;
    for (int sw = 1; sw <= maxsw; ++sw) {
        u64 utotal = bt.binom(m, sw);
        int64_t total = utotal > (u64)COMBO_CAP ? COMBO_CAP : (int64_t)utotal;
        if (total <= 0) continue;

        plan.d = sw;
        qsf::CollectResult cr = be->collect(plan, keep_weight);
        hits.insert(hits.end(), cr.hits.begin(), cr.hits.end());
        H->combos = (H->combos > COMBO_CAP - total) ? COMBO_CAP : H->combos + total;
    }

    // dedup collected hits
    int64_t nrows = (int64_t)hits.size() / stride;
    std::vector<int64_t> idx(nrows);
    for (int64_t i = 0; i < nrows; ++i) idx[i] = i;
    RowLess less{stride, hits.data()};
    std::sort(idx.begin(), idx.end(), less);
    H->rows.reserve(hits.size());
    for (int64_t i = 0; i < nrows; ++i) {
        if (i > 0) {
            const uint64_t* a = hits.data() + idx[i] * stride;
            const uint64_t* b = hits.data() + idx[i - 1] * stride;
            if (std::memcmp(a, b, (size_t)stride * sizeof(uint64_t)) == 0) continue;
        }
        const uint64_t* a = hits.data() + idx[i] * stride;
        H->rows.insert(H->rows.end(), a, a + stride);
    }
    H->count = (int64_t)H->rows.size() / stride;
    H->ok = true;
    return H;
}

int32_t codeaut_bz_ok(void* h)       { return h && ((BZHandle*)h)->ok ? 1 : 0; }
int64_t codeaut_bz_count(void* h)    { return h ? ((BZHandle*)h)->count : -1; }
int64_t codeaut_bz_combos(void* h)   { return h ? ((BZHandle*)h)->combos : -1; }

void codeaut_bz_copy_rows(void* h, uint8_t* buf) {
    BZHandle* H = (BZHandle*)h;
    if (!H) return;
    const int stride = H->stride, n = H->n;
    for (int64_t r = 0; r < H->count; ++r)
        qsf::unpack_row(H->rows.data() + r * stride, n, buf + (size_t)r * n);
}

const char* codeaut_bz_backend(void* h) {
    BZHandle* H = (BZHandle*)h;
    return H ? H->backend.c_str() : "";
}

void codeaut_bz_free(void* h) { delete (BZHandle*)h; }

}  // extern "C"
