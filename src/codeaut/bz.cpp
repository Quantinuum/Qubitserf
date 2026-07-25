// Brouwer-Zimmermann low-weight enumeration: the C ABI (codeaut_bz_*) declared in capi.h.
//
// Packs the (m x n) generator matrix into bit-packed rows, runs the selected backend
// (CPU / CUDA / Metal) to collect every codeword of weight <= keep_weight obtained by XORing
// a size-(1..p) subset of the rows, then DEDUPLICATES the collected codewords here (so the
// backends need only emit hits, duplicates allowed).  The Python `lowweight` layer loops the
// BZ information-set generators, calls this per generator, and merges/certifies.

#include "capi.h"
#include "backend.hpp"
#include "bits.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <cstring>

using namespace codeaut;

namespace {

struct BZHandle {
    int n = 0, stride = 0;
    bool ok = false;
    bool overflow = false;
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
                         int64_t budget, int32_t backend, int32_t threads) {
    BZHandle* H = new BZHandle();
    H->n = n;
    H->stride = words_for(n);
    if (m <= 0 || n <= 0 || p <= 0) { H->ok = true; return H; }

    // pack G (row-major uint8) into bit-packed rows
    const int stride = H->stride;
    std::vector<uint64_t> packed((size_t)m * stride, 0);
    for (int r = 0; r < m; ++r) {
        const uint8_t* src = G + (size_t)r * n;
        uint64_t* dst = packed.data() + (size_t)r * stride;
        for (int j = 0; j < n; ++j) if (src[j] & 1) dst[j >> 6] |= (uint64_t)1 << (j & 63);
    }

    BZEnumPlan plan;
    plan.rows = packed.data();
    plan.m = m; plan.n = n; plan.stride = stride;
    plan.p = p; plan.keep_weight = keep_weight;
    plan.budget = budget; plan.threads = threads > 0 ? threads : 0;

    Backend* be = select_backend(backend);
    BZEnumResult er = be->enumerate(plan);
    H->backend = be->name();
    H->combos = er.combos;
    H->overflow = er.overflow;

    // dedup collected hits
    int64_t nrows = (int64_t)er.hits.size() / stride;
    std::vector<int64_t> idx(nrows);
    for (int64_t i = 0; i < nrows; ++i) idx[i] = i;
    RowLess less{stride, er.hits.data()};
    std::sort(idx.begin(), idx.end(), less);
    H->rows.reserve(er.hits.size());
    for (int64_t i = 0; i < nrows; ++i) {
        if (i > 0) {
            const uint64_t* a = er.hits.data() + idx[i] * stride;
            const uint64_t* b = er.hits.data() + idx[i - 1] * stride;
            if (std::memcmp(a, b, (size_t)stride * sizeof(uint64_t)) == 0) continue;
        }
        const uint64_t* a = er.hits.data() + idx[i] * stride;
        H->rows.insert(H->rows.end(), a, a + stride);
    }
    H->count = (int64_t)H->rows.size() / stride;
    H->ok = true;
    return H;
}

int32_t codeaut_bz_ok(void* h)       { return h && ((BZHandle*)h)->ok ? 1 : 0; }
int32_t codeaut_bz_overflow(void* h) { return h && ((BZHandle*)h)->overflow ? 1 : 0; }
int64_t codeaut_bz_count(void* h)    { return h ? ((BZHandle*)h)->count : -1; }
int64_t codeaut_bz_combos(void* h)   { return h ? ((BZHandle*)h)->combos : -1; }

void codeaut_bz_copy_rows(void* h, uint8_t* buf) {
    BZHandle* H = (BZHandle*)h;
    if (!H) return;
    const int stride = H->stride, n = H->n;
    for (int64_t r = 0; r < H->count; ++r) {
        const uint64_t* src = H->rows.data() + r * stride;
        uint8_t* dst = buf + (size_t)r * n;
        for (int j = 0; j < n; ++j) dst[j] = (uint8_t)((src[j >> 6] >> (j & 63)) & 1);
    }
}

const char* codeaut_bz_backend(void* h) {
    BZHandle* H = (BZHandle*)h;
    return H ? H->backend.c_str() : "";
}

void codeaut_bz_free(void* h) { delete (BZHandle*)h; }

}  // extern "C"
