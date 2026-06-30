// Device-agnostic enumeration plan. The host builds this once per weight level `d`
// and hands it to a backend (CPU / GPU). The backend enumerates every
// weight-d combination of the K message rows, through every information-set generator,
// and returns the smallest full Hamming weight of a non-trivial logical codeword.
#pragma once
#include <vector>
#include <string>
#include "qubitserf/bits.hpp"

namespace qubitserf {

constexpr int WEIGHT_NONE = 1 << 30;

struct EnumPlan {
    int n = 0;                 // codeword length in bits
    int stride = 0;            // u64 words per codeword row = words_for(n)
    int K = 0;                 // message dimension (rows per gamma)
    int d = 0;                 // combination weight to enumerate at this level
    int num_gamma = 0;         // number of information sets
    const u64* gamma = nullptr;// num_gamma * K * stride, row-major (g, row, word)
    int kcheck = 0;            // logical-detector rows; 0 => every codeword is logical
    const u64* check = nullptr;// kcheck * stride
    const u64* binom = nullptr;// flattened (maxN+1)*(maxK+1) saturating binomials
    int binom_maxN = 0, binom_maxK = 0;
    int current_best = WEIGHT_NONE; // best upper bound so far (for pruning)
    // Unique per-solve token. gamma/check/binom are constant across weight levels within
    // one solve but the host buffers are freed and reallocated (often at the SAME address)
    // between solves -- GPU backends MUST key their device-buffer cache on this, never on
    // pointer identity, or they will read stale data.
    u64 buffers_key = 0;
};

struct Backend {
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual bool available() const = 0;
    // Smallest full weight of a logical codeword at this level, or >= current_best.
    virtual int enumerate(const EnumPlan& plan) = 0;
};

Backend* cpu_backend();
Backend* metal_backend();   // nullptr if Metal support not compiled in
Backend* cuda_backend();    // nullptr if CUDA support not compiled in

// Select by name ("cpu", "gpu", or "auto"). Never returns null; falls back
// to CPU when the requested accelerator is unavailable. "gpu" resolves to the
// machine-specific accelerator implementation (CUDA or Metal).
Backend* select_backend(const std::string& name);

// Thread count for the CPU backend (0 => hardware concurrency).
void set_cpu_threads(int n);

// ---- shared scalar kernel (used by the CPU backend; mirrored in MSL/CUDA) ----------

// Build codeword = XOR of rows pos[0..d-1] of gamma generator `g`, test the logical
// detector, and return its full weight if it is a logical, else WEIGHT_NONE.
inline int eval_combo(const EnumPlan& p, int g, const int* pos, u64* scratch) {
    const u64* G = p.gamma + (size_t)g * p.K * p.stride;
    for (int w = 0; w < p.stride; ++w) scratch[w] = 0;
    for (int i = 0; i < p.d; ++i) {
        const u64* r = G + (size_t)pos[i] * p.stride;
        for (int w = 0; w < p.stride; ++w) scratch[w] ^= r[w];
    }
    bool logical = (p.kcheck == 0);
    for (int c = 0; c < p.kcheck && !logical; ++c) {
        u64 acc = 0;
        const u64* chk = p.check + (size_t)c * p.stride;
        for (int w = 0; w < p.stride; ++w) acc ^= (chk[w] & scratch[w]);
        if (popcount64(acc) & 1) logical = true;
    }
    if (!logical) return WEIGHT_NONE;
    return vec_weight(scratch, p.stride);
}

} // namespace qubitserf
