// Device-agnostic enumeration plan and the unified backend interface shared by the
// distfind (minimum distance) and codeaut (low-weight collection) engines.
//
// One plan, two modes over the same enumeration (every weight-d combination of the K
// message rows, through every information-set generator):
//
//   * enumerate(plan)          -> smallest full Hamming weight of a non-trivial LOGICAL
//                                 codeword lighter than plan.current_best (distfind).
//   * collect(plan, keep)      -> every enumerated codeword with Hamming weight in
//                                 [1, keep], duplicates allowed (codeaut; the caller
//                                 dedups). collect ignores plan.current_best/kcheck.
#pragma once
#include <vector>
#include <string>
#include "qsf/bits.hpp"

namespace qsf {

constexpr int WEIGHT_NONE = 1 << 30;

struct EnumPlan {
    int n = 0;                 // codeword length in bits
    int stride = 0;            // u64 words per codeword row = words_for(n)
    int K = 0;                 // message dimension (rows per gamma)
    int d = 0;                 // combination weight to enumerate at this level
    int num_gamma = 0;         // information sets to ENUMERATE this level (active prefix)
    int num_gamma_total = 0;   // information sets in `gamma` (>= num_gamma; 0 => num_gamma).
                               // The BZ driver sorts sets by rank (descending) and skips the
                               // suffix whose lower-bound contribution is zero at this level;
                               // GPU backends size their cached upload on the TOTAL count.
    const u64* gamma = nullptr;// num_gamma_total * K * stride, row-major (g, row, word)
    int kcheck = 0;            // logical-detector rows; 0 => every codeword is logical
    const u64* check = nullptr;// kcheck * stride
    const u64* binom = nullptr;// flattened (maxN+1)*(maxK+1) saturating binomials
    int binom_maxN = 0, binom_maxK = 0;
    int current_best = WEIGHT_NONE; // best upper bound so far (for pruning; enumerate only)
    // Unique per-solve token. gamma/check/binom are constant across weight levels within
    // one solve but the host buffers are freed and reallocated (often at the SAME address)
    // between solves -- GPU backends MUST key their device-buffer cache on this, never on
    // pointer identity, or they will read stale data.
    u64 buffers_key = 0;
};

// Result of a collect() level: the qualifying codewords, flat, `stride` words each,
// duplicates allowed. ok=false means the backend could not produce a COMPLETE set
// (e.g. GPU output-capacity overflow after CPU fallback also failed -- never happens
// for the CPU backend, which is always complete).
struct CollectResult {
    std::vector<u64> hits;
    bool ok = true;
};

struct Backend {
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual bool available() const = 0;
    // Smallest full weight of a logical codeword at this level, or >= current_best.
    virtual int enumerate(const EnumPlan& plan) = 0;
    // Every codeword of weight in [1, keep_weight] at this level (dups allowed).
    virtual CollectResult collect(const EnumPlan& plan, int keep_weight) = 0;
};

Backend* cpu_backend();
Backend* metal_backend();   // nullptr if Metal support not compiled in
Backend* cuda_backend();    // nullptr if CUDA support not compiled in

// Select by name ("cpu", "gpu", "cuda", "metal", or "auto"). Never returns null; falls
// back to CPU when the requested accelerator is unavailable. "gpu"/"auto" resolve to the
// machine-specific accelerator implementation (CUDA or Metal), else CPU.
Backend* select_backend(const std::string& name);

// Thread count for the CPU backend (0 => hardware concurrency).
void set_cpu_threads(int n);

// Process-wide unique buffers_key for EnumPlan. Every driver MUST take its keys from
// here: the backends (and their device-buffer caches) are shared across engines, so
// per-driver counters would collide.
u64 next_buffers_key();

// ---- shared scalar kernel (used by tests as the brute-force oracle) ---------------

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

} // namespace qsf
