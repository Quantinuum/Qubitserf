// Backend abstraction for the Brouwer-Zimmermann low-weight enumeration kernel.
//
// Given `m` bit-packed generator rows, collect every codeword of weight <= keep_weight that
// is the XOR of some size-(1..p) subset of those rows.  The CPU backend (always present) and
// the optional CUDA / Metal backends all implement this same plan; dedup of the collected
// codewords is done by the caller (src/bz.cpp), so backends only need to emit the low-weight
// hits (duplicates across lanes are fine).
#ifndef CODEAUT_BACKEND_HPP
#define CODEAUT_BACKEND_HPP

#include <cstdint>
#include <vector>
#include <string>

namespace codeaut {

struct BZEnumPlan {
    const uint64_t* rows;   // m rows, each `stride` u64 words, row-major: rows[j*stride + w]
    int m;                  // number of generator rows (= dim)
    int n;                  // number of coordinates
    int stride;             // words per row = (n+63)/64
    int p;                  // max subset size (enumerate sizes 1..p)
    int keep_weight;        // collect XORs with Hamming weight <= keep_weight
    int64_t budget;         // abort once this many combinations are evaluated (<=0 => unlimited)
    int threads;            // CPU worker threads (0 => hardware concurrency)
};

struct BZEnumResult {
    std::vector<uint64_t> hits;   // flat, `stride` u64 words per collected codeword (with dups)
    int64_t combos = 0;           // subset-combinations evaluated
    bool overflow = false;        // budget exceeded (result is then incomplete)
};

struct Backend {
    virtual ~Backend() {}
    virtual const char* name() const = 0;        // "cpu" / "cuda" / "metal"
    virtual bool available() const = 0;
    virtual BZEnumResult enumerate(const BZEnumPlan& plan) = 0;
};

Backend* cpu_backend();
Backend* cuda_backend();   // nullptr unless built with CODEAUT_CUDA and a device is present
Backend* metal_backend();  // nullptr unless built with CODEAUT_METAL on Apple

// which: 0 = cpu, 1 = gpu (cuda then metal, else cpu), 2 = auto (gpu if available else cpu)
Backend* select_backend(int which);

}  // namespace codeaut
#endif
