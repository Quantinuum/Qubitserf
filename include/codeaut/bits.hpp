// Bit-packed GF(2) row helpers for the Brouwer-Zimmermann enumeration kernel.
// A "row" is `stride` little-endian uint64 words; bit i lives in word (i>>6) at offset (i&63).
#ifndef CODEAUT_BITS_HPP
#define CODEAUT_BITS_HPP

#include <cstdint>
#include <cstddef>

namespace codeaut {

inline int words_for(int nbits) { return (nbits + 63) / 64; }

inline int popcount_row(const uint64_t* r, int stride) {
    int w = 0;
    for (int i = 0; i < stride; ++i) w += __builtin_popcountll(r[i]);
    return w;
}

// dst ^= src   (stride words)
inline void xor_into(uint64_t* dst, const uint64_t* src, int stride) {
    for (int i = 0; i < stride; ++i) dst[i] ^= src[i];
}

// dst = a XOR b  (stride words), returning popcount(dst)
inline int xor_weight(uint64_t* dst, const uint64_t* a, const uint64_t* b, int stride) {
    int w = 0;
    for (int i = 0; i < stride; ++i) { dst[i] = a[i] ^ b[i]; w += __builtin_popcountll(dst[i]); }
    return w;
}

}  // namespace codeaut
#endif
