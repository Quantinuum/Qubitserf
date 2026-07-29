// Bit-packed GF(2) word primitives shared by every qubitserf engine (distfind + codeaut),
// host and (conceptually) device code. A "row" is `stride` little-endian uint64 words;
// bit i lives in word (i>>6) at offset (i&63).
#pragma once
#include <cstdint>
#include <cstddef>

namespace qsf {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using u8  = std::uint8_t;

// Number of 64-bit words needed to hold `bits` bits.
inline int words_for(int bits) { return (bits + 63) >> 6; }

inline int popcount64(u64 x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0full;
    return (int)((x * 0x0101010101010101ull) >> 56);
#endif
}

// popcount of a packed bit-vector of `stride` words.
inline int vec_weight(const u64* v, int stride) {
    int w = 0;
    for (int i = 0; i < stride; ++i) w += popcount64(v[i]);
    return w;
}

// dst ^= src  (stride words)
inline void vec_xor(u64* dst, const u64* src, int stride) {
    for (int i = 0; i < stride; ++i) dst[i] ^= src[i];
}

// parity of (a AND b) over stride words -> 0/1 (the GF(2) inner product)
inline int vec_dot(const u64* a, const u64* b, int stride) {
    u64 acc = 0;
    for (int i = 0; i < stride; ++i) acc ^= (a[i] & b[i]);
    return popcount64(acc) & 1;
}

// Pack a row-major `rows x cols` 0/1-byte matrix into bit-packed rows of `stride` words.
// `dst` must hold rows * words_for(cols) words and is fully overwritten.
inline void pack_rows(const u8* src, int rows, int cols, u64* dst) {
    const int stride = words_for(cols);
    for (int r = 0; r < rows; ++r) {
        u64* d = dst + (size_t)r * stride;
        const u8* s = src + (size_t)r * cols;
        for (int w = 0; w < stride; ++w) d[w] = 0ull;
        for (int j = 0; j < cols; ++j)
            if (s[j] & 1) d[j >> 6] |= (u64)1 << (j & 63);
    }
}

// Unpack a bit-packed row of `cols` bits into `cols` 0/1 bytes.
inline void unpack_row(const u64* src, int cols, u8* dst) {
    for (int j = 0; j < cols; ++j) dst[j] = (u8)((src[j >> 6] >> (j & 63)) & 1ull);
}

} // namespace qsf
