// Bit-packed GF(2) word primitives shared by host and (conceptually) device code.
#pragma once
#include <cstdint>
#include <cstddef>

namespace qubitserf {

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

} // namespace qubitserf
