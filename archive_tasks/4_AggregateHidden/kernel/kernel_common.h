#pragma once
#include <cstdint>

inline __aicore__ uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1U) / b;
}

template <typename TilingT>
__aicore__ inline void CopyTiling(TilingT &local, __gm__ uint8_t *gm) {
    const uint32_t n = sizeof(TilingT) / sizeof(int32_t);
    auto *dst = reinterpret_cast<int32_t *>(&local);
    auto *src = reinterpret_cast<__gm__ int32_t *>(gm);
    for (uint32_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}
