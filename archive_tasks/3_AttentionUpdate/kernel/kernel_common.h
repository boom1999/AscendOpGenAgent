#pragma once
#include "kernel_operator.h"

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

template <typename TilingT>
__aicore__ inline void CopyTiling(TilingT &tiling, __gm__ uint8_t *tilingGm) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(&tiling);
    __gm__ uint32_t *src = reinterpret_cast<__gm__ uint32_t *>(tilingGm);
    for (int32_t i = 0; i < static_cast<int32_t>(sizeof(TilingT) / sizeof(uint32_t)); ++i) {
        dst[i] = src[i];
    }
}
