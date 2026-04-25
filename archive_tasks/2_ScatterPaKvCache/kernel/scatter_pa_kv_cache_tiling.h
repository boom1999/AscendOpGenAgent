#pragma once

#include <cstdint>

constexpr int32_t SCATTER_NUM_PHYSICAL_CORES = 20;
constexpr int32_t SLICE_LEN_I32 = 8;  // 32 bytes / 4 bytes per int32 = 8

struct ScatterPaKvCacheTiling {
    int32_t nTokens;
    int32_t numKvSlices;
    int32_t numVvSlices;
    int32_t numBlocks;
    int32_t blockSize;
    int32_t keyFlatDimI32;   // numKvSlices * SLICE_LEN_I32
    int32_t valFlatDimI32;   // numVvSlices * SLICE_LEN_I32
    int32_t usedCoreNum;
    int32_t tokensPerCore;
};
