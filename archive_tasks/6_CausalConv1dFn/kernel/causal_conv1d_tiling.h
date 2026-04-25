#pragma once
#include <cstdint>

struct CausalConv1dTiling {
    int32_t cuSeqLen;
    int32_t dim;
    int32_t numStatesXSl;
    int32_t batchCount;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t blockN;
    int32_t nTiles;
    int32_t residual;
    int32_t padSlotId;
};
