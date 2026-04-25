#pragma once
#include <cstdint>

struct AdvanceStepV2Tiling {
    int32_t numReqs;
    int32_t tokenEachReqs;
    int32_t sampledCols;
    int32_t maxNumBlocks;
    int32_t blockSize;
    int32_t usedCoreNum;
    int32_t reqsPerCore;
};
