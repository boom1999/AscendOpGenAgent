#pragma once
#include <cstdint>

struct ScatterBlockUpdateTiling {
    int32_t D1;
    int32_t D2;
    int32_t K;
    int32_t elemSize;
    int32_t usedCoreNum;
    int32_t kPerCore;
};
