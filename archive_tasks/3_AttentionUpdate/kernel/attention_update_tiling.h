#pragma once
#include <cstdint>

struct AttentionUpdateTiling {
    uint32_t K;
    uint32_t N;
    uint32_t H;
    uint32_t usedCoreNum;
    uint32_t tasksPerCore;
    uint32_t hAlign;
};
