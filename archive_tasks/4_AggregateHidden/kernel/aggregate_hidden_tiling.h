#pragma once
#include <cstdint>

constexpr int32_t DEFAULT_BLOCK_H = 1024;
constexpr int32_t DEFAULT_NUM_CORES = 20;

struct AggregateHiddenTiling {
    int32_t S;
    int32_t B;
    int32_t H;
    int32_t blockH;
    int32_t hNum;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
};
