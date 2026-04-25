#pragma once
#include <cstdint>

struct ClippedSwigluTiling {
    int32_t M;
    int32_t N;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t blockN;
    int32_t nLoops;
    float alpha;
    float limit;
    float biasVal;
};
