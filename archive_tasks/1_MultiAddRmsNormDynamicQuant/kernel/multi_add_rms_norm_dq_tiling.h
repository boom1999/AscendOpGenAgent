#ifndef MULTI_ADD_RMS_NORM_DQ_TILING_H
#define MULTI_ADD_RMS_NORM_DQ_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_M = 64;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct MultiAddRmsNormDqTiling {
    int32_t M;
    int32_t N;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    float eps;
    float invN;
    float inv127;
};

#endif
