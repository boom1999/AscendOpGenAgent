#ifndef APPLY_ROTARY_POS_EMB_TILING_H
#define APPLY_ROTARY_POS_EMB_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_M = 64;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct ApplyRotaryPosEmbTiling {
    int32_t M;            // total rows
    int32_t D;            // head dimension
    int32_t D_rot;        // rotary dimension (<= D)
    int32_t split;        // D_rot / 2
    int32_t blockM;       // rows per block
    int32_t usedCoreNum;  // actual cores used
    int32_t tasksPerCore; // blocks per core
};

#endif
