#ifndef ROTARY_POS_EMB_TILING_H
#define ROTARY_POS_EMB_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_M = 64;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct RotaryPosEmbTiling {
    int32_t M;            // total rows (B * H * S)
    int32_t D;            // head dimension
    int32_t split;        // D / 2
    int32_t blockM;       // rows per block
    int32_t usedCoreNum;  // actual cores used
    int32_t tasksPerCore; // blocks per core
};

#endif
