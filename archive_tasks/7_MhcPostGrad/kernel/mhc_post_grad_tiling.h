#ifndef MHC_POST_GRAD_TILING_H
#define MHC_POST_GRAD_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_NUM_CORES = 20;

#pragma pack(push, 4)
struct MhcPostGradTiling {
    int32_t B;          // batch size (flattened)
    int32_t n;          // small dim (4, 6, 8)
    int32_t nPad;       // padded n for 32B alignment
    int32_t D;          // D dimension (padded to block_D multiple)
    int32_t blockD;     // D tile size
    int32_t dTiles;     // = D / blockD
    int32_t usedCoreNum;
    int32_t tasksPerCore;
};
#pragma pack(pop)

#endif // MHC_POST_GRAD_TILING_H
