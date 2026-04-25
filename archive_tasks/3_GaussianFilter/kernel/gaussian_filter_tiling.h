#pragma once
#include <cstdint>

#pragma pack(push, 4)
struct GaussianFilterTiling {
    int32_t BC;
    int32_t B;
    int32_t N_padded;
    int32_t M_padded;
    int32_t TILE_N;
    int32_t n_tiles;
    int32_t bytes_per_tile;
    float near_plane;
    float far_plane;
    float width;
    float height;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
};
#pragma pack(pop)
