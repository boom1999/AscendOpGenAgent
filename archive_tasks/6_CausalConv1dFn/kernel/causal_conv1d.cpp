#include "causal_conv1d_kernel.h"

extern "C" __global__ __aicore__ void causal_conv1d_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR convStatesFlat,
    GM_ADDR queryStartLoc,
    GM_ADDR cacheIndices,
    GM_ADDR initialStateMode,
    GM_ADDR output,
    GM_ADDR cacheUpdates,
    GM_ADDR tilingGm
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV);
    CausalConv1dKernel kernel;
    kernel.Init(
        tilingGm,
        reinterpret_cast<__gm__ float *>(x),
        reinterpret_cast<__gm__ float *>(weight),
        reinterpret_cast<__gm__ float *>(convStatesFlat),
        reinterpret_cast<__gm__ int32_t *>(queryStartLoc),
        reinterpret_cast<__gm__ int32_t *>(cacheIndices),
        reinterpret_cast<__gm__ int32_t *>(initialStateMode),
        reinterpret_cast<__gm__ float *>(output),
        reinterpret_cast<__gm__ float *>(cacheUpdates)
    );
    kernel.InitBuffers();
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void causal_conv1d_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *weight,
    uint8_t *convStatesFlat,
    uint8_t *queryStartLoc,
    uint8_t *cacheIndices,
    uint8_t *initialStateMode,
    uint8_t *output,
    uint8_t *cacheUpdates,
    uint8_t *tilingGm
) {
    causal_conv1d_custom<<<blockDim, nullptr, stream>>>(
        x, weight, convStatesFlat,
        queryStartLoc, cacheIndices, initialStateMode,
        output, cacheUpdates, tilingGm
    );
}
#endif
