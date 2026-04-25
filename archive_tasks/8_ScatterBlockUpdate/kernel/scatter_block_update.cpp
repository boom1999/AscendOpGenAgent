#include "scatter_block_update_kernel.h"

extern "C" __global__ __aicore__ void scatter_block_update_custom(
    GM_ADDR outputData,
    GM_ADDR indices,
    GM_ADDR updateData,
    GM_ADDR tilingGm
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV);
    ScatterBlockUpdateKernel kernel;
    kernel.Init(
        tilingGm,
        reinterpret_cast<__gm__ uint8_t *>(outputData),
        reinterpret_cast<__gm__ int32_t *>(indices),
        reinterpret_cast<__gm__ uint8_t *>(updateData)
    );
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void scatter_block_update_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *outputData,
    uint8_t *indices,
    uint8_t *updateData,
    uint8_t *tilingGm
) {
    scatter_block_update_custom<<<blockDim, nullptr, stream>>>(
        outputData, indices, updateData, tilingGm
    );
}
#endif
