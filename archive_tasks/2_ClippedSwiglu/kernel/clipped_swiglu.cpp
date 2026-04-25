#include "clipped_swiglu_kernel.h"

extern "C" __global__ __aicore__ void clipped_swiglu_custom(
    GM_ADDR A,
    GM_ADDR B,
    GM_ADDR Y,
    GM_ADDR tilingGm
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV);
    ClippedSwigluKernel kernel;
    kernel.Init(
        tilingGm,
        reinterpret_cast<__gm__ float *>(A),
        reinterpret_cast<__gm__ float *>(B),
        reinterpret_cast<__gm__ float *>(Y)
    );
    kernel.InitBuffers();
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void clipped_swiglu_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *A,
    uint8_t *B,
    uint8_t *Y,
    uint8_t *tilingGm
) {
    clipped_swiglu_custom<<<blockDim, nullptr, stream>>>(
        A, B, Y, tilingGm
    );
}
#endif
