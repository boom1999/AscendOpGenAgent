#include "no_smooth_kernel.h"

extern "C" __global__ __aicore__ void no_smooth_custom_bf16(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma,
    GM_ADDR xSum, GM_ADDR yNorm,
    GM_ADDR y1, GM_ADDR scale1,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    NoSmoothKernel<bfloat16_t> kernel;
    kernel.Init(x1, x2, gamma, xSum, yNorm, y1, scale1, tiling, &pipe);
    kernel.Process();
}

extern "C" void no_smooth_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x1, uint8_t *x2, uint8_t *gamma,
    uint8_t *xSum, uint8_t *yNorm,
    uint8_t *y1, uint8_t *scale1,
    uint8_t *tiling)
{
    no_smooth_custom_bf16<<<blockDim, nullptr, stream>>>(
        x1, x2, gamma, xSum, yNorm, y1, scale1, tiling);
}
