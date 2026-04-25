#include "dual_smooth_kernel.h"

extern "C" __global__ __aicore__ void dual_smooth_custom_bf16(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR ss1, GM_ADDR ss2,
    GM_ADDR xSum, GM_ADDR yNorm,
    GM_ADDR y1, GM_ADDR scale1,
    GM_ADDR y2, GM_ADDR scale2,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    DualSmoothKernel<bfloat16_t> kernel;
    kernel.Init(x1, x2, gamma, ss1, ss2, xSum, yNorm, y1, scale1, y2, scale2, tiling, &pipe);
    kernel.Process();
}

extern "C" void dual_smooth_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x1, uint8_t *x2, uint8_t *gamma, uint8_t *ss1, uint8_t *ss2,
    uint8_t *xSum, uint8_t *yNorm,
    uint8_t *y1, uint8_t *scale1,
    uint8_t *y2, uint8_t *scale2,
    uint8_t *tiling)
{
    dual_smooth_custom_bf16<<<blockDim, nullptr, stream>>>(
        x1, x2, gamma, ss1, ss2, xSum, yNorm, y1, scale1, y2, scale2, tiling);
}
