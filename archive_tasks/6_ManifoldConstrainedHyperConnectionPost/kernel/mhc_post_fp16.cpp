/**
 * @file mhc_post_fp16.cpp
 *
 * Entry point for MhcPost kernel with float16 data type.
 */
#include "mhc_post_kernel.h"

extern "C" __global__ __aicore__ void mhc_post_custom_fp16(
    GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
    GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    MhcPostKernel<half> kernel;
    kernel.Init(x, hRes, hOut, hPost, y, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void mhc_post_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *hRes, uint8_t *hOut, uint8_t *hPost,
    uint8_t *y, uint8_t *tiling)
{
    mhc_post_custom_fp16<<<blockDim, nullptr, stream>>>(
        x, hRes, hOut, hPost, y, tiling);
}
#endif
