/**
 * @file mhc_post_grad_bf16.cpp
 *
 * Entry point for MhcPostGrad kernel with bfloat16 data type.
 */
#include "mhc_post_grad_kernel.h"

extern "C" __global__ __aicore__ void mhc_post_grad_custom_bf16(
    GM_ADDR gradOutput, GM_ADDR x, GM_ADDR hRes,
    GM_ADDR hOut, GM_ADDR hPost,
    GM_ADDR gradX, GM_ADDR gradHRes,
    GM_ADDR gradHOut, GM_ADDR gradHPost,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    MhcPostGradKernel<bfloat16_t> kernel;
    kernel.Init(gradOutput, x, hRes, hOut, hPost,
                gradX, gradHRes, gradHOut, gradHPost,
                tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void mhc_post_grad_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *gradOutput, uint8_t *x, uint8_t *hRes,
    uint8_t *hOut, uint8_t *hPost,
    uint8_t *gradX, uint8_t *gradHRes,
    uint8_t *gradHOut, uint8_t *gradHPost,
    uint8_t *tiling)
{
    mhc_post_grad_custom_bf16<<<blockDim, nullptr, stream>>>(
        gradOutput, x, hRes, hOut, hPost,
        gradX, gradHRes, gradHOut, gradHPost,
        tiling);
}
#endif
