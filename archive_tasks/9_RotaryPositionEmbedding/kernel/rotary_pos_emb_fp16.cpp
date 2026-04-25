#include "rotary_pos_emb_kernel.h"

extern "C" __global__ __aicore__ void rotary_pos_emb_custom_fp16(
    GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR out, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    RotaryPosEmbKernel<half> kernel;
    kernel.Init(x, cos, sin, out, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void rotary_pos_emb_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *cos, uint8_t *sin, uint8_t *out, uint8_t *tiling)
{
    rotary_pos_emb_custom_fp16<<<blockDim, nullptr, stream>>>(
        x, cos, sin, out, tiling);
}
#endif
