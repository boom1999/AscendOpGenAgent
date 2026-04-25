#include "aggregate_hidden_kernel.h"

extern "C" __global__ __aicore__ void aggregate_hidden_custom(
    GM_ADDR gradOut,
    GM_ADDR input,
    GM_ADDR weight,
    GM_ADDR mask,
    GM_ADDR output,
    GM_ADDR gradInput,
    GM_ADDR gradWeight,
    GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    AggregateHiddenKernel kernel;
    kernel.Init(gradOut, input, weight, mask,
                output, gradInput, gradWeight,
                tilingGm, &pipe);
    kernel.InitBuffers();
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void aggregate_hidden_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *gradOut,
    uint8_t *input,
    uint8_t *weight,
    uint8_t *mask,
    uint8_t *output,
    uint8_t *gradInput,
    uint8_t *gradWeight,
    uint8_t *tilingGm)
{
    aggregate_hidden_custom<<<blockDim, nullptr, stream>>>(
        gradOut, input, weight, mask,
        output, gradInput, gradWeight,
        tilingGm);
}
#endif
