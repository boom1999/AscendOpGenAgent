#include "kernel_operator.h"
#include "attention_update_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void attention_update_custom(
    GM_ADDR lse0, GM_ADDR lse1, GM_ADDR lse2,
    GM_ADDR out0, GM_ADDR out1, GM_ADDR out2,
    GM_ADDR resultOut, GM_ADDR lseOut,
    GM_ADDR tilingGm
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV);
    AttentionUpdateKernel op;
    op.Init(
        tilingGm,
        reinterpret_cast<__gm__ float *>(lse0),
        reinterpret_cast<__gm__ float *>(lse1),
        reinterpret_cast<__gm__ float *>(lse2),
        reinterpret_cast<__gm__ float *>(out0),
        reinterpret_cast<__gm__ float *>(out1),
        reinterpret_cast<__gm__ float *>(out2),
        reinterpret_cast<__gm__ float *>(resultOut),
        reinterpret_cast<__gm__ float *>(lseOut)
    );
    op.InitBuffers();
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void attention_update_do(
    uint32_t blockDim, void *stream,
    uint8_t *lse0, uint8_t *lse1, uint8_t *lse2,
    uint8_t *out0, uint8_t *out1, uint8_t *out2,
    uint8_t *resultOut, uint8_t *lseOut,
    uint8_t *tilingGm
) {
    attention_update_custom<<<blockDim, nullptr, stream>>>(
        lse0, lse1, lse2,
        out0, out1, out2,
        resultOut, lseOut,
        tilingGm
    );
}
#endif
