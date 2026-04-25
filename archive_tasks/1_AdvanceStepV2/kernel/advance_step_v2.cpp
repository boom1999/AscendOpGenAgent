#include "advance_step_v2_kernel.h"

extern "C" __global__ __aicore__ void advance_step_v2_custom(
    GM_ADDR inputTokens,
    GM_ADDR sampledTokens,
    GM_ADDR inputPositions,
    GM_ADDR acceptedNum,
    GM_ADDR blockTableFlat,
    GM_ADDR specTokensFlat,
    GM_ADDR outInputTokens,
    GM_ADDR outInputPositions,
    GM_ADDR outSeqLens,
    GM_ADDR outSlotMapping,
    GM_ADDR tilingGm
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV);
    AdvanceStepV2Kernel kernel;
    kernel.Init(
        tilingGm,
        reinterpret_cast<__gm__ int64_t *>(inputTokens),
        reinterpret_cast<__gm__ int64_t *>(sampledTokens),
        reinterpret_cast<__gm__ int64_t *>(inputPositions),
        reinterpret_cast<__gm__ int64_t *>(acceptedNum),
        reinterpret_cast<__gm__ int64_t *>(blockTableFlat),
        reinterpret_cast<__gm__ int64_t *>(specTokensFlat),
        reinterpret_cast<__gm__ int64_t *>(outInputTokens),
        reinterpret_cast<__gm__ int64_t *>(outInputPositions),
        reinterpret_cast<__gm__ int64_t *>(outSeqLens),
        reinterpret_cast<__gm__ int64_t *>(outSlotMapping)
    );
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void advance_step_v2_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *inputTokens,
    uint8_t *sampledTokens,
    uint8_t *inputPositions,
    uint8_t *acceptedNum,
    uint8_t *blockTableFlat,
    uint8_t *specTokensFlat,
    uint8_t *outInputTokens,
    uint8_t *outInputPositions,
    uint8_t *outSeqLens,
    uint8_t *outSlotMapping,
    uint8_t *tilingGm
) {
    advance_step_v2_custom<<<blockDim, nullptr, stream>>>(
        inputTokens, sampledTokens, inputPositions,
        acceptedNum, blockTableFlat, specTokensFlat,
        outInputTokens, outInputPositions, outSeqLens,
        outSlotMapping, tilingGm
    );
}
#endif
