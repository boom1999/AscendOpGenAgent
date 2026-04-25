#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "advance_step_v2_tiling.h"

namespace py = pybind11;

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
);

static uint8_t *TensorAddr(const at::Tensor &t) {
    return static_cast<uint8_t *>(const_cast<void *>(t.storage().data()));
}

std::vector<at::Tensor> run_advance_step_v2(
    const at::Tensor &inputTokens,
    const at::Tensor &sampledTokens,
    const at::Tensor &inputPositions,
    const at::Tensor &acceptedNum,
    const at::Tensor &blockTableFlat,
    const at::Tensor &specTokensFlat,
    int64_t numReqs,
    int64_t tokenEachReqs,
    int64_t sampledCols,
    int64_t maxNumBlocks,
    int64_t blockSize
) {
    TORCH_CHECK(inputTokens.dtype() == at::kLong, "inputTokens must be int64");

    const int32_t totalElements = static_cast<int32_t>(numReqs * tokenEachReqs);

    auto opts = inputTokens.options();
    at::Tensor outInputTokens = at::empty({totalElements}, opts);
    at::Tensor outInputPositions = at::empty({totalElements}, opts);
    at::Tensor outSeqLens = at::empty({totalElements}, opts);
    at::Tensor outSlotMapping = at::empty({totalElements}, opts);

    int32_t usedCoreNum = static_cast<int32_t>(numReqs);
    if (usedCoreNum > 40) usedCoreNum = 40;
    if (usedCoreNum < 1) usedCoreNum = 1;

    int32_t reqsPerCore = (static_cast<int32_t>(numReqs) + usedCoreNum - 1) / usedCoreNum;

    AdvanceStepV2Tiling tiling;
    tiling.numReqs = static_cast<int32_t>(numReqs);
    tiling.tokenEachReqs = static_cast<int32_t>(tokenEachReqs);
    tiling.sampledCols = static_cast<int32_t>(sampledCols);
    tiling.maxNumBlocks = static_cast<int32_t>(maxNumBlocks);
    tiling.blockSize = static_cast<int32_t>(blockSize);
    tiling.usedCoreNum = usedCoreNum;
    tiling.reqsPerCore = reqsPerCore;

    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(AdvanceStepV2Tiling))},
        at::TensorOptions().dtype(at::kByte).device(inputTokens.device())
    );
    auto tilingHost = at::from_blob(
        &tiling,
        {static_cast<int64_t>(sizeof(AdvanceStepV2Tiling))},
        at::TensorOptions().dtype(at::kByte)
    );
    tilingTensor.copy_(tilingHost);

    auto stream = c10_npu::getCurrentNPUStream();

    advance_step_v2_do(
        usedCoreNum,
        stream,
        TensorAddr(inputTokens),
        TensorAddr(sampledTokens),
        TensorAddr(inputPositions),
        TensorAddr(acceptedNum),
        TensorAddr(blockTableFlat),
        TensorAddr(specTokensFlat),
        TensorAddr(outInputTokens),
        TensorAddr(outInputPositions),
        TensorAddr(outSeqLens),
        TensorAddr(outSlotMapping),
        TensorAddr(tilingTensor)
    );

    return {outInputTokens, outInputPositions, outSeqLens, outSlotMapping};
}

PYBIND11_MODULE(_advance_step_v2_ext, m) {
    m.def("run_advance_step_v2", &run_advance_step_v2,
          "AdvanceStepV2 kernel (AscendC)");
}
