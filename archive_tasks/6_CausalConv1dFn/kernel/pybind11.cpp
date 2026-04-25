#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "causal_conv1d_tiling.h"

namespace py = pybind11;

extern "C" void causal_conv1d_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *weight,
    uint8_t *convStatesFlat,
    uint8_t *queryStartLoc,
    uint8_t *cacheIndices,
    uint8_t *initialStateMode,
    uint8_t *output,
    uint8_t *cacheUpdates,
    uint8_t *tilingGm
);

static uint8_t *TensorAddr(const at::Tensor &t) {
    return static_cast<uint8_t *>(const_cast<void *>(t.storage().data()));
}

std::vector<at::Tensor> run_causal_conv1d(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &convStatesFlat,
    const at::Tensor &queryStartLoc,
    const at::Tensor &cacheIndices,
    const at::Tensor &initialStateMode,
    int64_t residual,
    int64_t padSlotId
) {
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");

    int32_t cuSeqLen = static_cast<int32_t>(x.size(0));
    int32_t dim = static_cast<int32_t>(x.size(1));
    int32_t numStatesXSl = static_cast<int32_t>(convStatesFlat.size(0));
    int32_t batchCount = static_cast<int32_t>(queryStartLoc.size(0)) - 1;

    at::Tensor output = at::empty({cuSeqLen, dim}, x.options());
    int32_t statelen = 2;
    at::Tensor cacheUpdates = at::empty({batchCount * statelen, dim}, x.options());

    int32_t numCores = 20;
    int32_t usedCoreNum = batchCount < numCores ? batchCount : numCores;
    if (usedCoreNum < 1) usedCoreNum = 1;
    int32_t tasksPerCore = (batchCount + usedCoreNum - 1) / usedCoreNum;
    int32_t blockN = dim < 1024 ? dim : 1024;
    int32_t nTiles = (dim + blockN - 1) / blockN;

    CausalConv1dTiling tiling;
    tiling.cuSeqLen = cuSeqLen;
    tiling.dim = dim;
    tiling.numStatesXSl = numStatesXSl;
    tiling.batchCount = batchCount;
    tiling.usedCoreNum = usedCoreNum;
    tiling.tasksPerCore = tasksPerCore;
    tiling.blockN = blockN;
    tiling.nTiles = nTiles;
    tiling.residual = static_cast<int32_t>(residual);
    tiling.padSlotId = static_cast<int32_t>(padSlotId);

    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(CausalConv1dTiling))},
        at::TensorOptions().dtype(at::kByte).device(x.device())
    );
    auto tilingHost = at::from_blob(
        &tiling,
        {static_cast<int64_t>(sizeof(CausalConv1dTiling))},
        at::TensorOptions().dtype(at::kByte)
    );
    tilingTensor.copy_(tilingHost);

    auto stream = c10_npu::getCurrentNPUStream();

    causal_conv1d_do(
        usedCoreNum,
        stream,
        TensorAddr(x),
        TensorAddr(weight),
        TensorAddr(convStatesFlat),
        TensorAddr(queryStartLoc),
        TensorAddr(cacheIndices),
        TensorAddr(initialStateMode),
        TensorAddr(output),
        TensorAddr(cacheUpdates),
        TensorAddr(tilingTensor)
    );

    return {output, cacheUpdates};
}

PYBIND11_MODULE(_causal_conv1d_ext, m) {
    m.def("run_causal_conv1d", &run_causal_conv1d,
          "CausalConv1d kernel (AscendC)");
}
