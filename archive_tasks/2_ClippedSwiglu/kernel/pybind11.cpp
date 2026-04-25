#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "clipped_swiglu_tiling.h"

namespace py = pybind11;

extern "C" void clipped_swiglu_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *A,
    uint8_t *B,
    uint8_t *Y,
    uint8_t *tilingGm
);

static uint8_t *TensorAddr(const at::Tensor &t) {
    return static_cast<uint8_t *>(const_cast<void *>(t.storage().data()));
}

at::Tensor run_clipped_swiglu(
    const at::Tensor &A,
    const at::Tensor &B,
    int64_t M,
    int64_t N,
    double alpha,
    double limit,
    double biasVal
) {
    TORCH_CHECK(A.dtype() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.dtype() == at::kFloat, "B must be float32");

    at::Tensor Y = at::empty({M, N}, A.options());

    int32_t m = static_cast<int32_t>(M);
    int32_t n = static_cast<int32_t>(N);
    int32_t numCores = 20;
    int32_t usedCoreNum = m < numCores ? m : numCores;
    if (usedCoreNum < 1) usedCoreNum = 1;
    int32_t tasksPerCore = (m + usedCoreNum - 1) / usedCoreNum;
    int32_t blockN = n < 1024 ? n : 1024;
    int32_t nLoops = (n + blockN - 1) / blockN;

    ClippedSwigluTiling tiling;
    tiling.M = m;
    tiling.N = n;
    tiling.usedCoreNum = usedCoreNum;
    tiling.tasksPerCore = tasksPerCore;
    tiling.blockN = blockN;
    tiling.nLoops = nLoops;
    tiling.alpha = static_cast<float>(alpha);
    tiling.limit = static_cast<float>(limit);
    tiling.biasVal = static_cast<float>(biasVal);

    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(ClippedSwigluTiling))},
        at::TensorOptions().dtype(at::kByte).device(A.device())
    );
    auto tilingHost = at::from_blob(
        &tiling,
        {static_cast<int64_t>(sizeof(ClippedSwigluTiling))},
        at::TensorOptions().dtype(at::kByte)
    );
    tilingTensor.copy_(tilingHost);

    auto stream = c10_npu::getCurrentNPUStream();

    clipped_swiglu_do(
        usedCoreNum,
        stream,
        TensorAddr(A),
        TensorAddr(B),
        TensorAddr(Y),
        TensorAddr(tilingTensor)
    );

    return Y;
}

PYBIND11_MODULE(_clipped_swiglu_ext, m) {
    m.def("run_clipped_swiglu", &run_clipped_swiglu,
          "ClippedSwiglu kernel (AscendC)");
}
