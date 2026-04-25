#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "scatter_block_update_tiling.h"

namespace py = pybind11;

extern "C" void scatter_block_update_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *outputData,
    uint8_t *indices,
    uint8_t *updateData,
    uint8_t *tilingGm
);

at::Tensor run_scatter_block_update(
    const at::Tensor &input,
    const at::Tensor &indices,
    const at::Tensor &update,
    int64_t D0,
    int64_t D1,
    int64_t D2,
    int64_t K,
    int64_t elemSize
) {
    at::Tensor output = input.clone();
    at::Tensor output_flat = output.reshape({D0 * D1, D2}).contiguous();

    int32_t usedCoreNum = static_cast<int32_t>(K);
    if (usedCoreNum > 20) usedCoreNum = 20;
    if (usedCoreNum < 1) usedCoreNum = 1;

    // When D2==1, each output row is a single element (1-4 bytes).
    // Many rows share a single cache line and multi-core scalar writes
    // cause false sharing / write tearing. Fall back to single core.
    if (D2 <= 1) {
        usedCoreNum = 1;
    }

    int32_t kPerCore = (static_cast<int32_t>(K) + usedCoreNum - 1) / usedCoreNum;

    ScatterBlockUpdateTiling tiling;
    tiling.D1 = static_cast<int32_t>(D1);
    tiling.D2 = static_cast<int32_t>(D2);
    tiling.K = static_cast<int32_t>(K);
    tiling.elemSize = static_cast<int32_t>(elemSize);
    tiling.usedCoreNum = usedCoreNum;
    tiling.kPerCore = kPerCore;

    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(ScatterBlockUpdateTiling))},
        at::TensorOptions().dtype(at::kByte).device(output.device())
    );
    auto tilingHost = at::from_blob(
        &tiling,
        {static_cast<int64_t>(sizeof(ScatterBlockUpdateTiling))},
        at::TensorOptions().dtype(at::kByte)
    );
    tilingTensor.copy_(tilingHost);

    auto stream = c10_npu::getCurrentNPUStream();

    scatter_block_update_do(
        usedCoreNum,
        stream,
        static_cast<uint8_t *>(output_flat.data_ptr()),
        static_cast<uint8_t *>(indices.data_ptr()),
        static_cast<uint8_t *>(update.data_ptr()),
        static_cast<uint8_t *>(tilingTensor.data_ptr())
    );

    return output.reshape({D0, D1, D2});
}

PYBIND11_MODULE(_scatter_block_update_ext, m) {
    m.def("run_scatter_block_update", &run_scatter_block_update,
          "ScatterBlockUpdate kernel (AscendC)");
}
