#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "aggregate_hidden_tiling.h"

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
    uint8_t *tilingGm);

static inline uint8_t *TensorPtr(const at::Tensor &t)
{
    return static_cast<uint8_t *>(const_cast<void *>(t.storage().data()));
}

pybind11::tuple run_aggregate_hidden(
    const at::Tensor &grad_out_2d,
    const at::Tensor &input_2d,
    const at::Tensor &weight,
    const at::Tensor &mask_f32,
    int64_t S, int64_t B, int64_t H)
{
    TORCH_CHECK(grad_out_2d.dtype() == at::kFloat, "grad_out must be float32");
    TORCH_CHECK(input_2d.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(mask_f32.dtype() == at::kFloat, "mask must be float32");

    int32_t s = static_cast<int32_t>(S);
    int32_t b = static_cast<int32_t>(B);
    int32_t h = static_cast<int32_t>(H);
    int32_t sb = s * b;

    // Allocate output tensors
    at::Tensor output = at::empty({sb, h}, grad_out_2d.options());
    at::Tensor gradInput = at::empty({sb, h}, grad_out_2d.options());
    at::Tensor gradWeight = at::empty({3, h}, grad_out_2d.options());

    // Compute tiling
    int32_t blockH = h < DEFAULT_BLOCK_H ? h : DEFAULT_BLOCK_H;
    int32_t hNum = (h + blockH - 1) / blockH;
    int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_CORES, hNum);
    if (usedCoreNum < 1) usedCoreNum = 1;
    int32_t tasksPerCore = (hNum + usedCoreNum - 1) / usedCoreNum;

    AggregateHiddenTiling tiling;
    tiling.S = s;
    tiling.B = b;
    tiling.H = h;
    tiling.blockH = blockH;
    tiling.hNum = hNum;
    tiling.usedCoreNum = usedCoreNum;
    tiling.tasksPerCore = tasksPerCore;

    // Copy tiling to device
    at::Tensor tilingCpu = at::from_blob(
        &tiling,
        {static_cast<long>(sizeof(AggregateHiddenTiling))},
        at::TensorOptions().dtype(at::kByte));
    at::Tensor tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    aggregate_hidden_do(
        usedCoreNum,
        aclStream,
        TensorPtr(grad_out_2d),
        TensorPtr(input_2d),
        TensorPtr(weight),
        TensorPtr(mask_f32),
        TensorPtr(output),
        TensorPtr(gradInput),
        TensorPtr(gradWeight),
        TensorPtr(tilingNpu));

    return pybind11::make_tuple(output, gradInput, gradWeight);
}

PYBIND11_MODULE(_aggregate_hidden_ext, m)
{
    m.doc() = "AggregateHidden AscendC kernel";
    m.def("run_aggregate_hidden", &run_aggregate_hidden,
          "Fused forward conv + backward grad_input/grad_weight");
}
