#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "rotary_pos_emb_tiling.h"

extern "C" void rotary_pos_emb_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *cos, uint8_t *sin, uint8_t *out, uint8_t *tiling);

extern "C" void rotary_pos_emb_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *cos, uint8_t *sin, uint8_t *out, uint8_t *tiling);

extern "C" void rotary_pos_emb_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *cos, uint8_t *sin, uint8_t *out, uint8_t *tiling);

namespace rotary_pos_emb_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

at::Tensor run_rotary_pos_emb(
    const at::Tensor &x,
    const at::Tensor &cos,
    const at::Tensor &sin)
{
    TORCH_CHECK(x.dim() == 2, "x must be [M, D]");
    TORCH_CHECK(cos.dim() == 2, "cos must be [M, D]");
    TORCH_CHECK(sin.dim() == 2, "sin must be [M, D]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(cos.is_contiguous(), "cos must be contiguous");
    TORCH_CHECK(sin.is_contiguous(), "sin must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat ||
        x.scalar_type() == at::kHalf  ||
        x.scalar_type() == at::kBFloat16,
        "x must be float16, float32, or bfloat16");
    TORCH_CHECK(cos.scalar_type() == x.scalar_type(), "cos must have same dtype as x");
    TORCH_CHECK(sin.scalar_type() == x.scalar_type(), "sin must have same dtype as x");

    const auto M = static_cast<int32_t>(x.sizes()[0]);
    const auto D = static_cast<int32_t>(x.sizes()[1]);
    const int32_t split = D / 2;

    TORCH_CHECK(cos.sizes()[0] == M && cos.sizes()[1] == D, "cos shape must match x");
    TORCH_CHECK(sin.sizes()[0] == M && sin.sizes()[1] == D, "sin shape must match x");
    TORCH_CHECK(D % 2 == 0, "D must be even");

    const int32_t mNum = (M + static_cast<int32_t>(DEFAULT_BLOCK_M) - 1)
                       / static_cast<int32_t>(DEFAULT_BLOCK_M);
    const int32_t usedCoreNum =
        std::min<int32_t>(static_cast<int32_t>(DEFAULT_NUM_PHYSICAL_CORES), mNum);
    const int32_t tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;

    at::Tensor out = at::empty_like(x);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(RotaryPosEmbTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<RotaryPosEmbTiling *>(tilingCpu.data_ptr());
    tiling->M            = M;
    tiling->D            = D;
    tiling->split        = split;
    tiling->blockM       = static_cast<int32_t>(DEFAULT_BLOCK_M);
    tiling->usedCoreNum  = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    LaunchFn launch = nullptr;
    if (x.scalar_type() == at::kFloat) {
        launch = rotary_pos_emb_do_fp32;
    } else if (x.scalar_type() == at::kHalf) {
        launch = rotary_pos_emb_do_fp16;
    } else if (x.scalar_type() == at::kBFloat16) {
        launch = rotary_pos_emb_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(x.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(cos.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(sin.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(out.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return out;
}

}  // namespace rotary_pos_emb_ext

PYBIND11_MODULE(_rotary_pos_emb_ext, m)
{
    m.doc() = "rotary_pos_emb extension";
    m.def("run_rotary_pos_emb",
          &rotary_pos_emb_ext::run_rotary_pos_emb, "");
}
