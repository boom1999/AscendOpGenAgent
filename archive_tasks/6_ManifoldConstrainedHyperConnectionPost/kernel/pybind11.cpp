/**
 * @file pybind11.cpp
 *
 * Python binding for MhcPost AscendC kernel.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cmath>
#include <algorithm>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"

#include "mhc_post_tiling.h"

extern "C" void mhc_post_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *hRes, uint8_t *hOut, uint8_t *hPost,
    uint8_t *y, uint8_t *tiling);

extern "C" void mhc_post_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *hRes, uint8_t *hOut, uint8_t *hPost,
    uint8_t *y, uint8_t *tiling);

namespace mhc_post_ext {

using LaunchFn = void (*)(uint32_t, void *,
                          uint8_t *, uint8_t *, uint8_t *,
                          uint8_t *, uint8_t *, uint8_t *);

at::Tensor run_mhc_post(
    const at::Tensor &x,        // (B, n, D), fp16/bf16
    const at::Tensor &hRes,     // (B, n, nPad), float32
    const at::Tensor &hOut,     // (B, D), fp16/bf16
    const at::Tensor &hPost,    // (B, nPad), float32
    int64_t n,                  // actual n (before padding)
    int64_t nPad,               // padded n
    int64_t blockD)             // D tile size
{
    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, n, D)");
    TORCH_CHECK(hRes.dim() == 3, "h_res must be 3D (B, n, nPad)");
    TORCH_CHECK(hOut.dim() == 2, "h_out must be 2D (B, D)");
    TORCH_CHECK(hPost.dim() == 2, "h_post must be 2D (B, nPad)");
    TORCH_CHECK(hRes.scalar_type() == at::kFloat, "h_res must be float32");
    TORCH_CHECK(hPost.scalar_type() == at::kFloat, "h_post must be float32");
    TORCH_CHECK(x.scalar_type() == at::kHalf ||
                x.scalar_type() == at::kBFloat16,
                "x must be float16 or bfloat16");

    int32_t B = static_cast<int32_t>(x.sizes()[0]);
    int32_t D = static_cast<int32_t>(x.sizes()[2]);

    TORCH_CHECK(D % blockD == 0, "D must be divisible by blockD");
    int32_t dTiles = D / static_cast<int32_t>(blockD);

    // Select launch function
    LaunchFn launchFn;
    if (x.scalar_type() == at::kHalf) {
        launchFn = mhc_post_do_fp16;
    } else {
        launchFn = mhc_post_do_bf16;
    }

    auto origDtype = x.scalar_type();

    // Allocate output
    at::Tensor y = at::empty({B, n, D}, at::device(at::kPrivateUse1).dtype(origDtype));

    // Compute core count
    uint32_t usedCoreNum = std::min<uint32_t>(DEFAULT_NUM_CORES, static_cast<uint32_t>(B));
    int32_t tasksPerCore = (B + static_cast<int32_t>(usedCoreNum) - 1) / static_cast<int32_t>(usedCoreNum);

    // Fill tiling struct
    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(MhcPostTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tp = reinterpret_cast<MhcPostTiling *>(tilingCpu.data_ptr());
    tp->B = B;
    tp->n = static_cast<int32_t>(n);
    tp->nPad = static_cast<int32_t>(nPad);
    tp->D = D;
    tp->blockD = static_cast<int32_t>(blockD);
    tp->dTiles = dTiles;
    tp->usedCoreNum = static_cast<int32_t>(usedCoreNum);
    tp->tasksPerCore = tasksPerCore;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    launchFn(usedCoreNum, aclStream,
             static_cast<uint8_t *>(const_cast<void *>(x.storage().data())),
             static_cast<uint8_t *>(const_cast<void *>(hRes.storage().data())),
             static_cast<uint8_t *>(const_cast<void *>(hOut.storage().data())),
             static_cast<uint8_t *>(const_cast<void *>(hPost.storage().data())),
             static_cast<uint8_t *>(const_cast<void *>(y.storage().data())),
             static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return y;
}

}  // namespace mhc_post_ext

PYBIND11_MODULE(_mhc_post_ext, m)
{
    m.doc() = "MhcPost AscendC kernel extension";
    m.def("run_mhc_post", &mhc_post_ext::run_mhc_post, "");
}
