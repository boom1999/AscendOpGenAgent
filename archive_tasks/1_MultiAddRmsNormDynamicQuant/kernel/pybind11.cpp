#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "multi_add_rms_norm_dq_tiling.h"

// --- NoSmooth launch declarations ---
extern "C" void no_smooth_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *x1, uint8_t *x2, uint8_t *gamma,
    uint8_t *xSum, uint8_t *yNorm,
    uint8_t *y1, uint8_t *scale1,
    uint8_t *tiling);

extern "C" void no_smooth_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *x1, uint8_t *x2, uint8_t *gamma,
    uint8_t *xSum, uint8_t *yNorm,
    uint8_t *y1, uint8_t *scale1,
    uint8_t *tiling);

// --- Smooth1 launch declarations ---
extern "C" void smooth1_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *x1, uint8_t *x2, uint8_t *gamma, uint8_t *ss1,
    uint8_t *xSum, uint8_t *yNorm,
    uint8_t *y1, uint8_t *scale1,
    uint8_t *tiling);

extern "C" void smooth1_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *x1, uint8_t *x2, uint8_t *gamma, uint8_t *ss1,
    uint8_t *xSum, uint8_t *yNorm,
    uint8_t *y1, uint8_t *scale1,
    uint8_t *tiling);

// --- DualSmooth launch declarations ---
extern "C" void dual_smooth_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *x1, uint8_t *x2, uint8_t *gamma, uint8_t *ss1, uint8_t *ss2,
    uint8_t *xSum, uint8_t *yNorm,
    uint8_t *y1, uint8_t *scale1,
    uint8_t *y2, uint8_t *scale2,
    uint8_t *tiling);

extern "C" void dual_smooth_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *x1, uint8_t *x2, uint8_t *gamma, uint8_t *ss1, uint8_t *ss2,
    uint8_t *xSum, uint8_t *yNorm,
    uint8_t *y1, uint8_t *scale1,
    uint8_t *y2, uint8_t *scale2,
    uint8_t *tiling);

namespace multi_add_rms_norm_dq_ext {

using NoSmoothLaunchFn = void (*)(uint32_t, void *,
    uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint8_t *,
    uint8_t *, uint8_t *,
    uint8_t *);

using Smooth1LaunchFn = void (*)(uint32_t, void *,
    uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint8_t *,
    uint8_t *, uint8_t *,
    uint8_t *);

using DualSmoothLaunchFn = void (*)(uint32_t, void *,
    uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint8_t *,
    uint8_t *, uint8_t *,
    uint8_t *, uint8_t *,
    uint8_t *);

static inline uint8_t *TensorPtr(const at::Tensor &t)
{
    return static_cast<uint8_t *>(const_cast<void *>(t.storage().data()));
}

static inline at::Tensor MakeTiling(int32_t m, int32_t n, float eps)
{
    const int32_t mNum = (m + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    const int32_t tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(MultiAddRmsNormDqTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<MultiAddRmsNormDqTiling *>(tilingCpu.data_ptr());
    tiling->M = m;
    tiling->N = n;
    tiling->blockM = DEFAULT_BLOCK_M;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->eps = eps;
    tiling->invN = 1.0f / static_cast<float>(n);
    tiling->inv127 = 1.0f / 127.0f;
    return tilingCpu;
}

pybind11::tuple run_no_smooth(
    const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &gamma, double eps)
{
    TORCH_CHECK(x1.dim() == 2, "x1 must be [M, N]");
    TORCH_CHECK(x2.dim() == 2, "x2 must be [M, N]");
    TORCH_CHECK(gamma.dim() == 2, "gamma must be [M, N]");

    const auto m = static_cast<int32_t>(x1.sizes()[0]);
    const auto n = static_cast<int32_t>(x1.sizes()[1]);

    at::Tensor xSum = at::empty_like(x1);
    at::Tensor yNorm = at::empty_like(x1);
    at::Tensor y1 = at::empty({m, n}, x1.options().dtype(at::kChar));
    at::Tensor scale1 = at::empty({m}, x1.options().dtype(at::kFloat));

    at::Tensor tilingCpu = MakeTiling(m, n, static_cast<float>(eps));
    const int32_t mNum = (m + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    NoSmoothLaunchFn launch = nullptr;
    if (x1.scalar_type() == at::kHalf) {
        launch = no_smooth_do_fp16;
    } else if (x1.scalar_type() == at::kBFloat16) {
        launch = no_smooth_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype for no_smooth");
    }

    launch(usedCoreNum, aclStream,
        TensorPtr(x1), TensorPtr(x2), TensorPtr(gamma),
        TensorPtr(xSum), TensorPtr(yNorm),
        TensorPtr(y1), TensorPtr(scale1),
        TensorPtr(tilingNpu));

    return pybind11::make_tuple(xSum, yNorm, y1, scale1);
}

pybind11::tuple run_smooth1(
    const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &gamma,
    const at::Tensor &ss1, double eps)
{
    TORCH_CHECK(x1.dim() == 2, "x1 must be [M, N]");

    const auto m = static_cast<int32_t>(x1.sizes()[0]);
    const auto n = static_cast<int32_t>(x1.sizes()[1]);

    at::Tensor xSum = at::empty_like(x1);
    at::Tensor yNorm = at::empty_like(x1);
    at::Tensor y1 = at::empty({m, n}, x1.options().dtype(at::kChar));
    at::Tensor scale1 = at::empty({m}, x1.options().dtype(at::kFloat));

    at::Tensor tilingCpu = MakeTiling(m, n, static_cast<float>(eps));
    const int32_t mNum = (m + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    Smooth1LaunchFn launch = nullptr;
    if (x1.scalar_type() == at::kHalf) {
        launch = smooth1_do_fp16;
    } else if (x1.scalar_type() == at::kBFloat16) {
        launch = smooth1_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype for smooth1");
    }

    launch(usedCoreNum, aclStream,
        TensorPtr(x1), TensorPtr(x2), TensorPtr(gamma), TensorPtr(ss1),
        TensorPtr(xSum), TensorPtr(yNorm),
        TensorPtr(y1), TensorPtr(scale1),
        TensorPtr(tilingNpu));

    return pybind11::make_tuple(xSum, yNorm, y1, scale1);
}

pybind11::tuple run_dual_smooth(
    const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &gamma,
    const at::Tensor &ss1, const at::Tensor &ss2, double eps)
{
    TORCH_CHECK(x1.dim() == 2, "x1 must be [M, N]");

    const auto m = static_cast<int32_t>(x1.sizes()[0]);
    const auto n = static_cast<int32_t>(x1.sizes()[1]);

    at::Tensor xSum = at::empty_like(x1);
    at::Tensor yNorm = at::empty_like(x1);
    at::Tensor y1 = at::empty({m, n}, x1.options().dtype(at::kChar));
    at::Tensor scale1 = at::empty({m}, x1.options().dtype(at::kFloat));
    at::Tensor y2 = at::empty({m, n}, x1.options().dtype(at::kChar));
    at::Tensor scale2 = at::empty({m}, x1.options().dtype(at::kFloat));

    at::Tensor tilingCpu = MakeTiling(m, n, static_cast<float>(eps));
    const int32_t mNum = (m + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    DualSmoothLaunchFn launch = nullptr;
    if (x1.scalar_type() == at::kHalf) {
        launch = dual_smooth_do_fp16;
    } else if (x1.scalar_type() == at::kBFloat16) {
        launch = dual_smooth_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype for dual_smooth");
    }

    launch(usedCoreNum, aclStream,
        TensorPtr(x1), TensorPtr(x2), TensorPtr(gamma),
        TensorPtr(ss1), TensorPtr(ss2),
        TensorPtr(xSum), TensorPtr(yNorm),
        TensorPtr(y1), TensorPtr(scale1),
        TensorPtr(y2), TensorPtr(scale2),
        TensorPtr(tilingNpu));

    return pybind11::make_tuple(xSum, yNorm, y1, scale1, y2, scale2);
}

}  // namespace multi_add_rms_norm_dq_ext

PYBIND11_MODULE(_multi_add_rms_norm_dq_ext, m)
{
    m.doc() = "multi_add_rms_norm_dynamic_quant extension";
    m.def("run_no_smooth", &multi_add_rms_norm_dq_ext::run_no_smooth, "");
    m.def("run_smooth1", &multi_add_rms_norm_dq_ext::run_smooth1, "");
    m.def("run_dual_smooth", &multi_add_rms_norm_dq_ext::run_dual_smooth, "");
}
