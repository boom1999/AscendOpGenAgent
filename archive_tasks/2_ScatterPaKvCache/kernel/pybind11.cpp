#include <algorithm>
#include <cstring>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "scatter_pa_kv_cache_tiling.h"

extern "C" void scatter_pa_kv_cache_h_h_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *key,
    uint8_t *keyCache,
    uint8_t *slotMapping,
    uint8_t *value,
    uint8_t *valueCache,
    uint8_t *tiling);

extern "C" void scatter_pa_kv_cache_i8_h_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *key,
    uint8_t *keyCache,
    uint8_t *slotMapping,
    uint8_t *value,
    uint8_t *valueCache,
    uint8_t *tiling);

namespace current_task_ext {

using LaunchFn = void (*)(
    uint32_t, void *,
    uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

inline int32_t CeilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

void run_scatter(
    const at::Tensor &key,
    const at::Tensor &keyCache,
    const at::Tensor &slotMapping,
    const at::Tensor &value,
    const at::Tensor &valueCache)
{
    TORCH_CHECK(key.device().is_privateuseone(), "key must be on NPU");
    TORCH_CHECK(keyCache.device().is_privateuseone(), "keyCache must be on NPU");
    TORCH_CHECK(slotMapping.device().is_privateuseone(), "slotMapping must be on NPU");
    TORCH_CHECK(value.device().is_privateuseone(), "value must be on NPU");
    TORCH_CHECK(valueCache.device().is_privateuseone(), "valueCache must be on NPU");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(keyCache.is_contiguous(), "keyCache must be contiguous");
    TORCH_CHECK(slotMapping.is_contiguous(), "slotMapping must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(valueCache.is_contiguous(), "valueCache must be contiguous");
    TORCH_CHECK(slotMapping.scalar_type() == at::kInt, "slotMapping must be int32");
    TORCH_CHECK(key.dim() == 2, "key must be 2D [nTokens, keyFlatDim]");
    TORCH_CHECK(keyCache.dim() == 3, "keyCache must be 3D [totalKvRows, blockSize, lastDimK]");
    TORCH_CHECK(value.dim() == 2, "value must be 2D [nTokens, valFlatDim]");
    TORCH_CHECK(valueCache.dim() == 3, "valueCache must be 3D [totalVvRows, blockSize, lastDimV]");

    const int32_t nTokens = static_cast<int32_t>(key.size(0));
    const int32_t keyFlatDim = static_cast<int32_t>(key.size(1));
    const int32_t totalKvRows = static_cast<int32_t>(keyCache.size(0));
    const int32_t blockSize = static_cast<int32_t>(keyCache.size(1));
    const int32_t lastDimK = static_cast<int32_t>(keyCache.size(2));
    const int32_t valFlatDim = static_cast<int32_t>(value.size(1));
    const int32_t totalVvRows = static_cast<int32_t>(valueCache.size(0));
    const int32_t lastDimV = static_cast<int32_t>(valueCache.size(2));

    const int32_t numKvSlices = keyFlatDim / lastDimK;
    const int32_t numVvSlices = valFlatDim / lastDimV;
    const int32_t numBlocks = totalKvRows / numKvSlices;

    const int32_t usedCoreNum = std::min(
        static_cast<int32_t>(SCATTER_NUM_PHYSICAL_CORES),
        std::max(nTokens, 1));
    const int32_t tokensPerCore = CeilDiv(std::max(nTokens, 1), usedCoreNum);

    // Select kernel variant based on key element size
    const int keyElemBytes = static_cast<int>(key.element_size());
    LaunchFn launch = (keyElemBytes == 1)
        ? scatter_pa_kv_cache_i8_h_do
        : scatter_pa_kv_cache_h_h_do;

    // Fill tiling
    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(ScatterPaKvCacheTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<ScatterPaKvCacheTiling *>(tilingCpu.data_ptr());
    tiling->nTokens = nTokens;
    tiling->numKvSlices = numKvSlices;
    tiling->numVvSlices = numVvSlices;
    tiling->numBlocks = numBlocks;
    tiling->blockSize = blockSize;
    tiling->keyFlatDimI32 = numKvSlices * SLICE_LEN_I32;
    tiling->valFlatDimI32 = numVvSlices * SLICE_LEN_I32;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tokensPerCore = tokensPerCore;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    launch(
        static_cast<uint32_t>(usedCoreNum),
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(key.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(keyCache.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(slotMapping.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(value.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(valueCache.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
}

} // namespace current_task_ext

PYBIND11_MODULE(_current_task_ext, m)
{
    m.doc() = "ScatterPaKvCache AscendC extension";
    m.def("run_scatter", &current_task_ext::run_scatter, "Scatter key/value into PA KV cache");
}
