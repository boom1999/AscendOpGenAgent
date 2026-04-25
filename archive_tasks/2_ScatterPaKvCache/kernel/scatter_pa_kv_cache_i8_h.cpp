#include "scatter_pa_kv_cache_kernel.h"

extern "C" __global__ __aicore__ void scatter_pa_kv_cache_i8_h_kernel(
    GM_ADDR key,
    GM_ADDR keyCache,
    GM_ADDR slotMapping,
    GM_ADDR value,
    GM_ADDR valueCache,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV);
    ScatterPaKvCacheKernel op;
    AscendC::TPipe pipe;
    op.Init(key, keyCache, slotMapping, value, valueCache, tiling, &pipe);
    op.Process();
}

extern "C" void scatter_pa_kv_cache_i8_h_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *key,
    uint8_t *keyCache,
    uint8_t *slotMapping,
    uint8_t *value,
    uint8_t *valueCache,
    uint8_t *tiling)
{
    scatter_pa_kv_cache_i8_h_kernel<<<blockDim, nullptr, stream>>>(
        key, keyCache, slotMapping, value, valueCache, tiling);
}
