#include "gaussian_filter_kernel.h"

extern "C" __global__ __aicore__ void gaussian_filter_custom(
    GM_ADDR means, GM_ADDR colors, GM_ADDR det, GM_ADDR opacities,
    GM_ADDR means2d, GM_ADDR depths, GM_ADDR radius, GM_ADDR conics,
    GM_ADDR covars2d, GM_ADDR bMap,
    GM_ADDR meansOut, GM_ADDR colorsOut, GM_ADDR means2dOut,
    GM_ADDR depthsOut, GM_ADDR radiusOut, GM_ADDR covars2dOut,
    GM_ADDR conicsOut, GM_ADDR opacitiesOut,
    GM_ADDR filterUint8, GM_ADDR cntOut,
    GM_ADDR tilingGm
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV);
    GaussianFilterKernel kernel;
    kernel.Init(
        tilingGm,
        reinterpret_cast<__gm__ float *>(means),
        reinterpret_cast<__gm__ float *>(colors),
        reinterpret_cast<__gm__ float *>(det),
        reinterpret_cast<__gm__ float *>(opacities),
        reinterpret_cast<__gm__ float *>(means2d),
        reinterpret_cast<__gm__ float *>(depths),
        reinterpret_cast<__gm__ float *>(radius),
        reinterpret_cast<__gm__ float *>(conics),
        reinterpret_cast<__gm__ float *>(covars2d),
        reinterpret_cast<__gm__ int32_t *>(bMap),
        reinterpret_cast<__gm__ float *>(meansOut),
        reinterpret_cast<__gm__ float *>(colorsOut),
        reinterpret_cast<__gm__ float *>(means2dOut),
        reinterpret_cast<__gm__ float *>(depthsOut),
        reinterpret_cast<__gm__ float *>(radiusOut),
        reinterpret_cast<__gm__ float *>(covars2dOut),
        reinterpret_cast<__gm__ float *>(conicsOut),
        reinterpret_cast<__gm__ float *>(opacitiesOut),
        reinterpret_cast<__gm__ uint8_t *>(filterUint8),
        reinterpret_cast<__gm__ int32_t *>(cntOut)
    );
    kernel.InitBuffers();
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void gaussian_filter_do(
    uint32_t blockDim, void *stream,
    uint8_t *means, uint8_t *colors, uint8_t *det, uint8_t *opacities,
    uint8_t *means2d, uint8_t *depths, uint8_t *radius, uint8_t *conics,
    uint8_t *covars2d, uint8_t *bMap,
    uint8_t *meansOut, uint8_t *colorsOut, uint8_t *means2dOut,
    uint8_t *depthsOut, uint8_t *radiusOut, uint8_t *covars2dOut,
    uint8_t *conicsOut, uint8_t *opacitiesOut,
    uint8_t *filterUint8, uint8_t *cntOut,
    uint8_t *tilingGm
) {
    gaussian_filter_custom<<<blockDim, nullptr, stream>>>(
        means, colors, det, opacities,
        means2d, depths, radius, conics, covars2d, bMap,
        meansOut, colorsOut, means2dOut,
        depthsOut, radiusOut, covars2dOut,
        conicsOut, opacitiesOut,
        filterUint8, cntOut, tilingGm
    );
}
#endif
