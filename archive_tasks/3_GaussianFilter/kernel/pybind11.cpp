#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "gaussian_filter_tiling.h"

namespace py = pybind11;

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
);

static uint8_t *TensorAddr(const at::Tensor &t) {
    return static_cast<uint8_t *>(const_cast<void *>(t.storage().data()));
}

std::vector<at::Tensor> run_gaussian_filter(
    const at::Tensor &means,
    const at::Tensor &colors,
    const at::Tensor &det,
    const at::Tensor &opacities,
    const at::Tensor &means2d,
    const at::Tensor &depths,
    const at::Tensor &radius,
    const at::Tensor &conics,
    const at::Tensor &covars2d,
    const at::Tensor &bMap,
    int64_t BC, int64_t B, int64_t N_padded, int64_t M_padded,
    double near_plane, double far_plane, double width, double height
) {
    auto fOpts = at::device(at::kPrivateUse1).dtype(at::kFloat);
    at::Tensor meansOut = at::empty({BC, 3, N_padded}, fOpts);
    at::Tensor colorsOut = at::empty({BC, 3, N_padded}, fOpts);
    at::Tensor means2dOut = at::empty({BC, 2, N_padded}, fOpts);
    at::Tensor depthsOut = at::empty({BC, N_padded}, fOpts);
    at::Tensor radiusOut = at::empty({BC, 2, N_padded}, fOpts);
    at::Tensor covars2dOut = at::empty({BC, 3, N_padded}, fOpts);
    at::Tensor conicsOut = at::empty({BC, 3, N_padded}, fOpts);
    at::Tensor opacitiesOut = at::empty({BC, N_padded}, fOpts);
    at::Tensor filterUint8 = at::empty({BC, M_padded},
        at::device(at::kPrivateUse1).dtype(at::kByte));
    at::Tensor cntOut = at::empty({BC},
        at::device(at::kPrivateUse1).dtype(at::kInt));

    const int32_t TILE_N = 256;
    int32_t nTiles = static_cast<int32_t>((N_padded + TILE_N - 1) / TILE_N);
    int32_t bytesPerTile = TILE_N / 8;

    int32_t usedCoreNum = static_cast<int32_t>(BC);
    if (usedCoreNum > 20) usedCoreNum = 20;
    if (usedCoreNum < 1) usedCoreNum = 1;
    int32_t tasksPerCore = (static_cast<int32_t>(BC) + usedCoreNum - 1) / usedCoreNum;

    GaussianFilterTiling tiling;
    tiling.BC = static_cast<int32_t>(BC);
    tiling.B = static_cast<int32_t>(B);
    tiling.N_padded = static_cast<int32_t>(N_padded);
    tiling.M_padded = static_cast<int32_t>(M_padded);
    tiling.TILE_N = TILE_N;
    tiling.n_tiles = nTiles;
    tiling.bytes_per_tile = bytesPerTile;
    tiling.near_plane = static_cast<float>(near_plane);
    tiling.far_plane = static_cast<float>(far_plane);
    tiling.width = static_cast<float>(width);
    tiling.height = static_cast<float>(height);
    tiling.usedCoreNum = usedCoreNum;
    tiling.tasksPerCore = tasksPerCore;

    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(GaussianFilterTiling))},
        at::TensorOptions().dtype(at::kByte).device(means.device())
    );
    auto tilingHost = at::from_blob(
        &tiling,
        {static_cast<int64_t>(sizeof(GaussianFilterTiling))},
        at::TensorOptions().dtype(at::kByte)
    );
    tilingTensor.copy_(tilingHost);

    auto stream = c10_npu::getCurrentNPUStream();
    gaussian_filter_do(
        usedCoreNum, stream,
        TensorAddr(means), TensorAddr(colors), TensorAddr(det),
        TensorAddr(opacities), TensorAddr(means2d), TensorAddr(depths),
        TensorAddr(radius), TensorAddr(conics), TensorAddr(covars2d),
        TensorAddr(bMap),
        TensorAddr(meansOut), TensorAddr(colorsOut), TensorAddr(means2dOut),
        TensorAddr(depthsOut), TensorAddr(radiusOut), TensorAddr(covars2dOut),
        TensorAddr(conicsOut), TensorAddr(opacitiesOut),
        TensorAddr(filterUint8), TensorAddr(cntOut),
        TensorAddr(tilingTensor)
    );

    return {meansOut, colorsOut, means2dOut, depthsOut,
            radiusOut, covars2dOut, conicsOut, opacitiesOut,
            filterUint8, cntOut};
}

PYBIND11_MODULE(_gaussian_filter_ext, m) {
    m.def("run_gaussian_filter", &run_gaussian_filter,
          "GaussianFilter kernel (AscendC)");
}
