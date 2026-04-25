#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <cstring>
#include "attention_update_tiling.h"

extern "C" void attention_update_do(
    uint32_t blockDim, void *stream,
    uint8_t *lse0, uint8_t *lse1, uint8_t *lse2,
    uint8_t *out0, uint8_t *out1, uint8_t *out2,
    uint8_t *resultOut, uint8_t *lseOut,
    uint8_t *tilingGm
);

static uint8_t *TensorAddr(const at::Tensor &t) {
    return static_cast<uint8_t *>(const_cast<void *>(t.storage().data()));
}

std::vector<at::Tensor> run_attention_update(
    at::Tensor lse0, at::Tensor lse1, at::Tensor lse2,
    at::Tensor out0, at::Tensor out1, at::Tensor out2,
    int64_t K
) {
    uint32_t N = static_cast<uint32_t>(lse0.size(0));
    uint32_t H = static_cast<uint32_t>(out0.size(1));
    uint32_t numCores = 20;
    uint32_t usedCoreNum = (N < numCores) ? N : numCores;
    uint32_t tasksPerCore = (N + usedCoreNum - 1) / usedCoreNum;
    uint32_t hAlign = ((H + 7) / 8) * 8;

    auto resultOut = at::empty({static_cast<long>(N), static_cast<long>(H)},
                               out0.options());
    auto lseOut = at::empty({static_cast<long>(N)}, lse0.options());

    AttentionUpdateTiling tiling;
    tiling.K = static_cast<uint32_t>(K);
    tiling.N = N;
    tiling.H = H;
    tiling.usedCoreNum = usedCoreNum;
    tiling.tasksPerCore = tasksPerCore;
    tiling.hAlign = hAlign;

    at::Tensor tilingTensor = at::empty(
        {static_cast<long>(sizeof(tiling))},
        at::TensorOptions().dtype(at::kByte).device(lse0.device())
    );
    auto tilingHost = at::from_blob(
        &tiling, {static_cast<long>(sizeof(tiling))},
        at::TensorOptions().dtype(at::kByte)
    );
    tilingTensor.copy_(tilingHost);

    auto stream = c10_npu::getCurrentNPUStream();
    attention_update_do(
        usedCoreNum, stream,
        TensorAddr(lse0), TensorAddr(lse1), TensorAddr(lse2),
        TensorAddr(out0), TensorAddr(out1), TensorAddr(out2),
        TensorAddr(resultOut), TensorAddr(lseOut),
        TensorAddr(tilingTensor)
    );

    return {resultOut, lseOut};
}

PYBIND11_MODULE(_attention_update_ext, m) {
    m.def("run_attention_update", &run_attention_update,
          "AttentionUpdate AscendC kernel",
          py::arg("lse0"), py::arg("lse1"), py::arg("lse2"),
          py::arg("out0"), py::arg("out1"), py::arg("out2"),
          py::arg("K"));
}
