#pragma once
#include "kernel_operator.h"
#include "scatter_block_update_tiling.h"
#include "kernel_common.h"

using namespace AscendC;

class ScatterBlockUpdateKernel {
public:
    __aicore__ inline ScatterBlockUpdateKernel() {}

    __aicore__ inline void Init(
        __gm__ uint8_t *tilingGm,
        __gm__ uint8_t *outputData,
        __gm__ int32_t *indices,
        __gm__ uint8_t *updateData
    ) {
        CopyTiling(tiling_, tilingGm);

        indicesGm_.SetGlobalBuffer(indices);
        outputGm8_.SetGlobalBuffer(outputData);
        updateGm8_.SetGlobalBuffer(updateData);
        outputGm16_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(outputData));
        updateGm16_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(updateData));
        outputGm32_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(outputData));
        updateGm32_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(updateData));
    }

    __aicore__ inline void Process() {
        const int32_t coreIdx = GetBlockIdx();
        const int32_t K = tiling_.K;
        const int32_t kPerCore = tiling_.kPerCore;
        const int32_t D1 = tiling_.D1;
        const int32_t D2 = tiling_.D2;
        const int32_t elemSize = tiling_.elemSize;

        const int32_t kStart = coreIdx * kPerCore;
        int32_t kEnd = kStart + kPerCore;
        if (kEnd > K) kEnd = K;
        if (kStart >= K) return;

        if (elemSize == 4) {
            CopyRows32(kStart, kEnd, D1, D2);
        } else if (elemSize == 2) {
            CopyRows16(kStart, kEnd, D1, D2);
        } else {
            CopyRows8(kStart, kEnd, D1, D2);
        }
    }

private:
    __aicore__ inline void CopyRows32(int32_t kStart, int32_t kEnd, int32_t D1, int32_t D2) {
        for (int32_t k = kStart; k < kEnd; ++k) {
            int32_t idx0 = indicesGm_.GetValue(k * 2);
            int32_t idx1 = indicesGm_.GetValue(k * 2 + 1);
            int32_t targetRow = idx0 * D1 + idx1;
            int32_t srcBase = k * D2;
            int32_t dstBase = targetRow * D2;
            for (int32_t e = 0; e < D2; ++e) {
                int32_t val = updateGm32_.GetValue(srcBase + e);
                outputGm32_.SetValue(dstBase + e, val);
            }
        }
    }

    __aicore__ inline void CopyRows16(int32_t kStart, int32_t kEnd, int32_t D1, int32_t D2) {
        for (int32_t k = kStart; k < kEnd; ++k) {
            int32_t idx0 = indicesGm_.GetValue(k * 2);
            int32_t idx1 = indicesGm_.GetValue(k * 2 + 1);
            int32_t targetRow = idx0 * D1 + idx1;
            int32_t srcBase = k * D2;
            int32_t dstBase = targetRow * D2;
            for (int32_t e = 0; e < D2; ++e) {
                int16_t val = updateGm16_.GetValue(srcBase + e);
                outputGm16_.SetValue(dstBase + e, val);
            }
        }
    }

    __aicore__ inline void CopyRows8(int32_t kStart, int32_t kEnd, int32_t D1, int32_t D2) {
        for (int32_t k = kStart; k < kEnd; ++k) {
            int32_t idx0 = indicesGm_.GetValue(k * 2);
            int32_t idx1 = indicesGm_.GetValue(k * 2 + 1);
            int32_t targetRow = idx0 * D1 + idx1;
            int32_t srcBase = k * D2;
            int32_t dstBase = targetRow * D2;
            for (int32_t e = 0; e < D2; ++e) {
                uint8_t val = updateGm8_.GetValue(srcBase + e);
                outputGm8_.SetValue(dstBase + e, val);
            }
        }
    }

    ScatterBlockUpdateTiling tiling_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<uint8_t> outputGm8_;
    GlobalTensor<uint8_t> updateGm8_;
    GlobalTensor<int16_t> outputGm16_;
    GlobalTensor<int16_t> updateGm16_;
    GlobalTensor<int32_t> outputGm32_;
    GlobalTensor<int32_t> updateGm32_;
};
