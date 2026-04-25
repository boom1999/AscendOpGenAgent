#pragma once
#include "kernel_operator.h"
#include "attention_update_tiling.h"
#include "kernel_common.h"

using namespace AscendC;

class AttentionUpdateKernel {
public:
    __aicore__ inline AttentionUpdateKernel() {}

    __aicore__ inline void Init(
        __gm__ uint8_t *tilingGm,
        __gm__ float *lse0, __gm__ float *lse1, __gm__ float *lse2,
        __gm__ float *out0, __gm__ float *out1, __gm__ float *out2,
        __gm__ float *resultOut, __gm__ float *lseOut
    ) {
        CopyTiling(tiling_, tilingGm);
        lse0Gm_.SetGlobalBuffer(lse0);
        lse1Gm_.SetGlobalBuffer(lse1);
        lse2Gm_.SetGlobalBuffer(lse2);
        out0Gm_.SetGlobalBuffer(out0);
        out1Gm_.SetGlobalBuffer(out1);
        out2Gm_.SetGlobalBuffer(out2);
        resultOutGm_.SetGlobalBuffer(resultOut);
        lseOutGm_.SetGlobalBuffer(lseOut);
    }

    __aicore__ inline void InitBuffers() {
        uint32_t hAlign = tiling_.hAlign;
        uint32_t rowBufBytes = hAlign * static_cast<uint32_t>(sizeof(float));
        uint32_t scalarBufBytes = 8 * static_cast<uint32_t>(sizeof(float));
        pipe_.InitBuffer(scalarQue_, 1, scalarBufBytes);
        pipe_.InitBuffer(accQue_, 1, rowBufBytes);
        pipe_.InitBuffer(rowQue_, 1, rowBufBytes);
    }

    __aicore__ inline void Process() {
        const uint32_t coreIdx = GetBlockIdx();
        const uint32_t K = tiling_.K;
        const uint32_t N = tiling_.N;
        const uint32_t H = tiling_.H;
        const uint32_t tasksPerCore = tiling_.tasksPerCore;
        const uint32_t hAlign = tiling_.hAlign;

        LocalTensor<float> scalarUb = scalarQue_.AllocTensor<float>();
        LocalTensor<float> accUb = accQue_.AllocTensor<float>();
        LocalTensor<float> rowUb = rowQue_.AllocTensor<float>();

        const uint32_t copyBytesH = H * static_cast<uint32_t>(sizeof(float));
        const uint32_t tailPad = hAlign - H;
        const DataCopyExtParams rowLoadP{1, copyBytesH, 0, 0, 0};
        const DataCopyPadExtParams<float> rowPadP{true, 0, static_cast<uint8_t>(tailPad), 0.0f};
        const DataCopyExtParams rowStoreP{1, copyBytesH, 0, 0, 0};
        const DataCopyExtParams scaLoadP{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> scaPadP{false, 0, 0, 0.0f};
        const DataCopyExtParams scaStoreP{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};

        for (uint32_t li = 0; li < tasksPerCore; ++li) {
            uint32_t row = coreIdx * tasksPerCore + li;
            if (row >= N) break;

            // ======== Phase A: Compute LSE weights ========

            // Load lse_0[row]
            DataCopyPad(scalarUb, lse0Gm_[row], scaLoadP, scaPadP);
            pipe_barrier(PIPE_ALL);
            float l0 = scalarUb.GetValue(0);
            float lseMax = l0;
            float l1 = 0.0f, l2 = 0.0f;

            if (K >= 2) {
                DataCopyPad(scalarUb, lse1Gm_[row], scaLoadP, scaPadP);
                pipe_barrier(PIPE_ALL);
                l1 = scalarUb.GetValue(0);
                if (l1 > lseMax) lseMax = l1;
            }
            if (K >= 3) {
                DataCopyPad(scalarUb, lse2Gm_[row], scaLoadP, scaPadP);
                pipe_barrier(PIPE_ALL);
                l2 = scalarUb.GetValue(0);
                if (l2 > lseMax) lseMax = l2;
            }

            // Compute exp weights using vector Exp on 8-element buffers
            Duplicate(scalarUb, l0 - lseMax, 8);
            Exp(scalarUb, scalarUb, 8);
            pipe_barrier(PIPE_ALL);
            float w0 = scalarUb.GetValue(0);
            float wsum = w0;
            float w1 = 0.0f, w2 = 0.0f;

            if (K >= 2) {
                Duplicate(scalarUb, l1 - lseMax, 8);
                Exp(scalarUb, scalarUb, 8);
                pipe_barrier(PIPE_ALL);
                w1 = scalarUb.GetValue(0);
                wsum += w1;
            }
            if (K >= 3) {
                Duplicate(scalarUb, l2 - lseMax, 8);
                Exp(scalarUb, scalarUb, 8);
                pipe_barrier(PIPE_ALL);
                w2 = scalarUb.GetValue(0);
                wsum += w2;
            }

            // lse_out = lseMax + ln(wsum)
            Duplicate(scalarUb, wsum, 8);
            Ln(scalarUb, scalarUb, 8);
            pipe_barrier(PIPE_ALL);
            float lseOutVal = lseMax + scalarUb.GetValue(0);

            // Final weights: wk / wsum  (= exp(lse_k - lse_out))
            float invWsum = 1.0f / wsum;
            float fw0 = w0 * invWsum;
            float fw1 = (K >= 2) ? w1 * invWsum : 0.0f;
            float fw2 = (K >= 3) ? w2 * invWsum : 0.0f;

            // ======== Phase B: Weighted output sum (vectorized over H) ========
            uint32_t rowOff = row * H;

            // acc = out_0[row, :] * fw0
            DataCopyPad(accUb, out0Gm_[rowOff], rowLoadP, rowPadP);
            pipe_barrier(PIPE_ALL);
            Muls(accUb, accUb, fw0, hAlign);

            if (K >= 2) {
                pipe_barrier(PIPE_ALL);
                DataCopyPad(rowUb, out1Gm_[rowOff], rowLoadP, rowPadP);
                pipe_barrier(PIPE_ALL);
                Muls(rowUb, rowUb, fw1, hAlign);
                Add(accUb, accUb, rowUb, hAlign);
            }

            if (K >= 3) {
                pipe_barrier(PIPE_ALL);
                DataCopyPad(rowUb, out2Gm_[rowOff], rowLoadP, rowPadP);
                pipe_barrier(PIPE_ALL);
                Muls(rowUb, rowUb, fw2, hAlign);
                Add(accUb, accUb, rowUb, hAlign);
            }

            // Store result_out[row, :]
            pipe_barrier(PIPE_ALL);
            DataCopyPad(resultOutGm_[rowOff], accUb, rowStoreP);

            // Store lse_out[row]
            Duplicate(scalarUb, lseOutVal, 8);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(lseOutGm_[row], scalarUb, scaStoreP);

            pipe_barrier(PIPE_ALL);
        }

        scalarQue_.FreeTensor(scalarUb);
        accQue_.FreeTensor(accUb);
        rowQue_.FreeTensor(rowUb);
    }

private:
    AttentionUpdateTiling tiling_;
    GlobalTensor<float> lse0Gm_, lse1Gm_, lse2Gm_;
    GlobalTensor<float> out0Gm_, out1Gm_, out2Gm_;
    GlobalTensor<float> resultOutGm_, lseOutGm_;
    TQue<QuePosition::VECIN, 1> scalarQue_;
    TQue<QuePosition::VECIN, 1> accQue_;
    TQue<QuePosition::VECIN, 1> rowQue_;

public:
    TPipe pipe_;
};
