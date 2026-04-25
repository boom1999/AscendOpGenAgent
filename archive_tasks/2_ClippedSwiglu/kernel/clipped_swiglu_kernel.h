#pragma once
#include "kernel_operator.h"
#include "clipped_swiglu_tiling.h"
#include "kernel_common.h"

using namespace AscendC;

class ClippedSwigluKernel {
public:
    __aicore__ inline ClippedSwigluKernel() {}

    __aicore__ inline void Init(
        __gm__ uint8_t *tilingGm,
        __gm__ float *A,
        __gm__ float *B,
        __gm__ float *Y
    ) {
        CopyTiling(tiling_, tilingGm);
        aGm_.SetGlobalBuffer(A);
        bGm_.SetGlobalBuffer(B);
        yGm_.SetGlobalBuffer(Y);
    }

    __aicore__ inline void Process() {
        const int32_t coreIdx = GetBlockIdx();
        const int32_t M = tiling_.M;
        const int32_t N = tiling_.N;
        const int32_t tasksPerCore = tiling_.tasksPerCore;
        const int32_t blockN = tiling_.blockN;
        const int32_t nLoops = tiling_.nLoops;
        const float alpha = tiling_.alpha;
        const float limit = tiling_.limit;
        const float biasVal = tiling_.biasVal;
        const float negLimit = -limit;
        const float negOne = -1.0f;
        const float one = 1.0f;

        // Align blockN to 8 floats (32 bytes)
        const int32_t alignedBlockN = ((blockN + 7) / 8) * 8;

        // Allocate UB buffers
        LocalTensor<float> aUb = aQue_.AllocTensor<float>();
        LocalTensor<float> bUb = bQue_.AllocTensor<float>();
        LocalTensor<float> tmpUb = tmpBuf_.Get<float>();
        LocalTensor<float> yUb = yQue_.AllocTensor<float>();

        for (int32_t localIdx = 0; localIdx < tasksPerCore; ++localIdx) {
            int32_t row = coreIdx * tasksPerCore + localIdx;
            if (row >= M) break;

            for (int32_t ni = 0; ni < nLoops; ++ni) {
                int32_t colStart = ni * blockN;
                int32_t curN = blockN;
                if (colStart + curN > N) curN = N - colStart;
                int32_t gmOffset = row * N + colStart;
                int32_t copyLen = curN * static_cast<int32_t>(sizeof(float));

                // CopyIn A
                DataCopyExtParams copyParamsIn{1, static_cast<uint32_t>(copyLen), 0, 0, 0};
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
                DataCopyPad(aUb, aGm_[gmOffset], copyParamsIn, padParams);
                // CopyIn B
                DataCopyPad(bUb, bGm_[gmOffset], copyParamsIn, padParams);

                pipe_barrier(PIPE_ALL);

                // Clamp A: min(A, limit)
                Mins(aUb, aUb, limit, curN);
                // Clamp B: min(B, limit) then max(B, -limit)
                Mins(bUb, bUb, limit, curN);
                Maxs(bUb, bUb, negLimit, curN);

                // sigmoid: tmp = alpha * A
                Muls(tmpUb, aUb, alpha, curN);
                // tmp = -tmp
                Muls(tmpUb, tmpUb, negOne, curN);
                // tmp = exp(-alpha * A)
                Exp(tmpUb, tmpUb, curN);
                // tmp = 1 + exp(...)
                Adds(tmpUb, tmpUb, one, curN);
                // tmp = 1 / (1 + exp(...)) = sigmoid
                Reciprocal(tmpUb, tmpUb, curN);

                // y = A_clamped * sigmoid
                Mul(yUb, aUb, tmpUb, curN);
                // B = B_clamped + bias
                Adds(bUb, bUb, biasVal, curN);
                // y = y * (B + bias)
                Mul(yUb, yUb, bUb, curN);

                pipe_barrier(PIPE_ALL);

                // CopyOut
                DataCopyExtParams copyParamsOut{1, static_cast<uint32_t>(copyLen), 0, 0, 0};
                DataCopyPad(yGm_[gmOffset], yUb, copyParamsOut);
            }
        }

        aQue_.FreeTensor(aUb);
        bQue_.FreeTensor(bUb);
        yQue_.FreeTensor(yUb);
    }

private:
    ClippedSwigluTiling tiling_;
    GlobalTensor<float> aGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> yGm_;

    // blockN max 1024, aligned to 8 -> 1024 floats = 4096 bytes
    TQue<QuePosition::VECIN, 1> aQue_;
    TQue<QuePosition::VECIN, 1> bQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

public:
    __aicore__ inline void InitBuffers() {
        const int32_t blockN = tiling_.blockN;
        const int32_t alignedBlockN = ((blockN + 7) / 8) * 8;
        const int32_t bufSize = alignedBlockN * static_cast<int32_t>(sizeof(float));
        pipe_.InitBuffer(aQue_, 1, bufSize);
        pipe_.InitBuffer(bQue_, 1, bufSize);
        pipe_.InitBuffer(yQue_, 1, bufSize);
        pipe_.InitBuffer(tmpBuf_, bufSize);
    }

    TPipe pipe_;
};
