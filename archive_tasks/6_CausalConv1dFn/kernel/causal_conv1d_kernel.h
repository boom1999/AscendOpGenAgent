#pragma once
#include "kernel_operator.h"
#include "causal_conv1d_tiling.h"
#include "kernel_common.h"

using namespace AscendC;

class CausalConv1dKernel {
public:
    __aicore__ inline CausalConv1dKernel() {}

    __aicore__ inline void Init(
        __gm__ uint8_t *tilingGm,
        __gm__ float *x,
        __gm__ float *weight,
        __gm__ float *convStatesFlat,
        __gm__ int32_t *queryStartLoc,
        __gm__ int32_t *cacheIndices,
        __gm__ int32_t *initialStateMode,
        __gm__ float *output,
        __gm__ float *cacheUpdates
    ) {
        CopyTiling(tiling_, tilingGm);
        xGm_.SetGlobalBuffer(x);
        weightGm_.SetGlobalBuffer(weight);
        convStatesGm_.SetGlobalBuffer(convStatesFlat);
        queryStartLocGm_.SetGlobalBuffer(queryStartLoc);
        cacheIndicesGm_.SetGlobalBuffer(cacheIndices);
        initialStateModeGm_.SetGlobalBuffer(initialStateMode);
        outputGm_.SetGlobalBuffer(output);
        cacheUpdatesGm_.SetGlobalBuffer(cacheUpdates);
    }

    __aicore__ inline void InitBuffers() {
        const int32_t blockN = tiling_.blockN;
        const int32_t alignedN = ((blockN + 7) / 8) * 8;
        const int32_t bufSize = alignedN * static_cast<int32_t>(sizeof(float));
        pipe_.InitBuffer(w0Buf_, bufSize);
        pipe_.InitBuffer(w1Buf_, bufSize);
        pipe_.InitBuffer(w2Buf_, bufSize);
        pipe_.InitBuffer(prev2Buf_, bufSize);
        pipe_.InitBuffer(prev1Buf_, bufSize);
        pipe_.InitBuffer(currBuf_, bufSize);
        pipe_.InitBuffer(outBuf_, bufSize);
        pipe_.InitBuffer(tmpBuf_, bufSize);
    }

    __aicore__ inline void Process() {
        const int32_t coreIdx = GetBlockIdx();
        const int32_t dim = tiling_.dim;
        const int32_t batchCount = tiling_.batchCount;
        const int32_t tasksPerCore = tiling_.tasksPerCore;
        const int32_t blockN = tiling_.blockN;
        const int32_t nTiles = tiling_.nTiles;
        const int32_t residual = tiling_.residual;
        const int32_t padSlotId = tiling_.padSlotId;
        const int32_t alignedN = ((blockN + 7) / 8) * 8;
        const int32_t statelen = 2;

        LocalTensor<float> w0Ub = w0Buf_.Get<float>();
        LocalTensor<float> w1Ub = w1Buf_.Get<float>();
        LocalTensor<float> w2Ub = w2Buf_.Get<float>();
        LocalTensor<float> prev2Ub = prev2Buf_.Get<float>();
        LocalTensor<float> prev1Ub = prev1Buf_.Get<float>();
        LocalTensor<float> currUb = currBuf_.Get<float>();
        LocalTensor<float> outUb = outBuf_.Get<float>();
        LocalTensor<float> tmpUb = tmpBuf_.Get<float>();

        for (int32_t localIdx = 0; localIdx < tasksPerCore; ++localIdx) {
            int32_t batchIdx = coreIdx * tasksPerCore + localIdx;
            if (batchIdx >= batchCount) break;

            // Read per-batch scalar metadata from GM
            int32_t start = queryStartLocGm_.GetValue(batchIdx);
            int32_t end = queryStartLocGm_.GetValue(batchIdx + 1);
            int32_t cacheIdx = cacheIndicesGm_.GetValue(batchIdx);
            int32_t mode = initialStateModeGm_.GetValue(batchIdx);
            int32_t seqLen = end - start;

            for (int32_t dt = 0; dt < nTiles; ++dt) {
                int32_t dStart = dt * blockN;
                int32_t curN = blockN;
                if (dStart + blockN > dim) curN = dim - dStart;
                int32_t copyLen = curN * static_cast<int32_t>(sizeof(float));

                // Load weight rows (zero-fill first for tail tile padding)
                Duplicate(w0Ub, 0.0f, alignedN);
                Duplicate(w1Ub, 0.0f, alignedN);
                Duplicate(w2Ub, 0.0f, alignedN);
                PipeBarrier<PIPE_ALL>();

                DataCopyExtParams wCopyParams{1, static_cast<uint32_t>(copyLen), 0, 0, 0};
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
                DataCopyPad(w0Ub, weightGm_[0 * dim + dStart], wCopyParams, padParams);
                DataCopyPad(w1Ub, weightGm_[1 * dim + dStart], wCopyParams, padParams);
                DataCopyPad(w2Ub, weightGm_[2 * dim + dStart], wCopyParams, padParams);
                PipeBarrier<PIPE_ALL>();

                // Load or zero-init cache for this dim tile
                int32_t cacheBase = cacheIdx * statelen;
                Duplicate(prev2Ub, 0.0f, alignedN);
                Duplicate(prev1Ub, 0.0f, alignedN);
                PipeBarrier<PIPE_ALL>();

                if (mode == 1) {
                    DataCopyPad(prev2Ub, convStatesGm_[cacheBase * dim + dStart], wCopyParams, padParams);
                    DataCopyPad(prev1Ub, convStatesGm_[(cacheBase + 1) * dim + dStart], wCopyParams, padParams);
                    PipeBarrier<PIPE_ALL>();
                }

                // Process sequence positions with sliding window
                for (int32_t i = 0; i < seqLen; ++i) {
                    int32_t pos = start + i;
                    int32_t gmOffset = pos * dim + dStart;

                    // Load x[pos] into curr
                    Duplicate(currUb, 0.0f, alignedN);
                    PipeBarrier<PIPE_ALL>();
                    DataCopyPad(currUb, xGm_[gmOffset], wCopyParams, padParams);
                    PipeBarrier<PIPE_ALL>();

                    // Conv: out = w0*prev2 + w1*prev1 + w2*curr
                    Mul(outUb, w0Ub, prev2Ub, alignedN);
                    PipeBarrier<PIPE_ALL>();
                    Mul(tmpUb, w1Ub, prev1Ub, alignedN);
                    PipeBarrier<PIPE_ALL>();
                    Add(outUb, outUb, tmpUb, alignedN);
                    PipeBarrier<PIPE_ALL>();
                    Mul(tmpUb, w2Ub, currUb, alignedN);
                    PipeBarrier<PIPE_ALL>();
                    Add(outUb, outUb, tmpUb, alignedN);
                    PipeBarrier<PIPE_ALL>();

                    // Mode 2: zero first 2 output rows of batch
                    if (mode == 2 && i < 2) {
                        Duplicate(outUb, 0.0f, alignedN);
                        PipeBarrier<PIPE_ALL>();
                    }

                    // Residual connection: out += x[pos]
                    if (residual == 1) {
                        Add(outUb, outUb, currUb, alignedN);
                        PipeBarrier<PIPE_ALL>();
                    }

                    // Write output
                    DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(copyLen), 0, 0, 0};
                    DataCopyPad(outputGm_[gmOffset], outUb, outCopyParams);
                    PipeBarrier<PIPE_ALL>();

                    // Rotate sliding window: prev2 <- prev1, prev1 <- curr
                    DataCopy(prev2Ub, prev1Ub, alignedN);
                    PipeBarrier<PIPE_ALL>();
                    DataCopy(prev1Ub, currUb, alignedN);
                    PipeBarrier<PIPE_ALL>();
                }

                // Write cache updates for this dim tile
                int32_t cacheOutBase = batchIdx * statelen;
                DataCopyExtParams cacheCopyParams{1, static_cast<uint32_t>(copyLen), 0, 0, 0};
                DataCopyPad(cacheUpdatesGm_[cacheOutBase * dim + dStart], prev2Ub, cacheCopyParams);
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(cacheUpdatesGm_[(cacheOutBase + 1) * dim + dStart], prev1Ub, cacheCopyParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

private:
    CausalConv1dTiling tiling_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> weightGm_;
    GlobalTensor<float> convStatesGm_;
    GlobalTensor<int32_t> queryStartLocGm_;
    GlobalTensor<int32_t> cacheIndicesGm_;
    GlobalTensor<int32_t> initialStateModeGm_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<float> cacheUpdatesGm_;

    TBuf<QuePosition::VECCALC> w0Buf_;
    TBuf<QuePosition::VECCALC> w1Buf_;
    TBuf<QuePosition::VECCALC> w2Buf_;
    TBuf<QuePosition::VECCALC> prev2Buf_;
    TBuf<QuePosition::VECCALC> prev1Buf_;
    TBuf<QuePosition::VECCALC> currBuf_;
    TBuf<QuePosition::VECCALC> outBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

public:
    TPipe pipe_;
};
