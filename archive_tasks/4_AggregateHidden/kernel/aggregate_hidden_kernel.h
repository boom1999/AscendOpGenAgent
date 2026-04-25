#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "aggregate_hidden_tiling.h"
#include "kernel_common.h"
#include "vector_tile.h"

using namespace AscendC;

/*
 * AggregateHidden kernel: fused forward conv + backward grad_input/grad_weight.
 *
 * All inputs/outputs are float32 (host casts bf16/fp16 -> f32 before calling).
 * Data layout: (S*B, H) for main tensors, (B, S) for mask, (3, H) for weight.
 *
 * Partition H across AI cores. Each core iterates B then S.
 * Single-pass pipeline: forward + grad_weight accumulation + 2-step delayed grad_input.
 */
class AggregateHiddenKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR gradOutGm, GM_ADDR inputGm, GM_ADDR weightGm, GM_ADDR maskGm,
        GM_ADDR outputGm, GM_ADDR gradInputGm, GM_ADDR gradWeightGm,
        GM_ADDR tilingGm, TPipe *pipe)
    {
        CopyTiling(tiling_, tilingGm);

        int32_t SBH = tiling_.S * tiling_.B * tiling_.H;
        gradOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradOutGm), SBH);
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputGm), SBH);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputGm), SBH);
        gradInputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradInputGm), SBH);

        int32_t WH = 3 * tiling_.H;
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightGm), WH);
        gradWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradWeightGm), WH);

        int32_t BS = tiling_.B * tiling_.S;
        maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(maskGm), BS);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
        }
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            int32_t blockH = tiling_.blockH;
            int32_t alignedBlockH = ((blockH + 7) / 8) * 8;
            int32_t bufSize = alignedBlockH * static_cast<int32_t>(sizeof(float));

            // Weight buffers (loaded once per H-block, reused)
            pipe_->InitBuffer(w0Buf_, bufSize);
            pipe_->InitBuffer(w1Buf_, bufSize);
            pipe_->InitBuffer(w2Buf_, bufSize);

            // Input sliding window
            pipe_->InitBuffer(inpCurBuf_, bufSize);
            pipe_->InitBuffer(inpPrev1Buf_, bufSize);
            pipe_->InitBuffer(inpPrev2Buf_, bufSize);

            // Grad_eff sliding window
            pipe_->InitBuffer(geCurBuf_, bufSize);
            pipe_->InitBuffer(gePrev1Buf_, bufSize);
            pipe_->InitBuffer(gePrev2Buf_, bufSize);

            // GM load buffers
            pipe_->InitBuffer(goQue_, 1, bufSize);
            pipe_->InitBuffer(inpLoadQue_, 1, bufSize);

            // GM store buffers
            pipe_->InitBuffer(outQue_, 1, bufSize);
            pipe_->InitBuffer(giQue_, 1, bufSize);

            // Grad_weight accumulators
            pipe_->InitBuffer(gw0Buf_, bufSize);
            pipe_->InitBuffer(gw1Buf_, bufSize);
            pipe_->InitBuffer(gw2Buf_, bufSize);

            // Working buffers
            pipe_->InitBuffer(tmpBuf_, bufSize);
            pipe_->InitBuffer(maskVecBuf_, bufSize);
            pipe_->InitBuffer(maskLoadBuf_, 32);  // 8 floats minimum for alignment
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
            if (GetSubBlockIdx() != 0) return;

            const int32_t S = tiling_.S;
            const int32_t B = tiling_.B;
            const int32_t H = tiling_.H;
            const int32_t blockH = tiling_.blockH;
            const int32_t tasksPerCore = tiling_.tasksPerCore;
            const int32_t hNum = tiling_.hNum;

            // Alloc TQue tensors once
            LocalTensor<float> goUb = goQue_.AllocTensor<float>();
            LocalTensor<float> inpLoadUb = inpLoadQue_.AllocTensor<float>();
            LocalTensor<float> outUb = outQue_.AllocTensor<float>();
            LocalTensor<float> giUb = giQue_.AllocTensor<float>();

            // Get TBuf tensors
            auto w0 = w0Buf_.Get<float>();
            auto w1 = w1Buf_.Get<float>();
            auto w2 = w2Buf_.Get<float>();
            auto inpCur = inpCurBuf_.Get<float>();
            auto inpPrev1 = inpPrev1Buf_.Get<float>();
            auto inpPrev2 = inpPrev2Buf_.Get<float>();
            auto geCur = geCurBuf_.Get<float>();
            auto gePrev1 = gePrev1Buf_.Get<float>();
            auto gePrev2 = gePrev2Buf_.Get<float>();
            auto gw0 = gw0Buf_.Get<float>();
            auto gw1 = gw1Buf_.Get<float>();
            auto gw2 = gw2Buf_.Get<float>();
            auto tmp = tmpBuf_.Get<float>();
            auto maskVec = maskVecBuf_.Get<float>();
            auto maskLoad = maskLoadBuf_.Get<float>();

            for (int32_t localIdx = 0; localIdx < tasksPerCore; ++localIdx) {
                int32_t bx = coreIdx * tasksPerCore + localIdx;
                if (bx >= hNum) break;

                int32_t hStart = bx * blockH;
                int32_t curH = blockH;
                if (hStart + curH > H) curH = H - hStart;
                uint32_t copyBytes = static_cast<uint32_t>(curH) * static_cast<uint32_t>(sizeof(float));

                // Load weight rows from GM
                DataCopyExtParams loadParams{1, copyBytes, 0, 0, 0};
                DataCopyPadExtParams<float> padParams{true, 0, 0, 0.0f};
                DataCopyPad(w0, weightGm_[0 * H + hStart], loadParams, padParams);
                DataCopyPad(w1, weightGm_[1 * H + hStart], loadParams, padParams);
                DataCopyPad(w2, weightGm_[2 * H + hStart], loadParams, padParams);
                pipe_barrier(PIPE_ALL);

                // Zero-init grad_weight accumulators
                Duplicate(gw0, 0.0f, curH);
                Duplicate(gw1, 0.0f, curH);
                Duplicate(gw2, 0.0f, curH);
                pipe_barrier(PIPE_ALL);

                for (int32_t bi = 0; bi < B; ++bi) {
                    // Zero-init sliding windows
                    Duplicate(inpPrev1, 0.0f, curH);
                    Duplicate(inpPrev2, 0.0f, curH);
                    Duplicate(gePrev1, 0.0f, curH);
                    Duplicate(gePrev2, 0.0f, curH);
                    pipe_barrier(PIPE_ALL);

                    for (int32_t si = 0; si < S; ++si) {
                        int32_t flatIdx = si * B + bi;
                        int32_t gmOff = flatIdx * H + hStart;

                        // Load input and grad_out from GM
                        DataCopyPad(inpLoadUb, inputGm_[gmOff], loadParams, padParams);
                        DataCopyPad(goUb, gradOutGm_[gmOff], loadParams, padParams);

                        // Load mask scalar from GM (aligned to 32 bytes = 8 floats)
                        int32_t maskIdx = bi * S + si;
                        int32_t alignedIdx = (maskIdx / 8) * 8;
                        int32_t localIdx = maskIdx - alignedIdx;
                        int32_t totalMask = B * S;
                        int32_t avail = totalMask - alignedIdx;
                        int32_t loadN = avail < 8 ? avail : 8;
                        DataCopyExtParams maskCopyParams{1, static_cast<uint32_t>(loadN) * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                        DataCopyPadExtParams<float> maskPadParams{true, 0, 0, 0.0f};
                        DataCopyPad(maskLoad, maskGm_[alignedIdx], maskCopyParams, maskPadParams);
                        pipe_barrier(PIPE_ALL);

                        // Copy loaded input to sliding window buffer
                        Adds(inpCur, inpLoadUb, 0.0f, curH);
                        pipe_barrier(PIPE_ALL);

                        // Broadcast mask value to vector
                        float maskVal = maskLoad.GetValue(localIdx);
                        Duplicate(maskVec, maskVal, curH);
                        pipe_barrier(PIPE_ALL);

                        // ---- Forward convolution ----
                        // out = w2*inp_cur + w1*inp_prev1 + w0*inp_prev2
                        Mul(outUb, w2, inpCur, curH);
                        Mul(tmp, w1, inpPrev1, curH);
                        Add(outUb, outUb, tmp, curH);
                        Mul(tmp, w0, inpPrev2, curH);
                        Add(outUb, outUb, tmp, curH);
                        // Apply mask
                        Mul(outUb, outUb, maskVec, curH);
                        pipe_barrier(PIPE_ALL);

                        // Store output
                        DataCopyExtParams storeParams{1, copyBytes, 0, 0, 0};
                        DataCopyPad(outputGm_[gmOff], outUb, storeParams);

                        // ---- Grad_eff = grad_out * mask ----
                        Mul(geCur, goUb, maskVec, curH);
                        pipe_barrier(PIPE_ALL);

                        // ---- Accumulate grad_weight ----
                        // gw2 += ge * inp_cur
                        Mul(tmp, geCur, inpCur, curH);
                        Add(gw2, gw2, tmp, curH);
                        // gw1 += ge * inp_prev1
                        Mul(tmp, geCur, inpPrev1, curH);
                        Add(gw1, gw1, tmp, curH);
                        // gw0 += ge * inp_prev2
                        Mul(tmp, geCur, inpPrev2, curH);
                        Add(gw0, gw0, tmp, curH);
                        pipe_barrier(PIPE_ALL);

                        // ---- Grad_input (delayed by 2) ----
                        if (si >= 2) {
                            // gi[s-2] = w0*ge[s] + w1*ge[s-1] + w2*ge[s-2]
                            Mul(giUb, w0, geCur, curH);
                            Mul(tmp, w1, gePrev1, curH);
                            Add(giUb, giUb, tmp, curH);
                            Mul(tmp, w2, gePrev2, curH);
                            Add(giUb, giUb, tmp, curH);
                            pipe_barrier(PIPE_ALL);

                            int32_t giOff = ((si - 2) * B + bi) * H + hStart;
                            DataCopyExtParams giStoreParams{1, copyBytes, 0, 0, 0};
                            DataCopyPad(gradInputGm_[giOff], giUb, giStoreParams);
                        }

                        // ---- Shift sliding windows ----
                        // inp_prev2 = inp_prev1; inp_prev1 = inp_cur
                        Adds(inpPrev2, inpPrev1, 0.0f, curH);
                        Adds(inpPrev1, inpCur, 0.0f, curH);
                        // ge_prev2 = ge_prev1; ge_prev1 = ge_cur
                        Adds(gePrev2, gePrev1, 0.0f, curH);
                        Adds(gePrev1, geCur, 0.0f, curH);
                        pipe_barrier(PIPE_ALL);
                    }

                    // ---- Tail: last 2 grad_input positions ----
                    if (S >= 2) {
                        // grad_input[S-2] = w1*ge[S-1] + w2*ge[S-2]
                        Mul(giUb, w1, gePrev1, curH);
                        Mul(tmp, w2, gePrev2, curH);
                        Add(giUb, giUb, tmp, curH);
                        pipe_barrier(PIPE_ALL);

                        int32_t giOff1 = ((S - 2) * B + bi) * H + hStart;
                        DataCopyExtParams giStoreParams1{1, copyBytes, 0, 0, 0};
                        DataCopyPad(gradInputGm_[giOff1], giUb, giStoreParams1);
                        // Barrier to ensure MTE3 finishes reading giUb before
                        // the next Vector write overwrites it (WAR hazard).
                        pipe_barrier(PIPE_ALL);
                    }

                    // grad_input[S-1] = w2*ge[S-1]
                    Mul(giUb, w2, gePrev1, curH);
                    pipe_barrier(PIPE_ALL);

                    int32_t giOff2 = ((S - 1) * B + bi) * H + hStart;
                    DataCopyExtParams giStoreParams2{1, copyBytes, 0, 0, 0};
                    DataCopyPad(gradInputGm_[giOff2], giUb, giStoreParams2);
                    pipe_barrier(PIPE_ALL);
                }

                // Store grad_weight
                DataCopyExtParams gwStoreParams{1, copyBytes, 0, 0, 0};
                DataCopyPad(gradWeightGm_[0 * H + hStart], gw0, gwStoreParams);
                DataCopyPad(gradWeightGm_[1 * H + hStart], gw1, gwStoreParams);
                DataCopyPad(gradWeightGm_[2 * H + hStart], gw2, gwStoreParams);
                pipe_barrier(PIPE_ALL);
            }

            goQue_.FreeTensor(goUb);
            inpLoadQue_.FreeTensor(inpLoadUb);
            outQue_.FreeTensor(outUb);
            giQue_.FreeTensor(giUb);
        }
    }

private:
    AggregateHiddenTiling tiling_{};
    TPipe *pipe_{nullptr};

    GlobalTensor<float> gradOutGm_;
    GlobalTensor<float> inputGm_;
    GlobalTensor<float> weightGm_;
    GlobalTensor<float> maskGm_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<float> gradInputGm_;
    GlobalTensor<float> gradWeightGm_;

    // GM load queues
    TQue<QuePosition::VECIN, 1> goQue_;
    TQue<QuePosition::VECIN, 1> inpLoadQue_;

    // GM store queues
    TQue<QuePosition::VECOUT, 1> outQue_;
    TQue<QuePosition::VECOUT, 1> giQue_;

    // Weight (loaded once per H-block)
    TBuf<QuePosition::VECCALC> w0Buf_;
    TBuf<QuePosition::VECCALC> w1Buf_;
    TBuf<QuePosition::VECCALC> w2Buf_;

    // Input sliding window
    TBuf<QuePosition::VECCALC> inpCurBuf_;
    TBuf<QuePosition::VECCALC> inpPrev1Buf_;
    TBuf<QuePosition::VECCALC> inpPrev2Buf_;

    // Grad_eff sliding window
    TBuf<QuePosition::VECCALC> geCurBuf_;
    TBuf<QuePosition::VECCALC> gePrev1Buf_;
    TBuf<QuePosition::VECCALC> gePrev2Buf_;

    // Grad_weight accumulators
    TBuf<QuePosition::VECCALC> gw0Buf_;
    TBuf<QuePosition::VECCALC> gw1Buf_;
    TBuf<QuePosition::VECCALC> gw2Buf_;

    // Temporary buffers
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> maskVecBuf_;
    TBuf<QuePosition::VECCALC> maskLoadBuf_;
};
