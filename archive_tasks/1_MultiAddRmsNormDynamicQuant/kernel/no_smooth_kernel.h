#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"

#include "kernel_common.h"
#include "multi_add_rms_norm_dq_tiling.h"
#include "vector_tile.h"

template <typename dataType>
class NoSmoothKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma,
        GM_ADDR xSum, GM_ADDR yNorm,
        GM_ADDR y1, GM_ADDR scale1,
        GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        int32_t totalElem = tiling_.M * tiling_.N;
        x1GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x1), totalElem);
        x2GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x2), totalElem);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamma), totalElem);
        xSumGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(xSum), totalElem);
        yNormGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(yNorm), totalElem);
        y1GM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y1), totalElem);
        scale1GM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale1), tiling_.M);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
            int32_t N = tiling_.N;

            pipe_->InitBuffer(inQueue_, 1, N * sizeof(dataType));
            pipe_->InitBuffer(dtypeOutQueue_, 1, N * sizeof(dataType));
            pipe_->InitBuffer(i8OutQueue_, 1, N * sizeof(int8_t));
            pipe_->InitBuffer(scaleOutQueue_, 1, 64);

            pipe_->InitBuffer(xsumBuf_, N * sizeof(float));
            pipe_->InitBuffer(ynormBuf_, N * sizeof(float));
            pipe_->InitBuffer(tempBuf_, N * sizeof(float));
            pipe_->InitBuffer(f16CastBuf_, N * sizeof(half));
            pipe_->InitBuffer(reduceBuf_, N * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 64);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) continue;

                for (int row = 0; row < subBlockRows_; ++row) {
                    const int rowIdx = bx * tiling_.blockM + subBlockIdx * subBlockRows_ + row;
                    if (rowIdx < tiling_.M) {
                        ProcessRow(rowIdx);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void LoadToF32(
        AscendC::LocalTensor<float> &dst,
        AscendC::GlobalTensor<dataType> src,
        int32_t count)
    {
        AscendC::LocalTensor<dataType> inLocal;
        inQueue_.AllocTensor<dataType>(inLocal);
        LoadGmToUb(inLocal, src, static_cast<uint32_t>(count));
        inQueue_.EnQue(inLocal);
        inQueue_.DeQue<dataType>(inLocal);
        if constexpr (std::is_same_v<dataType, float>) {
            dst = inLocal.template ReinterpretCast<float>();
        } else {
            AscendC::Cast(dst, inLocal, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        inQueue_.FreeTensor(inLocal);
    }

    __aicore__ inline void StoreFromF32(
        AscendC::GlobalTensor<dataType> dst,
        AscendC::LocalTensor<float> &src,
        int32_t count)
    {
        AscendC::LocalTensor<dataType> outLocal;
        dtypeOutQueue_.AllocTensor<dataType>(outLocal);
        if constexpr (std::is_same_v<dataType, float>) {
            outLocal = src.template ReinterpretCast<dataType>();
        } else {
            AscendC::Cast(outLocal, src, OutputRoundMode(), count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        dtypeOutQueue_.EnQue(outLocal);
        dtypeOutQueue_.DeQue<dataType>(outLocal);
        StoreUbToGm(dst, outLocal, static_cast<uint32_t>(count));
        dtypeOutQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void ProcessRow(int rowIdx)
    {
        int32_t N = tiling_.N;
        int32_t offset = rowIdx * N;

        auto xsumLocal = xsumBuf_.Get<float>();
        auto ynormLocal = ynormBuf_.Get<float>();
        auto tempLocal = tempBuf_.Get<float>();
        auto reduceLocal = reduceBuf_.Get<float>();
        auto sumLocal = sumBuf_.Get<float>();
        auto f16Local = f16CastBuf_.Get<half>();

        // --- Add: xsum = x1 + x2 ---
        LoadToF32(tempLocal, x1GM_[offset], N);
        LoadToF32(xsumLocal, x2GM_[offset], N);
        AscendC::Add(xsumLocal, tempLocal, xsumLocal, N);
        AscendC::PipeBarrier<PIPE_V>();

        // --- Store xSum ---
        StoreFromF32(xSumGM_[offset], xsumLocal, N);

        // --- RmsNorm ---
        AscendC::Mul(tempLocal, xsumLocal, xsumLocal, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(sumLocal, tempLocal, reduceLocal, N);
        AscendC::PipeBarrier<PIPE_V>();

        float meanSq = sumLocal.GetValue(0) * tiling_.invN + tiling_.eps;
        AscendC::Duplicate(sumLocal, meanSq, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Rsqrt(sumLocal, sumLocal, 1);
        AscendC::PipeBarrier<PIPE_V>();
        float invRms = sumLocal.GetValue(0);

        // --- Load gamma -> tempLocal (reuse) ---
        LoadToF32(tempLocal, gammaGM_[offset], N);

        // --- yNorm = xsum * invRms * gamma ---
        AscendC::Muls(ynormLocal, xsumLocal, invRms, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(ynormLocal, ynormLocal, tempLocal, N);
        AscendC::PipeBarrier<PIPE_V>();

        // --- Store yNorm ---
        StoreFromF32(yNormGM_[offset], ynormLocal, N);

        // --- Dynamic Quant (no smooth) ---
        AscendC::Abs(tempLocal, ynormLocal, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax<float>(sumLocal, tempLocal, reduceLocal, N);
        AscendC::PipeBarrier<PIPE_V>();

        float maxAbs = sumLocal.GetValue(0);
        float scaleVal = maxAbs * tiling_.inv127;

        // Store scale
        AscendC::LocalTensor<float> scaleOutLocal;
        scaleOutQueue_.AllocTensor<float>(scaleOutLocal);
        AscendC::Duplicate(scaleOutLocal, scaleVal, 1);
        scaleOutQueue_.EnQue(scaleOutLocal);
        scaleOutQueue_.DeQue<float>(scaleOutLocal);
        StoreUbToGm(scale1GM_[rowIdx], scaleOutLocal, 1);
        scaleOutQueue_.FreeTensor(scaleOutLocal);

        // y_quant = round(yNorm / scale) as int8
        float invScale = (scaleVal > 0.0f) ? (1.0f / scaleVal) : 0.0f;
        AscendC::Muls(tempLocal, ynormLocal, invScale, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(f16Local, tempLocal, AscendC::RoundMode::CAST_NONE, N);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<int8_t> i8OutLocal;
        i8OutQueue_.AllocTensor<int8_t>(i8OutLocal);
        AscendC::Cast(i8OutLocal, f16Local, AscendC::RoundMode::CAST_ROUND, N);
        AscendC::PipeBarrier<PIPE_V>();
        i8OutQueue_.EnQue(i8OutLocal);
        i8OutQueue_.DeQue<int8_t>(i8OutLocal);
        StoreUbToGm(y1GM_[offset], i8OutLocal, static_cast<uint32_t>(N));
        i8OutQueue_.FreeTensor(i8OutLocal);
    }

private:
    MultiAddRmsNormDqTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<dataType> x1GM_;
    AscendC::GlobalTensor<dataType> x2GM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> xSumGM_;
    AscendC::GlobalTensor<dataType> yNormGM_;
    AscendC::GlobalTensor<int8_t> y1GM_;
    AscendC::GlobalTensor<float> scale1GM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> dtypeOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> i8OutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> scaleOutQueue_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xsumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ynormBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> f16CastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
};
