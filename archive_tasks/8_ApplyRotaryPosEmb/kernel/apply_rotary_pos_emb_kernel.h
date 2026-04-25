#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"
#include "apply_rotary_pos_emb_tiling.h"
#include "kernel_common.h"

using namespace AscendC;

template <typename dataType>
class ApplyRotaryPosEmbKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR out,
        GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        pipe_ = pipe;

        xGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ dataType *>(x),
            tiling_.M * tiling_.D);
        cosGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ dataType *>(cos),
            tiling_.M * tiling_.D_rot);
        sinGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ dataType *>(sin),
            tiling_.M * tiling_.D_rot);
        outGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ dataType *>(out),
            tiling_.M * tiling_.D);

        if ASCEND_IS_AIV {
            int32_t D     = tiling_.D;
            int32_t split = tiling_.split;

            pipe_->InitBuffer(fullRowBuf_, D * sizeof(dataType));
            pipe_->InitBuffer(x1Buf_,      split * sizeof(float));
            pipe_->InitBuffer(x2Buf_,      split * sizeof(float));
            pipe_->InitBuffer(cos1Buf_,    split * sizeof(float));
            pipe_->InitBuffer(cos2Buf_,    split * sizeof(float));
            pipe_->InitBuffer(sin1Buf_,    split * sizeof(float));
            pipe_->InitBuffer(sin2Buf_,    split * sizeof(float));
            pipe_->InitBuffer(res1Buf_,    split * sizeof(float));
            pipe_->InitBuffer(res2Buf_,    split * sizeof(float));
            pipe_->InitBuffer(tmpBuf_,     split * sizeof(float));

            if constexpr (needCast_) {
                pipe_->InitBuffer(castInBuf_,   split * sizeof(dataType));
                pipe_->InitBuffer(castOut1Buf_, split * sizeof(dataType));
                pipe_->InitBuffer(castOut2Buf_, split * sizeof(dataType));
            }

            fullRowLocal_ = fullRowBuf_.Get<dataType>();
            x1Local_      = x1Buf_.Get<float>();
            x2Local_      = x2Buf_.Get<float>();
            cos1Local_    = cos1Buf_.Get<float>();
            cos2Local_    = cos2Buf_.Get<float>();
            sin1Local_    = sin1Buf_.Get<float>();
            sin2Local_    = sin2Buf_.Get<float>();
            res1Local_    = res1Buf_.Get<float>();
            res2Local_    = res2Buf_.Get<float>();
            tmpLocal_     = tmpBuf_.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            int coreIdx = AscendC::GetBlockIdx();
            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }
                for (int row = 0; row < tiling_.blockM; ++row) {
                    int rowIdx = bx * tiling_.blockM + row;
                    if (rowIdx < tiling_.M) {
                        ProcessRow(rowIdx);
                    }
                }
            }
        }
    }

private:
    static constexpr bool needCast_ = !std::is_same_v<dataType, float>;

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

    __aicore__ inline void ProcessRow(int rowIdx)
    {
        const int32_t D     = tiling_.D;
        const int32_t D_rot = tiling_.D_rot;
        const int32_t split = tiling_.split;
        const int32_t gmRowX   = rowIdx * D;
        const int32_t gmRowCos = rowIdx * D_rot;

        // -- Step 1: copy full x row -> out (implicit passthrough) --
        AscendC::DataCopy(fullRowLocal_, xGM_[gmRowX], D);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(outGM_[gmRowX], fullRowLocal_, D);
        AscendC::PipeBarrier<PIPE_ALL>();

        // -- Step 2: load halves & cast to float32 --
        if constexpr (!needCast_) {
            // dataType == float: load directly into fp32 buffers
            AscendC::DataCopy(x1Local_, xGM_[gmRowX], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(x2Local_, xGM_[gmRowX + split], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(cos1Local_, cosGM_[gmRowCos], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(cos2Local_, cosGM_[gmRowCos + split], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(sin1Local_, sinGM_[gmRowCos], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(sin2Local_, sinGM_[gmRowCos + split], split);
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            // dataType == half/bf16: load into cast buffer, cast to fp32
            AscendC::LocalTensor<dataType> castIn = castInBuf_.Get<dataType>();

            AscendC::DataCopy(castIn, xGM_[gmRowX], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(x1Local_, castIn, AscendC::RoundMode::CAST_NONE, split);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(castIn, xGM_[gmRowX + split], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(x2Local_, castIn, AscendC::RoundMode::CAST_NONE, split);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(castIn, cosGM_[gmRowCos], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(cos1Local_, castIn, AscendC::RoundMode::CAST_NONE, split);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(castIn, cosGM_[gmRowCos + split], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(cos2Local_, castIn, AscendC::RoundMode::CAST_NONE, split);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(castIn, sinGM_[gmRowCos], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(sin1Local_, castIn, AscendC::RoundMode::CAST_NONE, split);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(castIn, sinGM_[gmRowCos + split], split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(sin2Local_, castIn, AscendC::RoundMode::CAST_NONE, split);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // -- Step 3: rotation compute (all in fp32) --
        // res1 = x1 * cos1 - x2 * sin1
        AscendC::Mul(res1Local_, x1Local_, cos1Local_, split);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(tmpLocal_, x2Local_, sin1Local_, split);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sub(res1Local_, res1Local_, tmpLocal_, split);
        AscendC::PipeBarrier<PIPE_ALL>();

        // res2 = x2 * cos2 + x1 * sin2
        AscendC::Mul(res2Local_, x2Local_, cos2Local_, split);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(tmpLocal_, x1Local_, sin2Local_, split);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Add(res2Local_, res2Local_, tmpLocal_, split);
        AscendC::PipeBarrier<PIPE_ALL>();

        // -- Step 4: cast back & store rotation portion --
        if constexpr (!needCast_) {
            // dataType == float: store directly
            AscendC::DataCopy(outGM_[gmRowX], res1Local_, split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(outGM_[gmRowX + split], res2Local_, split);
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            AscendC::LocalTensor<dataType> castOut1 = castOut1Buf_.Get<dataType>();
            AscendC::LocalTensor<dataType> castOut2 = castOut2Buf_.Get<dataType>();

            AscendC::Cast(castOut1, res1Local_, OutputRoundMode(), split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(outGM_[gmRowX], castOut1, split);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Cast(castOut2, res2Local_, OutputRoundMode(), split);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(outGM_[gmRowX + split], castOut2, split);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    ApplyRotaryPosEmbTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> cosGM_;
    AscendC::GlobalTensor<dataType> sinGM_;
    AscendC::GlobalTensor<dataType> outGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> fullRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cos1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cos2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sin1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sin2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> res1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> res2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castInBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castOut1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castOut2Buf_;

    AscendC::LocalTensor<dataType> fullRowLocal_;
    AscendC::LocalTensor<float> x1Local_;
    AscendC::LocalTensor<float> x2Local_;
    AscendC::LocalTensor<float> cos1Local_;
    AscendC::LocalTensor<float> cos2Local_;
    AscendC::LocalTensor<float> sin1Local_;
    AscendC::LocalTensor<float> sin2Local_;
    AscendC::LocalTensor<float> res1Local_;
    AscendC::LocalTensor<float> res2Local_;
    AscendC::LocalTensor<float> tmpLocal_;
};
