/**
 * @file mhc_post_kernel.h
 *
 * MhcPost AscendC kernel -- pure Vector, persistent-core partitioning.
 *
 * Computes:
 *   y[i, d] = h_post[i] * h_out[d] + sum_j(h_res[j, i] * x[j, d])
 *
 * Inputs:
 *   x:      (B, n, D)     dataType (fp16/bf16)
 *   h_res:  (B, n, n_pad) float32
 *   h_out:  (B, D)        dataType (fp16/bf16)
 *   h_post: (B, n_pad)    float32
 * Output:
 *   y:      (B, n, D)     dataType (fp16/bf16)
 */
#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_common.h"
#include "mhc_post_tiling.h"

using namespace AscendC;

template <typename dataType>
class MhcPostKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hRes,
                                GM_ADDR hOut, GM_ADDR hPost,
                                GM_ADDR y,
                                GM_ADDR tilingGM, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

private:
    static constexpr bool needCast_ = !std::is_same_v<dataType, float>;

    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void LoadToFp32(
        AscendC::LocalTensor<float> &dst,
        AscendC::GlobalTensor<dataType> &gm,
        uint32_t offset, int32_t count)
    {
        if constexpr (needCast_) {
            AscendC::LocalTensor<dataType> castIn = castInBuf_.Get<dataType>();
            AscendC::DataCopy(castIn, gm[offset], count);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(dst, castIn, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            auto fGM = gm.template ReinterpretCast<float>();
            AscendC::DataCopy(dst, fGM[offset], count);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void StoreFp32(
        AscendC::GlobalTensor<dataType> &gm,
        uint32_t offset,
        AscendC::LocalTensor<float> &src,
        int32_t count)
    {
        if constexpr (needCast_) {
            AscendC::LocalTensor<dataType> castOut = castOutBuf_.Get<dataType>();
            AscendC::Cast(castOut, src, OutputRoundMode(), count);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(gm[offset], castOut, count);
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            auto fGM = gm.template ReinterpretCast<float>();
            AscendC::DataCopy(fGM[offset], src, count);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessBatch(int32_t batchId);

    MhcPostTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    // Global memory tensors
    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<float>    hResGM_;
    AscendC::GlobalTensor<dataType> hOutGM_;
    AscendC::GlobalTensor<float>    hPostGM_;
    AscendC::GlobalTensor<dataType> yGM_;

    // UB buffers
    AscendC::TBuf<AscendC::TPosition::VECCALC> xRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> houtTileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castInBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hresRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hpostBuf_;

    // Cached local tensors
    AscendC::LocalTensor<float> xRowLocal_;
    AscendC::LocalTensor<float> houtTileLocal_;
    AscendC::LocalTensor<float> yRowLocal_;
    AscendC::LocalTensor<float> tmpLocal_;
    AscendC::LocalTensor<float> hresRowLocal_;
    AscendC::LocalTensor<float> hpostLocal_;
};

template <typename dataType>
__aicore__ inline void MhcPostKernel<dataType>::Init(
    GM_ADDR x, GM_ADDR hRes,
    GM_ADDR hOut, GM_ADDR hPost,
    GM_ADDR y,
    GM_ADDR tilingGM, AscendC::TPipe *pipe)
{
    CopyTiling(&tiling_, tilingGM);
    pipe_ = pipe;

    int32_t B = tiling_.B;
    int32_t n = tiling_.n;
    int32_t nPad = tiling_.nPad;
    int32_t D = tiling_.D;
    int32_t blockD = tiling_.blockD;

    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), B * n * D);
    hResGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hRes), B * n * nPad);
    hOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(hOut), B * D);
    hPostGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hPost), B * nPad);
    yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), B * n * D);

    if ASCEND_IS_AIV {
        uint32_t blockDBytes = blockD * sizeof(float);
        pipe_->InitBuffer(xRowBuf_, blockDBytes);
        pipe_->InitBuffer(houtTileBuf_, blockDBytes);
        pipe_->InitBuffer(yRowBuf_, blockDBytes);
        pipe_->InitBuffer(tmpBuf_, blockDBytes);

        if constexpr (needCast_) {
            pipe_->InitBuffer(castInBuf_, blockD * sizeof(dataType));
            pipe_->InitBuffer(castOutBuf_, blockD * sizeof(dataType));
        }

        uint32_t nPadBytes = nPad * sizeof(float);
        pipe_->InitBuffer(hresRowBuf_, nPadBytes);
        pipe_->InitBuffer(hpostBuf_, nPadBytes);

        xRowLocal_ = xRowBuf_.Get<float>();
        houtTileLocal_ = houtTileBuf_.Get<float>();
        yRowLocal_ = yRowBuf_.Get<float>();
        tmpLocal_ = tmpBuf_.Get<float>();
        hresRowLocal_ = hresRowBuf_.Get<float>();
        hpostLocal_ = hpostBuf_.Get<float>();
    }
}

template <typename dataType>
__aicore__ inline void MhcPostKernel<dataType>::Process()
{
    if ASCEND_IS_AIV {
        int coreIdx = AscendC::GetBlockIdx();
        for (int32_t localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
            int32_t batchId = coreIdx * tiling_.tasksPerCore + localIdx;
            if (batchId < tiling_.B) {
                ProcessBatch(batchId);
            }
        }
    }
}

template <typename dataType>
__aicore__ inline void MhcPostKernel<dataType>::ProcessBatch(int32_t batchId)
{
    const int32_t n = tiling_.n;
    const int32_t nPad = tiling_.nPad;
    const int32_t D = tiling_.D;
    const int32_t blockD = tiling_.blockD;
    const int32_t dTiles = tiling_.dTiles;

    // Load h_post once per batch
    AscendC::DataCopy(hpostLocal_, hPostGM_[batchId * nPad], nPad);
    AscendC::PipeBarrier<PIPE_ALL>();

    for (int32_t dt = 0; dt < dTiles; ++dt) {
        const int32_t dStart = dt * blockD;

        // Load h_out tile, cast to fp32
        LoadToFp32(houtTileLocal_, hOutGM_, batchId * D + dStart, blockD);

        // For each output row i
        for (int32_t i = 0; i < n; ++i) {
            // y_row = h_post[i] * h_out_tile
            float hpostI = hpostLocal_.GetValue(i);
            AscendC::Muls(yRowLocal_, houtTileLocal_, hpostI, blockD);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Accumulate h_res^T @ x: sum_j(h_res[j,i] * x[j, d_tile])
            for (int32_t j = 0; j < n; ++j) {
                // Load h_res row j
                AscendC::DataCopy(hresRowLocal_, hResGM_[batchId * n * nPad + j * nPad], nPad);
                AscendC::PipeBarrier<PIPE_ALL>();

                float hresJI = hresRowLocal_.GetValue(i);

                // Load x[j, d_tile], cast to fp32
                LoadToFp32(xRowLocal_, xGM_, batchId * n * D + j * D + dStart, blockD);

                // y_row += h_res[j,i] * x[j, d_tile]
                AscendC::Muls(tmpLocal_, xRowLocal_, hresJI, blockD);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(yRowLocal_, yRowLocal_, tmpLocal_, blockD);
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // Cast and write output y[batchId, i, dStart:dStart+blockD]
            StoreFp32(yGM_, batchId * n * D + i * D + dStart, yRowLocal_, blockD);
        }
    }
}
