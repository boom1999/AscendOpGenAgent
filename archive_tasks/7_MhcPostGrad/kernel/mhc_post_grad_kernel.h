/**
 * @file mhc_post_grad_kernel.h
 *
 * MhcPostGrad AscendC kernel -- pure Vector, persistent-core partitioning.
 *
 * Computes four backward gradients fused into a single kernel:
 *   grad_x      = h_res @ grad_output          -> (B, n, D), cast to orig_dtype
 *   grad_h_res  = x @ grad_output^T            -> (B, n, n_pad), float32
 *   grad_h_out  = sum(grad_output * h_post, -2) -> (B, D), cast to orig_dtype
 *   grad_h_post = sum(grad_output * h_out, -1)  -> (B, n_pad), float32
 */
#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_common.h"
#include "mhc_post_grad_tiling.h"

using namespace AscendC;

template <typename dataType>
class MhcPostGradKernel {
public:
    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR x, GM_ADDR hRes,
                                GM_ADDR hOut, GM_ADDR hPost,
                                GM_ADDR gradX, GM_ADDR gradHRes,
                                GM_ADDR gradHOut, GM_ADDR gradHPost,
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

    // Load blockD elements from GM (dataType) into fp32 UB buffer
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

    // Store blockD fp32 elements from UB to GM (dataType), casting if needed
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
    __aicore__ inline void ComputeGradX(int32_t batchId);
    __aicore__ inline void ComputeGradHOutAndHPost(int32_t batchId);
    __aicore__ inline void ComputeGradHRes(int32_t batchId);

    MhcPostGradTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    // Global memory tensors
    AscendC::GlobalTensor<dataType> goGM_;
    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<float>    hResGM_;
    AscendC::GlobalTensor<dataType> hOutGM_;
    AscendC::GlobalTensor<float>    hPostGM_;
    AscendC::GlobalTensor<dataType> gradXGM_;
    AscendC::GlobalTensor<float>    gradHResGM_;
    AscendC::GlobalTensor<dataType> gradHOutGM_;
    AscendC::GlobalTensor<float>    gradHPostGM_;

    // UB buffers
    AscendC::TBuf<AscendC::TPosition::VECCALC> goRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> houtTileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castInBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hresRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hpostBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> nAccumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceDstBuf_;

    // Cached local tensors
    AscendC::LocalTensor<float> goRowLocal_;
    AscendC::LocalTensor<float> xRowLocal_;
    AscendC::LocalTensor<float> houtTileLocal_;
    AscendC::LocalTensor<float> accumLocal_;
    AscendC::LocalTensor<float> tmpLocal_;
    AscendC::LocalTensor<float> hresRowLocal_;
    AscendC::LocalTensor<float> hpostLocal_;
    AscendC::LocalTensor<float> nAccumLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> reduceDstLocal_;
};

template <typename dataType>
__aicore__ inline void MhcPostGradKernel<dataType>::Init(
    GM_ADDR gradOutput, GM_ADDR x, GM_ADDR hRes,
    GM_ADDR hOut, GM_ADDR hPost,
    GM_ADDR gradX, GM_ADDR gradHRes,
    GM_ADDR gradHOut, GM_ADDR gradHPost,
    GM_ADDR tilingGM, AscendC::TPipe *pipe)
{
    CopyTiling(&tiling_, tilingGM);
    pipe_ = pipe;

    int32_t B = tiling_.B;
    int32_t n = tiling_.n;
    int32_t nPad = tiling_.nPad;
    int32_t D = tiling_.D;
    int32_t blockD = tiling_.blockD;

    goGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gradOutput), B * n * D);
    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), B * n * D);
    hResGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hRes), B * n * nPad);
    hOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(hOut), B * D);
    hPostGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hPost), B * nPad);
    gradXGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gradX), B * n * D);
    gradHResGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradHRes), B * n * nPad);
    gradHOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gradHOut), B * D);
    gradHPostGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradHPost), B * nPad);

    if ASCEND_IS_AIV {
        uint32_t blockDBytes = blockD * sizeof(float);
        pipe_->InitBuffer(goRowBuf_, blockDBytes);
        pipe_->InitBuffer(xRowBuf_, blockDBytes);
        pipe_->InitBuffer(houtTileBuf_, blockDBytes);
        pipe_->InitBuffer(accumBuf_, blockDBytes);
        pipe_->InitBuffer(tmpBuf_, blockDBytes);

        if constexpr (needCast_) {
            pipe_->InitBuffer(castInBuf_, blockD * sizeof(dataType));
            pipe_->InitBuffer(castOutBuf_, blockD * sizeof(dataType));
        }

        uint32_t nPadBytes = nPad * sizeof(float);
        pipe_->InitBuffer(hresRowBuf_, nPadBytes);
        pipe_->InitBuffer(hpostBuf_, nPadBytes);
        pipe_->InitBuffer(nAccumBuf_, nPadBytes);

        pipe_->InitBuffer(reduceBuf_, blockDBytes);
        pipe_->InitBuffer(reduceDstBuf_, 8 * sizeof(float));

        goRowLocal_ = goRowBuf_.Get<float>();
        xRowLocal_ = xRowBuf_.Get<float>();
        houtTileLocal_ = houtTileBuf_.Get<float>();
        accumLocal_ = accumBuf_.Get<float>();
        tmpLocal_ = tmpBuf_.Get<float>();
        hresRowLocal_ = hresRowBuf_.Get<float>();
        hpostLocal_ = hpostBuf_.Get<float>();
        nAccumLocal_ = nAccumBuf_.Get<float>();
        reduceLocal_ = reduceBuf_.Get<float>();
        reduceDstLocal_ = reduceDstBuf_.Get<float>();
    }
}

template <typename dataType>
__aicore__ inline void MhcPostGradKernel<dataType>::Process()
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
__aicore__ inline void MhcPostGradKernel<dataType>::ProcessBatch(int32_t batchId)
{
    ComputeGradHOutAndHPost(batchId);
    ComputeGradX(batchId);
    ComputeGradHRes(batchId);
}

/**
 * Compute grad_x = h_res @ grad_output for one batch.
 * grad_x[i, d] = sum_j( h_res[i,j] * grad_output[j, d] )
 */
template <typename dataType>
__aicore__ inline void MhcPostGradKernel<dataType>::ComputeGradX(int32_t batchId)
{
    const int32_t n = tiling_.n;
    const int32_t nPad = tiling_.nPad;
    const int32_t D = tiling_.D;
    const int32_t blockD = tiling_.blockD;
    const int32_t dTiles = tiling_.dTiles;

    for (int32_t i = 0; i < n; ++i) {
        // Load h_res row for this i
        AscendC::DataCopy(hresRowLocal_, hResGM_[batchId * n * nPad + i * nPad], nPad);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (int32_t dt = 0; dt < dTiles; ++dt) {
            const int32_t dStart = dt * blockD;

            // Zero accumulator
            AscendC::Duplicate(accumLocal_, 0.0f, blockD);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (int32_t j = 0; j < n; ++j) {
                // Load grad_output[batchId, j, dStart:dStart+blockD] to fp32
                LoadToFp32(goRowLocal_, goGM_, batchId * n * D + j * D + dStart, blockD);

                // accum += go_row * h_res[i, j]
                float hresIJ = hresRowLocal_.GetValue(j);
                AscendC::Muls(tmpLocal_, goRowLocal_, hresIJ, blockD);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(accumLocal_, accumLocal_, tmpLocal_, blockD);
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // Store grad_x[batchId, i, dStart:dStart+blockD]
            StoreFp32(gradXGM_, batchId * n * D + i * D + dStart, accumLocal_, blockD);
        }
    }
}

/**
 * Compute grad_h_out and grad_h_post for one batch.
 * grad_h_out[d] = sum_j( grad_output[j, d] * h_post[j] )
 * grad_h_post[j] = sum_d( grad_output[j, d] * h_out[d] )
 */
template <typename dataType>
__aicore__ inline void MhcPostGradKernel<dataType>::ComputeGradHOutAndHPost(int32_t batchId)
{
    const int32_t n = tiling_.n;
    const int32_t nPad = tiling_.nPad;
    const int32_t D = tiling_.D;
    const int32_t blockD = tiling_.blockD;
    const int32_t dTiles = tiling_.dTiles;

    // Load h_post once per batch
    AscendC::DataCopy(hpostLocal_, hPostGM_[batchId * nPad], nPad);
    AscendC::PipeBarrier<PIPE_ALL>();

    // Zero grad_h_post accumulator
    AscendC::Duplicate(nAccumLocal_, 0.0f, nPad);
    AscendC::PipeBarrier<PIPE_ALL>();

    for (int32_t dt = 0; dt < dTiles; ++dt) {
        const int32_t dStart = dt * blockD;

        // Load h_out tile
        LoadToFp32(houtTileLocal_, hOutGM_, batchId * D + dStart, blockD);

        // Zero grad_h_out tile accumulator
        AscendC::Duplicate(accumLocal_, 0.0f, blockD);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (int32_t j = 0; j < n; ++j) {
            // Load grad_output tile
            LoadToFp32(goRowLocal_, goGM_, batchId * n * D + j * D + dStart, blockD);

            // grad_h_out_tile += go_row * h_post[j]
            float hpostJ = hpostLocal_.GetValue(j);
            AscendC::Muls(tmpLocal_, goRowLocal_, hpostJ, blockD);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(accumLocal_, accumLocal_, tmpLocal_, blockD);
            AscendC::PipeBarrier<PIPE_ALL>();

            // grad_h_post[j] += dot(go_row, h_out_tile)
            AscendC::Mul(tmpLocal_, goRowLocal_, houtTileLocal_, blockD);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::ReduceSum<float>(reduceDstLocal_, tmpLocal_, reduceLocal_, blockD);
            AscendC::PipeBarrier<PIPE_ALL>();
            float dotVal = reduceDstLocal_.GetValue(0);
            float curVal = nAccumLocal_.GetValue(j);
            nAccumLocal_.SetValue(j, curVal + dotVal);
        }

        // Write grad_h_out tile
        StoreFp32(gradHOutGM_, batchId * D + dStart, accumLocal_, blockD);
    }

    // Write accumulated grad_h_post
    AscendC::DataCopy(gradHPostGM_[batchId * nPad], nAccumLocal_, nPad);
    AscendC::PipeBarrier<PIPE_ALL>();
}

/**
 * Compute grad_h_res for one batch.
 * grad_h_res[i,j] = sum_d( x[i,d] * grad_output[j,d] )
 */
template <typename dataType>
__aicore__ inline void MhcPostGradKernel<dataType>::ComputeGradHRes(int32_t batchId)
{
    const int32_t n = tiling_.n;
    const int32_t nPad = tiling_.nPad;
    const int32_t D = tiling_.D;
    const int32_t blockD = tiling_.blockD;
    const int32_t dTiles = tiling_.dTiles;

    for (int32_t i = 0; i < n; ++i) {
        // Zero ghres_row accumulator
        AscendC::Duplicate(nAccumLocal_, 0.0f, nPad);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (int32_t dt = 0; dt < dTiles; ++dt) {
            const int32_t dStart = dt * blockD;

            // Load x[batchId, i, dStart:dStart+blockD] to fp32
            LoadToFp32(xRowLocal_, xGM_, batchId * n * D + i * D + dStart, blockD);

            for (int32_t j = 0; j < n; ++j) {
                // Load grad_output[batchId, j, dStart:dStart+blockD] to fp32
                LoadToFp32(goRowLocal_, goGM_, batchId * n * D + j * D + dStart, blockD);

                // ghres_row[j] += dot(x_row, go_row)
                AscendC::Mul(tmpLocal_, xRowLocal_, goRowLocal_, blockD);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::ReduceSum<float>(reduceDstLocal_, tmpLocal_, reduceLocal_, blockD);
                AscendC::PipeBarrier<PIPE_ALL>();
                float dotVal = reduceDstLocal_.GetValue(0);
                float curVal = nAccumLocal_.GetValue(j);
                nAccumLocal_.SetValue(j, curVal + dotVal);
            }
        }

        // Write completed row to GM
        AscendC::DataCopy(gradHResGM_[batchId * n * nPad + i * nPad], nAccumLocal_, nPad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
