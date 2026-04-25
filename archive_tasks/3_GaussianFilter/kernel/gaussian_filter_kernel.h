#pragma once
#include "kernel_operator.h"
#include "gaussian_filter_tiling.h"
#include "kernel_common.h"

using namespace AscendC;

class GaussianFilterKernel {
public:
    __aicore__ inline GaussianFilterKernel() {}

    __aicore__ inline void Init(
        __gm__ uint8_t *tilingGm,
        __gm__ float *means, __gm__ float *colors, __gm__ float *det,
        __gm__ float *opacities, __gm__ float *means2d, __gm__ float *depths,
        __gm__ float *radius, __gm__ float *conics, __gm__ float *covars2d,
        __gm__ int32_t *bMap,
        __gm__ float *meansOut, __gm__ float *colorsOut, __gm__ float *means2dOut,
        __gm__ float *depthsOut, __gm__ float *radiusOut, __gm__ float *covars2dOut,
        __gm__ float *conicsOut, __gm__ float *opacitiesOut,
        __gm__ uint8_t *filterUint8, __gm__ int32_t *cntOut
    ) {
        CopyTiling(tiling_, tilingGm);
        meansGm_.SetGlobalBuffer(means);
        colorsGm_.SetGlobalBuffer(colors);
        detGm_.SetGlobalBuffer(det);
        opacitiesGm_.SetGlobalBuffer(opacities);
        means2dGm_.SetGlobalBuffer(means2d);
        depthsGm_.SetGlobalBuffer(depths);
        radiusGm_.SetGlobalBuffer(radius);
        conicsGm_.SetGlobalBuffer(conics);
        covars2dGm_.SetGlobalBuffer(covars2d);
        bMapGm_.SetGlobalBuffer(bMap);
        meansOutGm_.SetGlobalBuffer(meansOut);
        colorsOutGm_.SetGlobalBuffer(colorsOut);
        means2dOutGm_.SetGlobalBuffer(means2dOut);
        depthsOutGm_.SetGlobalBuffer(depthsOut);
        radiusOutGm_.SetGlobalBuffer(radiusOut);
        covars2dOutGm_.SetGlobalBuffer(covars2dOut);
        conicsOutGm_.SetGlobalBuffer(conicsOut);
        opacitiesOutGm_.SetGlobalBuffer(opacitiesOut);
        filterUint8Gm_.SetGlobalBuffer(filterUint8);
        cntOutGm_.SetGlobalBuffer(cntOut);
    }

    __aicore__ inline void InitBuffers() {
        const int32_t fBytes = tiling_.TILE_N * static_cast<int32_t>(sizeof(float));
        pipe_.InitBuffer(detBuf_, fBytes);
        pipe_.InitBuffer(depthsBuf_, fBytes);
        pipe_.InitBuffer(m2dXBuf_, fBytes);
        pipe_.InitBuffer(m2dYBuf_, fBytes);
        pipe_.InitBuffer(radXBuf_, fBytes);
        pipe_.InitBuffer(radYBuf_, fBytes);
        pipe_.InitBuffer(validBuf_, fBytes);
        pipe_.InitBuffer(insideBuf_, fBytes);
        pipe_.InitBuffer(maskBuf_, fBytes);
        pipe_.InitBuffer(tmpBuf_, fBytes);
        pipe_.InitBuffer(onesBuf_, fBytes);
        pipe_.InitBuffer(zerosBuf_, fBytes);
        int32_t cmpSize = ((tiling_.TILE_N / 8 + 31) / 32) * 32;
        pipe_.InitBuffer(cmpBuf_, cmpSize);
    }

    __aicore__ inline void Process() {
        LocalTensor<float> onesUb = onesBuf_.Get<float>();
        LocalTensor<float> zerosUb = zerosBuf_.Get<float>();
        Duplicate(onesUb, 1.0f, tiling_.TILE_N);
        Duplicate(zerosUb, 0.0f, tiling_.TILE_N);
        pipe_barrier(PIPE_ALL);

        const int32_t coreIdx = GetBlockIdx();
        for (int32_t li = 0; li < tiling_.tasksPerCore; ++li) {
            int32_t bc = coreIdx * tiling_.tasksPerCore + li;
            if (bc < tiling_.BC) {
                ProcessBC(bc, onesUb, zerosUb);
            }
        }
    }

    TPipe pipe_;

private:
    __aicore__ inline void ProcessBC(int32_t bc,
                                     LocalTensor<float> &onesUb,
                                     LocalTensor<float> &zerosUb) {
        const int32_t N = tiling_.N_padded;
        const int32_t TILE_N = tiling_.TILE_N;
        const int32_t nTiles = tiling_.n_tiles;
        const int32_t bytesPerTile = tiling_.bytes_per_tile;
        const int32_t M = tiling_.M_padded;
        int32_t bIdx = bMapGm_.GetValue(bc);

        // Phase A: fill all outputs with 1.0
        FillOutputs(bc, onesUb);

        // Phase B: per-tile mask + compact + pack
        int32_t cntGlobal = 0;
        LocalTensor<float> detUb = detBuf_.Get<float>();
        LocalTensor<float> depthsUb = depthsBuf_.Get<float>();
        LocalTensor<float> m2dXUb = m2dXBuf_.Get<float>();
        LocalTensor<float> m2dYUb = m2dYBuf_.Get<float>();
        LocalTensor<float> radXUb = radXBuf_.Get<float>();
        LocalTensor<float> radYUb = radYBuf_.Get<float>();
        LocalTensor<float> validUb = validBuf_.Get<float>();
        LocalTensor<float> insideUb = insideBuf_.Get<float>();
        LocalTensor<float> maskUb = maskBuf_.Get<float>();
        LocalTensor<float> tmpUb = tmpBuf_.Get<float>();
        LocalTensor<uint8_t> cmpUb = cmpBuf_.Get<uint8_t>();

        for (int32_t t = 0; t < nTiles; ++t) {
            int32_t ts = t * TILE_N;

            // B.1: Load mask inputs
            DataCopyExtParams cpIn{1, static_cast<uint32_t>(TILE_N * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padP{false, 0, 0, 0.0f};
            DataCopyPad(detUb, detGm_[bc * N + ts], cpIn, padP);
            DataCopyPad(depthsUb, depthsGm_[bc * N + ts], cpIn, padP);
            DataCopyPad(m2dXUb, means2dGm_[(int64_t)bc * 2 * N + 0 * N + ts], cpIn, padP);
            DataCopyPad(m2dYUb, means2dGm_[(int64_t)bc * 2 * N + 1 * N + ts], cpIn, padP);
            DataCopyPad(radXUb, radiusGm_[(int64_t)bc * 2 * N + 0 * N + ts], cpIn, padP);
            DataCopyPad(radYUb, radiusGm_[(int64_t)bc * 2 * N + 1 * N + ts], cpIn, padP);
            pipe_barrier(PIPE_ALL);

            // B.2: Vectorized mask computation
            // valid = (det > 0)
            CompareScalar(cmpUb, detUb, 0.0f, CMPMODE::GT, TILE_N);
            Select(validUb, cmpUb, onesUb, zerosUb,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_N);

            // valid &= (depths > near_plane)
            CompareScalar(cmpUb, depthsUb, tiling_.near_plane, CMPMODE::GT, TILE_N);
            Select(tmpUb, cmpUb, onesUb, zerosUb,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_N);
            Mul(validUb, validUb, tmpUb, TILE_N);

            // valid &= (depths < far_plane)
            CompareScalar(cmpUb, depthsUb, tiling_.far_plane, CMPMODE::LT, TILE_N);
            Select(tmpUb, cmpUb, onesUb, zerosUb,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_N);
            Mul(validUb, validUb, tmpUb, TILE_N);

            // radius[~valid] = 0
            Mul(radXUb, radXUb, validUb, TILE_N);
            Mul(radYUb, radYUb, validUb, TILE_N);

            // inside = (m2d_x + rad_x > 0)
            Add(tmpUb, m2dXUb, radXUb, TILE_N);
            CompareScalar(cmpUb, tmpUb, 0.0f, CMPMODE::GT, TILE_N);
            Select(insideUb, cmpUb, onesUb, zerosUb,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_N);

            // inside &= (m2d_x - rad_x < width)
            Sub(tmpUb, m2dXUb, radXUb, TILE_N);
            CompareScalar(cmpUb, tmpUb, tiling_.width, CMPMODE::LT, TILE_N);
            Select(tmpUb, cmpUb, onesUb, zerosUb,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_N);
            Mul(insideUb, insideUb, tmpUb, TILE_N);

            // inside &= (m2d_y + rad_y > 0)
            Add(tmpUb, m2dYUb, radYUb, TILE_N);
            CompareScalar(cmpUb, tmpUb, 0.0f, CMPMODE::GT, TILE_N);
            Select(tmpUb, cmpUb, onesUb, zerosUb,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_N);
            Mul(insideUb, insideUb, tmpUb, TILE_N);

            // inside &= (m2d_y - rad_y < height)
            Sub(tmpUb, m2dYUb, radYUb, TILE_N);
            CompareScalar(cmpUb, tmpUb, tiling_.height, CMPMODE::LT, TILE_N);
            Select(tmpUb, cmpUb, onesUb, zerosUb,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_N);
            Mul(insideUb, insideUb, tmpUb, TILE_N);

            // radius[~inside] = 0
            Mul(radXUb, radXUb, insideUb, TILE_N);
            Mul(radYUb, radYUb, insideUb, TILE_N);

            // mask = valid & inside
            Mul(maskUb, validUb, insideUb, TILE_N);
            pipe_barrier(PIPE_ALL);

            // B.3: Build index buffer (reuse validBuf as int32)
            LocalTensor<int32_t> idxUb = validBuf_.Get<int32_t>();
            int32_t validCount = 0;
            for (int32_t i = 0; i < TILE_N; ++i) {
                if (maskUb.GetValue(i) > 0.5f) {
                    idxUb.SetValue(validCount, i);
                    ++validCount;
                }
            }

            // B.4: Compact and flush all output channels
            if (validCount > 0) {
                // Pre-fill compact buffer with 1.0 (reuse insideBuf)
                LocalTensor<float> compactUb = insideBuf_.Get<float>();
                Duplicate(compactUb, 1.0f, TILE_N);
                pipe_barrier(PIPE_ALL);

                int32_t copyCount = ((validCount + 7) / 8) * 8;
                if (copyCount > TILE_N) copyCount = TILE_N;
                DataCopyExtParams cpOut{1, static_cast<uint32_t>(copyCount * sizeof(float)), 0, 0, 0};

                // Use detUb as ch_ub (temp channel loader)
                LocalTensor<float> chUb = detUb;

                // --- means (3 channels, from meansGm_[bIdx, f, ts]) ---
                for (int32_t f = 0; f < 3; ++f) {
                    DataCopyPad(chUb, meansGm_[(int64_t)bIdx * 3 * N + f * N + ts], cpIn, padP);
                    pipe_barrier(PIPE_ALL);
                    for (int32_t i = 0; i < validCount; ++i)
                        compactUb.SetValue(i, chUb.GetValue(idxUb.GetValue(i)));
                    DataCopyPad(meansOutGm_[(int64_t)bc * 3 * N + f * N + cntGlobal], compactUb, cpOut);
                    pipe_barrier(PIPE_ALL);
                }

                // --- colors (3 channels, from colorsGm_[bIdx, f, ts]) ---
                for (int32_t f = 0; f < 3; ++f) {
                    DataCopyPad(chUb, colorsGm_[(int64_t)bIdx * 3 * N + f * N + ts], cpIn, padP);
                    pipe_barrier(PIPE_ALL);
                    for (int32_t i = 0; i < validCount; ++i)
                        compactUb.SetValue(i, chUb.GetValue(idxUb.GetValue(i)));
                    DataCopyPad(colorsOutGm_[(int64_t)bc * 3 * N + f * N + cntGlobal], compactUb, cpOut);
                    pipe_barrier(PIPE_ALL);
                }

                // --- conics (3 channels, from conicsGm_[bc, f, ts]) ---
                for (int32_t f = 0; f < 3; ++f) {
                    DataCopyPad(chUb, conicsGm_[(int64_t)bc * 3 * N + f * N + ts], cpIn, padP);
                    pipe_barrier(PIPE_ALL);
                    for (int32_t i = 0; i < validCount; ++i)
                        compactUb.SetValue(i, chUb.GetValue(idxUb.GetValue(i)));
                    DataCopyPad(conicsOutGm_[(int64_t)bc * 3 * N + f * N + cntGlobal], compactUb, cpOut);
                    pipe_barrier(PIPE_ALL);
                }

                // --- covars2d (3 channels, from covars2dGm_[bc, f, ts]) ---
                for (int32_t f = 0; f < 3; ++f) {
                    DataCopyPad(chUb, covars2dGm_[(int64_t)bc * 3 * N + f * N + ts], cpIn, padP);
                    pipe_barrier(PIPE_ALL);
                    for (int32_t i = 0; i < validCount; ++i)
                        compactUb.SetValue(i, chUb.GetValue(idxUb.GetValue(i)));
                    DataCopyPad(covars2dOutGm_[(int64_t)bc * 3 * N + f * N + cntGlobal], compactUb, cpOut);
                    pipe_barrier(PIPE_ALL);
                }

                // --- radius (2 channels, from radiusGm_[bc, f, ts]) ---
                for (int32_t f = 0; f < 2; ++f) {
                    DataCopyPad(chUb, radiusGm_[(int64_t)bc * 2 * N + f * N + ts], cpIn, padP);
                    pipe_barrier(PIPE_ALL);
                    for (int32_t i = 0; i < validCount; ++i)
                        compactUb.SetValue(i, chUb.GetValue(idxUb.GetValue(i)));
                    DataCopyPad(radiusOutGm_[(int64_t)bc * 2 * N + f * N + cntGlobal], compactUb, cpOut);
                    pipe_barrier(PIPE_ALL);
                }

                // --- means2d (2 channels, already in m2dXUb/m2dYUb — unmodified) ---
                for (int32_t i = 0; i < validCount; ++i)
                    compactUb.SetValue(i, m2dXUb.GetValue(idxUb.GetValue(i)));
                DataCopyPad(means2dOutGm_[(int64_t)bc * 2 * N + 0 * N + cntGlobal], compactUb, cpOut);
                pipe_barrier(PIPE_ALL);

                for (int32_t i = 0; i < validCount; ++i)
                    compactUb.SetValue(i, m2dYUb.GetValue(idxUb.GetValue(i)));
                DataCopyPad(means2dOutGm_[(int64_t)bc * 2 * N + 1 * N + cntGlobal], compactUb, cpOut);
                pipe_barrier(PIPE_ALL);

                // --- depths (1 channel, already in depthsUb — unmodified) ---
                for (int32_t i = 0; i < validCount; ++i)
                    compactUb.SetValue(i, depthsUb.GetValue(idxUb.GetValue(i)));
                DataCopyPad(depthsOutGm_[(int64_t)bc * N + cntGlobal], compactUb, cpOut);
                pipe_barrier(PIPE_ALL);

                // --- opacities (1 channel, from opacitiesGm_[bIdx, ts]) ---
                DataCopyPad(chUb, opacitiesGm_[(int64_t)bIdx * N + ts], cpIn, padP);
                pipe_barrier(PIPE_ALL);
                for (int32_t i = 0; i < validCount; ++i)
                    compactUb.SetValue(i, chUb.GetValue(idxUb.GetValue(i)));
                DataCopyPad(opacitiesOutGm_[(int64_t)bc * N + cntGlobal], compactUb, cpOut);
                pipe_barrier(PIPE_ALL);
            }

            cntGlobal += validCount;

            // B.5: Bit packing
            LocalTensor<uint8_t> packUb = cmpBuf_.Get<uint8_t>();
            for (int32_t g = 0; g < bytesPerTile; ++g) {
                int32_t packVal = 0;
                int32_t powVal = 1;
                for (int32_t bit = 0; bit < 8; ++bit) {
                    int32_t idx = g * 8 + bit;
                    if (maskUb.GetValue(idx) > 0.5f) {
                        packVal += powVal;
                    }
                    powVal += powVal;
                }
                packUb.SetValue(g, static_cast<uint8_t>(packVal));
            }
            int32_t packCopySize = ((bytesPerTile + 31) / 32) * 32;
            DataCopyExtParams cpPack{1, static_cast<uint32_t>(packCopySize), 0, 0, 0};
            DataCopyPad(filterUint8Gm_[(int64_t)bc * M + t * bytesPerTile], packUb, cpPack);
            pipe_barrier(PIPE_ALL);
        }

        // Write final count
        cntOutGm_.SetValue(bc, cntGlobal);
    }

    __aicore__ inline void FillOutputs(int32_t bc, LocalTensor<float> &onesUb) {
        const int32_t N = tiling_.N_padded;
        const int32_t TILE_N = tiling_.TILE_N;
        const int32_t nTiles = tiling_.n_tiles;
        DataCopyExtParams cp{1, static_cast<uint32_t>(TILE_N * sizeof(float)), 0, 0, 0};

        for (int32_t t = 0; t < nTiles; ++t) {
            int32_t ts = t * TILE_N;
            for (int32_t f = 0; f < 3; ++f)
                DataCopyPad(meansOutGm_[(int64_t)bc * 3 * N + f * N + ts], onesUb, cp);
            for (int32_t f = 0; f < 3; ++f)
                DataCopyPad(colorsOutGm_[(int64_t)bc * 3 * N + f * N + ts], onesUb, cp);
            for (int32_t f = 0; f < 2; ++f)
                DataCopyPad(means2dOutGm_[(int64_t)bc * 2 * N + f * N + ts], onesUb, cp);
            DataCopyPad(depthsOutGm_[(int64_t)bc * N + ts], onesUb, cp);
            for (int32_t f = 0; f < 2; ++f)
                DataCopyPad(radiusOutGm_[(int64_t)bc * 2 * N + f * N + ts], onesUb, cp);
            for (int32_t f = 0; f < 3; ++f)
                DataCopyPad(covars2dOutGm_[(int64_t)bc * 3 * N + f * N + ts], onesUb, cp);
            for (int32_t f = 0; f < 3; ++f)
                DataCopyPad(conicsOutGm_[(int64_t)bc * 3 * N + f * N + ts], onesUb, cp);
            DataCopyPad(opacitiesOutGm_[(int64_t)bc * N + ts], onesUb, cp);
        }
        pipe_barrier(PIPE_ALL);
    }

    GaussianFilterTiling tiling_;
    GlobalTensor<float> meansGm_, colorsGm_, detGm_, opacitiesGm_;
    GlobalTensor<float> means2dGm_, depthsGm_, radiusGm_, conicsGm_, covars2dGm_;
    GlobalTensor<int32_t> bMapGm_;
    GlobalTensor<float> meansOutGm_, colorsOutGm_, means2dOutGm_;
    GlobalTensor<float> depthsOutGm_, radiusOutGm_, covars2dOutGm_;
    GlobalTensor<float> conicsOutGm_, opacitiesOutGm_;
    GlobalTensor<uint8_t> filterUint8Gm_;
    GlobalTensor<int32_t> cntOutGm_;

    TBuf<QuePosition::VECCALC> detBuf_, depthsBuf_, m2dXBuf_, m2dYBuf_;
    TBuf<QuePosition::VECCALC> radXBuf_, radYBuf_;
    TBuf<QuePosition::VECCALC> validBuf_, insideBuf_, maskBuf_, tmpBuf_;
    TBuf<QuePosition::VECCALC> onesBuf_, zerosBuf_;
    TBuf<QuePosition::VECCALC> cmpBuf_;
};
