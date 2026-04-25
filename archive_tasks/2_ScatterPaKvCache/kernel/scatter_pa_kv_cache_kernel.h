#pragma once

#include "kernel_common.h"
#include "scatter_pa_kv_cache_tiling.h"

class ScatterPaKvCacheKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR key,
        GM_ADDR keyCache,
        GM_ADDR slotMapping,
        GM_ADDR value,
        GM_ADDR valueCache,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);

        const int32_t totalKvRows = tiling_.numBlocks * tiling_.numKvSlices;
        const int32_t totalVvRows = tiling_.numBlocks * tiling_.numVvSlices;

        keyGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(key),
            static_cast<int64_t>(tiling_.nTokens) * tiling_.keyFlatDimI32);
        keyCacheGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(keyCache),
            static_cast<int64_t>(totalKvRows) * tiling_.blockSize * SLICE_LEN_I32);
        slotGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(slotMapping),
            tiling_.nTokens);
        valGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(value),
            static_cast<int64_t>(tiling_.nTokens) * tiling_.valFlatDimI32);
        valCacheGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(valueCache),
            static_cast<int64_t>(totalVvRows) * tiling_.blockSize * SLICE_LEN_I32);

        pipe_ = pipe;
        constexpr uint32_t slotBytes = 32;
        constexpr uint32_t sliceBytes = SLICE_LEN_I32 * sizeof(int32_t); // 32
        pipe_->InitBuffer(slotBuf_, slotBytes);
        pipe_->InitBuffer(sliceBuf_, sliceBytes);
    }

    __aicore__ inline void Process()
    {
        const int32_t coreIdx = AscendC::GetBlockIdx();
        for (int32_t localIdx = 0; localIdx < tiling_.tokensPerCore; ++localIdx) {
            const int32_t tokenIdx = coreIdx * tiling_.tokensPerCore + localIdx;
            if (tokenIdx < tiling_.nTokens) {
                ProcessToken(tokenIdx);
            }
        }
    }

private:
    __aicore__ inline void ProcessToken(int32_t tokenIdx)
    {
        constexpr uint32_t sliceBytes = SLICE_LEN_I32 * sizeof(int32_t); // 32

        // 1. Read slot_mapping[tokenIdx]
        AscendC::LocalTensor<int32_t> slotLocal = slotBuf_.Get<int32_t>();
        {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
            AscendC::DataCopyPad(slotLocal, slotGM_[tokenIdx], params, padParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        const int32_t slot = slotLocal.GetValue(0);

        const int32_t blockIndex = slot / tiling_.blockSize;
        const int32_t blockOffset = slot % tiling_.blockSize;

        // 2. Scatter key slices (each slice = 8 int32 = 32 bytes)
        AscendC::LocalTensor<int32_t> sliceLocal = sliceBuf_.Get<int32_t>();
        for (int32_t s = 0; s < tiling_.numKvSlices; ++s) {
            const int64_t srcOff =
                static_cast<int64_t>(tokenIdx) * tiling_.keyFlatDimI32
                + static_cast<int64_t>(s) * SLICE_LEN_I32;

            // GM -> UB
            AscendC::DataCopyExtParams loadP{1, sliceBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> padP{true, 0, 0, 0};
            AscendC::DataCopyPad(sliceLocal, keyGM_[srcOff], loadP, padP);
            AscendC::PipeBarrier<PIPE_ALL>();

            // UB -> GM
            const int64_t cacheRow =
                static_cast<int64_t>(blockIndex) * tiling_.numKvSlices + s;
            const int64_t dstOff =
                cacheRow * tiling_.blockSize * SLICE_LEN_I32
                + static_cast<int64_t>(blockOffset) * SLICE_LEN_I32;
            AscendC::DataCopyExtParams storeP{1, sliceBytes, 0, 0, 0};
            AscendC::DataCopyPad(keyCacheGM_[dstOff], sliceLocal, storeP);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // 3. Scatter value slices (each slice = 8 int32 = 32 bytes)
        for (int32_t s = 0; s < tiling_.numVvSlices; ++s) {
            const int64_t srcOff =
                static_cast<int64_t>(tokenIdx) * tiling_.valFlatDimI32
                + static_cast<int64_t>(s) * SLICE_LEN_I32;

            // GM -> UB
            AscendC::DataCopyExtParams loadP{1, sliceBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> padP{true, 0, 0, 0};
            AscendC::DataCopyPad(sliceLocal, valGM_[srcOff], loadP, padP);
            AscendC::PipeBarrier<PIPE_ALL>();

            // UB -> GM
            const int64_t cacheRow =
                static_cast<int64_t>(blockIndex) * tiling_.numVvSlices + s;
            const int64_t dstOff =
                cacheRow * tiling_.blockSize * SLICE_LEN_I32
                + static_cast<int64_t>(blockOffset) * SLICE_LEN_I32;
            AscendC::DataCopyExtParams storeP{1, sliceBytes, 0, 0, 0};
            AscendC::DataCopyPad(valCacheGM_[dstOff], sliceLocal, storeP);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    ScatterPaKvCacheTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<int32_t> keyGM_;
    AscendC::GlobalTensor<int32_t> keyCacheGM_;
    AscendC::GlobalTensor<int32_t> slotGM_;
    AscendC::GlobalTensor<int32_t> valGM_;
    AscendC::GlobalTensor<int32_t> valCacheGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> slotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sliceBuf_;
};
