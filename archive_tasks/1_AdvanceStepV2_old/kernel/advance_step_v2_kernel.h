#pragma once
#include "kernel_operator.h"
#include "advance_step_v2_tiling.h"
#include "kernel_common.h"

using namespace AscendC;

class AdvanceStepV2Kernel {
public:
    __aicore__ inline AdvanceStepV2Kernel() {}

    __aicore__ inline void Init(
        __gm__ uint8_t *tilingGm,
        __gm__ int64_t *inputTokens,
        __gm__ int64_t *sampledTokens,
        __gm__ int64_t *inputPositions,
        __gm__ int64_t *acceptedNum,
        __gm__ int64_t *blockTableFlat,
        __gm__ int64_t *specTokensFlat,
        __gm__ int64_t *outInputTokens,
        __gm__ int64_t *outInputPositions,
        __gm__ int64_t *outSeqLens,
        __gm__ int64_t *outSlotMapping
    ) {
        CopyTiling(tiling_, tilingGm);

        inputTokensGm_.SetGlobalBuffer(inputTokens);
        sampledTokensGm_.SetGlobalBuffer(sampledTokens);
        inputPositionsGm_.SetGlobalBuffer(inputPositions);
        acceptedNumGm_.SetGlobalBuffer(acceptedNum);
        blockTableFlatGm_.SetGlobalBuffer(blockTableFlat);
        specTokensFlatGm_.SetGlobalBuffer(specTokensFlat);
        outInputTokensGm_.SetGlobalBuffer(outInputTokens);
        outInputPositionsGm_.SetGlobalBuffer(outInputPositions);
        outSeqLensGm_.SetGlobalBuffer(outSeqLens);
        outSlotMappingGm_.SetGlobalBuffer(outSlotMapping);
    }

    __aicore__ inline void Process() {
        const int32_t coreIdx = GetBlockIdx();
        const int32_t numReqs = tiling_.numReqs;
        const int32_t reqsPerCore = tiling_.reqsPerCore;
        const int32_t reqStart = coreIdx * reqsPerCore;
        int32_t reqEnd = reqStart + reqsPerCore;
        if (reqEnd > numReqs) reqEnd = numReqs;
        if (reqStart >= numReqs) return;

        const int32_t tokenEachReqs = tiling_.tokenEachReqs;
        const int32_t sampledCols = tiling_.sampledCols;
        const int32_t maxNumBlocks = tiling_.maxNumBlocks;
        const int32_t blockSize = tiling_.blockSize;
        const int32_t specNum = tokenEachReqs - 1;

        for (int32_t reqIdx = reqStart; reqIdx < reqEnd; ++reqIdx) {
            const int32_t elemStart = reqIdx * tokenEachReqs;

            // Step 1: Find last_token via argmin logic
            // Scan sampled_tokens[reqIdx, :] to find first negative value
            // The argmin of cat(sampled_tokens[reqIdx], [-1]) gives us the
            // position of the first -1, then we take the previous element.
            int64_t lastToken = 0;
            {
                const int64_t sampledRowBase = static_cast<int64_t>(reqIdx) * sampledCols;
                int32_t minIdx = sampledCols - 1;  // default: last valid column
                for (int32_t col = 0; col < sampledCols; ++col) {
                    int64_t val = sampledTokensGm_.GetValue(sampledRowBase + col);
                    if (val < 0) {
                        minIdx = col - 1;
                        break;
                    }
                }
                // Edge case: if col=0 is negative, minIdx=-1, but that shouldn't happen
                // per the algorithm semantics (at least one token is accepted)
                if (minIdx < 0) minIdx = 0;
                lastToken = sampledTokensGm_.GetValue(sampledRowBase + minIdx);
            }

            // Read accepted_num for this request
            const int64_t accNum = acceptedNumGm_.GetValue(reqIdx);

            // Process each token position for this request
            for (int32_t tokIdx = 0; tokIdx < tokenEachReqs; ++tokIdx) {
                const int32_t flatIdx = elemStart + tokIdx;

                // Step 2: out_input_positions = input_positions + accepted_num + 1
                int64_t oldPos = inputPositionsGm_.GetValue(flatIdx);
                int64_t newPos = oldPos + accNum + 1;
                outInputPositionsGm_.SetValue(flatIdx, newPos);

                // Step 3: out_seq_lens = out_input_positions + 1
                outSeqLensGm_.SetValue(flatIdx, newPos + 1);

                // Step 4: out_input_tokens
                if (tokIdx == 0) {
                    outInputTokensGm_.SetValue(flatIdx, lastToken);
                } else {
                    // spec_tokens[reqIdx, tokIdx - 1]
                    int64_t specOffset = static_cast<int64_t>(reqIdx) * specNum + (tokIdx - 1);
                    int64_t specTok = specTokensFlatGm_.GetValue(specOffset);
                    outInputTokensGm_.SetValue(flatIdx, specTok);
                }

                // Step 5: out_slot_mapping
                // block_table_idx = reqIdx * maxNumBlocks + newPos // blockSize
                // block_number = blockTableFlat[block_table_idx]
                // block_offset = newPos % blockSize
                // slot = block_number * blockSize + block_offset
                int64_t blockTableIdx = static_cast<int64_t>(reqIdx) * maxNumBlocks + newPos / blockSize;
                int64_t blockNumber = blockTableFlatGm_.GetValue(blockTableIdx);
                int64_t blockOffset = newPos % blockSize;
                int64_t slot = blockNumber * blockSize + blockOffset;
                outSlotMappingGm_.SetValue(flatIdx, slot);
            }
        }
    }

private:
    AdvanceStepV2Tiling tiling_;
    GlobalTensor<int64_t> inputTokensGm_;
    GlobalTensor<int64_t> sampledTokensGm_;
    GlobalTensor<int64_t> inputPositionsGm_;
    GlobalTensor<int64_t> acceptedNumGm_;
    GlobalTensor<int64_t> blockTableFlatGm_;
    GlobalTensor<int64_t> specTokensFlatGm_;
    GlobalTensor<int64_t> outInputTokensGm_;
    GlobalTensor<int64_t> outInputPositionsGm_;
    GlobalTensor<int64_t> outSeqLensGm_;
    GlobalTensor<int64_t> outSlotMappingGm_;
};
