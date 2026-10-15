// One full warp merges up to 32 ordered shard lists. Pending A5 validation.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/device_warp_functions.h"

namespace {
constexpr uint32_t kWarp = 32;
constexpr int32_t kNoId = 2147483647;

__simt_vf__ __aicore__ LAUNCH_BOUND(kWarp) inline void MergeTopKSimt(
    __gm__ const float *partialScores, __gm__ const int32_t *partialIds,
    __gm__ float *topScores, __gm__ int32_t *topIds,
    uint32_t queryCount, uint32_t shards, uint32_t docsPerShard, uint32_t k,
    uint32_t blockId, uint32_t blockNum)
{
    const uint32_t lane = AscendC::Simt::GetThreadIdx<0>();
    for (uint64_t query = blockId; query < queryCount; query += blockNum) {
        const uint64_t base = (query * shards + lane) * k;
        uint32_t bad = 0;
        if (lane < shards) {
            float previousScore = 0.0F;
            int32_t previousId = -1;
            for (uint32_t rank = 0; rank < k; ++rank) {
                const float score = partialScores[base + rank];
                const int32_t id = partialIds[base + rank];
                if (!(score >= 0.0F) || score > 3.402823466e+38F ||
                    id < 0 || static_cast<uint32_t>(id) >= docsPerShard ||
                    (rank != 0 && (score > previousScore || (score == previousScore && id <= previousId))))
                    bad = 1;
                previousScore = score;
                previousId = id;
            }
        }
        for (int32_t mask = 16; mask > 0; mask >>= 1)
            bad |= asc_shfl_xor(bad, mask, 32);
        // This branch is uniform across the entire warp. Invalid input cannot become a plausible ranking.
        if (bad != 0) {
            if (lane == 0) {
                for (uint32_t rank = 0; rank < k; ++rank) {
                    topScores[query * k + rank] = -1.0F;
                    topIds[query * k + rank] = -1;
                }
            }
            continue;
        }
        uint32_t head = 0;
        float currentScore = -1.0F;
        int32_t currentId = kNoId;
        if (lane < shards) {
            currentScore = partialScores[base];
            currentId = static_cast<int32_t>(static_cast<uint64_t>(lane) * docsPerShard + partialIds[base]);
        }
        for (uint32_t rank = 0; rank < k; ++rank) {
            float winnerScore = currentScore;
            int32_t winnerId = currentId;
            // Every lane participates; inactive lists supply the sentinel, not an inactive shuffle source.
            for (int32_t mask = 16; mask > 0; mask >>= 1) {
                const float otherScore = asc_shfl_xor(winnerScore, mask, 32);
                const int32_t otherId = asc_shfl_xor(winnerId, mask, 32);
                if (otherScore > winnerScore || (otherScore == winnerScore && otherId < winnerId)) {
                    winnerScore = otherScore;
                    winnerId = otherId;
                }
            }
            if (lane == 0) {
                topScores[query * k + rank] = winnerId == kNoId ? -1.0F : winnerScore;
                topIds[query * k + rank] = winnerId == kNoId ? -1 : winnerId;
            }
            // Original document IDs are unique across disjoint shards, so only the winning lane advances.
            if (lane < shards && currentId == winnerId && winnerId != kNoId) {
                ++head;
                if (head < k) {
                    currentScore = partialScores[base + head];
                    currentId = static_cast<int32_t>(static_cast<uint64_t>(lane) * docsPerShard + partialIds[base + head]);
                } else {
                    currentScore = -1.0F;
                    currentId = kNoId;
                }
            }
        }
    }
}
}

extern "C" __global__ __aicore__ void bm25_top_k_merge(
    GM_ADDR partial_scores, GM_ADDR partial_ids, GM_ADDR top_scores, GM_ADDR top_ids,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    AscendC::Simt::VF_CALL<MergeTopKSimt>(
        AscendC::Simt::Dim3{kWarp, 1, 1}, reinterpret_cast<__gm__ const float *>(partial_scores),
        reinterpret_cast<__gm__ const int32_t *>(partial_ids), reinterpret_cast<__gm__ float *>(top_scores),
        reinterpret_cast<__gm__ int32_t *>(top_ids), data.queryCount, data.shardCount,
        data.docsPerShard, data.k, static_cast<uint32_t>(AscendC::GetBlockIdx()), data.blockNum);
}
