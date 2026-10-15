// Experimental self-developed striped local Top-K + cooperative 32-lane merge.
// Same Bm25TopK IR/signature; install as a separate vendor. Pending A5 validation.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"

namespace {
constexpr uint32_t kMaxThreads = 256;
constexpr uint32_t kMaxK = 32;
constexpr uint32_t kWarp = 32;
constexpr uint32_t kListsPerLane = kMaxThreads / kWarp;
constexpr int32_t kNoId = 2147483647;

__simt_vf__ __aicore__ LAUNCH_BOUND(kMaxThreads) inline void ParallelTopKSimt(
    __gm__ const float *scores, __gm__ float *topScores, __gm__ int32_t *topIds,
    __gm__ float *candidateScores, __gm__ int32_t *candidateIds, __gm__ uint32_t *badFlags,
    uint32_t queryCount, uint32_t docCount, uint32_t k, uint32_t partitions,
    uint32_t blockId, uint32_t blockNum)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    // Entire block has the same query/loop count, so every thread reaches both barriers.
    for (uint64_t query = blockId; query < queryCount; query += blockNum) {
        const uint64_t inputBase = query * docCount;
        const uint64_t candidateBase = query * partitions * k;
        const uint64_t flagBase = query * partitions;
        float bestScores[kMaxK];
        int32_t bestIds[kMaxK];
        for (uint32_t rank = 0; rank < k; ++rank) {
            bestScores[rank] = -1.0F;
            bestIds[rank] = kNoId;
        }
        uint32_t bad = 0;
        // Striped documents provide contiguous GM reads across adjacent lanes.
        for (uint64_t doc = thread; doc < docCount; doc += partitions) {
            const float score = scores[inputBase + doc];
            if (!(score >= 0.0F) || score > 3.402823466e+38F) {
                bad = 1;
                continue;
            }
            const int32_t id = static_cast<int32_t>(doc);
            for (uint32_t slot = 0; slot < k; ++slot) {
                if (score > bestScores[slot] || (score == bestScores[slot] && id < bestIds[slot])) {
                    for (uint32_t move = k - 1; move > slot; --move) {
                        bestScores[move] = bestScores[move - 1];
                        bestIds[move] = bestIds[move - 1];
                    }
                    bestScores[slot] = score;
                    bestIds[slot] = id;
                    break;
                }
            }
        }
        // Rank-major candidate layout: threads write contiguous elements at each rank.
        for (uint32_t rank = 0; rank < k; ++rank) {
            const uint64_t at = candidateBase + static_cast<uint64_t>(rank) * partitions + thread;
            candidateScores[at] = bestScores[rank];
            candidateIds[at] = bestIds[rank];
        }
        badFlags[flagBase + thread] = bad;
        asc_threadfence();
        asc_syncthreads();

        // Exactly one FULL warp merges this query. No inactive shuffle source lanes.
        if (thread < kWarp) {
            const uint32_t lane = thread;
            uint32_t queryBad = 0;
            for (uint32_t list = 0; list < kListsPerLane; ++list) {
                const uint32_t owner = lane + list * kWarp;
                if (owner < partitions) queryBad |= badFlags[flagBase + owner];
            }
            for (int32_t mask = 16; mask > 0; mask >>= 1)
                queryBad |= asc_shfl_xor(queryBad, mask, 32);
            uint32_t heads[kListsPerLane];
            for (uint32_t list = 0; list < kListsPerLane; ++list) heads[list] = 0;
            for (uint32_t rank = 0; rank < k; ++rank) {
                float localScore = -1.0F;
                int32_t localId = kNoId;
                int32_t localList = -1;
                for (uint32_t list = 0; list < kListsPerLane; ++list) {
                    const uint32_t owner = lane + list * kWarp;
                    if (owner >= partitions || heads[list] >= k) continue;
                    const uint64_t at = candidateBase + static_cast<uint64_t>(heads[list]) * partitions + owner;
                    const float score = candidateScores[at];
                    const int32_t id = candidateIds[at];
                    if (score > localScore || (score == localScore && id < localId)) {
                        localScore = score;
                        localId = id;
                        localList = static_cast<int32_t>(list);
                    }
                }
                float winnerScore = localScore;
                int32_t winnerId = localId;
                for (int32_t mask = 16; mask > 0; mask >>= 1) {
                    const float otherScore = asc_shfl_xor(winnerScore, mask, 32);
                    const int32_t otherId = asc_shfl_xor(winnerId, mask, 32);
                    if (otherScore > winnerScore || (otherScore == winnerScore && otherId < winnerId)) {
                        winnerScore = otherScore;
                        winnerId = otherId;
                    }
                }
                if (lane == 0) {
                    const uint64_t at = query * k + rank;
                    const bool invalid = queryBad != 0 || winnerId == kNoId;
                    topScores[at] = invalid ? -1.0F : winnerScore;
                    topIds[at] = invalid ? -1 : winnerId;
                }
                // IDs are unique; only the lane owning the selected list advances it.
                if (localList >= 0 && localId == winnerId)
                    ++heads[static_cast<uint32_t>(localList)];
            }
        }
        asc_threadfence();
        asc_syncthreads();
    }
}
}

extern "C" __global__ __aicore__ void bm25_top_k(
    GM_ADDR scores, GM_ADDR top_scores, GM_ADDR top_ids, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    if (workspace == nullptr) return; // Poisoned caller outputs make this fail checking.
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) return;
    const uint64_t candidates = static_cast<uint64_t>(data.queryCount) * data.partitionCount * data.k;
    auto *candidateScores = reinterpret_cast<__gm__ float *>(user);
    auto *candidateIds = reinterpret_cast<__gm__ int32_t *>(user + candidates * sizeof(float));
    auto *badFlags = reinterpret_cast<__gm__ uint32_t *>(user + candidates * (sizeof(float) + sizeof(int32_t)));
    AscendC::Simt::VF_CALL<ParallelTopKSimt>(
        AscendC::Simt::Dim3{data.partitionCount, 1, 1}, reinterpret_cast<__gm__ const float *>(scores),
        reinterpret_cast<__gm__ float *>(top_scores), reinterpret_cast<__gm__ int32_t *>(top_ids),
        candidateScores, candidateIds, badFlags, data.queryCount, data.docCount, data.k,
        data.partitionCount, static_cast<uint32_t>(AscendC::GetBlockIdx()), data.blockNum);
}
