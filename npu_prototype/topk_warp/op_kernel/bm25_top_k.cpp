// Self-developed full-warp sorting network + cached-head merge. Pending A5 compile/run.
// No private Top-K arrays: each lane retains one score/ID, each warp retains its best 32.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"

namespace {
constexpr uint32_t kMaxThreads = 256;
constexpr uint32_t kWarp = 32;
constexpr int32_t kNoId = 2147483647;

// SIMT helper annotation follows the group's A5 reference; comparator is self-developed.
__simt_callee__ __aicore__ inline bool Better(float lhs, int32_t lhsId, float rhs, int32_t rhsId)
{
    return lhs > rhs || (lhs == rhs && lhsId < rhsId);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kMaxThreads) inline void WarpTopKSimt(
    __gm__ const float *scores, __gm__ float *topScores, __gm__ int32_t *topIds,
    __gm__ float *candidateScores, __gm__ int32_t *candidateIds, __gm__ uint32_t *badFlags,
    uint32_t queryCount, uint32_t docCount, uint32_t k, uint32_t partitions,
    uint32_t blockId, uint32_t blockNum)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t lane = thread % kWarp;
    const uint32_t warpCount = partitions / kWarp;
    // Uniform query and tile loops: every launched thread reaches both block barriers.
    for (uint64_t query = blockId; query < queryCount; query += blockNum) {
        const uint64_t inputBase = query * docCount;
        const uint64_t candidateBase = query * partitions;
        float retainedScore = -1.0F;
        int32_t retainedId = kNoId;
        uint32_t bad = 0;
        for (uint64_t tile = 0; tile < docCount; tile += partitions) {
            const uint64_t doc = tile + thread;
            float tileScore = -1.0F;
            int32_t tileId = kNoId;
            if (doc < docCount) {
                const float value = scores[inputBase + doc];
                if (!(value >= 0.0F) || value > 3.402823466e+38F) {
                    bad = 1;
                } else {
                    tileScore = value;
                    tileId = static_cast<int32_t>(doc);
                }
            }
            // Sort the new 32 values worst-first. All 32 lanes, including padding, participate.
#pragma unroll
            for (uint32_t span = 2; span <= kWarp; span <<= 1) {
#pragma unroll
                for (uint32_t stride = span >> 1; stride > 0; stride >>= 1) {
                    const float otherScore = asc_shfl_xor(tileScore, static_cast<int32_t>(stride), 32);
                    const int32_t otherId = asc_shfl_xor(tileId, static_cast<int32_t>(stride), 32);
                    const bool wantBetter = ((lane & span) == 0) != ((lane & stride) == 0);
                    if ((wantBetter && Better(otherScore, otherId, tileScore, tileId)) ||
                        (!wantBetter && Better(tileScore, tileId, otherScore, otherId))) {
                        tileScore = otherScore;
                        tileId = otherId;
                    }
                }
            }
            // Retained values are best-first, the new tile worst-first: compare across the
            // two halves of a bitonic sequence, retain the best half, then merge-sort it.
            if (Better(tileScore, tileId, retainedScore, retainedId)) {
                retainedScore = tileScore;
                retainedId = tileId;
            }
#pragma unroll
            for (uint32_t stride = kWarp >> 1; stride > 0; stride >>= 1) {
                const float otherScore = asc_shfl_xor(retainedScore, static_cast<int32_t>(stride), 32);
                const int32_t otherId = asc_shfl_xor(retainedId, static_cast<int32_t>(stride), 32);
                const bool wantBetter = (lane & stride) == 0;
                if ((wantBetter && Better(otherScore, otherId, retainedScore, retainedId)) ||
                    (!wantBetter && Better(retainedScore, retainedId, otherScore, otherId))) {
                    retainedScore = otherScore;
                    retainedId = otherId;
                }
            }
        }
        // One contiguous best-first list of 32 entries per warp, at most 8 lists per block.
        candidateScores[candidateBase + thread] = retainedScore;
        candidateIds[candidateBase + thread] = retainedId;
        badFlags[candidateBase + thread] = bad;
        asc_threadfence();
        asc_syncthreads();

        if (thread < kWarp) {
            uint32_t queryBad = 0;
            for (uint32_t group = 0; group < kMaxThreads / kWarp; ++group) {
                const uint32_t owner = lane + group * kWarp;
                if (owner < partitions) queryBad |= badFlags[candidateBase + owner];
            }
            for (int32_t mask = 16; mask > 0; mask >>= 1)
                queryBad |= asc_shfl_xor(queryBad, mask, 32);
            uint32_t head = 0;
            float currentScore = -1.0F;
            int32_t currentId = kNoId;
            if (lane < warpCount) {
                currentScore = candidateScores[candidateBase + lane * kWarp];
                currentId = candidateIds[candidateBase + lane * kWarp];
            }
            for (uint32_t rank = 0; rank < k; ++rank) {
                float winnerScore = currentScore;
                int32_t winnerId = currentId;
                for (int32_t mask = 16; mask > 0; mask >>= 1) {
                    const float otherScore = asc_shfl_xor(winnerScore, mask, 32);
                    const int32_t otherId = asc_shfl_xor(winnerId, mask, 32);
                    if (Better(otherScore, otherId, winnerScore, winnerId)) {
                        winnerScore = otherScore;
                        winnerId = otherId;
                    }
                }
                if (lane == 0) {
                    const bool invalid = queryBad != 0 || winnerId == kNoId;
                    topScores[query*k + rank] = invalid ? -1.0F : winnerScore;
                    topIds[query*k + rank] = invalid ? -1 : winnerId;
                }
                // Only the owner of this document advances; other list heads stay in registers.
                if (lane < warpCount && currentId == winnerId && winnerId != kNoId) {
                    ++head;
                    if (head < kWarp) {
                        currentScore = candidateScores[candidateBase + lane*kWarp + head];
                        currentId = candidateIds[candidateBase + lane*kWarp + head];
                    } else {
                        currentScore = -1.0F;
                        currentId = kNoId;
                    }
                }
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
    if (workspace == nullptr) return;
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) return;
    const uint64_t candidates = static_cast<uint64_t>(data.queryCount) * data.partitionCount;
    auto *candidateScores = reinterpret_cast<__gm__ float *>(user);
    auto *candidateIds = reinterpret_cast<__gm__ int32_t *>(user + candidates*sizeof(float));
    auto *badFlags = reinterpret_cast<__gm__ uint32_t *>(user + candidates*(sizeof(float) + sizeof(int32_t)));
    AscendC::Simt::VF_CALL<WarpTopKSimt>(
        AscendC::Simt::Dim3{data.partitionCount, 1, 1}, reinterpret_cast<__gm__ const float *>(scores),
        reinterpret_cast<__gm__ float *>(top_scores), reinterpret_cast<__gm__ int32_t *>(top_ids),
        candidateScores, candidateIds, badFlags, data.queryCount, data.docCount, data.k,
        data.partitionCount, static_cast<uint32_t>(AscendC::GetBlockIdx()), data.blockNum);
}
