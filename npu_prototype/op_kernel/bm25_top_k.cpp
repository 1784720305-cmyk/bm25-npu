// Own insertion Top-K; exact FP32 ties use ascending document ID.
// One SIMT thread per query. Pending A5 compilation/execution; not a performance claim.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/device_warp_functions.h"

namespace {
constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxK = 32;
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void TopKSimt(
    __gm__ const float *scores, __gm__ float *topScores, __gm__ int32_t *topIds,
    uint32_t queryCount, uint32_t docCount, uint32_t k, uint32_t blockId, uint32_t blockNum)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    const uint64_t stride = static_cast<uint64_t>(blockNum) * kThreads;
    for (uint64_t query = static_cast<uint64_t>(blockId) * kThreads + thread;
         query < queryCount; query += stride) {
        const uint64_t inputBase = query * docCount;
        const uint64_t outputBase = query * k;
        float bestScores[kMaxK];
        int32_t bestIds[kMaxK];
        for (uint32_t i = 0; i < k; ++i) {
            bestScores[i] = -1.0F; // All valid BM25 scores are nonnegative, including zero.
            bestIds[i] = 2147483647;
        }
        bool invalid = false;
        for (uint32_t doc = 0; doc < docCount; ++doc) {
            const float score = scores[inputBase + doc];
            if (!(score >= 0.0F) || score > 3.402823466e+38F) {
                invalid = true; // Reject negatives, NaN and +infinity without silently skipping documents.
                break;
            }
            const auto id = static_cast<int32_t>(doc);
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
        for (uint32_t i = 0; i < k; ++i) {
            topScores[outputBase + i] = invalid ? -1.0F : bestScores[i];
            topIds[outputBase + i] = invalid ? -1 : bestIds[i];
        }
    }
}
}

extern "C" __global__ __aicore__ void bm25_top_k(
    GM_ADDR scores, GM_ADDR top_scores, GM_ADDR top_ids, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    AscendC::Simt::VF_CALL<TopKSimt>(
        AscendC::Simt::Dim3{kThreads, 1, 1}, reinterpret_cast<__gm__ const float *>(scores),
        reinterpret_cast<__gm__ float *>(top_scores), reinterpret_cast<__gm__ int32_t *>(top_ids),
        data.queryCount, data.docCount, data.k, static_cast<uint32_t>(AscendC::GetBlockIdx()), data.blockNum);
}
