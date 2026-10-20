// Own range-owned query scoring over a resident inverted index. No float atomics.
// Pending actual CANN compilation and Ascend950 execution; no performance claim.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"

namespace {
constexpr uint32_t kThreads = 128;
constexpr uint32_t kDocTile = 2048;
constexpr uint32_t kScratchBytes = kDocTile * sizeof(float) + 64;
constexpr float kMaxFloat = 3.402823466e+38F;

__simt_callee__ __aicore__ inline int64_t LowerBound(
    __gm__ const int32_t *docs, int64_t begin, int64_t end, uint32_t target)
{
    while (begin < end) {
        const int64_t middle = begin + (end - begin) / 2;
        if (docs[middle] < static_cast<int32_t>(target)) begin = middle + 1;
        else end = middle;
    }
    return begin;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void QueryScoreSimt(
    __gm__ const int64_t *termOffsets, __gm__ const int32_t *postingDocs,
    __gm__ const float *postingTf, __gm__ const float *docNorm, __gm__ const float *idf,
    __gm__ const int64_t *queryOffsets, __gm__ const int32_t *queryTerms,
    __gm__ float *output, __ubuf__ float *scores,
    uint32_t queryCount, uint32_t docCount, uint32_t vocabSize, uint32_t postingCount,
    uint32_t queryStorageCount, uint32_t tilesPerQuery, uint32_t blockId, uint32_t blockNum)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    // TBuf base is aligned. bounds uses 32 bytes, control starts at the next 32-byte boundary.
    auto *bounds = reinterpret_cast<__ubuf__ int64_t *>(scores + kDocTile);
    auto *control = reinterpret_cast<__ubuf__ uint32_t *>(scores + kDocTile + 8);
    const uint64_t tasks = static_cast<uint64_t>(queryCount) * tilesPerQuery;
    for (uint64_t task = blockId; task < tasks; task += blockNum) {
        const uint32_t query = static_cast<uint32_t>(task / tilesPerQuery);
        const uint32_t firstDoc = static_cast<uint32_t>(task % tilesPerQuery) * kDocTile;
        const uint32_t lastDoc = firstDoc + kDocTile < docCount ? firstDoc + kDocTile : docCount;
        for (uint32_t i = thread; i < kDocTile; i += kThreads) scores[i] = 0.0F;
        if (thread == 0) {
            bounds[0] = queryOffsets[query];
            bounds[1] = queryOffsets[query + 1];
            control[0] = bounds[0] < 0 || bounds[1] < bounds[0] ||
                         static_cast<uint64_t>(bounds[1]) > queryStorageCount ? 1U : 0U;
        }
        asc_syncthreads();
        const int64_t queryBegin = bounds[0], queryEnd = bounds[1];
        const bool validQuery = control[0] == 0;
        asc_syncthreads(); // Every lane snapshots the decision before lane 0 can change control.
        if (validQuery) {
            for (int64_t token = queryBegin; token < queryEnd; ++token) {
                if (thread == 0) {
                    const int32_t term = queryTerms[token];
                    bounds[2] = bounds[3] = 0;
                    if (term < 0) control[0] = 1;
                    else if (static_cast<uint32_t>(term) < vocabSize) {
                        const int64_t begin = termOffsets[term], end = termOffsets[term + 1];
                        if (begin < 0 || end < begin || static_cast<uint64_t>(end) > postingCount)
                            control[0] = 1;
                        else {
                            bounds[2] = LowerBound(postingDocs, begin, end, firstDoc);
                            bounds[3] = LowerBound(postingDocs, bounds[2], end, lastDoc);
                        }
                    } // Unknown nonnegative terms have an empty span; never read their IDF.
                }
                asc_syncthreads();
                const bool validSpan = control[0] == 0;
                const int64_t spanBegin = bounds[2], spanEnd = bounds[3];
                asc_syncthreads(); // Protect the uniform decision from per-lane error writes.
                if (!validSpan) break;
                const int32_t term = queryTerms[token];
                for (int64_t pos = spanBegin + thread; pos < spanEnd; pos += kThreads) {
                    const int32_t doc = postingDocs[pos];
                    if (doc < static_cast<int32_t>(firstDoc) || doc >= static_cast<int32_t>(lastDoc)) {
                        atomicAdd(&control[0], 1U);
                        continue;
                    }
                    const float tf = postingTf[pos], norm = docNorm[doc], weight = idf[term];
                    if (!(tf >= 1.0F && tf <= 16777216.0F && norm > 0.0F && norm <= kMaxFloat &&
                          weight >= 0.0F && weight <= kMaxFloat)) {
                        atomicAdd(&control[0], 1U);
                        continue;
                    }
                    const float numerator = tf * 2.2F;
                    const float contribution = (weight * numerator) / (tf + norm);
                    const uint32_t localDoc = static_cast<uint32_t>(doc) - firstDoc;
                    const float total = scores[localDoc] + contribution;
                    if (!(contribution >= 0.0F && contribution <= kMaxFloat && total >= 0.0F && total <= kMaxFloat))
                        atomicAdd(&control[0], 1U);
                    else scores[localDoc] = total;
                    // Validated posting IDs are unique per term: one writer per localDoc.
                }
                asc_syncthreads(); // Complete this term before the next, including repeated terms.
                const bool validAccumulation = control[0] == 0;
                asc_syncthreads(); // All lanes read before next term's lane-0 control writes.
                if (!validAccumulation) break;
            }
        }
        const bool invalid = control[0] != 0;
        const uint64_t base = static_cast<uint64_t>(query) * docCount;
        for (uint32_t doc = firstDoc + thread; doc < lastDoc; doc += kThreads)
            output[base + doc] = invalid ? -1.0F : scores[doc - firstDoc];
        // Empty queries/ranges overwrite all output positions with zero, never reuse old scores.
        asc_syncthreads();
    }
}
}

extern "C" __global__ __aicore__ void bm25_query_score(
    GM_ADDR term_offsets, GM_ADDR posting_doc_ids, GM_ADDR posting_tf, GM_ADDR doc_norm,
    GM_ADDR idf, GM_ADDR query_offsets, GM_ADDR query_term_ids, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, kScratchBytes);
    auto local = buffer.Get<float>();
    AscendC::Simt::VF_CALL<QueryScoreSimt>(
        AscendC::Simt::Dim3{kThreads, 1, 1}, reinterpret_cast<__gm__ const int64_t *>(term_offsets),
        reinterpret_cast<__gm__ const int32_t *>(posting_doc_ids), reinterpret_cast<__gm__ const float *>(posting_tf),
        reinterpret_cast<__gm__ const float *>(doc_norm), reinterpret_cast<__gm__ const float *>(idf),
        reinterpret_cast<__gm__ const int64_t *>(query_offsets), reinterpret_cast<__gm__ const int32_t *>(query_term_ids),
        reinterpret_cast<__gm__ float *>(output), reinterpret_cast<__ubuf__ float *>(local.GetPhyAddr()),
        data.queryCount, data.docCount, data.vocabSize, data.postingCount, data.queryStorageCount,
        data.tilesPerQuery, static_cast<uint32_t>(AscendC::GetBlockIdx()), data.blockNum);
}
