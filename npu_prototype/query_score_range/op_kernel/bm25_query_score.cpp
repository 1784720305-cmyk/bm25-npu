// Isolated short-query experiment: contiguous article owners and monotone posting cursors.
// Reuses the accepted FP32 formula/term order; no float atomics, no Top-K changes.
// Source only: requires real CANN compilation, generated-header review and A5 acceptance.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"

namespace {
constexpr uint32_t kThreads=128, kWarp=32, kTile=2048, kTerms=64;
// Explicit TBuf layout in uint32 words; 64-bit cursor arrays begin on 32-byte boundaries.
constexpr uint32_t kCursor=kTile, kEnds=kCursor+2*kTerms, kIds=kEnds+2*kTerms;
constexpr uint32_t kTotals=kIds+kTerms, kControl=kTotals+8;
constexpr uint32_t kScratchBytes=(kControl+16)*sizeof(uint32_t); // 9568, aligned to 32 bytes
constexpr float kMaxFloat=3.402823466e+38F;

__simt_callee__ __aicore__ inline int64_t LowerBound(
    __gm__ const int32_t *docs,int64_t begin,int64_t end,uint32_t target)
{
    while (begin<end) {
        const int64_t mid=begin+(end-begin)/2;
        if (docs[mid]<static_cast<int32_t>(target)) begin=mid+1;
        else end=mid;
    }
    return begin;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void RangeQueryScoreSimt(
    __gm__ const int64_t *termOffsets,__gm__ const int32_t *postingDocs,
    __gm__ const float *postingTf,__gm__ const float *docNorm,__gm__ const float *idf,
    __gm__ const int64_t *queryOffsets,__gm__ const int32_t *queryTerms,
    __gm__ float *output,__ubuf__ uint32_t *scratch,
    uint32_t queryCount,uint32_t docCount,uint32_t vocabSize,uint32_t postingCount,
    uint32_t queryStorageCount,uint32_t partitions,uint32_t blockId,uint32_t blockNum)
{
    const uint32_t thread=AscendC::Simt::GetThreadIdx<0>();
    auto *scores=reinterpret_cast<__ubuf__ float *>(scratch);
    auto *cursors=reinterpret_cast<__ubuf__ int64_t *>(scratch+kCursor);
    auto *ends=reinterpret_cast<__ubuf__ int64_t *>(scratch+kEnds);
    auto *terms=reinterpret_cast<__ubuf__ int32_t *>(scratch+kIds);
    auto *control=scratch+kControl;
    auto *bounds=reinterpret_cast<__ubuf__ int64_t *>(control+8);
    const uint64_t tasks=static_cast<uint64_t>(queryCount)*partitions;
    for (uint64_t task=blockId;task<tasks;task+=blockNum) {
        const uint32_t query=static_cast<uint32_t>(task/partitions);
        const uint32_t part=static_cast<uint32_t>(task%partitions);
        const uint32_t first=static_cast<uint32_t>(static_cast<uint64_t>(docCount)*part/partitions);
        const uint32_t last=static_cast<uint32_t>(static_cast<uint64_t>(docCount)*(part+1)/partitions);
        const uint64_t base=static_cast<uint64_t>(query)*docCount;
        if (thread==0) {
            bounds[0]=queryOffsets[query]; bounds[1]=queryOffsets[query+1];
            const bool valid=bounds[0]>=0 && bounds[1]>=bounds[0] &&
                static_cast<uint64_t>(bounds[1])<=queryStorageCount && bounds[1]-bounds[0]<=kTerms;
            control[0]=valid?0U:1U;
            control[1]=valid?static_cast<uint32_t>(bounds[1]-bounds[0]):0U;
        }
        asc_syncthreads();
        const int64_t queryBegin=bounds[0];
        const uint32_t count=control[1]; const bool invalidQuery=control[0]!=0;
        asc_syncthreads();
        if (invalidQuery) {
            for (uint32_t doc=first+thread;doc<last;doc+=kThreads) output[base+doc]=-1.0F;
            asc_syncthreads(); continue;
        }
        // Each active term gets its own lane for the initial two searches. Afterwards
        // its cursor only moves forward, across all tiles owned by this range.
        if (thread<count) {
            const int32_t term=queryTerms[queryBegin+thread];
            terms[thread]=-1; cursors[thread]=ends[thread]=0;
            if (term<0) atomicAdd(&control[0],1U);
            else if (static_cast<uint32_t>(term)<vocabSize) {
                const int64_t begin=termOffsets[term],end=termOffsets[term+1];
                if (begin<0 || end<begin || static_cast<uint64_t>(end)>postingCount) atomicAdd(&control[0],1U);
                else {
                    const int64_t low=LowerBound(postingDocs,begin,end,first);
                    cursors[thread]=low; ends[thread]=LowerBound(postingDocs,low,end,last); terms[thread]=term;
                }
            }
        }
        asc_syncthreads();
        const bool badSpans=control[0]!=0; asc_syncthreads();
        if (!badSpans) {
            for (uint32_t tile=first;tile<last;tile+=kTile) {
                const uint32_t stop=last-tile<kTile?last:tile+kTile;
                for (uint32_t i=thread;i<kTile;i+=kThreads) scores[i]=0.0F;
                asc_syncthreads();
                for (uint32_t slot=0;slot<count;++slot) {
                    const int64_t begin=cursors[slot],end=ends[slot]; const int32_t term=terms[slot];
                    uint32_t consumed=0;
                    if (term>=0) {
                        for (int64_t pos=begin+thread;pos<end;pos+=kThreads) {
                            const int32_t doc=postingDocs[pos];
                            if (doc>=static_cast<int32_t>(stop)) break; // sorted list: this lane is done for this tile
                            ++consumed;
                            if (doc<static_cast<int32_t>(tile)) { atomicAdd(&control[0],1U); continue; }
                            const float tf=postingTf[pos],norm=docNorm[doc],weight=idf[term];
                            if (!(tf>=1.0F && tf<=16777216.0F && norm>0.0F && norm<=kMaxFloat &&
                                  weight>=0.0F && weight<=kMaxFloat)) { atomicAdd(&control[0],1U); continue; }
                            const float numerator=tf*2.2F;
                            const float contribution=(weight*numerator)/(tf+norm);
                            const uint32_t local=static_cast<uint32_t>(doc)-tile;
                            const float total=scores[local]+contribution;
                            if (!(contribution>=0.0F && contribution<=kMaxFloat && total>=0.0F && total<=kMaxFloat))
                                atomicAdd(&control[0],1U);
                            else scores[local]=total;
                        }
                    }
                    // Integer warp sums count the contiguous consumed posting prefix.
                    // No racing quota and no float atomics; duplicate query words have independent cursors.
#pragma unroll
                    for (int32_t offset=16;offset>0;offset>>=1)
                        consumed+=asc_shfl_xor(consumed,offset,32);
                    if (thread%kWarp==0) scratch[kTotals+thread/kWarp]=consumed;
                    asc_syncthreads(); // all score additions for this term and four warp totals are complete
                    if (thread==0) {
                        uint32_t total=0;
#pragma unroll
                        for (uint32_t warp=0;warp<kThreads/kWarp;++warp) total+=scratch[kTotals+warp];
                        if (static_cast<uint64_t>(end-begin)<total) atomicAdd(&control[0],1U);
                        else cursors[slot]=begin+total;
                    }
                    asc_syncthreads();
                    const bool failed=control[0]!=0; asc_syncthreads();
                    if (failed) break;
                }
                const bool failedTile=control[0]!=0; asc_syncthreads();
                if (failedTile) break;
                for (uint32_t doc=tile+thread;doc<stop;doc+=kThreads) output[base+doc]=scores[doc-tile];
                asc_syncthreads();
            }
        }
        // Full overwrite for malformed requests/data. A valid range must consume every
        // referenced posting before its last article; empty queries write all zeros above.
        if (thread<count && cursors[thread]!=ends[thread]) atomicAdd(&control[0],1U);
        asc_syncthreads();
        const bool failedRange=control[0]!=0; asc_syncthreads();
        if (failedRange)
            for (uint32_t doc=first+thread;doc<last;doc+=kThreads) output[base+doc]=-1.0F;
        asc_syncthreads(); // protect UB/cursors before this block takes the next task
    }
}
}

extern "C" __global__ __aicore__ void bm25_query_score(
    GM_ADDR term_offsets,GM_ADDR posting_doc_ids,GM_ADDR posting_tf,GM_ADDR doc_norm,
    GM_ADDR idf,GM_ADDR query_offsets,GM_ADDR query_term_ids,GM_ADDR output,GM_ADDR workspace,GM_ADDR tiling)
{
    GET_TILING_DATA(data,tiling);
    AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer,kScratchBytes); auto local=buffer.Get<uint32_t>();
    AscendC::Simt::VF_CALL<RangeQueryScoreSimt>(AscendC::Simt::Dim3{kThreads,1,1},
        reinterpret_cast<__gm__ const int64_t *>(term_offsets),reinterpret_cast<__gm__ const int32_t *>(posting_doc_ids),
        reinterpret_cast<__gm__ const float *>(posting_tf),reinterpret_cast<__gm__ const float *>(doc_norm),
        reinterpret_cast<__gm__ const float *>(idf),reinterpret_cast<__gm__ const int64_t *>(query_offsets),
        reinterpret_cast<__gm__ const int32_t *>(query_term_ids),reinterpret_cast<__gm__ float *>(output),
        reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr()),data.queryCount,data.docCount,data.vocabSize,
        data.postingCount,data.queryStorageCount,data.partitionsPerQuery,static_cast<uint32_t>(AscendC::GetBlockIdx()),data.blockNum);
}
