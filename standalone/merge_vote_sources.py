"""950 Merge唯一最高分快速路径；同分和异常路径保留原始精确归约。"""
from standalone import large_search_sources as base

simd=base.simd;r=base.r
FUSED=base.FUSED
MERGE='bm25_topk_merge_unique_vote'
VENDOR=MERGE
KERNEL='op_kernel/bm25_top_k_merge.cpp'
INCLUDE_OLD='#include "simt_api/device_warp_functions.h"'
INCLUDE_NEW=INCLUDE_OLD+'\n#include "simt_api/device_functions.h"'
REDUCE_OLD="""            float winnerScore = currentScore;
            int32_t winnerId = currentId;
            // Every lane participates; inactive lists supply the sentinel, not an inactive shuffle source.
            for (int32_t mask = 16; mask > 0; mask >>= 1) {
                const float otherScore = asc_shfl_xor(winnerScore, mask, 32);
                const int32_t otherId = asc_shfl_xor(winnerId, mask, 32);
                if (otherScore > winnerScore || (otherScore == winnerScore && otherId < winnerId)) {
                    winnerScore = otherScore;
                    winnerId = otherId;
                }
            }"""
REDUCE_NEW="""            float winnerScore = currentScore;
            int32_t winnerId = currentId;
            // 先做五次纯分数归约。主负载通常只有一个最高分，不再为这些轮次执行完整ID归约。
            for (int32_t mask = 16; mask > 0; mask >>= 1) {
                const float otherScore = asc_shfl_xor(winnerScore, mask, 32);
                if (otherScore > winnerScore) winnerScore = otherScore;
            }
            const uint32_t winners = asc_ballot(currentScore == winnerScore && currentId != kNoId);
            if (winners != 0 && (winners & (winners-1U)) == 0) {
                // 唯一bit的索引等于popc(winners-1)；把该lane的原始分数和ID广播给全warp。
                const uint32_t winnerLane = static_cast<uint32_t>(__popc(winners-1U));
                winnerScore = asc_shfl(currentScore, winnerLane, kWarp);
                winnerId = asc_shfl(currentId, winnerLane, kWarp);
            } else {
                // 同分、空榜单及所有非唯一情况逐字保留原算法，包括ID次序和带符号零分数位。
                winnerScore = currentScore;
                winnerId = currentId;
                for (int32_t mask = 16; mask > 0; mask >>= 1) {
                    const float otherScore = asc_shfl_xor(winnerScore, mask, 32);
                    const int32_t otherId = asc_shfl_xor(winnerId, mask, 32);
                    if (otherScore > winnerScore || (otherScore == winnerScore && otherId < winnerId)) {
                        winnerScore = otherScore;
                        winnerId = otherId;
                    }
                }
            }"""


def candidate_sources(original):
    sources=base.merge_sources(original)
    text=sources[KERNEL].decode('utf-8')
    text=simd.t.replace_once(text,INCLUDE_OLD,INCLUDE_NEW)
    text=simd.t.replace_once(text,REDUCE_OLD,REDUCE_NEW)
    sources[KERNEL]=text.encode('utf-8')
    return sources


# large_search.build_operator 的只读接口。
merge_sources=candidate_sources


def prepare(original,target,kind='merge'):
    r.require(kind=='merge','唯一赢家候选只构建Merge')
    metadata=base.prepare(original,target,'merge')
    sources=candidate_sources(original)
    for name,raw in sources.items():(target/name).write_bytes(raw)
    base.vendor_preset(target,VENDOR)
    metadata.update(status='merge_unique_vote_source_prepared_not_device_verified',vendor_name=VENDOR,
        source_sha256={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
        merge_fast_path='five score-only butterfly steps; one ballot and two broadcasts for a unique valid maximum',
        exact_fallback='all tied, empty or non-unique cases rerun the original score/ID butterfly bit-for-bit',
        scope='NPU Merge only. Reuse the accepted Warp Append leaf unchanged. No FP32 scoring, leaf, host, CPU, ABI or ranking-rule change. CANN compilation,36 merge checks and paired full-Search timing required.')
    simd.r.write_json(target/'bm25_project.json',metadata)
    return metadata