"""仅在NPU叶算子内以warp自主分配替代全块前缀和同步的候选。"""
from standalone import sparse_pack_sources as base

VENDOR='bm25_fused_score_topk_warp_pipeline_large'
KERNEL=base.KERNEL
replace=base.base.simd.t.replace_once

PACK_MARKER="""template <bool BitmapReady>
__simt_callee__ __aicore__ inline void PackForSimd("""
WARP_ALLOCATE="""// 每warp一次原子分配连续段；各线程仍独占一段并保持文档号升序。
__simt_callee__ __aicore__ inline uint32_t WarpAllocate(
    uint32_t count, __ubuf__ uint32_t *cursor)
{
    const uint32_t lane = AscendC::Simt::GetThreadIdx<0>()%kWarp;
    uint32_t sum = count;
#pragma unroll
    for (int offset = 1; offset < kWarp; offset <<= 1) {
        const uint32_t other = __shfl_up(sum, offset, kWarp);
        if (lane >= static_cast<uint32_t>(offset)) sum += other;
    }
    uint32_t warpBegin = 0;
    if (lane == kWarp-1 && sum != 0) warpBegin = atomicAdd(cursor, sum);
    warpBegin = asc_shfl(warpBegin, static_cast<int32_t>(kWarp-1), kWarp);
    return warpBegin+sum-count;
}

"""
PACK_PREFIX_OLD="""    uint32_t begin, unused, total, bad;
    PrefixPair(positive, invalid, scratch, begin, unused, total, bad);
    if (tid == 0) {
        scratch[kControl] = total;
        scratch[kControl+1] = bad != 0 ? 2U : (total > k && total <= kCapacity ? 1U : 0U);
    }
    scratch[kOwnerBegin+tid] = begin;
    scratch[kOwnerCount+tid] = positive;
    if (bad == 0 && total > k && total <= kCapacity) {"""
PACK_PREFIX_NEW="""    uint32_t begin, unused, total = 0, bad = 0;
    if constexpr (BitmapReady) {
        begin = WarpAllocate(positive, scratch+kControl);
    } else {
        PrefixPair(positive, invalid, scratch, begin, unused, total, bad);
    }
    if (tid == 0) {
        if constexpr (!BitmapReady) {
            scratch[kControl] = total;
            scratch[kControl+1] = bad != 0 ? 2U : (total > k && total <= kCapacity ? 1U : 0U);
        }
        // Collect的严格候选游标和越界诊断；与阈值prefix/rank及全局ID基址不重叠。
        scratch[kControl+4] = 0;
        scratch[kControl+6] = 0;
    }
    scratch[kOwnerBegin+tid] = begin;
    scratch[kOwnerCount+tid] = positive;
    // 原子分配时总数尚未可知；只写完整落在容量内的段，最终统一决定是否需要回退。
    const bool bounded = BitmapReady ? begin+positive <= kCapacity : bad == 0 && total > k && total <= kCapacity;
    if (bounded) {"""
PACK_END_OLD="""    }
    asc_syncthreads();
}

/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd."""
PACK_END_NEW="""    }
    asc_syncthreads();
    if constexpr (BitmapReady) {
        total = scratch[kControl];
        if (tid == 0) scratch[kControl+1] = total > k && total <= kCapacity ? 1U : 0U;
    }
}

/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd."""
ZERO_NEW="""        for (uint32_t doc = thread; doc < width; doc += kThreads) scores[doc] = 0.0F;
        if (thread == 0) bitmap[kControl] = 0;
        bitmap[kOwnerBegin+thread] = 0;
        bitmap[kOwnerCount+thread] = 0;
        asc_syncthreads();"""
COLLECT_INIT_OLD="""        const uint32_t strictCount = k - remaining;
        if (thread < kWarp) {
            scratch[kCandidateKeys + thread] = 0xffffffffU;
            scratch[kCandidateIds + thread] = kNoId;
        }
        asc_syncthreads();"""
COLLECT_INIT_NEW="""        const uint32_t strictCount = k - remaining;
        if (thread < kWarp) {
            scratch[kCandidateKeys + thread] = 0xffffffffU;
            scratch[kCandidateIds + thread] = kNoId;
        }
        // 只需补1个阈值同分候选时，各owner对固定槽位的文档号取最小值。
        if (remaining == 1 && thread == strictCount) scratch[kCandidateKeys + thread] = prefix;
        asc_syncthreads();"""
COLLECT_PREFIX_OLD="""        uint32_t strictBegin, ignoredRank, strictTotal, ignoredTotal;
        PrefixPair(privateStrict, 0, scratch, strictBegin, ignoredRank, strictTotal, ignoredTotal);
        if (strictTotal != strictCount) {
            if (thread < k) { topScores[query*k + thread] = 0xbf800000U; topIds[query*k + thread] = -1; }
            asc_syncthreads();
            return; // uniform failure, never return a partial answer
        }
        // Only owners of strict candidates rescan. Disjoint ranges come from the
        // single whole-row prefix above; stop at the last local strict candidate.
        if (privateStrict != 0) {
            uint32_t written = 0;
            for (uint64_t doc = sourceBegin; doc < sourceEnd; doc += sourceStep) {
                const uint32_t key = ScoreKey(ReadBits(scores, scratch, base, doc, compact));
                if (key < prefix) {
                    const uint32_t slot = strictBegin + written;
                    scratch[kCandidateKeys + slot] = key;
                    scratch[kCandidateIds + slot] = ReadId(scratch, doc, compact);
                    if (++written == privateStrict) break;
                }
            }
        }
        asc_syncthreads(); // all strict candidates visible before head merging/final sorting"""
COLLECT_PREFIX_NEW="""        const uint32_t strictBegin = WarpAllocate(privateStrict, scratch+kControl+4);
        // 各warp段可乱序到达，最终32槽精确排序不依赖段间顺序；槽位必须保持有界。
        if (privateStrict != 0) {
            if (strictBegin+privateStrict <= strictCount) {
                uint32_t written = 0;
                for (uint64_t doc = sourceBegin; doc < sourceEnd; doc += sourceStep) {
                    const uint32_t key = ScoreKey(ReadBits(scores, scratch, base, doc, compact));
                    if (key < prefix) {
                        const uint32_t slot = strictBegin + written;
                        scratch[kCandidateKeys + slot] = key;
                        scratch[kCandidateIds + slot] = ReadId(scratch, doc, compact);
                        if (++written == privateStrict) break;
                    }
                }
            } else {
                atomicAdd(scratch+kControl+6, 1U);
            }
        }
        if (remaining == 1 && equalHead != kNoId)
            atomicMin(scratch+kCandidateIds+strictCount, equalHead);
        asc_syncthreads(); // 所有严格候选、游标及同分最小ID在此统一可见
        const uint32_t strictTotal = scratch[kControl+4];
        if (strictTotal != strictCount || scratch[kControl+6] != 0 ||
            (remaining == 1 && scratch[kCandidateIds+strictCount] >= docCount)) {
            if (thread < k) { topScores[query*k + thread] = 0xbf800000U; topIds[query*k + thread] = -1; }
            asc_syncthreads();
            return;
        }
        if (remaining == 1) {
            WriteSelected(scores, topScores, topIds, scratch, base, query, k, docCount, true);
            return;
        }"""
COMMON=((PACK_MARKER,WARP_ALLOCATE+PACK_MARKER),(PACK_PREFIX_OLD,PACK_PREFIX_NEW),
    (PACK_END_OLD,PACK_END_NEW),(base.base.ZERO_NEW,ZERO_NEW))
COLLECT=((COLLECT_INIT_OLD,COLLECT_INIT_NEW),(COLLECT_PREFIX_OLD,COLLECT_PREFIX_NEW))
COLLECT_MARKER='__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedCollectSimt('


def candidate_sources(original):
    sources=base.candidate_sources(original)
    text=sources[KERNEL].decode('utf-8')
    for old,new in COMMON:text=replace(text,old,new)
    begin=text.index(COLLECT_MARKER);head,tail=text[:begin],text[begin:]
    for old,new in COLLECT:tail=replace(tail,old,new)
    sources[KERNEL]=(head+tail).encode('utf-8')
    return sources


def prepare(original,target):
    metadata=base.prepare(original,target)
    sources=candidate_sources(original)
    for name,raw in sources.items():(target/name).write_bytes(raw)
    preset=target/'CMakePresets.json';data=base.base.simd.json.loads(preset.read_text(encoding='utf-8'))
    changed=0
    for item in data.get('configurePresets',[]):
        variables=item.get('cacheVariables',{});value=variables.get('vendor_name')
        if isinstance(value,dict) and value.get('value')==base.VENDOR:value['value']=VENDOR;changed+=1
        elif value==base.VENDOR:variables['vendor_name']=VENDOR;changed+=1
    base.base.simd.r.require(changed>0,'warp流水候选vendor配置缺失')
    base.base.simd.r.write_json(preset,data)
    metadata.update(status='warp_pipeline_source_prepared_not_device_verified',vendor_name=VENDOR,
        source_sha256={name:base.base.simd.hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
        npu_execution='8 autonomous warps: warp shuffle prefix plus bounded UB atomic segment allocation in Pack and strict Collect; atomic minimum ID when exactly one threshold-score candidate is needed',
        scope='NPU-only candidate derived from verified Sparse Pack. No CPU source/build/binary/configuration change. FP32 scoring, exact threshold, tie rule, dense fallback, global IDs, tree, host, ABI and UB allocation unchanged. Device compile, exactness and timing required; no full-card utilization claim.')
    base.base.simd.r.write_json(target/'bm25_project.json',metadata)
    return metadata
