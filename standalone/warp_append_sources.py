"""950专用warp聚合稀疏追加：评分直接产出无序紧凑ID，删除正常路径的位图Pack。"""
from standalone import sparse_pack_sources as base

simd=base.base.simd
VENDOR='bm25_fused_score_topk_warp_append_large'
KERNEL=base.KERNEL
SCORE_MARKER='__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedScorePackSimt('
COLLECT_MARKER='__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedCollectSimt('
PACK_MARKER='template <bool BitmapReady>\n__simt_callee__ __aicore__ inline void PackForSimd('


def pack_function(text):
    """Replace the complete generated function, never just PACK_NEW's prefix."""
    start=text.index(PACK_MARKER)
    body=text.index('{',start);depth=1;end=body+1
    while depth and end<len(text):
        depth += (text[end]=='{')-(text[end]=='}');end+=1
    simd.r.require(depth==0,'Pack function braces are unbalanced')
    return text[start:end]

DIRECT_PACK="""template <bool BitmapReady>
__simt_callee__ __aicore__ inline void PackForSimd(
    __ubuf__ const uint32_t *scores, __ubuf__ uint32_t *scratch, uint32_t width, uint32_t k)
{
    const uint32_t tid = AscendC::Simt::GetThreadIdx<0>();
    if constexpr (BitmapReady) {
        // 正常评分已用warp聚合追加局部文档ID；这里只补连续分数，不再读取位图或做块前缀。
        const uint32_t total = scratch[kControl];
        if (tid == 0) {
            scratch[kControl+1] = total > k && total <= kCapacity ? 1U : 0U;
            scratch[kControl+4] = 1U; // 无序、按线程跨步访问的直接追加布局
        }
        if (total > k && total <= kCapacity) {
            for (uint32_t pos = tid; pos < total; pos += kThreads) {
                const uint32_t doc = scratch[kPositiveIds+pos];
                scratch[kPositiveBits+pos] = scores[doc];
            }
        }
        asc_syncthreads();
    } else {
        // debug=2原始分数检查保留已验收的稠密验证、保序Pack和Sparse发射。
        uint32_t positive = 0, invalid = 0, lowMask = 0, highMask = 0, ordinal = 0;
        for (uint32_t doc = tid; doc < width; doc += kThreads, ++ordinal) {
            const uint32_t bits = scores[doc];
            invalid |= !ValidBits(bits);
            if ((bits & 0x7fffffffU) != 0) {
                ++positive;
                if (ordinal < 32) lowMask |= 1U << ordinal;
                else highMask |= 1U << (ordinal-32);
            }
        }
        uint32_t begin, unused, total, bad;
        PrefixPair(positive, invalid, scratch, begin, unused, total, bad);
        if (tid == 0) {
            scratch[kControl] = total;
            scratch[kControl+1] = bad != 0 ? 2U : (total > k && total <= kCapacity ? 1U : 0U);
            scratch[kControl+4] = 0U; // owner连续段、文档号升序布局
        }
        scratch[kOwnerBegin+tid] = begin;
        scratch[kOwnerCount+tid] = positive;
        if (bad == 0 && total > k && total <= kCapacity) {
            uint32_t written = 0;
            for (uint32_t half = 0; half < 2 && written < positive; ++half) {
                uint32_t remaining = half == 0 ? lowMask : highMask;
                while (remaining != 0 && written < positive) {
                    const uint32_t lowest = remaining & (0U-remaining);
                    const uint32_t bit = static_cast<uint32_t>(__popc(lowest-1U));
                    const uint32_t pos = half*32+bit;
                    const uint32_t doc = tid+pos*kThreads;
                    scratch[kPositiveBits+begin+written] = scores[doc];
                    scratch[kPositiveIds+begin+written] = doc;
                    ++written;
                    remaining ^= lowest;
                }
            }
        }
        asc_syncthreads();
    }
}"""

ZERO_OLD=base.base.ZERO_NEW
ZERO_NEW="""        for (uint32_t doc = thread; doc < width; doc += kThreads) scores[doc] = 0.0F;
        if (thread == 0) bitmap[kControl] = 0; // 本任务直接追加游标
        asc_syncthreads();"""

LOOP_HEADER_OLD="""                for (int64_t pos = begins[slot]+thread; pos < ends[slot]; pos += kThreads) {"""
LOOP_HEADER_NEW="""                // 每个warp具有相同迭代次数，ballot/shuffle始终在warp完整汇合点执行。
                const uint32_t lane = thread%kWarp;
                const int64_t warpBegin = begins[slot]+(thread/kWarp)*kWarp;
                const int64_t span = ends[slot]-warpBegin;
                const uint32_t iterations = span > 0 ? static_cast<uint32_t>((span+kThreads-1)/kThreads) : 0U;
                int64_t pos = warpBegin+lane;
                for (uint32_t iteration = 0; iteration < iterations; ++iteration, pos += kThreads) {
                    const bool active = pos < ends[slot];
                    uint32_t local = 0;
                    bool firstPositive = false;
                    if (active) {"""
LOCAL_OLD="""                    const uint32_t local = static_cast<uint32_t>(doc)-first;"""
LOCAL_NEW="""                    local = static_cast<uint32_t>(doc)-first;"""
VALIDATE_OLD="""                    const int32_t doc = postingDocs[pos];
                    if (doc < static_cast<int32_t>(first) || doc >= static_cast<int32_t>(last)) {
                        atomicAdd(&control[0], 1U); continue;
                    }
                    const float tf = postingTf[pos], norm = docNorm[doc], weight = idf[term];
                    if (!(tf>=1.0F && tf<=16777216.0F && norm>0.0F && norm<=kMaxFloat &&
                          weight>=0.0F && weight<=kMaxFloat)) { atomicAdd(&control[0], 1U); continue; }
                    const float numerator = tf*2.2F;"""
VALIDATE_NEW="""                    const int32_t doc = postingDocs[pos];
                    if (doc < static_cast<int32_t>(first) || doc >= static_cast<int32_t>(last)) {
                        atomicAdd(&control[0], 1U);
                    } else {
                    const float tf = postingTf[pos], norm = docNorm[doc], weight = idf[term];
                    if (!(tf>=1.0F && tf<=16777216.0F && norm>0.0F && norm<=kMaxFloat &&
                          weight>=0.0F && weight<=kMaxFloat)) { atomicAdd(&control[0], 1U); }
                    else {
                    const float numerator = tf*2.2F;"""
BITMAP_OLD="""                        // 同词 posting 的文档唯一，词间已有屏障；只在文档首次由零变正时置位。
                        if (previous == 0.0F && total > 0.0F) {
                            // Scoring threads own posting positions, not document residues.
                            // Distinct writers can set different bits of the SAME owner word.
                            const uint32_t owner = local%kThreads;
                            const uint32_t part = local/kThreads;
                            if (part < 32) asc_atomic_or(bitmap+kOwnerBegin+owner, 1U << part);
                            else asc_atomic_or(bitmap+kOwnerCount+owner, 1U << (part-32));
                        }"""
BITMAP_NEW="""                        // 同词文档唯一、词间有屏障；首次由零变正的文档至多追加一次。
                        if (previous == 0.0F && total > 0.0F) firstPositive = true;"""
LOOP_END_OLD="""                    }
                }
            }
            asc_syncthreads();"""
LOOP_END_NEW="""                    }
                    }
                    }
                    }
                    // 一次warp原子分配替代每个正分一次原子位图更新。槽位总数包含有界数组外的溢出。
                    const uint32_t hits = asc_ballot(static_cast<int32_t>(firstPositive));
                    if (hits != 0) {
                        const uint32_t lower = (1U << lane)-1U;
                        const uint32_t rank = static_cast<uint32_t>(__popc(hits & lower));
                        uint32_t warpBase = 0;
                        if (lane == 0) warpBase = atomicAdd(bitmap+kControl, static_cast<uint32_t>(__popc(hits)));
                        warpBase = asc_shfl(warpBase, 0, kWarp);
                        if (firstPositive) {
                            const uint32_t slot = warpBase+rank;
                            if (slot < kCapacity) bitmap[kPositiveIds+slot] = local;
                        }
                    }
                }
            }
            asc_syncthreads();"""
COMMON=((ZERO_OLD,ZERO_NEW),(LOOP_HEADER_OLD,LOOP_HEADER_NEW),(VALIDATE_OLD,VALIDATE_NEW),(LOCAL_OLD,LOCAL_NEW),
    (BITMAP_OLD,BITMAP_NEW),(LOOP_END_OLD,LOOP_END_NEW))

SOURCE_OLD="""    const bool compact = true;
    const uint64_t sourceBegin = scratch[kOwnerBegin+thread];
    const uint64_t sourceEnd = sourceBegin + scratch[kOwnerCount+thread], sourceStep = 1;"""
SOURCE_NEW="""    const bool compact = true;
    const bool direct = scratch[kControl+4] != 0;
    const uint64_t sourceBegin = direct ? thread : scratch[kOwnerBegin+thread];
    const uint64_t sourceEnd = direct ? scratch[kControl] : sourceBegin+scratch[kOwnerCount+thread];
    const uint64_t sourceStep = direct ? kThreads : 1;"""
EQUAL_HEAD_OLD="""            if (key == prefix && equalHead == kNoId) { equalHead = ReadId(scratch, doc, compact); equalPosition = doc; }"""
EQUAL_HEAD_NEW="""            if (key == prefix) {
                const uint32_t id = ReadId(scratch, doc, compact);
                if (id < equalHead) equalHead = id; // 直接追加无需有序，先取本线程最小同分ID
            }"""
EQUAL_DECL_OLD="""        uint64_t equalPosition = 0;
"""
EQUAL_DECL_NEW="""        // 无序流按ID寻找下一名，不保存紧凑位置。
"""
EQUAL_NEXT_OLD="""                if (rank + 1 < remaining) {
                    equalHead = kNoId;
                    for (uint64_t doc = equalPosition + sourceStep;
                         doc < sourceEnd; doc += sourceStep) {
                        if (ScoreKey(ReadBits(scores, scratch, base, doc, compact)) == prefix) {
                            equalHead = ReadId(scratch, doc, compact);
                            equalPosition = doc;
                            break;
                        }
                    }
                }"""
EQUAL_NEXT_NEW="""                if (rank + 1 < remaining) {
                    equalHead = kNoId;
                    // 无序紧凑流不能从旧位置前进；严格选择大于上个赢家ID的最小同分文档。
                    for (uint64_t doc = sourceBegin; doc < sourceEnd; doc += sourceStep) {
                        if (ScoreKey(ReadBits(scores, scratch, base, doc, compact)) == prefix) {
                            const uint32_t id = ReadId(scratch, doc, compact);
                            if (id > winner && id < equalHead) equalHead = id;
                        }
                    }
                }"""
COLLECT=((SOURCE_OLD,SOURCE_NEW),(EQUAL_DECL_OLD,EQUAL_DECL_NEW),(EQUAL_HEAD_OLD,EQUAL_HEAD_NEW),(EQUAL_NEXT_OLD,EQUAL_NEXT_NEW))


def candidate_sources(original):
    sources=base.candidate_sources(original)
    text=sources[KERNEL].decode('utf-8')
    text=simd.t.replace_once(text,pack_function(text),DIRECT_PACK)
    score_at=text.index(SCORE_MARKER);collect_at=text.index(COLLECT_MARKER)
    prefix,score,suffix=text[:score_at],text[score_at:collect_at],text[collect_at:]
    for old,new in COMMON:score=simd.t.replace_once(score,old,new)
    for old,new in COLLECT:suffix=simd.t.replace_once(suffix,old,new)
    sources[KERNEL]=(prefix+score+suffix).encode('utf-8')
    return sources


def prepare(original,target):
    metadata=base.prepare(original,target)
    sources=candidate_sources(original)
    for name,raw in sources.items():(target/name).write_bytes(raw)
    preset=target/'CMakePresets.json';data=simd.json.loads(preset.read_text(encoding='utf-8'))
    changed=0
    for item in data.get('configurePresets',[]):
        variables=item.get('cacheVariables',{});value=variables.get('vendor_name')
        if isinstance(value,dict) and value.get('value')==base.VENDOR:value['value']=VENDOR;changed+=1
        elif value==base.VENDOR:variables['vendor_name']=VENDOR;changed+=1
    simd.r.require(changed>0,'warp聚合追加候选vendor配置缺失')
    simd.r.write_json(preset,data)
    metadata.update(status='warp_append_source_prepared_not_device_verified',vendor_name=VENDOR,
        source_sha256={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
        compaction='per-warp ballot/popcount plus one bounded UB atomic segment allocation per nonzero scoring iteration; direct unordered IDs and contiguous threshold bits',
        tie_policy='unordered compact streams retain exact threshold; per-thread minimum equal IDs and strict greater-than previous winner preserve global ID ascending ties',
        fallback='small, overflowing, invalid or raw-debug inputs retain dense/existing exact paths; no candidate truncation',
        scope='NPU-only candidate derived from accepted Sparse Pack. Normal bitmap mask reads/count/prefix/ID emission removed; accepted baseline already avoided a full8192-score Pack scan. Positive score gather remains. FP32 posting and query-term order,256 threads,UB,threshold,merge,host,CPU and ABI unchanged. Actual950 compilation, exactness and timing required; no utilization or speedup claim before measurement.')
    simd.r.write_json(target/'bm25_project.json',metadata)
    return metadata
