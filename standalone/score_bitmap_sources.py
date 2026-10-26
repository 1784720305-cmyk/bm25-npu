"""仅在FP32评分写回时生成正分位图的独立算子候选。"""
from pathlib import Path
from standalone import register_select_sources as base,large_search_sources as large

simd=base.simd
ROOT=Path(__file__).resolve().parents[1]
VENDOR='bm25_fused_score_topk_scorebitmap_large'
HEADER=base.HEADER
KERNEL='op_kernel/bm25_fused_score_top_k.cpp'

PACK_OLD="""__simt_callee__ __aicore__ inline void PackForSimd(
    __ubuf__ const uint32_t *scores, __ubuf__ uint32_t *scratch, uint32_t width, uint32_t k)
{
    const uint32_t tid = AscendC::Simt::GetThreadIdx<0>();
    uint32_t positive = 0, invalid = 0, lowMask = 0, highMask = 0, ordinal = 0;
    for (uint32_t doc = tid; doc < width; doc += kThreads, ++ordinal) {
        const uint32_t bits = scores[doc];
        invalid |= !ValidBits(bits);
        if ((bits & 0x7fffffffU) != 0) {
            ++positive;
            if (ordinal < 32) lowMask |= 1U << ordinal;
            else highMask |= 1U << (ordinal-32);
        }
    }"""
PACK_NEW="""template <bool BitmapReady>
__simt_callee__ __aicore__ inline void PackForSimd(
    __ubuf__ const uint32_t *scores, __ubuf__ uint32_t *scratch, uint32_t width, uint32_t k)
{
    const uint32_t tid = AscendC::Simt::GetThreadIdx<0>();
    uint32_t positive = 0, invalid = 0, lowMask = 0, highMask = 0;
    if constexpr (BitmapReady) {
        // 评分暂存在 owner scratch 中的位图先读入寄存器，随后该空间仍按原流程复用。
        lowMask = scratch[kOwnerBegin+tid];
        highMask = scratch[kOwnerCount+tid];
        positive = static_cast<uint32_t>(__popc(lowMask)+__popc(highMask));
    } else {
        // debug=2 直接提供原始分数，保留原稠密验证和位图生成路径。
        uint32_t ordinal = 0;
        for (uint32_t doc = tid; doc < width; doc += kThreads, ++ordinal) {
            const uint32_t bits = scores[doc];
            invalid |= !ValidBits(bits);
            if ((bits & 0x7fffffffU) != 0) {
                ++positive;
                if (ordinal < 32) lowMask |= 1U << ordinal;
                else highMask |= 1U << (ordinal-32);
            }
        }
    }"""
POINTER_OLD="    auto *bounds = reinterpret_cast<__ubuf__ int64_t *>(control+8);"
POINTER_NEW=POINTER_OLD+"\n    auto *bitmap = memory+kSelection;"
ZERO_OLD="""        for (uint32_t doc = thread; doc < width; doc += kThreads) scores[doc] = 0.0F;
        asc_syncthreads();"""
ZERO_NEW="""        for (uint32_t doc = thread; doc < width; doc += kThreads) scores[doc] = 0.0F;
        bitmap[kOwnerBegin+thread] = 0;
        bitmap[kOwnerCount+thread] = 0;
        asc_syncthreads();"""
SCORE_OLD="""                    const float total = scores[local]+contribution;
                    if (!(contribution>=0.0F && contribution<=kMaxFloat && total>=0.0F && total<=kMaxFloat))
                        atomicAdd(&control[0], 1U);
                    else scores[local] = total;"""
SCORE_NEW="""                    const float previous = scores[local];
                    const float total = previous+contribution;
                    if (!(contribution>=0.0F && contribution<=kMaxFloat && total>=0.0F && total<=kMaxFloat))
                        atomicAdd(&control[0], 1U);
                    else {
                        scores[local] = total;
                        // 同词 posting 的文档唯一，词间已有屏障；只在文档首次由零变正时置位。
                        if (previous == 0.0F && total > 0.0F) {
                            // Scoring threads own posting positions, not document residues.
                            // Distinct writers can set different bits of the SAME owner word.
                            const uint32_t owner = local%kThreads;
                            const uint32_t part = local/kThreads;
                            if (part < 32) asc_atomic_or(bitmap+kOwnerBegin+owner, 1U << part);
                            else asc_atomic_or(bitmap+kOwnerCount+owner, 1U << (part-32));
                        }
                    }"""
NORMAL_PACK_OLD="""        } else {
            PackForSimd(memory, memory+kSelection, width, k);
        }"""
NORMAL_PACK_NEW="""        } else {
            PackForSimd<true>(memory, memory+kSelection, width, k);
        }"""
DEBUG_PACK_OLD="        asc_syncthreads(); PackForSimd(memory, memory+kSelection, width, k); return;"
DEBUG_PACK_NEW="        asc_syncthreads(); PackForSimd<false>(memory, memory+kSelection, width, k); return;"
REPLACEMENTS=((PACK_OLD,PACK_NEW),(POINTER_OLD,POINTER_NEW),(ZERO_OLD,ZERO_NEW),
    (SCORE_OLD,SCORE_NEW),(NORMAL_PACK_OLD,NORMAL_PACK_NEW),(DEBUG_PACK_OLD,DEBUG_PACK_NEW))


def candidate_sources(original):
    # Preserve the hardware-accepted bounded large-library/global-ID implementation.
    sources=large.fused_sources(original)
    text=sources[KERNEL].decode('utf-8')
    text=simd.t.replace_once(text,'#include "simt_api/device_sync_functions.h"',
        '#include "simt_api/device_sync_functions.h"\n#include "simt_api/device_atomic_functions.h"\n#include "simt_api/device_functions.h"')
    for old,new in REPLACEMENTS:text=simd.t.replace_once(text,old,new)
    sources[KERNEL]=text.encode('utf-8')
    return sources


def prepare(original,target):
    metadata=large.prepare(original,target,'fused')
    sources=candidate_sources(original)
    for name,raw in sources.items():(target/name).write_bytes(raw)
    preset=target/'CMakePresets.json';data=simd.json.loads(preset.read_text(encoding='utf-8'))
    changed=0
    for item in data.get('configurePresets',[]):
        variables=item.get('cacheVariables',{});value=variables.get('vendor_name')
        if isinstance(value,dict) and value.get('value')==large.FUSED:value['value']=VENDOR;changed+=1
        elif value==large.FUSED:variables['vendor_name']=VENDOR;changed+=1
    simd.r.require(changed>0,'评分位图候选 vendor 配置缺失')
    simd.r.write_json(preset,data)
    metadata.update(status='score_bitmap_source_prepared_not_device_verified',vendor_name=VENDOR,
        source_sha256={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
        positive_bitmap='first_positive_transition_atomic_OR_document_residue_owner_then_popcount',
        atomic_reference='https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/atomic_operations/asc_atomic_or.md',
        scope='Independent candidate of verified large leaf: same FP32/order, global IDs, register threshold, compact order, fallback, UB, threads and ABI; reuse tested tree/host. Only bitmap production/count changes. Atomic overhead may outweigh avoided scan; device evidence required.')
    simd.r.write_json(target/'bm25_project.json',metadata)
    return metadata
