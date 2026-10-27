"""仅优化正分位图发射的独立候选；保留位图评分和全部选择语义。"""
from pathlib import Path
from standalone import score_bitmap_sources as base

ROOT=Path(__file__).resolve().parents[1]
VENDOR='bm25_fused_score_topk_sparsepack_large'
KERNEL=base.KERNEL

EMIT_OLD="""        for (uint32_t pos = 0; written < positive && pos < 64; ++pos) {
            const uint32_t membership = pos < 32 ? lowMask : highMask;
            const uint32_t bit = pos < 32 ? pos : pos-32;
            if ((membership & (1U << bit)) != 0) {
                const uint32_t doc = tid+pos*kThreads;
                scratch[kPositiveBits+begin+written] = scores[doc];
                scratch[kPositiveIds+begin+written] = doc;
                ++written;
            }
        }"""
EMIT_NEW="""        for (uint32_t half = 0; half < 2 && written < positive; ++half) {
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
        }"""


def candidate_sources(original):
    sources=base.candidate_sources(original)
    text=sources[KERNEL].decode('utf-8')
    text=base.simd.t.replace_once(text,EMIT_OLD,EMIT_NEW)
    sources[KERNEL]=text.encode('utf-8')
    return sources


def prepare(original,target):
    # 先复用位图版工程生成，再完整覆盖独立候选的 vendor 和元数据。
    metadata=base.prepare(original,target)
    sources=candidate_sources(original)
    for name,raw in sources.items():(target/name).write_bytes(raw)
    preset=target/'CMakePresets.json';data=base.simd.json.loads(preset.read_text(encoding='utf-8'))
    changed=0
    for item in data.get('configurePresets',[]):
        variables=item.get('cacheVariables',{});value=variables.get('vendor_name')
        if isinstance(value,dict) and value.get('value')==base.VENDOR:value['value']=VENDOR;changed+=1
        elif value==base.VENDOR:variables['vendor_name']=VENDOR;changed+=1
    base.simd.r.require(changed>0,'稀疏 Pack 候选 vendor 配置缺失')
    base.simd.r.write_json(preset,data)
    metadata.update(status='sparse_pack_source_prepared_not_device_verified',vendor_name=VENDOR,
        source_sha256={name:base.simd.hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
        pack_emission='consume_set_bits_low_to_high_with_popc_instead_of_testing_every_ordinal',
        scope='Independent candidate of verified score-bitmap large leaf. Only compact emission loop changes; ascending document order, output slots, FP32, threshold, collector, fallback, global IDs, tree, host, CPU and ABI remain unchanged. Device timing required.')
    base.simd.r.write_json(target/'bm25_project.json',metadata)
    return metadata
