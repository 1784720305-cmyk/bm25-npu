"""Operator-only candidate: four histogram passes plus boundary control in one Reg scope."""
from pathlib import Path
from standalone import shard_directory_sources as directory

simd=directory.simd
ROOT=Path(__file__).resolve().parents[1]
VENDOR='bm25_fused_score_topk_regselect'
HEADER='npu_prototype/fused_reg_select/register_threshold.h'

def candidate_sources(original):
    sources=directory.candidate_sources(original)
    name='op_kernel/bm25_fused_score_top_k.cpp'
    baseline=(ROOT/simd.source.HEADER).read_text(encoding='utf-8')
    begin=baseline.index('// Hardware ACCUMULATE returns a CDF')
    replacement=baseline[:begin]+(ROOT/HEADER).read_text(encoding='utf-8')
    sources[name]=simd.t.replace_once(sources[name].decode('utf-8'),baseline,replacement).encode()
    return sources

def prepare(original,target):
    metadata=directory.prepare(original,target)
    sources=candidate_sources(original)
    for name,raw in sources.items():(target/name).write_bytes(raw)
    preset=target/'CMakePresets.json';data=simd.json.loads(preset.read_text(encoding='utf-8'))
    changed=0
    for item in data.get('configurePresets',[]):
        variables=item.get('cacheVariables',{});value=variables.get('vendor_name')
        if isinstance(value,dict) and value.get('value')==directory.VENDOR:value['value']=VENDOR;changed+=1
        elif value==directory.VENDOR:variables['vendor_name']=VENDOR;changed+=1
    simd.r.require(changed>0,'寄存器选择候选 vendor 配置缺失')
    simd.r.write_json(preset,data)
    metadata.update(status='register_selector_source_prepared_not_device_verified',vendor_name=VENDOR,
        source_sha256={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
        threshold_control='four_exact_byte_passes_register_resident_no_per_byte_scalar_CDF_search',
        scope='Only threshold control changes. Same scoring, directory, packing, strict-ID collector, fallback, UB, threads, shapes, ABI, merge and host Search. No CPU optimization.')
    simd.r.write_json(target/'bm25_project.json',metadata)
    return metadata
