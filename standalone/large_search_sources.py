"""Bounded8192 leaf scoring and global-ID32-way hierarchical merge; independent vendors."""
import json
import os
from pathlib import Path
import shutil
from standalone import register_select_sources as base

ROOT=Path(__file__).resolve().parents[1]
simd=base.simd;r=simd.r
FUSED='bm25_fused_score_topk_large'
MERGE='bm25_topk_merge_tree'
replace=simd.t.replace_once


def fused_sources(original):
    sources=base.candidate_sources(original)
    name='op_host/bm25_fused_score_top_k.cpp';text=sources[name].decode()
    text=replace(text,'(*shards > 32 && *shards != 64)','*shards > 65536')
    # The UB limit and per-block8192 limit remain intact; only the number of bounded leaves grows.
    sources[name]=text.encode()
    name='op_kernel/bm25_fused_score_top_k.cpp';text=sources[name].decode()
    text=replace(text,'            topIds[query*k + lane] = valid ? static_cast<int32_t>(id) : -1;',
        '            topIds[query*k + lane] = valid ? static_cast<int32_t>(scratch[kControl+7]+id) : -1;')
    text=replace(text,'    const uint32_t width = docCount/shards;',
        '    const uint32_t width = docCount/shards;\n'
        '    if (thread==0) memory[kSelection+kControl+7]=(blockId%shards)*width;')
    sources[name]=text.encode()
    return sources


def merge_sources(original):
    names=('op_host/bm25_top_k_merge.cpp','op_host/bm25_top_k_merge_tiling.h','op_kernel/bm25_top_k_merge.cpp')
    sources={name:(original/name).read_bytes() for name in names}
    name=names[0];text=sources[name].decode()
    text=replace(text,'constexpr int64_t kMaxShards = 32;','constexpr int64_t kMaxShards = 65536;')
    begin=text.index('bool ValidShape(');end=text.index('\nge::graphStatus InferShape',begin)
    text=text[:begin]+'''bool ValidShape(const gert::Shape &scores,const gert::Shape &ids,int64_t k,int64_t docs)
{
    if(scores.GetDimNum()!=3 || ids.GetDimNum()!=3) return false;
    for(std::size_t d=0;d<3;++d) if(scores.GetDim(d)!=ids.GetDim(d))return false;
    const int64_t q=scores.GetDim(0),s=scores.GetDim(1),limit=std::numeric_limits<int32_t>::max();
    return docs>=k && docs<=limit && q>=1 && q<=limit/docs && s>=1 && s<=kMaxShards &&
        k>=1 && k<=kMaxK && scores.GetDim(2)==k;
}
'''+text[end:]
    text=replace(text,'gert::Shape({scores->GetDim(0), *k})','gert::Shape({scores->GetDim(0), (scores->GetDim(1)+31)/32, *k})')
    text=replace(text,'        if (outShape.GetDimNum() != 2 || outShape.GetDim(0) != shape.GetDim(0) || outShape.GetDim(1) != *k)',
        '        if (outShape.GetDimNum() != 3 || outShape.GetDim(0) != shape.GetDim(0) ||\n'
        '            outShape.GetDim(1) != (shape.GetDim(1)+31)/32 || outShape.GetDim(2) != *k)')
    text=replace(text,'const uint32_t blocks = std::min(cores, queries);',
        'const uint32_t blocks = std::min<uint64_t>(cores, static_cast<uint64_t>(queries)*((shape.GetDim(1)+31)/32));')
    text=replace(text,'data.set_docsPerShard(static_cast<uint32_t>(*docs / shape.GetDim(1)));',
        'data.set_docsPerShard(static_cast<uint32_t>(*docs)); // global ID range; no per-list offset')
    sources[name]=text.encode()
    name=names[2];text=sources[name].decode()
    text=replace(text,'    for (uint64_t query = blockId; query < queryCount; query += blockNum) {',
        '    const uint32_t groups=(shards+31)/32;\n'
        '    for (uint64_t task=blockId;task<static_cast<uint64_t>(queryCount)*groups;task+=blockNum) {\n'
        '        const uint64_t query=task/groups;\n'
        '        const uint32_t inputLane=static_cast<uint32_t>(task%groups)*32+lane;')
    text=replace(text,'const uint64_t base = (query * shards + lane) * k;',
        'const uint64_t base = (query * shards + inputLane) * k;')
    text=text.replace('lane < shards','inputLane < shards').replace('query * k','task * k')
    old='static_cast<int32_t>(static_cast<uint64_t>(lane) * docsPerShard + partialIds[base'
    r.require(text.count(old)==2,'Global-ID merge conversion sites changed')
    text=text.replace(old,'static_cast<int32_t>(partialIds[base')
    sources[name]=text.encode()
    return sources


def vendor_preset(target,name):
    path=target/'CMakePresets.json';data=json.loads(path.read_text(encoding='utf-8'));changed=0
    for preset in data.get('configurePresets',[]):
        variables=preset.get('cacheVariables',{})
        if 'ASCEND_COMPUTE_UNIT' not in variables:continue
        value=variables.get('vendor_name')
        if isinstance(value,dict):value['value']=name
        else:variables['vendor_name']={'type':'STRING','value':name}
        changed+=1
    r.require(changed>0,'Actual vendor preset missing');r.write_json(path,data)


def prepare(original,target,kind):
    if kind=='fused':
        metadata=base.prepare(original,target);sources=fused_sources(original);vendor=FUSED
    else:
        r.require(kind=='merge' and not target.exists() and target.resolve().is_relative_to((ROOT/'results').resolve()) and
            not target.resolve().is_relative_to(original.resolve()) and not original.resolve().is_relative_to(target.resolve()),'Independent new merge project required')
        metadata=json.loads((original/'bm25_project.json').read_text(encoding='utf-8'))
        r.require(metadata.get('compute_unit')=='ascend950','Actual950 merge template required')
        for name in ('op_host/bm25_top_k_merge.cpp','op_host/bm25_top_k_merge_tiling.h','op_kernel/bm25_top_k_merge.cpp'):
            r.require((original/name).read_bytes()==(ROOT/'npu_prototype/topk_merge'/name).read_bytes(),'Accepted original merge source changed')
        for root,dirs,files in os.walk(original,followlinks=False):
            dirs[:]=[d for d in dirs if d not in simd.t.resource.EXCLUDED]
            for name in dirs+files:r.require(not (Path(root)/name).is_symlink(),'Merge source contains symlink')
        shutil.copytree(original,target,ignore=lambda unused,names:[n for n in names if n in simd.t.resource.EXCLUDED])
        sources=merge_sources(original);vendor=MERGE
    for name,raw in sources.items():(target/name).write_bytes(raw)
    vendor_preset(target,vendor)
    metadata.update(status='large_search_source_not_device_verified',vendor_name=vendor,kind=kind,
        source_sha256={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
        scope='Bounded8192 leaves, unchanged FP32/order/selector/UB; global IDs and bounded32-way merge tree. No CPU change.')
    r.write_json(target/'bm25_project.json',metadata)
    return metadata
