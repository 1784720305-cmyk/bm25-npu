"""Same accepted SIMD operator ABI; tensor0 becomes an index-only int64 shard directory."""
from standalone import fused_simd_select as simd

VENDOR='bm25_fused_score_topk_directory'
replace_once=simd.t.replace_once

def candidate_sources(original):
    sources=simd.source.candidate_sources(original)
    host='op_host/bm25_fused_score_top_k.cpp'
    text=sources[host].decode('utf-8')
    text=replace_once(text,'length[0] != vocab+1','length[0] != vocab*(*shards+1)')
    sources[host]=text.encode()
    kernel='op_kernel/bm25_fused_score_top_k.cpp'
    text=sources[kernel].decode('utf-8')
    text=replace_once(text,
        '                const int64_t begin = termOffsets[term], end = termOffsets[term+1];',
        '                const uint64_t entry = static_cast<uint64_t>(term)*(shards+1)+task%shards;\n'
        '                const int64_t begin = termOffsets[entry], end = termOffsets[entry+1];')
    text=replace_once(text,
        '                    const int64_t lo = LowerBound(postingDocs, begin, end, first);\n'
        '                    begins[thread] = lo; ends[thread] = LowerBound(postingDocs, lo, end, last);',
        '                    begins[thread] = begin; ends[thread] = end;')
    sources[kernel]=text.encode()
    return sources

def prepare(original,target):
    metadata=simd.prepare(original,target)
    sources=candidate_sources(original)
    for name,data in sources.items(): (target/name).write_bytes(data)
    preset=target/'CMakePresets.json'
    data=simd.json.loads(preset.read_text(encoding='utf-8'))
    changed=0
    for item in data.get('configurePresets',[]):
        variables=item.get('cacheVariables',{}); value=variables.get('vendor_name')
        if isinstance(value,dict) and value.get('value')==simd.VENDOR: value['value']=VENDOR; changed+=1
        elif value==simd.VENDOR: variables['vendor_name']=VENDOR; changed+=1
    simd.r.require(changed>0,'Directory vendor preset missing')
    simd.r.write_json(preset,data)
    metadata.update(status='directory_source_prepared_not_device_verified',vendor_name=VENDOR,
        source_sha256={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
        first_tensor_contract='int64_absolute_posting_offsets_vocab_times_shards_plus_one',
        scope='Accepted256/SIMD formula, order, selector and ABI retained. Tensor0 shape and two LowerBound calls replaced by prebuilt directory reads.')
    simd.r.write_json(target/'bm25_project.json',metadata)
    return metadata
