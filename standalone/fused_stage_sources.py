"""Diagnostic-only copy of the device-verified register selector. Normal implementation retained."""
from standalone import register_select_sources as base

simd=base.simd
VENDOR='bm25_fused_score_topk_stages'
MAGIC=0x424d323550483031
WORDS=8


def candidate_sources(original):
    sources=base.candidate_sources(original)
    name='op_kernel/bm25_fused_score_top_k.cpp';text=sources[name].decode()
    begin=text.index('__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedScorePackSimt(')
    end=text.index('\n__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedCollectSimt(',begin)
    score=text[begin:end]
    assert score.count('PackForSimd(memory, memory+kSelection, width, k);')==2
    score=score.replace('FusedScorePackSimt(', 'FusedScoreOnlyDiagnostic(').replace(
        'PackForSimd(memory, memory+kSelection, width, k);','/* separate diagnostic pack dispatch */')
    pack='''
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void PackOnlyDiagnostic(
    __ubuf__ uint32_t *memory,uint32_t width,uint32_t k)
{
    if (memory[kScoreControl]==0) PackForSimd(memory,memory+kSelection,width,k);
}
'''
    text=text[:end]+score+pack+text[end:]
    wrapper=text[text.index('extern "C" __global__'):]
    start=wrapper.index('    AscendC::Simt::VF_CALL<FusedScorePackSimt>')
    stop=wrapper.index(';',start)+1
    score_call=wrapper[start:stop].replace('FusedScorePackSimt>','FusedScoreOnlyDiagnostic>').replace(
        'data.shards, data.k, data.debug,','data.shards, data.k, 0U,')
    assert 'data.debug' not in score_call
    collect_begin=wrapper.index('    AscendC::Simt::VF_CALL<FusedCollectSimt>')
    collect=wrapper[collect_begin:wrapper.index(';',collect_begin)+1]
    diagnostic='''
    // debug3 is diagnostic-only. Its [Q,N] FLOAT output carries raw uint64 records,
    // NOT scores. Each task owns 8 words; no device print or intermediate H2D/D2H.
    if (data.debug==3) {
        auto *records=reinterpret_cast<__gm__ uint64_t *>(debug_scores);
        const uint32_t tasks=data.queryCount*data.shards;
        for (uint32_t task=AscendC::GetBlockIdx();task<tasks;task+=data.blockNum) {
            AscendC::PipeBarrier<PIPE_ALL>();
            const uint64_t t0=AscendC::GetSystemCycle();
'''+score_call+'''
            AscendC::PipeBarrier<PIPE_ALL>();
            const uint64_t t1=AscendC::GetSystemCycle();
            AscendC::Simt::VF_CALL<PackOnlyDiagnostic>(AscendC::Simt::Dim3{kThreads,1,1},
                reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr()),data.docCount/data.shards,data.k);
            AscendC::PipeBarrier<PIPE_ALL>();
            const uint64_t t2=AscendC::GetSystemCycle();
            FindSimdThreshold(local,data.k);
            AscendC::PipeBarrier<PIPE_ALL>();
            const uint64_t t3=AscendC::GetSystemCycle();
'''+collect+'''
            AscendC::PipeBarrier<PIPE_ALL>();
            const uint64_t t4=AscendC::GetSystemCycle();
            auto *record=records+static_cast<uint64_t>(task)*8;
            record[0]=0x424d323550483031ULL; record[1]=task; record[2]=AscendC::GetBlockIdx();
            record[3]=t0;record[4]=t1;record[5]=t2;record[6]=t3;record[7]=t4;
        }
        return;
    }
'''
    marker='    const uint32_t tasks = data.queryCount*data.shards;'
    text=simd.t.replace_once(text,marker,diagnostic+'\n'+marker)
    sources[name]=text.encode()
    host='op_host/bm25_fused_score_top_k.cpp';text=sources[host].decode()
    text=simd.t.replace_once(text,'(*debug != 0 && *debug != 1 && *debug != 2)',
        '(*debug != 0 && *debug != 1 && *debug != 2 && *debug != 3)')
    text=simd.t.replace_once(text,'    dims = {static_cast<uint32_t>(queries),',
        '    if (*debug==3 && docs/(*shards)<16) return false; // 64 record bytes per task\n'
        '    dims = {static_cast<uint32_t>(queries),')
    sources[host]=text.encode()
    return sources


def prepare(original,target):
    metadata=base.prepare(original,target);sources=candidate_sources(original)
    for name,raw in sources.items():(target/name).write_bytes(raw)
    path=target/'CMakePresets.json';preset=simd.json.loads(path.read_text(encoding='utf-8'));changed=0
    for item in preset['configurePresets']:
        variables=item.get('cacheVariables',{});vendor=variables.get('vendor_name')
        if isinstance(vendor,dict) and vendor.get('value')==base.VENDOR:vendor['value']=VENDOR;changed+=1
        elif vendor==base.VENDOR:variables['vendor_name']=VENDOR;changed+=1
    simd.r.require(changed>0,'Diagnostic vendor preset missing');simd.r.write_json(path,preset)
    metadata.update(status='diagnostic_stages_not_performance_candidate',vendor_name=VENDOR,
        source_sha256={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
        scope='Debug3 only: split score/pack dispatch and clock/barrier overhead. No performance acceptance claim; no CPU changes.')
    simd.r.write_json(target/'bm25_project.json',metadata)
    return metadata
