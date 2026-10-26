"""One bounded mixed SIMT/SIMD candidate, derived from the accepted 256-thread source."""
from pathlib import Path
from standalone import fused_threads256 as t

ROOT=Path(__file__).resolve().parents[1]
HEADER='npu_prototype/fused_simd_select/simd_selection.h'


def candidate_sources(original):
    sources=t.candidate_sources(original)
    name='op_kernel/bm25_fused_score_top_k.cpp'
    old=sources[name].decode()
    marker='__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedScoreTopKSimt('
    begin=old.index(marker)
    prefix=old[:begin]
    score=old[begin:old.index('\n}\n}\n\nextern "C"',begin)+2]
    score=t.replace_once(score,'FusedScoreTopKSimt(', 'FusedScorePackSimt(')
    score=t.replace_once(score,
        '    const uint64_t tasks = static_cast<uint64_t>(queryCount)*shards;\n'
        '    for (uint64_t task = blockId; task < tasks; task += blockNum) {',
        '    const uint64_t task = blockId;\n    {')
    score=t.replace_once(score,
        '            SelectLocal(memory, partialScores, partialIds, memory+kSelection, task, width, k);',
        '            PackForSimd(memory, memory+kSelection, width, k);')
    score=t.replace_once(score,
        '        if (bad) {\n            if (thread < k)',
        '        if (bad) {\n            if (thread == 0) memory[kSelection+kControl+1] = 2U;\n'
        '            if (thread < k)')
    # One diagnostic mode in THIS operator, not a second experimental kernel.
    # Actual host tiling also checks postingCount == queries*docs before this raw read.
    insert='''    if (debug == 2) {
        const uint32_t query = blockId/shards, first = (blockId%shards)*width;
        const auto *bits = reinterpret_cast<__gm__ const uint32_t *>(postingTf);
        auto *out = reinterpret_cast<__gm__ uint32_t *>(debugScores);
        for (uint32_t d = thread; d < width; d += kThreads) {
            const uint64_t at = static_cast<uint64_t>(query)*docCount+first+d;
            memory[d] = bits[at]; out[at] = bits[at];
        }
        asc_syncthreads(); PackForSimd(memory, memory+kSelection, width, k); return;
    }
'''
    score=t.replace_once(score,'    // A disabled debug output',insert+'    // A disabled debug output')
    tail=old[old.index('        const uint32_t strictCount = k - remaining;'):old.index('// Three disjoint regions:')]
    collect='''
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedCollectSimt(
    __ubuf__ const uint32_t *scores, __ubuf__ uint32_t *scratch,
    __gm__ uint32_t *topScores, __gm__ int32_t *topIds,
    uint32_t query, uint32_t docCount, uint32_t k)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t mode = scratch[kControl+1];
    if (mode == 0) { SelectLocal(scores, topScores, topIds, scratch, query, docCount, k); return; }
    if (mode != 1) {
        if (thread < k) { topScores[query*k+thread] = 0xbf800000U; topIds[query*k+thread] = -1; }
        asc_syncthreads(); return;
    }
    const uint32_t prefix = scratch[kControl+2], remaining = scratch[kControl+3];
    const uint64_t base = 0;
    const bool compact = true;
    const uint64_t sourceBegin = scratch[kOwnerBegin+thread];
    const uint64_t sourceEnd = sourceBegin + scratch[kOwnerCount+thread], sourceStep = 1;
    uint32_t badInput = 0;
    asc_syncthreads(); // consume owner metadata before the accepted collector reuses scan scratch
'''+tail
    wrapper=old[old.index('extern "C" __global__'):]
    wrapper=t.replace_once(wrapper,'    AscendC::Simt::VF_CALL<FusedScoreTopKSimt>',
        '    const uint32_t tasks = data.queryCount*data.shards;\n'
        '    for (uint32_t task = AscendC::GetBlockIdx(); task < tasks; task += data.blockNum) {\n'
        '    AscendC::Simt::VF_CALL<FusedScorePackSimt>')
    wrapper=t.replace_once(wrapper,
        '        static_cast<uint32_t>(AscendC::GetBlockIdx()), data.blockNum);\n}',
        '''        task, data.blockNum);
    AscendC::PipeBarrier<PIPE_ALL>();
    FindSimdThreshold(local, data.k);
    AscendC::Simt::VF_CALL<FusedCollectSimt>(AscendC::Simt::Dim3{kThreads,1,1},
        reinterpret_cast<__ubuf__ const uint32_t *>(local.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr())+kSelection,
        reinterpret_cast<__gm__ uint32_t *>(partial_scores), reinterpret_cast<__gm__ int32_t *>(partial_ids),
        task, data.docCount/data.shards, data.k);
    AscendC::PipeBarrier<PIPE_ALL>();
    }
}''')
    header=(ROOT/HEADER).read_text(encoding='utf-8')
    sources[name]=(prefix+'\n'+header+'\n'+score+'\n'+collect+'}\n\n'+wrapper).encode()
    name='op_host/bm25_fused_score_top_k.cpp';host=sources[name].decode()
    host=t.replace_once(host,'(*debug != 0 && *debug != 1)', '(*debug != 0 && *debug != 1 && *debug != 2)')
    host=t.replace_once(host,'    dims = {static_cast<uint32_t>(queries),',
        '    // debug2 is a selector-only diagnostic: never used by public Search.\n'
        '    if (*debug == 2 && static_cast<uint64_t>(length[1]) != static_cast<uint64_t>(queries)*docs) return false;\n'
        '    dims = {static_cast<uint32_t>(queries),')
    sources[name]=host.encode()
    return sources
