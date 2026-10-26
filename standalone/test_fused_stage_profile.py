"""Parser, counter units, diagnostic-source isolation; not device execution or speed evidence."""
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import fused_stage_profile as x


def records():
    result=[]
    for task in range(8):
        block=task%4;start=100000+block*100+task//4*15000
        result.append([x.source.MAGIC,task,block,start,start+1000,start+3000,start+6000,start+10000])
    return result


def encode(rows):return b''.join(struct.pack('<8Q',*row) for row in rows)


def parser_tests():
    rows=records();result=x.analyze(encode(rows),1,8)
    assert result['blocks']==4 and result['tasks']==8
    assert result['all_block_work_us']==dict(zip(x.STAGES,(8,16,24,32)))
    assert result['all_block_work_fraction']==dict(zip(x.STAGES,(.1,.2,.3,.4)))
    assert result['observed_work_span_us']==25.3
    assert result['last_finishing_block']['block']==3
    assert result['last_finishing_block']['stage_us']==dict(zip(x.STAGES,(2,4,6,8)))
    assert result['last_finishing_block']['work_sum_us']==20
    assert result['last_finishing_block']['block_span_us']==25
    failures=0
    for position,index,value in ((0,0,0),(0,1,1),(0,2,8),(0,3,0),(0,4,100000),(0,5,100500),
        (7,2,2),(4,3,100001),(0,2,3)):
        bad=[row[:] for row in rows];bad[position][index]=value
        try:x.analyze(encode(bad),1,8)
        except ValueError:failures+=1
        else:raise AssertionError('Malformed records accepted')
    for raw in (b'',encode(rows)[:-1],encode(rows)+b'\0'*64):
        try:x.analyze(raw,1,8)
        except ValueError:failures+=1
        else:raise AssertionError('Wrong record length accepted')
    return failures


def source_tests():
    original=ROOT/'npu_prototype/fused_score_topk'
    before=x.source.base.candidate_sources(original);after=x.source.candidate_sources(original)
    kernel='op_kernel/bm25_fused_score_top_k.cpp';host='op_host/bm25_fused_score_top_k.cpp'
    assert {k:v for k,v in before.items() if k not in (kernel,host)}=={k:v for k,v in after.items() if k not in (kernel,host)}
    old=before[kernel].decode();new=after[kernel].decode()
    # Normal score/packing/selection/collector and normal dispatch stay as the accepted source.
    name='__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void '
    begin=old.index(name+'FusedScorePackSimt(')
    end=old.index('\n'+name+'FusedCollectSimt(',begin)
    score=old[begin:end]
    assert score in new
    expected=score.replace('FusedScorePackSimt(', 'FusedScoreOnlyDiagnostic(').replace(
        'PackForSimd(memory, memory+kSelection, width, k);','/* separate diagnostic pack dispatch */')
    assert expected in new
    tail='    const uint32_t tasks = data.queryCount*data.shards;'
    assert new[new.rindex(tail):]==old[old.rindex(tail):]
    marker='    if (data.debug==3) {';phase=new[new.index(marker):new.rindex(tail)]
    assert phase.count('GetSystemCycle()')==5 and phase.count('PipeBarrier<PIPE_ALL>()')==5
    assert 'data.shards, data.k, 0U,' in phase
    assert phase.index('const uint64_t t4')<phase.index('record[0]=')
    assert 'printf' not in phase and 'data.debug,' not in phase
    assert 'docs/(*shards)<16' in after[host].decode()
    runner=(ROOT/x.OWN[2]).read_text(encoding='utf-8')
    assert 'Accepted output bits differ' in runner and 'Unexpected full-score write' in runner
    assert 'sizes[i]+2*guard' in runner and 'if(cycle>=5)' in runner and 'cycle<15' in runner
    cmake=(ROOT/x.OWN[3]).read_text(encoding='utf-8')
    assert 'OpenMP' not in cmake and 'cpu_benchmark' not in cmake


if __name__=='__main__':
    failures=parser_tests();source_tests()
    print(f'PASS: stage-source isolation, exact1GHz conversion, per-block grid-stride accounting, {failures} malformed-record rejections. No CANN/device/performance evidence.')
