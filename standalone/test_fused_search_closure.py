"""Affected closure gates, raw-ID command, metadata and result rejection only. MOCK child, not C++/device proof."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import close_fused_search as t, fused_search_checks as fc, inspect_fused_hardware as hw
from standalone import scale_fixture, search_scale_fixture, run_search as search
from standalone.test_resident_chain import mock_outputs


def session(index,records):
    d={'status':'search_benchmark_calls_completed_checks_pending','backend':'npu','docs':128,'queries':8,'k':10,
       'partitions':8,'capacity_queries':8,'capacity_words':24,'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory',
       'timer_policy':t.checks.TIMER,'index_load_batches':1,'index_bytes':index['index_bytes'],'cold_init_ms':1.,
       'cold_synthetic_empty_queries':8,'checkpoint_search_calls':4,'warmup_cycles':5,'warmup_search_calls':10,
       'warmup_topk_bit_checks':10,'measured_cycles':30,'measured_search_calls':60,'measured_topk_bit_checks':60,
       'measured_debug_download_calls':0,'measured_poison_calls':0,'measured_file_io_calls':0,
       'diagnostic_checkpoints_outside_measurement':True,'index_readback_byte_identical':True,
       'host_source_sha256':t.checks.sources(fc.HOST_SOURCES),'device':5,'index_tensor_h2d_calls':5,
       'pipeline_stages':2,'normal_full_score_elements':0,'debug_placeholder_elements':1,'diagnostic_extra_chains':4,
       'debug_index_d2h_calls':5,'intermediate_host_roundtrips':0,'workspace_capacity_bytes':[16777216]*2,'samples':[]}
    for cycle in range(30):
        for slot in range(2):
            name='B' if slot^(cycle&1) else 'A'
            d['samples'].append({'cycle':cycle,'case':name,'call_wall_ms':.1,'internal_search_ms':.09,
                'query_h2d_bytes':records[name]['query_input_bytes'],'topk_d2h_bytes':640})
    return d


def main():
    checks=0
    def reject(call):
        nonlocal checks
        try:call()
        except (ValueError,OSError):checks+=1
        else:raise AssertionError('Invalid closure evidence accepted')
    rows=[{'baseline':{'summary':{case:{'median_batch_ms':.33,'p95_batch_ms':.35} for case in ('A','B')}},
           'candidate':{'summary':{case:{'median_batch_ms':.2,'p95_batch_ms':.22} for case in ('A','B')}}} for i in range(6)]
    assert t.decision(rows)['keep_fused_candidate'];checks+=1
    for mode in ('one_slow','only_A','tiny_gain','p95_regression'):
        bad=copy.deepcopy(rows)
        if mode=='one_slow':bad[0]['candidate']['summary']['A']['median_batch_ms']=.34
        if mode=='only_A':
            for x in bad:x['candidate']['summary']['B']['median_batch_ms']=.4
        if mode=='tiny_gain':
            for x in bad:x['candidate']['summary']['A']['median_batch_ms']=.32
        if mode=='p95_regression':
            for x in bad:x['candidate']['summary']['B']['p95_batch_ms']=.36
        assert not t.decision(bad)['keep_fused_candidate'];checks+=1
    reject(lambda:t.decision(rows[:5]))
    with tempfile.TemporaryDirectory(prefix='fused_search_MOCK_ONLY_',dir=ROOT/'results') as tmp,redirect_stdout(io.StringIO()):
        base=Path(tmp);accept=base/'accept';accept.mkdir()
        for state in ('running','failed','interrupted'):
            t.r.write_json(accept/'report.json',{'status':state})
            with patch.object(t.fused,'run',side_effect=AssertionError('must not restart')):
                reject(lambda:t.acceptance(accept,base,base,base,base,3,2,True))
        source=base/'source';inputs=base/'inputs'
        scale_fixture.generate(source,128,8,10,128,20261014,0,'performance',3)
        search_scale_fixture.prepare(source,inputs);index,records,unused=search_scale_fixture.validate_inputs(inputs)
        cache=t.checks.cache_inputs(inputs,index,records);raw=base/'raw';raw.mkdir()
        for name in ('A','B'):search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
        runner=base/'FAKE_NOT_EXECUTABLE';runner.write_bytes(b'MOCK ONLY')
        for mode in ('pass','bad_ids','bad_timer','full_score_in_timed_path','stale_sources'):
            output=base/mode
            def child(command,log,env):
                assert len(command)==15 and command[2:4]==[str(raw/'A.txt'),str(raw/'B.txt')]
                assert command[11:]==['5','30',str(inputs/'A/expected_scores.bin'),str(inputs/'B/expected_scores.bin')]
                assert env['ASCEND_CUSTOM_OPP_PATH']==':'.join(str(base/x) for x in ('bm25_fused_score_topk','bm25_topk_merge'))
                actual=Path(command[4]);actual.mkdir()
                for label in t.checks.CHECKPOINTS:
                    name=label[-1];mock_outputs(actual/label,cache[name]['scores'],128,8,10,8)
                data=session(index,records)
                if mode=='bad_ids':(actual/'before_A/top_ids.bin').write_bytes(bytes(320))
                if mode=='bad_timer':data['samples'][0]['query_h2d_bytes']=0
                if mode=='full_score_in_timed_path':data['normal_full_score_elements']=1024
                if mode=='stale_sources':data['host_source_sha256']={}
                t.r.write_json(actual/'benchmark_session.json',data);log.write_bytes(b'MOCK NO DEVICE')
            with patch.object(t.fused.score,'execute',child):
                action=lambda:t.run_fused(output,inputs,index,records,cache,raw,runner,base,5,8,8,24,5,30)
                if mode=='pass':assert action()['summary']['A']['median_batch_ms']==.1;checks+=1
                else:reject(action)
    assert hw.fields({'kernels':[{'stackSize':0,'registerCount':127,'not_resource':8}]})==[
        {'field':'/kernels/0/stackSize','value':0},{'field':'/kernels/0/registerCount','value':127}];checks+=1
    # New orchestrator only: prove reject/keep branching and first-error stop, without real indexes or children.
    with tempfile.TemporaryDirectory(prefix='fused_closure_ORCHESTRATOR_MOCK_',dir=ROOT/'results') as tmp,redirect_stdout(io.StringIO()):
        base=Path(tmp)
        exe=base/'FAKE';exe.write_bytes(b'MOCK ONLY '+b' '.join(x.encode('ascii') for x in t.checks.sources(fc.HOST_SOURCES).values()))
        records={x:{'k':10,'num_queries':8,'query_term_count':24} for x in ('A','B','empty')}
        cache={x:{'queries':[[i]*3 for i in range(8)]} for x in records}
        for mode in ('lose','win','first_error'):
            output=base/mode;calls=[];installation={'MOCK':True}
            a=SimpleNamespace(input=base,output=output,vendors=base,project=base,merge_project=base,
                fused_acceptance=base,small_input=base,shards=32,baseline_runner=exe,runner=exe,cpu_runner=exe,
                baseline_acceptance=base,range_project=base,compact_project=base,device=5)
            def row(value,directory):return {'directory':str(directory),'summary':{x:{'median_batch_ms':value,'p95_batch_ms':value*1.1} for x in ('A','B')},'actual_threads':32}
            def old(directory,backend,*args):calls.append(backend);return row(.074 if backend=='cpu' else .33,directory)
            def candidate(directory,*args):
                calls.append('candidate')
                if mode=='first_error':raise ValueError('MOCK first Search error')
                return row(.34 if mode=='lose' else .2,directory)
            with patch.object(t,'acceptance',return_value={'installation':installation}), \
                 patch.object(t.fixture,'validate_inputs',return_value=({'num_docs':262144},records,{})), \
                 patch.object(t.checks,'cache_inputs',return_value=cache), \
                 patch.object(t.prior,'validate_selected',return_value={'targets':{},'installation':installation,'evidence_sha256':{}}), \
                 patch.object(t.prior.selected.previous,'create_runtime',return_value={}), \
                 patch.object(t.prior.selected.previous,'validate_runtime',return_value={}), \
                 patch.object(t.fused,'installed',return_value=installation), \
                 patch.object(t.bench,'run_one',side_effect=old),patch.object(t,'run_fused',side_effect=candidate), \
                 patch.object(t,'cross'):
                if mode=='first_error':reject(lambda:t.run(a))
                else:t.run(a);checks+=1
            report=json.loads((output/'report.json').read_text(encoding='utf-8'))
            if mode=='lose':assert len(report['per_round'])==6 and not report['cpu_rounds'] and 'cpu' not in calls
            elif mode=='win':assert len(report['cpu_rounds'])==6 and calls.count('cpu')==6
            else:assert report['status']=='failed' and calls==['npu','candidate']
            checks+=1
    h=(ROOT/'npu_prototype/fused_search_host/fused_search.h').read_text(encoding='utf-8')
    assert 'const size_t elements[5]={1,' in h and 'const size_t mode=debugMemory?1:0' in h
    assert h.index('auto packed = PackQueries')<h.index('search upload offsets')<h.index('search fused execute')<h.index('search download final IDs')<h.index('result.searchMs =')
    assert h[h.index('Result Search(const'):h.index('void PoisonForDiagnostic')].count('Clock::now()')==1
    assert 'CompareReference(request); CheckRanking' in h and 'for (size_t i=0;i<2;++i)' in h;checks+=1
    print(f'PASS: {checks} affected closure/decision/command/metadata/result/hardware-field checks; MOCK child only. No C++ build or device evidence.')


if __name__=='__main__':main()
