"""Search wrapper/provenance/byte/timing/failure tests with FAKE artifacts and MOCK device child."""
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import run_search as tool, resident_index as r, query_score_fixture as fixture
from standalone.test_resident_chain import fake_install,mock_outputs,TOP_DECLS,MERGE_DECLS


def main():
    checks=0
    def fails(call):
        nonlocal checks
        try: call()
        except (ValueError,OSError,subprocess.SubprocessError): checks+=1
        else: raise AssertionError('invalid state accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='search_HOST_MOCK_ONLY_',dir=ROOT/'results') as temp:
        base=Path(temp); inputs=base/'input'
        with redirect_stdout(io.StringIO()): fixture.prepare(inputs)
        score_vendor,sp=fake_install(base,'query_score','bm25_query_score',tool.chain.score.PROTOTYPES)
        top_vendor,tp=fake_install(base,'topk_radix_collect','bm25_top_k',TOP_DECLS)
        merge_vendor,mp=fake_install(base,'topk_merge','bm25_top_k_merge',MERGE_DECLS)
        acceptance=base/'TEST_ONLY_MOCK_prior_acceptance'
        accepted={'directory':str(acceptance),'report_sha256':'MOCK ONLY','workspace_bytes':16777216}
        runner=base/'FAKE_NOT_EXECUTABLE'; runner.write_bytes(b'TEST ONLY')
        args=(base/'vendors',sp,tp,mp,acceptance,2,3,2,65,512)
        index,records,unused=tool.chain.score.validate_inputs(inputs)
        unused,arrays=r.validate_index(inputs/'index')
        cpu={}
        for name in records:
            unused,queries=r.validate_queries(inputs/name,index)
            cpu[name]=r.diagnostic_scores(index,arrays,queries)
        file=base/'raw.txt'; tool.write_queries(file,[[2,2,2147483646],[],[0]])
        assert file.read_bytes()==b'BM25_QUERY_IDS_V1\n3\n3 2 2 2147483646\n0\n1 0\n'; checks+=1
        for bad in ([],[[True]],[[-1]],[[2147483647]],['bad']): fails(lambda:tool.write_queries(file,bad))
        for bad in (True,None,float('nan'),float('inf'),-1,0): assert not tool.finite_positive(bad); checks+=1
        with patch.object(tool.chain,'validate_radix_acceptance',return_value=accepted):
            for mode in ('pass','child_failure','bad_timer_policy','bad_contract','prepacked','early_stop',
                         'debug_inside','stale_executable','index_twice','wrong_device','wrong_capacity',
                         'missing_bootstrap','bad_cold_time','nan_time','bool_time','negative_time',
                         'wrong_query_bytes','extra_result_bytes','wrong_poison','repeat_bits','bad_ranking','bad_workspace'):
                output=base/('run_'+mode); calls=[]
                def child(command,logfile,env):
                    calls.append(command)
                    assert command[-5:]==['3','2','65','512','16777216']
                    assert env['ASCEND_CUSTOM_OPP_PATH']==':'.join(str(v.resolve()) for v in (score_vendor,top_vendor,merge_vendor))
                    plan=Path(command[2]).read_text().splitlines(); assert len(plan)==5
                    assert all(Path(line.split('\t')[1]).read_text().startswith('BM25_QUERY_IDS_V1\n') for line in plan)
                    logfile.write_bytes(b'TEST ONLY MOCK DEVICE CHILD -- NEVER EXECUTES AN NPU\n')
                    if mode=='child_failure': raise subprocess.CalledProcessError(8,command)
                    actual=Path(command[3]); actual.mkdir()
                    session={'status':'device_search_calls_completed_independent_check_pending','device':2,'docs':6,'k':3,'shards':2,
                        'capacity_queries':65,'capacity_words':512,'index_upload_batches':1,'index_tensor_h2d_calls':5,
                        'index_upload_bytes':index['index_bytes'],'index_readback_byte_identical':True,
                        'pipeline_stages':3,'intermediate_host_roundtrips':0,
                        'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory','timer_policy':tool.TIMER_POLICY,
                        'cold_index_load_runtime_upload_capacity_bootstrap_ms':10.0,
                        'cold_bootstrap':{'synthetic_empty_queries':65,'query_h2d_calls':2,'query_h2d_bytes':532,
                                          'topk_d2h_calls':2,'topk_d2h_bytes':1560},
                        'workspace_capacity_bytes':[16777216]*3,'debug_index_d2h_calls':5,
                        'debug_index_d2h_bytes':index['index_bytes'],
                        'host_source_sha256':{name:tool.digest(ROOT/name) for name in tool.HOST_SOURCES},'requests':[]}
                    for i,name in enumerate(tool.chain.SEQUENCE):
                        source=records[name]; q=source['num_queries']; logical=source['query_input_bytes']; label=f'call{i}_{name}'
                        mock_outputs(actual/label,cpu[name],6,q,3,2)
                        session['requests'].append({'label':label,'queries':q,'words':source['query_term_count'],
                            'search_call_ms':.1,'logical_query_bytes':logical,'query_h2d_calls':2,
                            'query_h2d_bytes':logical+(4 if source['query_term_count']==0 else 0),
                            'topk_d2h_calls':2,'topk_d2h_bytes':8*q*3,'debug_poison_h2d_calls':5,
                            'debug_poison_h2d_bytes':4*65*6+8*65*2*3+8*65*3,
                            'debug_score_candidate_d2h_calls':3,'debug_score_candidate_d2h_bytes':4*q*6+8*q*2*3,
                            'packing_inside_Search':True,'topk_received_before_stop':True,'debug_outside_Search':True})
                    first=session['requests'][0]
                    if mode=='bad_timer_policy': session['timer_policy']='start after packing'
                    if mode=='bad_contract': session['input_contract']='preexpanded TF arrays'
                    if mode=='prepacked': first['packing_inside_Search']=False
                    if mode=='early_stop': first['topk_received_before_stop']=False
                    if mode=='debug_inside': first['debug_outside_Search']=False
                    if mode=='stale_executable': session['host_source_sha256'][tool.HOST_SOURCES[0]]='0'*64
                    if mode=='index_twice': session['index_upload_batches']=2
                    if mode=='wrong_device': session['device']=4
                    if mode=='wrong_capacity': session['capacity_queries']=6
                    if mode=='missing_bootstrap': del session['cold_bootstrap']
                    if mode=='bad_cold_time': session['cold_index_load_runtime_upload_capacity_bootstrap_ms']=float('nan')
                    if mode=='nan_time': first['search_call_ms']=float('nan')
                    if mode=='bool_time': first['search_call_ms']=True
                    if mode=='negative_time': first['search_call_ms']=-1
                    if mode=='wrong_query_bytes': first['query_h2d_bytes']+=4
                    if mode=='extra_result_bytes': first['topk_d2h_bytes']+=4
                    if mode=='wrong_poison': first['debug_poison_h2d_bytes']-=4
                    if mode=='repeat_bits': (actual/'call2_A/top_ids.bin').write_bytes(bytes(6*3*4))
                    if mode=='bad_ranking':
                        for label in ('call1_B','call4_B'): (actual/label/'top_ids.bin').write_bytes(bytes(6*3*4))
                    if mode=='bad_workspace': session['workspace_capacity_bytes'][1]+=64
                    r.write_json(actual/'search_session.json',session)
                with patch.object(tool.chain.score,'execute',child),patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':'PARENT_UNCHANGED'}),redirect_stdout(io.StringIO()):
                    invoke=lambda:tool.run(inputs,output,runner,*args)
                    if mode=='pass': invoke()
                    else: fails(invoke)
                    assert os.environ['ASCEND_CUSTOM_OPP_PATH']=='PARENT_UNCHANGED'
                report=json.loads((output/'report.json').read_text())
                assert len(calls)==1 and report['status']==('search_boundary_device_calls_and_all_checks_passed' if mode=='pass' else 'failed')
                assert (output/'device').exists()==(mode!='child_failure')
                assert not (output/'npu_timing.json').exists() # No old resident benchmark naming or speedup.
                checks+=4
            with patch.object(tool.chain.score,'execute') as child:
                fails(lambda:tool.run(inputs,base/'run_pass',runner,*args))
                fails(lambda:tool.run(inputs,base/'bad_cap',runner,*args[:-2],1,512))
                assert not child.called; checks+=1
        # Source-order guards, not a compiled C++ or runtime proof of timer placement.
        source=(ROOT/'npu_prototype/search_host/npu_search.h').read_text()
        public=source[source.index('    Result Search(const Queries &queries)'):source.index('    void PoisonForDiagnostic()')]
        assert public.index('Clock::now()')<public.index('SearchImpl(queries,start)'); checks+=1
        body=source[source.index('    Result SearchImpl('):source.index('\npublic:')]
        assert body.index('PackQueries(')<body.index('search upload offsets')<body.index('search score execute')
        assert body.index('search complete chain synchronize')<body.index('search download final IDs')<body.index('result.searchMs =')
        assert 'DownloadDebug' not in body and 'PoisonForDiagnostic' not in body and 'expected_scores.bin' not in body
        checks+=3
    print(f'PASS: {checks} Search wrapper/raw-ID/provenance/timing-record/ranking/fail-stop and source-order guards. '
          'FAKE artifacts/MOCK child removed; source guards are not compiled/runtime timing proof. No C++ build or NPU execution.')


if __name__=='__main__': main()
