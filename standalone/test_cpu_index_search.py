"""CPU Search protocol/oracle/cross-check tests, FAKE executables and MOCK child only."""
from contextlib import redirect_stdout
from array import array
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import run_cpu_index_search as tool,resident_index as r,query_score_fixture as fixture
from standalone.test_resident_chain import mock_outputs


def make_prior(path,inputs,index,records,snapshot,cpu):
    """Completely MOCK prior NPU report for file-validation unit tests; no device invoked."""
    path.mkdir(); actual=path/'device'; actual.mkdir(); raw=path/'raw_queries'; raw.mkdir()
    executable=path/'FAKE_NPU_NEVER_EXECUTED'; executable.write_bytes(b'TEST ONLY FAKE NPU RUNNER')
    (path/'device.log').write_bytes(b'TEST ONLY MOCK NPU LOG')
    raw_hashes={}; lines=[]
    for name in records:
        unused,queries=r.validate_queries(inputs/name,index); file=raw/(name+'.txt')
        tool.npu.write_queries(file,queries); raw_hashes[str(file)]=tool.digest(file)
    for i,name in enumerate(tool.chain.SEQUENCE): lines.append(f'call{i}_{name}\t{raw/(name+".txt")}')
    plan=path/'requests.tsv'; plan.write_bytes(('\n'.join(lines)+'\n').encode()); raw_hashes[str(plan)]=tool.digest(plan)
    session={'status':'device_search_calls_completed_independent_check_pending','device':2,'docs':6,'k':3,'shards':2,
        'capacity_queries':65,'capacity_words':512,'index_upload_batches':1,'index_tensor_h2d_calls':5,
        'index_upload_bytes':index['index_bytes'],'index_readback_byte_identical':True,'pipeline_stages':3,
        'intermediate_host_roundtrips':0,'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory','timer_policy':tool.npu.TIMER_POLICY,
        'cold_index_load_runtime_upload_capacity_bootstrap_ms':10.,
        'cold_bootstrap':{'synthetic_empty_queries':65,'query_h2d_calls':2,'query_h2d_bytes':532,'topk_d2h_calls':2,'topk_d2h_bytes':1560},
        'workspace_capacity_bytes':[16777216]*3,'debug_index_d2h_calls':5,'debug_index_d2h_bytes':index['index_bytes'],
        'host_source_sha256':{name:tool.digest(ROOT/name) for name in tool.npu.HOST_SOURCES},'requests':[]}
    for i,name in enumerate(tool.chain.SEQUENCE):
        label=f'call{i}_{name}'; source=records[name]; q=source['num_queries']; logical=source['query_input_bytes']
        mock_outputs(actual/label,cpu[name],6,q,3,2)
        session['requests'].append({'label':label,'queries':q,'words':source['query_term_count'],'search_call_ms':.1,
            'logical_query_bytes':logical,'query_h2d_calls':2,'query_h2d_bytes':logical+(4 if source['query_term_count']==0 else 0),
            'topk_d2h_calls':2,'topk_d2h_bytes':8*q*3,'debug_poison_h2d_calls':5,'debug_poison_h2d_bytes':6240,
            'debug_score_candidate_d2h_calls':3,'debug_score_candidate_d2h_bytes':4*q*6+8*q*2*3,
            'packing_inside_Search':True,'topk_received_before_stop':True,'debug_outside_Search':True})
    r.write_json(actual/'search_session.json',session)
    command=[str(executable),str(inputs/'index'),str(plan),str(actual),'2','TEST_ONLY_UNUSED_VENDOR','3','2','65','512','16777216']
    report={'status':'search_boundary_device_calls_and_all_checks_passed','index_id':index['index_id'],'k':3,
        'input_sha256':snapshot,'device':2,'shards':2,'capacity_queries':65,'capacity_words':512,
        'host_source_sha256':session['host_source_sha256'],'tool_source_sha256':{name:tool.digest(ROOT/name) for name in tool.npu.TOOL_SOURCES},
        'command':command,'runner_sha256':tool.digest(executable),'device_log_sha256':tool.digest(path/'device.log'),
        'output_sha256':{f'call{i}_{name}/{file}':tool.digest(actual/f'call{i}_{name}'/file) for i,name in enumerate(tool.chain.SEQUENCE) for file in tool.chain.FILES},
        'raw_query_sha256':raw_hashes,'installation':{'topk_acceptance':{'workspace_bytes':16777216}},'TEST_ONLY_MOCK':True}
    r.write_json(path/'report.json',report); return report


def main():
    checks=0
    def fails(call):
        nonlocal checks
        try: call()
        except (ValueError,OSError,subprocess.SubprocessError): checks+=1
        else: raise AssertionError('invalid CPU Search state accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cpu_index_search_MOCK_ONLY_',dir=ROOT/'results') as temp:
        base=Path(temp); inputs=base/'input'
        with redirect_stdout(io.StringIO()): fixture.prepare(inputs)
        index,records,snapshot=tool.chain.score.validate_inputs(inputs); unused,arrays=r.validate_index(inputs/'index')
        cpu={}
        for name in records:
            unused,queries=r.validate_queries(inputs/name,index); cpu[name]=r.diagnostic_scores(index,arrays,queries)
        prior=base/'MOCK_NPU_PRIOR'; original=make_prior(prior,inputs,index,records,snapshot,cpu)
        with redirect_stdout(io.StringIO()): tool.validate_npu_reference(prior,inputs,index,records,snapshot,3)
        checks+=1
        for key,value in (('status','old_prepared_array_compute_passed'),('k',1),('index_id','WRONG_INDEX')):
            bad=dict(original); bad[key]=value; r.write_json(prior/'report.json',bad)
            with redirect_stdout(io.StringIO()): fails(lambda:tool.validate_npu_reference(prior,inputs,index,records,snapshot,3))
        r.write_json(prior/'report.json',original)
        runner=base/'FAKE_CPU_NEVER_EXECUTED'; runner.write_bytes(b'TEST ONLY FAKE CPU RUNNER')
        for mode in ('pass','child_failure','wrong_contract','old_timer','precomputed','bad_threads','bad_time',
                     'index_changed','stale_host','wrong_words','wrong_bits','bad_ids','repeat_bits','no_lookup'):
            output=base/('run_'+mode); calls=[]
            def child(command,logfile,env):
                calls.append(command); assert command[-6:]==['4','3','2','65','512','raw_query_ids_v1']
                assert env.get('ASCEND_CUSTOM_OPP_PATH')=='PARENT_UNCHANGED'
                logfile.write_bytes(b'TEST ONLY MOCK CPU CHILD; NO C++ EXECUTED')
                if mode=='child_failure': raise subprocess.CalledProcessError(7,command)
                actual=Path(command[3]); actual.mkdir()
                session={'status':'cpu_index_search_calls_completed_independent_check_pending','docs':6,'k':3,
                    'document_shards':2,'requested_threads':4,'actual_threads':4,'capacity_queries':65,'capacity_words':512,
                    'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory','timer_policy':tool.TIMER_POLICY,
                    'index_storage':'host_memory','query_index_loads':1,'index_bytes':index['index_bytes'],
                    'host_index_byte_identical_to_source':True,'debug_source_reread_batches':1,
                    'cold_index_load_threads_capacity_bootstrap_ms':2.,'cold_bootstrap_empty_queries':65,
                    'fp32_flags':'-O3 -march=native -ffp-contract=off -fno-fast-math','cpu_needs_backend_word_packing':False,
                    'diagnostic_poison_before_calls':True,'diagnostic_files_outside_Search':True,
                    'host_source_sha256':{name:tool.digest(ROOT/name) for name in tool.HOST_SOURCES},'requests':[]}
                for i,name in enumerate(tool.chain.SEQUENCE):
                    label=f'call{i}_{name}'; q=records[name]['num_queries']; mock_outputs(actual/label,cpu[name],6,q,3,2)
                    for extra in ('partial_scores.bin','partial_ids.bin'): (actual/label/extra).unlink()
                    session['requests'].append({'label':label,'queries':q,'words':records[name]['query_term_count'],
                        'search_call_ms':.05,'topk_output_bytes':8*q*3,'debug_full_scores_bytes':4*q*6,
                        'index_lookup_inside_Search':True,'topk_ready_before_stop':True,'debug_outside_Search':True})
                if mode=='wrong_contract': session['input_contract']='expanded_TF_arrays'
                if mode=='old_timer': session['timer_policy']='compute only'
                if mode=='precomputed': session['cpu_needs_backend_word_packing']=True
                if mode=='bad_threads': session['actual_threads']=True
                if mode=='bad_time': session['requests'][0]['search_call_ms']=float('nan')
                if mode=='index_changed': session['host_index_byte_identical_to_source']=False
                if mode=='stale_host': session['host_source_sha256'][tool.HOST_SOURCES[0]]='0'*64
                if mode=='wrong_words': session['requests'][0]['words']+=1
                if mode=='no_lookup': session['requests'][0]['index_lookup_inside_Search']=False
                if mode=='repeat_bits': (actual/'call2_A/top_ids.bin').write_bytes(bytes(6*3*4))
                if mode=='bad_ids':
                    for label in ('call1_B','call4_B'): (actual/label/'top_ids.bin').write_bytes(bytes(6*3*4))
                if mode=='wrong_bits':
                    for label in ('call1_B','call4_B'):
                        file=actual/label/'scores.bin'; raw=file.read_bytes(); file.write_bytes(b'\x01\0\0\0'+raw[4:])
                r.write_json(actual/'cpu_search_session.json',session)
            with patch.object(tool.chain.score,'execute',child),patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':'PARENT_UNCHANGED'}),redirect_stdout(io.StringIO()):
                invoke=lambda:tool.run(inputs,output,runner,4,3,2,65,512,prior)
                if mode=='pass': invoke()
                else: fails(invoke)
            report=json.loads((output/'report.json').read_text()); assert len(calls)==1
            assert report['status']==('cpu_index_search_and_independent_checks_passed' if mode=='pass' else 'failed')
            assert (output/'cpu').exists()==(mode!='child_failure'); checks+=3
            if mode=='pass': assert len(report['npu_cross_checks'])==5; checks+=1
        with patch.object(tool.chain.score,'execute') as child:
            fails(lambda:tool.run(inputs,base/'run_pass',runner,4,3,2,65,512))
            fails(lambda:tool.run(inputs,base/'bad_cap',runner,4,3,2,1,512))
            fails(lambda:tool.run(inputs,base/'bad_shards',runner,4,3,5,65,512))
            assert not child.called; checks+=1
        source=(ROOT/'cpu_reference/index_search/cpu_index_search.h').read_text()
        body=source[source.index('    Result Search(const Queries &queries)'):source.index('    void PoisonForDiagnostic()')]
        assert body.index('Clock::now()')<body.index('std::lower_bound')<body.index('TopKDocumentShards')<body.index('result.searchMs=')
        assert 'PackQueries(' not in body and 'expected_scores.bin' not in source and 'acl/' not in source; checks+=3
    print(f'PASS: {checks} CPU index-Search protocol/oracle/strict-NPU-file-cross/timer-record/fail-stop guards. '
          'FAKE CPU/NPU executables, MOCK child/prior and temporary arrays removed. No C++ build, real CPU Search or NPU execution.')


if __name__=='__main__': main()
