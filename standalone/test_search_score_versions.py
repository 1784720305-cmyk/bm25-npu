"""Real host fixtures/checkers; FAKE binaries/install, MOCK links and children; no CANN/device/performance evidence."""
from contextlib import redirect_stdout
import copy
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
from standalone import compare_search_score_versions as tool, resident_index as r, scale_fixture
from standalone import validate_query_score_range_boundaries as boundary
from standalone.test_search_scale import mock_session
from standalone.test_resident_chain import mock_outputs


def main():
    count=0
    def fails(call):
        nonlocal count
        try: call()
        except (ValueError,OSError,subprocess.SubprocessError): count+=1
        else: raise AssertionError('Invalid comparison accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='search_score_versions_MOCK_ONLY_',dir=ROOT/'results') as tmp,redirect_stdout(io.StringIO()):
        base=Path(tmp); vendors=base/'vendors'; vendor=vendors/'bm25_query_score_range'
        diagnostic=base/'FAKE_DIAGNOSTIC'; runner=base/'FAKE_MEASURE'; score_runner=base/'FAKE_SCORE'
        for path in (diagnostic,runner,score_runner): path.write_bytes(b'TEST ONLY FAKE EXECUTABLE')
        for slot in tool.SLOTS+('bm25_query_score_range',):
            directory=vendors/slot; (directory/'op_api/lib').mkdir(parents=True)
            (directory/'op_api/lib/libcust_opapi.so').write_bytes(('FAKE '+slot).encode())
            inc=directory/'op_api/include'; inc.mkdir()
            for header in ('aclnn_bm25_query_score.h','aclnn_bm25_top_k.h','aclnn_bm25_top_k_merge.h'):
                (inc/header).write_bytes(b'FAKE DECLARATION -- NOT COMPILED')
        range_install={'vendor':str(vendor.resolve()),'TEST_ONLY_FAKE':True}
        baseline={'TEST_ONLY_FAKE':True,'score':{'variant':'old'},'topk_acceptance':{'workspace_bytes':16777216}}
        boundary_input=base/'boundary_input'; accepted=base/'boundary_accepted'
        boundary.boundaries.prepare(boundary_input)
        def boundary_child(command,logfile,env):
            inputs=Path(command[1]).parent; index,records,unused=boundary.validate_inputs(inputs)
            actual=Path(command[3]); actual.mkdir()
            session={'status':'device_score_session_completed_reference_checked','device':2,
                'index_upload_batches':1,'index_tensor_h2d_calls':5,'index_upload_bytes':index['index_bytes'],
                'index_readback_byte_identical':True,'index_device_allocations_reused':True,
                'debug_index_d2h_calls':5,'debug_index_d2h_bytes':index['index_bytes'],
                'host_source_sha256':tool.checks.sources(boundary.HOST_SOURCES),'requests':[]}
            for i,name in enumerate(boundary.SEQUENCE):
                item=records[name]
                (actual/f'call{i}_{name}_scores.bin').write_bytes((inputs/name/'expected_scores.bin').read_bytes())
                session['requests'].append({'label':f'call{i}_{name}','queries':item['num_queries'],
                    'logical_word_count':item['query_term_count'],'logical_query_bytes':item['query_input_bytes'],
                    'query_h2d_bytes':item['query_input_bytes']+(4 if item['query_term_count']==0 else 0),
                    'query_tensor_h2d_calls':2,'score_d2h_bytes':4*item['num_queries']*index['num_docs'],
                    'score_d2h_calls':1,'debug_poison_h2d_calls':1,
                    'debug_poison_h2d_bytes':4*max(x['num_queries'] for x in records.values())*index['num_docs'],
                    'workspace_bytes':16777216,'output_poisoned':True,'fresh_executor':True,'full_reference_checked':True,
                    'repeated_query_bits_checked':i in (2,4)})
            r.write_json(actual/'device_session.json',session); logfile.write_bytes(b'MOCK SCORE CHILD ONLY')
        with patch.object(boundary,'validate_install',return_value=range_install),patch.object(boundary,'execute',boundary_child):
            boundary.run_suite(boundary_input,accepted,score_runner,vendor,base,2)
        proof=tool.validate_boundaries(accepted,range_install); count+=1
        meta=accepted/'report.json'; saved=json.loads(meta.read_text())
        for key,value in (('status','compiled_only'),('kind','old'),('suite_sha256','wrong')):
            bad=copy.deepcopy(saved); bad[key]=value; r.write_json(meta,bad)
            fails(lambda:tool.validate_boundaries(accepted,range_install))
        r.write_json(meta,saved)
        file=accepted/'split_tail/device/call0_A_scores.bin'; data=file.read_bytes(); file.write_bytes(bytes(len(data)))
        fails(lambda:tool.validate_boundaries(accepted,range_install)); file.write_bytes(data)
        child_meta=accepted/'single_tail/report.json'
        child_saved=json.loads(child_meta.read_text())
        for key,value in (('host_source_sha256',{}),('tool_source_sha256',{}),
                          ('runner_sha256','0'*64),('actual_max_query_terms',{'A':63,'B':64,'empty':0})):
            bad=copy.deepcopy(child_saved); bad[key]=value; r.write_json(child_meta,bad)
            root_bad=copy.deepcopy(saved); root_bad['cases'][0]['report_sha256']=tool.digest(child_meta)
            r.write_json(meta,root_bad)
            fails(lambda:tool.validate_boundaries(accepted,range_install))
        r.write_json(child_meta,child_saved); r.write_json(meta,saved)
        # Validate ownership/hash logic with synthetic alias objects. Windows link
        # privilege is unavailable; this does not claim OS symlink execution.
        targets={slot:(vendors/slot) for slot in tool.SLOTS}
        class Alias:
            def __init__(self,target): self.target=target
            def is_symlink(self): return True
            def resolve(self,strict=False): return self.target.resolve(strict=strict)
            def __truediv__(self,name): return self.target/name
        class AliasDirectory:
            def __truediv__(self,slot): return Alias(targets[slot])
        mapped=tool.validate_runtime(AliasDirectory(),targets)
        assert mapped['bm25_query_score']['actual_directory']==str(targets['bm25_query_score'].resolve()); count+=1
        with patch.object(Alias,'is_symlink',return_value=False): fails(lambda:tool.validate_runtime(AliasDirectory(),targets))
        wrong=dict(targets); wrong['bm25_query_score']=vendor
        fails(lambda:tool.validate_runtime(AliasDirectory(),wrong))
        fails(lambda:tool.create_runtime(vendors,targets))

        source=base/'source'; inputs=base/'inputs'
        scale_fixture.generate(source,128,8,10,128,20261014,0,'performance',3)
        tool.fixture.prepare(source,inputs); index,records,snapshot=tool.fixture.validate_inputs(inputs)
        cache=tool.checks.cache_inputs(inputs,index,records); full={key:value['scores'] for key,value in cache.items()}
        def fake_runtime(path,targets):
            path.mkdir(parents=True)
            for slot,target in targets.items():
                (path/slot).mkdir(); (path/slot/'target.txt').write_text(str(target),encoding='utf-8')
            return fake_validate_runtime(path,targets)
        def fake_validate_runtime(path,targets):
            assert all((path/slot/'target.txt').read_text(encoding='utf-8')==str(target) for slot,target in targets.items())
            return {slot:str(target) for slot,target in targets.items()}
        for mode in ('pass','preflight_failure','preflight_bad_ids','measurement_failure','bad_rank','nan_time','stale_host','wrong_transfer'):
            output=base/('run_'+mode); commands=[]
            def child(command,logfile,env):
                commands.append(command)
                is_diagnostic=command[0]==str(diagnostic)
                runtime=Path(command[5] if is_diagnostic else command[6])
                candidate=runtime.parent.name=='runtime_candidate'
                assert command[1]==str(inputs/'index')
                assert env['ASCEND_CUSTOM_OPP_PATH']==':'.join(str((runtime/slot).resolve()) for slot in tool.SLOTS)
                logfile.write_bytes(b'MOCK SEARCH CHILD -- NO DEVICE')
                if (is_diagnostic and mode=='preflight_failure') or (not is_diagnostic and mode=='measurement_failure'):
                    raise subprocess.CalledProcessError(7,command)
                actual=Path(command[3] if is_diagnostic else command[4])
                if is_diagnostic:
                    assert [line.split('\t')[0] for line in Path(command[2]).read_text().splitlines()]==[f'call{i}_{name}' for i,name in enumerate(tool.score.SEQUENCE)]
                    session=mock_session(False,actual,index,records,10,8,8,1,8,24,full)
                    if mode=='preflight_bad_ids':
                        (actual/'call0_A/top_ids.bin').write_bytes(bytes(8*10*4))
                    r.write_json(actual/'search_session.json',session)
                    return
                assert command[2:4]==[str(output/'raw_queries/A.txt'),str(output/'raw_queries/B.txt')]
                actual.mkdir(); warmup,repeats=map(int,command[-2:])
                session={'status':'search_benchmark_calls_completed_checks_pending','backend':'npu','docs':128,
                    'queries':8,'k':10,'partitions':8,'capacity_queries':8,'capacity_words':24,
                    'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory','timer_policy':tool.checks.TIMER,
                    'index_load_batches':1,'index_bytes':index['index_bytes'],'cold_init_ms':1.,'cold_synthetic_empty_queries':8,
                    'checkpoint_search_calls':4,'warmup_cycles':warmup,'warmup_search_calls':2*warmup,'warmup_topk_bit_checks':2*warmup,
                    'measured_cycles':repeats,'measured_search_calls':2*repeats,'measured_topk_bit_checks':2*repeats,
                    'measured_debug_download_calls':0,'measured_poison_calls':0,'measured_file_io_calls':0,
                    'diagnostic_checkpoints_outside_measurement':True,'index_readback_byte_identical':True,
                    'host_source_sha256':tool.checks.sources(tool.checks.NPU_SOURCES),'device':2,'index_tensor_h2d_calls':5,
                    'pipeline_stages':3,'debug_index_d2h_calls':5,'intermediate_host_roundtrips':0,
                    'workspace_capacity_bytes':[16777216]*3,'samples':[]}
                for cycle in range(repeats):
                    for slot in range(2):
                        case='B' if slot^(cycle&1) else 'A'; elapsed=.1 if candidate else .2
                        session['samples'].append({'cycle':cycle,'case':case,'call_wall_ms':elapsed,
                            'internal_search_ms':elapsed*.99,'query_h2d_bytes':168,'topk_d2h_bytes':640})
                for label in tool.checks.CHECKPOINTS: mock_outputs(actual/label,full[label[-1]],128,8,10,8)
                if mode=='bad_rank':
                    for label in ('before_A','after_A'): (actual/label/'top_ids.bin').write_bytes(bytes(8*10*4))
                if mode=='nan_time': session['samples'][0]['call_wall_ms']=float('nan')
                if mode=='stale_host': session['host_source_sha256']={}
                if mode=='wrong_transfer': session['samples'][0]['query_h2d_bytes']=0
                r.write_json(actual/'benchmark_session.json',session)
            with patch.object(tool.scale.chain,'validate_install',return_value=baseline),patch.object(tool.score,'validate_install',return_value=range_install),patch.object(tool,'create_runtime',fake_runtime),patch.object(tool,'validate_runtime',fake_validate_runtime),patch.object(tool.score,'execute',child),patch.object(tool.scale.chain.score,'execute',child),patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':'PARENT'}):
                invoke=lambda:tool.run(inputs,output,runner,diagnostic,vendors,base,base,base,base,base,accepted,2,8,2,1,10)
                if mode=='pass': invoke()
                else: fails(invoke)
                assert os.environ['ASCEND_CUSTOM_OPP_PATH']=='PARENT'; count+=1
            report=json.loads((output/'report.json').read_text())
            assert report['status']==(tool.STATUS if mode=='pass' else 'failed'); count+=1
            if mode.startswith('preflight_'):
                assert len(commands)==1 and not report['per_round']; count+=1
            if mode=='pass':
                assert len(commands)==6 and report['metric']=='batch_latency_and_search_time_throughput'
                assert [row['process_order'] for row in report['per_round']]==[['baseline','candidate'],['candidate','baseline']]
                assert report['paired_summary']['A']['candidate_faster_rounds']==2
                assert report['paired_summary']['A']['paired_baseline_over_candidate_ratios']==[2.,2.]
                assert len(report['preflight'])==2 and all(len(row['cross_checks'])==4 for row in report['per_round']); count+=5
        with patch.object(tool.score,'execute') as child:
            fails(lambda:tool.run(inputs,base/'run_pass',runner,diagnostic,vendors,base,base,base,base,base,accepted))
            fails(lambda:tool.run(inputs,base/'odd_rounds',runner,diagnostic,vendors,base,base,base,base,base,accepted,rounds=3))
            assert not child.called; count+=1
        oversized_source=base/'source_q32'; oversized_input=base/'input_q32'
        scale_fixture.generate(oversized_source,128,32,10,128,20261014,0,'performance',3)
        tool.fixture.prepare(oversized_source,oversized_input)
        with patch.object(tool.score,'execute') as child:
            fails(lambda:tool.run(oversized_input,base/'over64_batch',runner,diagnostic,vendors,
                base,base,base,base,base,accepted,2,8,2,1,10))
            assert not child.called and not (base/'over64_batch').exists(); count+=1
        # Public total-capacity guard must remain inside the unchanged timed Search.
        search=(ROOT/'npu_prototype/search_host/npu_search.h').read_text()
        pack=(ROOT/'npu_prototype/search_host/search_inputs.h').read_text()
        assert 'auto packed = PackQueries(queries,maxQ,maxWords);' in search
        assert 'row.size() <= maxWords-total' in pack; count+=1
    print(f'PASS: {count} prior-device-record/provenance/full-Search preflight/paired-order/timing/strict-ranking/fail-stop checks. '
          'Real host fixtures and independent oracles; FAKE executables/install, synthetic links and MOCK children only. '
          'No OS symlink proof, C++ build, ACLNN or NPU/performance evidence.')


if __name__=='__main__': main()
