"""Affected256 profile orchestration/provenance/CSV checks. MOCK processes and synthetic CSV only."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import profile_fused_t256 as p, scale_fixture, search_scale_fixture
from standalone.test_fused_profile import csv_files
from standalone.test_fused_search_closure import session as fake_session
from standalone.test_resident_chain import mock_outputs
from standalone.test_search_profile import write_csv


def main():
    count=0
    def reject(call):
        nonlocal count
        try:call()
        except (ValueError,OSError,subprocess.SubprocessError):count+=1
        else:raise AssertionError('Bad256 profile/command/proof accepted')
    with tempfile.TemporaryDirectory(prefix='profile_t256_MOCK_ONLY_',dir=ROOT/'results') as tmp,redirect_stdout(io.StringIO()):
        base=Path(tmp);source=base/'source';inputs=base/'inputs'
        scale_fixture.generate(source,1024,8,10,128,20261014,0,'performance',3)
        search_scale_fixture.prepare(source,inputs)
        index,records,snapshot=search_scale_fixture.validate_inputs(inputs)
        cache=p.close.checks.cache_inputs(inputs,index,records)
        runner=base/'FAKE_NOT_EXECUTABLE';runner.write_bytes(b'MOCK NOT ELF')
        profiler=base/'FAKE_msprof';profiler.write_bytes(b'MOCK NOT TOOL')
        vendors=base/'installed/vendors';runtime=base/'aliases/vendors';prior=base/'prior_proof';prior.write_bytes(b'MOCK PRIOR')
        paths=[str(vendors/p.t.VENDOR),str(vendors/'bm25_topk_merge')]
        installation={'MOCK':'original128_proof'};candidate={'vendor':paths[0],'MOCK':'accepted256_proof'}
        evidence=(index,records,snapshot,installation,candidate,runtime,{str(prior):p.digest(prior)},
            [{'function':'FusedScoreTopKSimt_MOCK','used_register_number':125,'stack_size_bytes':68}])
        a=SimpleNamespace(input=inputs,previous=base/'previous',completed=base/'completed',project=source,vendors=vendors,
            merge_project=base/'merge',runner=runner,diagnostic_runner=runner,cpu_runner=runner,acceptance=base/'acceptance',
            device=5,output=base/'application')
        commands=[]
        def actual_mock(cmd,log,env):
            commands.append(cmd)
            assert cmd[0]==str(runner) and cmd[6]==str(runtime) and cmd[7:13]==['10','32','8','24','5','10']
            assert env['ASCEND_CUSTOM_OPP_PATH']==':'.join(paths)
            actual=Path(cmd[4]);actual.mkdir()
            session=fake_session(index,records);session.update(docs=1024,partitions=32,
                measured_cycles=10,measured_search_calls=20,measured_topk_bit_checks=20)
            session['samples']=session['samples'][:20]
            for item in session['samples']:item.update(call_wall_ms=.24,internal_search_ms=.23)
            for label in p.close.checks.CHECKPOINTS:mock_outputs(actual/label,cache[label[-1]]['scores'],1024,8,10,32)
            p.r.write_json(actual/'benchmark_session.json',session);log.write_bytes(b'MOCK actual runner, no device\n')
        with patch.object(p,'evidence',return_value=evidence),patch.object(p,'unchanged'), \
             patch.object(p.accepted,'opp_paths',return_value=paths), \
             patch.object(p.t.fused,'installed',return_value=installation),patch.object(p.t,'candidate_install',return_value=candidate), \
             patch.object(p.t.fused.score,'execute',actual_mock):
            p.application(a);assert len(commands)==1;count+=1
            record,session=p.validate_record(a.output)
            assert record['simt_threads']==256 and record['candidate_installation']==candidate
            assert record['reused_vf_reports'][0]['used_register_number']==125;count+=2
            root=base/'SYNTHETIC';summary,tasks=csv_files(root,record,session)
            value=p.analyze_rows(root,a.output,record,session)
            assert value['own_operator_rows']==78 and value['measured_search_calls']==20
            assert value['all_search_calls']==35 and value['all_device_chains']==39 and value['excluded_debug_chains']==4
            assert value['cases']['A']['operators']['Bm25FusedScoreTopK']['median_us']==100
            assert value['cases']['B']['operators']['Bm25FusedScoreTopK']['median_us']==120
            assert value['cases']['A']['device_chain_span_us']['median_us']==106
            assert value['cases']['A']['outside_device_chain_elapsed_us']['median_us']==134
            assert value['kernel_variant']==p.VARIANT and value['optional_raw_counters_by_task'];count+=7
            # Reject wrong device/stream/op/shape/type/blocks, lost tasks and debug phase mistakes.
            for key,bad in (('Device_id','4'),('Stream ID','57'),('OP Type','Bm25QueryScore'),
                            ('Input Shapes','1'),('Task Type','AI_CPU'),('Block Num','0')):
                changed=copy.deepcopy(summary);changed[0][key]=bad;write_csv(root/'op_summary_MOCK.csv',changed)
                reject(lambda:p.analyze_rows(root,a.output,record,session))
            changed=copy.deepcopy(summary);changed[4]['Output Shapes']='8,32,10;8,32,10;1'
            write_csv(root/'op_summary_MOCK.csv',changed);reject(lambda:p.analyze_rows(root,a.output,record,session))
            write_csv(root/'op_summary_MOCK.csv',summary[:-2]);reject(lambda:p.analyze_rows(root,a.output,record,session))
            write_csv(root/'op_summary_MOCK.csv',summary)
            changed=copy.deepcopy(tasks);changed[0]['task_time(us)']='1'
            write_csv(root/'task_time_MOCK.csv',changed);reject(lambda:p.analyze_rows(root,a.output,record,session))
            write_csv(root/'task_time_MOCK.csv',tasks)
            record_file=a.output/'report.json';saved=record_file.read_bytes()
            for key,bad in (('simt_threads',128),('kernel_variant','local32'),('runner_sha256','wrong')):
                changed=copy.deepcopy(record);changed[key]=bad;p.r.write_json(record_file,changed)
                reject(lambda:p.validate_record(a.output))
            record_file.write_bytes(saved)
            selected=a.output/'measurement/actual/after_B/top_ids.bin';old=selected.read_bytes();selected.write_bytes(bytes(len(old)))
            reject(lambda:p.validate_record(a.output));selected.write_bytes(old)
            reject(lambda:p.application(a))
            for mode in ('child_error','wrong_result'):
                arg=copy.copy(a);arg.output=base/mode
                def bad_child(cmd,log,env):
                    if mode=='child_error':raise subprocess.CalledProcessError(9,cmd)
                    actual_mock(cmd,log,env);(Path(cmd[4])/'before_A/top_ids.bin').write_bytes(bytes(320))
                with patch.object(p.t.fused.score,'execute',bad_child):reject(lambda:p.application(arg))
                assert json.loads((arg.output/'report.json').read_text())['status']=='failed';count+=1
        # Parent dispatches one profiler application; only --help subprocess is permitted here.
        for mode in ('pass','first_profiler_error','unsupported_help'):
            arg=copy.copy(a);arg.output=base/('parent_'+mode);seen=[]
            def help_only(cmd,**kwargs):
                assert cmd==[str(profiler),'--help'];seen.append('help')
                text=b'--runtime-api --task-time --ai-core --aic-metrics --python-path PipeUtilization'
                return SimpleNamespace(returncode=0,stdout=text if mode!='unsupported_help' else b'unsupported')
            def profiled(cmd,log,env):
                seen.append('profile')
                assert cmd[0]==str(profiler) and '--aic-metrics=PipeUtilization' in cmd
                assert '_application' in cmd and cmd.count('--cpu-runner')==1
                assert env['ASCEND_CUSTOM_OPP_PATH']==':'.join(paths)
                if mode=='first_profiler_error':raise subprocess.CalledProcessError(9,cmd)
                log.write_bytes(b'MOCK profiler no device\n')
            with patch.object(p,'evidence',return_value=evidence),patch.object(p,'unchanged'), \
                 patch.object(p.shutil,'which',return_value=str(profiler)),patch.object(p.subprocess,'run',help_only), \
                 patch.object(p.accepted,'opp_paths',return_value=paths),patch.object(p.t.fused.score,'execute',profiled), \
                 patch.object(p,'validate_record',return_value=(record,session)), \
                 patch.object(p,'analyze_rows',return_value=value) as analyzer:
                if mode=='pass':p.run(arg);assert seen==['help','profile'] and analyzer.call_count==1;count+=1
                else:reject(lambda:p.run(arg));assert analyzer.call_count==0;count+=1
            if mode=='first_profiler_error':assert json.loads((arg.output/'report.json').read_text())['status']=='failed';count+=1
            if mode=='unsupported_help':assert not arg.output.exists();count+=1
        # This entry contains no build/install/CPU dispatch, not even a resource-copy compilation.
        code=(ROOT/'standalone/profile_fused_t256.py').read_text(encoding='utf-8')
        assert "['bash','build.sh']" not in code and 'resource.run(' not in code and 'bench.run_one(' not in code;count+=1
    print(f'PASS: {count} affected256 profile phase/CSV/provenance/first-error checks; MOCK children and '
        'synthetic CSV only. No msprof execution, CANN compilation or NPU result.')


if __name__=='__main__':main()
