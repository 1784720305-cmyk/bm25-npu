"""Affected profile phases/provenance/CSV and isolated-resource-copy checks; MOCK child/synthetic counters only."""
import copy
from contextlib import redirect_stdout
from decimal import Decimal
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
from standalone import profile_fused_search as app, analyze_fused_search_profile as analysis, fused_resource_report as res
from standalone import scale_fixture, search_scale_fixture
from standalone.test_fused_search_closure import session as fake_session
from standalone.test_resident_chain import mock_outputs
from standalone.test_search_profile import write_csv


def csv_files(root,record,session):
    root.mkdir();summary=[];tasks=[]
    n,v,m=record['index']['num_docs'],record['index']['vocab_size'],record['index']['posting_count']
    for call,(phase,name,debug) in enumerate(analysis.phases(session)):
        words=1 if phase=='cold' else record['records'][name]['query_term_count']
        start=Decimal('1792081509347637.533')+call*100000
        duration=(100 if name=='A' else 120) if phase=='measured' else 5000
        for pos,(op,us) in enumerate(zip(analysis.OPERATORS,(duration,5))):
            tid=str(2*call+5+pos);stop=start+us
            inputs=f'{v+1};{m};{m};{n};{v};9;{words}' if pos==0 else '8,32,10;8,32,10'
            outputs=('8,32,10;8,32,10;'+(f'8,{n}' if debug else '1')) if pos==0 else '8,10;8,10'
            summary.append({'Device_id':'5','Task ID':tid,'Stream ID':'56','OP Type':op,'Task Type':'AI_VECTOR_CORE',
                'Task Start Time(us)':str(start)+'\t','Task Duration(us)':str(us),'Block Num':str(56 if pos==0 else 8),
                'Input Shapes':'"'+inputs+'"','Output Shapes':'"'+outputs+'"','aiv_vec_ratio':'0.1'})
            tasks.append({'Device_id':'5','kernel_name':op+'_MOCK_ONLY','kernel_type':'AI_VECTOR_CORE','stream_id':'56',
                'task_id':tid,'task_time(us)':str(us),'task_start(us)':str(start)+'\t','task_stop(us)':str(stop)+'\t'})
            start=stop+1
    write_csv(root/'op_summary_MOCK.csv',summary);write_csv(root/'task_time_MOCK.csv',tasks)
    write_csv(root/'api_statistic_MOCK.csv',[{'Device_id':'host','Level':'runtime','API Name':'DeviceReset',
        'Time(us)':'1000000','Count':'1','Avg(us)':'1000000','Min(us)':'1000000','Max(us)':'1000000'}])
    return summary,tasks


def main():
    count=0
    def reject(call):
        nonlocal count
        try:call()
        except (ValueError,OSError,subprocess.SubprocessError):count+=1
        else:raise AssertionError('bad profile evidence accepted')
    with tempfile.TemporaryDirectory(prefix='fused_profile_MOCK_ONLY_',dir=ROOT/'results') as tmp,redirect_stdout(io.StringIO()):
        base=Path(tmp);source=base/'source';inputs=base/'inputs'
        scale_fixture.generate(source,1024,8,10,128,20261014,0,'performance',3)
        search_scale_fixture.prepare(source,inputs);index,records,snapshot=search_scale_fixture.validate_inputs(inputs)
        cache=app.checks.cache_inputs(inputs,index,records);runner=base/'FAKE_NOT_EXECUTABLE';runner.write_bytes(b'MOCK ONLY')
        prior=base/'MOCK_PRIOR';prior.write_bytes(b'NO DEVICE EVIDENCE')
        installed={'fused':{'MOCK':True},'merge':{'MOCK':True}}
        evidence=(index,records,snapshot,installed,{str(prior):app.digest(prior)})
        for mode in ('pass','first_child_error','bad_ids'):
            output=base/mode
            def child(cmd,log,env):
                log.write_bytes(b'MOCK NO DEVICE')
                if mode=='first_child_error':raise subprocess.CalledProcessError(7,cmd)
                actual=Path(cmd[4]);actual.mkdir()
                d=fake_session(index,records);d.update(docs=1024,partitions=32,measured_cycles=10,measured_search_calls=20,measured_topk_bit_checks=20)
                d['samples']=d['samples'][:20]
                for x in d['samples']:x.update(call_wall_ms=.24,internal_search_ms=.23)
                for label in app.checks.CHECKPOINTS:mock_outputs(actual/label,cache[label[-1]]['scores'],1024,8,10,32)
                if mode=='bad_ids':(actual/'before_A/top_ids.bin').write_bytes(bytes(320))
                app.r.write_json(actual/'benchmark_session.json',d)
            with patch.object(app,'evidence',return_value=evidence),patch.object(app.close.fused,'installed',return_value=installed), \
                 patch.object(app.close.fused.score,'execute',child):
                action=lambda:app.application(inputs,base,output,runner,base,source,source,5)
                if mode=='pass':action();count+=1
                else:reject(action)
            assert json.loads((output/'report.json').read_text())['status']==(app.STATUS if mode=='pass' else 'failed');count+=1
        run=base/'pass'
        with patch.object(app.close.fused,'installed',return_value=installed):
            record,session=app.validate_record(run);count+=1
            assert len(analysis.phases(session))==39 and sum(p=='debug' for p,n,d in analysis.phases(session))==4;count+=1
            root=base/'SYNTHETIC_CSV';rows,tasks=csv_files(root,record,session)
            value=analysis.analyze(root,run)
            assert value['own_operator_rows']==78 and value['measured_search_calls']==20 and value['excluded_debug_chains']==4
            assert value['all_search_calls']==35 and value['all_device_chains']==39
            assert value['cases']['A']['operators']['Bm25FusedScoreTopK']['median_us']==100
            assert value['cases']['B']['operators']['Bm25FusedScoreTopK']['median_us']==120
            assert value['cases']['A']['device_chain_span_us']['median_us']==106
            assert value['cases']['A']['outside_device_chain_elapsed_us']['median_us']==134;count+=5
            for key,changed in (('Device_id','4'),('Task Type','AI_CPU'),('OP Type','Bm25TopK'),('Input Shapes','1'),('Block Num','0')):
                bad=copy.deepcopy(rows);bad[0][key]=changed;write_csv(root/'op_summary_MOCK.csv',bad)
                reject(lambda:analysis.analyze(root,run))
            bad=copy.deepcopy(rows);bad[4]['Output Shapes']='8,32,10;8,32,10;1' # first debug chain, not normal
            write_csv(root/'op_summary_MOCK.csv',bad);reject(lambda:analysis.analyze(root,run))
            write_csv(root/'op_summary_MOCK.csv',rows[:-2]);reject(lambda:analysis.analyze(root,run))
            write_csv(root/'op_summary_MOCK.csv',rows)
            bad=copy.deepcopy(tasks);bad[0]['task_time(us)']='1';write_csv(root/'task_time_MOCK.csv',bad)
            reject(lambda:analysis.analyze(root,run));write_csv(root/'task_time_MOCK.csv',tasks)
            old=(run/'measurement/actual/after_A/top_ids.bin').read_bytes()
            (run/'measurement/actual/after_A/top_ids.bin').write_bytes(bytes(320));reject(lambda:app.validate_record(run))
            (run/'measurement/actual/after_A/top_ids.bin').write_bytes(old)
        project=base/'RESOURCE_FAKE_PROJECT';(project/'op_kernel').mkdir(parents=True)
        for name in app.close.fused.generator.SOURCES+('CMakePresets.json','build.sh'):
            p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'FAKE COPY INPUT\n')
        cmake=project/'op_kernel/CMakeLists.txt'
        cmake.write_bytes(b'if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")\n add_ops_compile_options(ALL OPTIONS -g -O0)\nendif()\n')
        before={p.relative_to(project).as_posix():app.digest(p) for p in project.rglob('*') if p.is_file()}
        target=base/'RESOURCE_COPY';result=res.prepare(project,target)
        assert result['only_added_compile_option']=='--cce-res-usage' and '--cce-res-usage' in (target/'op_kernel/CMakeLists.txt').read_text()
        assert {p.relative_to(project).as_posix():app.digest(p) for p in project.rglob('*') if p.is_file()}==before;count+=2
        reject(lambda:res.prepare(project,project/'nested'))
        reject(lambda:res.prepare(project,target))
        cmake.write_bytes(b'# unknown actual template\n')
        assert res.prepare(project,base/'unsupported')['status']=='unsupported_actual_template';count+=1
        context=res.resource_lines('ignore\nFusedScoreTopKSimt\n Registers count: 70\n Stack size: 0\n')
        assert any('Stack size: 0' in x for x in context);count+=1
        # Resource build subprocess is MOCKED. Confirm only the copy is built and no install/runtime command is issued.
        fake_root=base/'RESOURCE_RUN_MOCK';(fake_root/'results').mkdir(parents=True)
        original=fake_root/'results/original';(original/'op_kernel').mkdir(parents=True)
        for name in app.close.fused.generator.SOURCES+('CMakePresets.json','build.sh'):
            p=original/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'MOCK ORIGINAL\n')
        (original/'op_kernel/CMakeLists.txt').write_bytes(b'if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")\n add_ops_compile_options(ALL OPTIONS -g -O0)\nendif()\n')
        (original/'build_out').mkdir();cfg='CMAKE_BUILD_TYPE:STRING=Release\nASCEND_COMPUTE_UNIT:STRING=ascend950\n'
        (original/'build_out/CMakeCache.txt').write_text(cfg)
        hw=fake_root/'results/fused_search_256k_q8_close1_hardware.json'
        app.r.write_json(hw,{'compiler_help':{'command':['/FAKE_COMPILER'],'resource_usage_option_lines':['--cce-res-usage Print resource usage']}})
        fake_install={'fused':{'device_object_sha256':{'MOCK.o':'not-real'}}}
        original_bytes={p.relative_to(original).as_posix():p.read_bytes() for p in original.rglob('*') if p.is_file()}
        for mode in ('pass','compile_failed','flags_changed'):
            report=fake_root/'results'/mode;commands=[]
            def build(cmd,**kwargs):
                commands.append(cmd);target=kwargs['cwd'];assert target==report/'project_copy' and cmd==['bash','build.sh']
                (target/'build_out').mkdir();(target/'build_out/CMakeCache.txt').write_text(cfg if mode!='flags_changed' else cfg.replace('Release','Debug'))
                kwargs['stdout'].write(b'FusedScoreTopKSimt\nRegisters count: 70\nStack size: 0\n')
                return SimpleNamespace(returncode=7 if mode=='compile_failed' else 0)
            with patch.object(res,'ROOT',fake_root),patch.object(res.fused,'installed',return_value=fake_install), \
                 patch.object(res.shutil,'which',return_value='/FAKE_COMPILER'),patch.object(res.subprocess,'run',build):
                action=lambda:res.run(original,report,runner,base,base)
                if mode=='pass':assert action()['accepted_artifacts_unchanged'];count+=1
                else:reject(action)
            assert commands==[['bash','build.sh']]
            assert {p.relative_to(original).as_posix():p.read_bytes() for p in original.rglob('*') if p.is_file()}==original_bytes;count+=1
    print(f'PASS: {count} affected frozen-profile/extra-debug-phase/CSV/first-error/resource-copy checks. MOCK children and synthetic CSV only; no compilation, msprof or NPU execution.')


if __name__=='__main__':main()
