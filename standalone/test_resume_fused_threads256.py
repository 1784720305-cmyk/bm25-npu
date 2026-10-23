"""MOCK path-resolution/reused-proof/first-error tests only; no CANN or NPU execution."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resume_fused_threads256 as t, scale_fixture, search_scale_fixture
from standalone.test_fused_search_closure import session as mock_session
from standalone.test_resident_chain import mock_outputs


def snapshot(path):return {p.relative_to(path).as_posix():p.read_bytes() for p in path.rglob('*') if p.is_file()}


def main():
    count=0
    def reject(call):
        nonlocal count
        try:call()
        except (ValueError,OSError,subprocess.SubprocessError):count+=1
        else:raise AssertionError('Invalid Search/proof accepted')
    with tempfile.TemporaryDirectory(prefix='resume_t256_MOCK_ONLY_',dir=ROOT/'results') as tmp,redirect_stdout(io.StringIO()):
        base=Path(tmp);source=base/'source';inputs=base/'inputs'
        scale_fixture.generate(source,1024,8,10,128,20261014,0,'performance',3)
        search_scale_fixture.prepare(source,inputs);index,records,unused=search_scale_fixture.validate_inputs(inputs)
        cache=t.close.checks.cache_inputs(inputs,index,records)
        raw=base/'raw';raw.mkdir()
        for name in ('A','B'):t.close.search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
        runner=base/'FAKE_NOT_EXECUTABLE';runner.write_bytes(b'MOCK NO DEVICE')
        vendors=base/'alias/vendors';real=base/'installed/vendors'
        real_fused=real/'bm25_fused_score_topk_t256';real_merge=real/'bm25_topk_merge'
        real_fused.mkdir(parents=True);real_merge.mkdir()
        mapping={vendors/'bm25_fused_score_topk':real_fused,vendors/'bm25_topk_merge':real_merge}
        original_resolve=Path.resolve
        def resolve(path,strict=False):
            return original_resolve(mapping[path],strict=strict) if path in mapping else original_resolve(path,strict=strict)
        with patch.object(Path,'resolve',resolve):
            assert t.opp_paths(vendors)==[str(real_fused),str(real_merge)];count+=1
            for mode in ('pass','preflight_pass','bad_ids','bad_timer','child_error'):
                output=base/mode;calls=[]
                def child(command,log,env):
                    calls.append(command)
                    assert command[6]==str(vendors) # Alias retained solely for the existing C++ API loader.
                    assert env['ASCEND_CUSTOM_OPP_PATH']==str(real_fused)+':'+str(real_merge)
                    assert os.environ['ASCEND_CUSTOM_OPP_PATH']=='PARENT_UNCHANGED'
                    if mode=='child_error':raise subprocess.CalledProcessError(1,command)
                    actual=Path(command[4]);actual.mkdir()
                    for label in t.close.checks.CHECKPOINTS:
                        mock_outputs(actual/label,cache[label[-1]]['scores'],1024,8,10,32)
                    data=mock_session(index,records);data.update(docs=1024,partitions=32)
                    if mode=='preflight_pass':
                        assert command[11:13]==['1','10']
                        data.update(warmup_cycles=1,warmup_search_calls=2,warmup_topk_bit_checks=2,
                            measured_cycles=10,measured_search_calls=20,measured_topk_bit_checks=20)
                        data['samples']=data['samples'][:20]
                    if mode=='bad_ids':(actual/'before_A/top_ids.bin').write_bytes(bytes(320))
                    if mode=='bad_timer':data['samples'][0]['query_h2d_bytes']=0
                    t.r.write_json(actual/'benchmark_session.json',data);log.write_bytes(b'MOCK Search execution\n')
                with patch.object(t.fused.score,'execute',child),patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':'PARENT_UNCHANGED'}):
                    action=lambda:t.measure(output,inputs,index,records,cache,raw,runner,vendors,5,8,24,
                        1 if mode=='preflight_pass' else 5,10 if mode=='preflight_pass' else 30)
                    if mode in ('pass','preflight_pass'):assert action()['summary']['A']['median_batch_ms']==.1;count+=1
                    else:reject(action)
                assert len(calls)==1
                assert json.loads((output/'report.json').read_text())['status']==(
                    'canonical_opp_complete_search_and_checkpoints_passed' if mode in ('pass','preflight_pass') else 'failed');count+=1
            reject(lambda:t.opp_paths(base/'missing'))
            reject(lambda:t.measure(base/'bad_bounds',inputs,index,records,cache,raw,runner,vendors,5,8,24,0,1))
            # Reuse finished acceptance by byte/provenance checks, never by running its requests again.
            old=base/'old';accept=old/'acceptance';device=accept/'device';device.mkdir(parents=True)
            binary_map={str(runner):t.digest(runner)};frozen=t.close.checks.sources(('standalone/fused_threads256.py',))
            session={'status':'device_fused_chain_debug_modes_and_rankings_checked','device':5,'docs':1024,'k':10,'shards':32,
                'pipeline_stages_per_mode':2,'index_upload_batches':1,'index_readback_byte_identical':True,
                'host_source_sha256':t.close.checks.sources(t.fused.HOST_SOURCES),'requests':[{}]*5}
            for i,name in enumerate(t.fused.SEQUENCE):
                for mode,extra in (('debug1','scores.bin'),('debug0','debug_placeholder.bin')):
                    for name2 in t.fused.CANDIDATES+(extra,):
                        file=device/f'call{i}_{name}'/mode/name2;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'MOCK')
            t.r.write_json(device/'device_session.json',session);(accept/'device.log').write_bytes(b'MOCK device log')
            (accept/'requests.tsv').write_bytes(('\n'.join(f'call{i}_{name}\t{inputs/name}' for i,name in enumerate(t.fused.SEQUENCE))+'\n').encode())
            runtime=old/'runtime/vendors';candidate={'MOCK_candidate':True};installation={'MOCK_baseline':True}
            accepted={'status':'t256_actual_workload_correctness_passed','session':session,'checks':[{}]*5,
                'command':[str(runner),str(inputs/'index'),str(accept/'requests.tsv'),str(device),'5',str(runtime),'10','32'],
                'output_sha256':{p.relative_to(device).as_posix():t.digest(p) for p in device.rglob('*.bin')},
                'log_sha256':t.digest(accept/'device.log')}
            t.r.write_json(accept/'report.json',accepted)
            record={'status':'failed','device':5,'input_sha256':{},'baseline_installation':installation,
                'candidate_installation':candidate,'frozen_source_sha256':frozen,'runner_sha256':binary_map,
                'prior_proofs_sha256':{},'acceptance':accepted}
            t.r.write_json(old/'report.json',record);before=snapshot(old)
            expected_paths=[str((base/'vendors'/t.previous.VENDOR).resolve()),str((base/'vendors/bm25_topk_merge').resolve())]
            with patch.object(t.previous,'candidate_install',return_value=candidate),patch.object(t,'opp_paths',return_value=expected_paths), \
                 patch.object(t.previous,'diagnostic',side_effect=AssertionError('do not repeat acceptance')):
                validate=lambda:t.validate_previous(old,inputs,base,base/'vendors',base,runner,runner,runner,5,index,records,{},installation)
                assert validate()[1]==candidate and snapshot(old)==before;count+=1
                file=next(device.rglob('*.bin'));saved=file.read_bytes();file.write_bytes(b'CHANGED')
                reject(validate);file.write_bytes(saved)
                bad=copy.deepcopy(record);bad['status']='running';t.r.write_json(old/'report.json',bad);reject(validate)
                bad=copy.deepcopy(record);bad['acceptance']['status']='failed';t.r.write_json(old/'report.json',bad);reject(validate)
                t.r.write_json(old/'report.json',record)
            # The recovered driver cannot compile/install/re-execute the five-request diagnostic.
            a=SimpleNamespace(input=inputs,previous=old,output=base/'resume_mock',project=base,vendors=base/'vendors',
                merge_project=base,runner=runner,diagnostic_runner=runner,cpu_runner=runner,acceptance=base,device=5)
            proof={str(old/'report.json'):t.digest(old/'report.json')}
            with patch.object(t.previous.prior,'evidence',return_value=({'num_docs':262144},{'A':{'num_queries':8,'k':10}}, {},installation,{})), \
                 patch.object(t,'validate_previous',return_value=(record,candidate,runtime,proof)), \
                 patch.object(t.close.checks,'cache_inputs',return_value=cache), \
                 patch.object(t.previous,'candidate_install',return_value=candidate), \
                 patch.object(t.fused,'installed',return_value=installation), \
                 patch.object(t,'opp_paths',return_value=expected_paths), \
                 patch.object(t,'compare',return_value={'decision':{'cases':{},'keep_fused_candidate':False}}), \
                 patch.object(t.previous,'prepare',side_effect=AssertionError('must not build')), \
                 patch.object(t.previous,'diagnostic',side_effect=AssertionError('must not repeat acceptance')), \
                 patch.object(subprocess,'run',side_effect=AssertionError('must not compile/install')):
                t.run(a)
                assert json.loads((a.output/'report.json').read_text())['status']==t.STATUS
                assert snapshot(old)==before;count+=1
        for mode in ('win','lose','preflight_error','first_pair_error','bad_cross'):
            output=base/('pairs_'+mode);output.mkdir();calls=[]
            def measured(directory,*args):
                role='candidate' if args[6]==base/'candidate_runtime' else 'baseline'
                preflight=directory.name=='candidate_path_check';calls.append((role,preflight))
                assert args[10:]==((1,10) if preflight else (5,30))
                if mode=='preflight_error' and preflight:raise ValueError('MOCK missing OPP')
                if mode=='first_pair_error' and not preflight and role=='candidate':raise ValueError('MOCK Search error')
                value=.2 if role=='baseline' else (.15 if mode=='win' else .21)
                return {'directory':str(directory),'summary':{x:{'median_batch_ms':value,'p95_batch_ms':value+.01} for x in ('A','B')}}
            def cpu(directory,*args):
                assert args[9:]==(32,32,32,5,8,24,5,30);calls.append(('cpu',False))
                return {'directory':str(directory),'actual_threads':32,
                    'summary':{x:{'median_batch_ms':.1,'p95_batch_ms':.11} for x in ('A','B')}}
            action=lambda:t.compare(output,base,index,records,{},base,runner,runner,base/'baseline_runtime',base/'candidate_runtime',5,lambda:None)
            with patch.object(t,'measure',measured),patch.object(t.close.bench,'run_one',cpu), \
                 patch.object(t.close,'cross',side_effect=ValueError('MOCK wrong result') if mode=='bad_cross' else None):
                if mode in ('preflight_error','first_pair_error','bad_cross'):reject(action)
                else:
                    result=action();assert len(result['per_round'])==6 and not result['old_partial_rounds_included']
                    assert result['decision']['keep_fused_candidate']==(mode=='win')
                    assert len(result['cpu_rounds'])==(6 if mode=='win' else 0);count+=1
            assert calls[0]==('candidate',True)
            if mode=='preflight_error':assert len(calls)==1
            else:assert calls[1:3]==[('baseline',False),('candidate',False)]
            if mode=='win':assert sum(role=='cpu' for role,unused in calls)==6
            count+=1
    print(f'PASS: {count} affected canonical-OPP/reused-acceptance/no-build/first-error/paired-gate checks; MOCK children only, no CANN or NPU execution.')


if __name__=='__main__':main()
