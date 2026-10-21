"""Real host fixture/oracles, FAKE install/binary and MOCK NPU child; not C++ or device evidence."""
from array import array
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import run_fused_chain as tool, scale_fixture
from standalone.test_resident_chain import mock_outputs


def fake_session(actual,index,records,references,k,shards,device):
    """Explicitly MOCKED device arrays/session; never used by the production wrapper."""
    actual.mkdir(); n=index['num_docs']; maxq=max(x['num_queries'] for x in records.values())
    stages=3 if shards==64 else 2
    result={'status':'device_fused_chain_debug_modes_and_rankings_checked','device':device,'docs':n,'k':k,'shards':shards,
        'pipeline_stages_per_mode':stages,'intermediate_host_roundtrips':0,'index_upload_batches':1,'index_tensor_h2d_calls':5,
        'index_upload_bytes':index['index_bytes'],'index_readback_byte_identical':True,'index_device_allocations_reused':True,
        'debug_index_d2h_calls':5,'debug_index_d2h_bytes':index['index_bytes'],
        'host_source_sha256':{name:tool.digest(ROOT/name) for name in tool.HOST_SOURCES},'requests':[]}
    poison=8*maxq*n+8*maxq*shards*k+8*maxq*k+(16*maxq*k if shards==64 else 0)+64
    for ordinal,name in enumerate(tool.SEQUENCE):
        record=records[name]; q=record['num_queries']; root=actual/f'call{ordinal}_{name}'; on,off=root/'debug1',root/'debug0'
        mock_outputs(on,references[name],n,q,k,shards)
        if shards==64:
            scores=array('f'); ids=array('i')
            for row in range(q):
                for group in range(2):
                    start=row*n+group*(n//2)
                    chosen=sorted(range(n//2),key=lambda doc:(-references[name][start+doc],doc))[:k]
                    scores.extend(references[name][start+doc] for doc in chosen); ids.extend(chosen)
            (on/'group_scores.bin').write_bytes(tool.r.encode(scores)); (on/'group_ids.bin').write_bytes(tool.r.encode(ids))
        off.mkdir()
        for file in tool.CANDIDATES+(tool.GROUPS if shards==64 else ()): shutil.copyfile(on/file,off/file)
        (off/'debug_placeholder.bin').write_bytes(bytes(4))
        result['requests'].append({'label':f'call{ordinal}_{name}','queries':q,'logical_word_count':record['query_term_count'],'mode_order':[1,0],
            'query_h2d_bytes_per_mode':record['query_input_bytes']+(4 if record['query_term_count']==0 else 0),
            'query_tensor_h2d_calls_per_mode':2,'fresh_executors_per_mode':stages,'chain_synchronizations_per_mode':1,
            'tensor_cleanup_synchronizations_per_mode':1,'debug_poison_h2d_calls_per_mode':8 if shards==64 else 6,
            'debug_poison_h2d_bytes_per_mode':poison,'placeholder_guard_bytes':4*(maxq*n+16),
            'debug_enabled_score_elements':q*n,'debug_disabled_score_elements':1,'outputs_poisoned':True,
            'logical_tail_guards_unchanged':True,'disabled_full_buffer_unchanged':True,'disabled_placeholder_guard_unchanged':True,
            'debug_modes_all_candidate_group_topk_bits_equal':True,'full_reference_checked':True,'exact_device_ranking_checked':True,
            'repeated_all_output_bits_checked':ordinal in (2,4),'workspace_bytes_by_mode':[[16777216]*stages for unused in range(2)]})
    tool.r.write_json(actual/'device_session.json',result)
    return result


def main():
    count=0
    def fails(call):
        nonlocal count
        try: call()
        except (ValueError,OSError,subprocess.SubprocessError): count+=1
        else: raise AssertionError('bad fused proof/result accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fused_chain_MOCK_ONLY_',dir=ROOT/'results') as tmp, redirect_stdout(io.StringIO()):
        base=Path(tmp); project=base/'project'; project.mkdir(); (project/'build.sh').write_bytes(b'FAKE TEMPLATE')
        for file in tool.generator.SOURCES:
            path=project/file; path.parent.mkdir(exist_ok=True); path.write_bytes(b'FAKE TEMPLATE SOURCE')
        tool.r.write_json(project/'CMakePresets.json',{'configurePresets':[{'cacheVariables':{'ASCEND_COMPUTE_UNIT':'old'}}]})
        tool.generator.populate(project)
        vendors=base/'vendors'; vendor=vendors/tool.generator.VENDOR
        header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'; header.parent.mkdir(parents=True)
        header.write_bytes(('\n'.join('aclnnStatus '+name+'('+params+');' for name,params in tool.PROTOTYPES.items())+'\n').encode())
        lib=vendor/'op_api/lib/libcust_opapi.so'; lib.parent.mkdir(); lib.write_bytes(b'FAKE NOT ELF')
        obj=vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_fused_score_top_k/FAKE.o'
        obj.parent.mkdir(parents=True); obj.write_bytes(b'FAKE NOT DEVICE OBJECT')
        build=project/'build_out'; build.mkdir()
        for file in (header,lib,obj): shutil.copyfile(file,build/file.name)
        valid=tool.validate_install(vendor,project); count+=1
        for file in (header,lib,obj,project/tool.generator.SOURCES[0]):
            original=file.read_bytes(); file.write_bytes(original+b'CORRUPTION'); fails(lambda:tool.validate_install(vendor,project)); file.write_bytes(original)
        for changed in (header.read_bytes().replace(b'int64_t debug',b'int32_t debug'),header.read_bytes()+header.read_bytes()):
            original=header.read_bytes(); header.write_bytes(changed); fails(lambda:tool.validate_install(vendor,project)); header.write_bytes(original)
        record_path=project/'bm25_project.json'; original=record_path.read_bytes()
        data=json.loads(original); data['explicit_ub_bytes']=8192; tool.r.write_json(record_path,data)
        fails(lambda:tool.validate_install(vendor,project)); record_path.write_bytes(original)
        merge_project=base/'merge_project'; (vendors/'bm25_topk_merge').mkdir()
        installation={'fused':valid,'merge':{'MOCK':True},'merge_device_objects':{'MOCK.o':'fake'}}
        runner=base/'FAKE_NOT_EXECUTABLE'; runner.write_bytes(b'MOCK RUNNER')
        small=base/'small'; tool.fixture.prepare(small)
        source=base/'source'; normal=base/'normal'
        scale_fixture.generate(source,128,2,1,128,20261014,0,'performance',3); tool.normal.prepare(source,normal)
        configurations=((small,3,2),(normal,1,64))
        for inputs,k,shards in configurations:
            index,records,snapshot,references=tool.validate_inputs(inputs)
            modes=('pass','child_failure','bad_off_ids','bad_full','bad_partial','negative_zero_placeholder',
                'old_host','two_index_uploads','wrong_mode_order','guard_failed','negative_workspace','extra_off_scores','plan_changes','installation_changes')
            if shards==64: modes+=('bad_group_ids',)
            for mode in modes:
                out=base/f's{shards}_{mode}'; calls=[]
                def child(command,log,env):
                    calls.append(command); log.write_bytes(b'MOCK CHILD -- NO DEVICE EXECUTED')
                    assert command==[str(runner),str(inputs/'index'),str(out/'requests.tsv'),str(out/'device'),'2',str(vendors),str(k),str(shards)]
                    assert env['ASCEND_CUSTOM_OPP_PATH']==str(vendor)+':'+str(vendors/'bm25_topk_merge')
                    if mode=='child_failure': raise subprocess.CalledProcessError(7,command)
                    actual=Path(command[3]); session=fake_session(actual,index,records,references,k,shards,2)
                    first=actual/'call0_A'; repeated=actual/'call2_A'
                    if mode=='bad_off_ids': (first/'debug0/top_ids.bin').write_bytes(bytes(records['A']['num_queries']*k*4))
                    if mode=='bad_full':
                        for parent in (first,repeated): (parent/'debug1/scores.bin').write_bytes(bytes(records['A']['num_queries']*index['num_docs']*4))
                    if mode in ('bad_partial','bad_group_ids'):
                        file='partial_ids.bin' if mode=='bad_partial' else 'group_ids.bin'
                        for parent in (first,repeated):
                            for variant in ('debug0','debug1'): (parent/variant/file).write_bytes(bytes((parent/variant/file).stat().st_size))
                    if mode=='negative_zero_placeholder': (first/'debug0/debug_placeholder.bin').write_bytes(b'\0\0\0\x80')
                    if mode=='old_host': session['host_source_sha256']={}
                    if mode=='two_index_uploads': session['index_upload_batches']=2
                    if mode=='wrong_mode_order': session['requests'][0]['mode_order']=[0,1]
                    if mode=='guard_failed': session['requests'][0]['disabled_placeholder_guard_unchanged']=False
                    if mode=='negative_workspace': session['requests'][0]['workspace_bytes_by_mode'][0][0]=-1
                    if mode=='extra_off_scores':
                        for parent in (first,repeated): shutil.copyfile(parent/'debug1/scores.bin',parent/'debug0/scores.bin')
                    if mode=='plan_changes': (out/'requests.tsv').write_bytes(b'CORRUPTED AFTER MOCK CHILD')
                    tool.r.write_json(actual/'device_session.json',session)
                install_results=[installation,{'CHANGED':True}] if mode=='installation_changes' else None
                with patch.object(tool,'installed',side_effect=install_results,return_value=installation), patch.object(tool.score,'execute',child):
                    invoke=lambda:tool.run(inputs,out,runner,vendors,project,merge_project,2,k,shards)
                    if mode=='pass': invoke()
                    else: fails(invoke)
                report=json.loads((out/'report.json').read_text())
                assert report['status']==(tool.STATUS if mode=='pass' else 'failed') and len(calls)==1; count+=2
                if mode=='pass':
                    assert len(report['checks'])==5 and len(report['output_sha256'])==(50 if shards!=64 else 70)
                    assert report['session']['pipeline_stages_per_mode']==(2 if shards!=64 else 3); count+=3
            with patch.object(tool.score,'execute') as child:
                fails(lambda:tool.run(inputs,base/f's{shards}_pass',runner,vendors,project,merge_project,2,k,shards))
                assert not child.called; count+=1
        for n,k,s in ((6,3,2),(8192,32,1),(65536,10,64),(262144,10,32)):
            tool.dimensions(n,k,s); count+=1
        for n,k,s in ((6,4,2),(8193,1,1),(128,1,33),(7,1,2),(128,True,1)):
            fails(lambda:tool.dimensions(n,k,s))
        # Per-row guard is exercised inside production validation, not just the child's kernel.
        index,records,proof=tool.score.validate_inputs(small)
        def oversized(path,unused_index):
            record=records[path.name]; return record,[[0]*65]+[[]]*(record['num_queries']-1)
        with patch.object(tool.score,'validate_inputs',return_value=(index,records,proof)),patch.object(tool.r,'validate_queries',oversized):
            fails(lambda:tool.validate_inputs(small))
        text=(ROOT/'npu_prototype/fused_chain_host/fused_runtime.h').read_text()
        assert 'aclnn_bm25_query_score.h' not in text and 'int64_t, int64_t, int64_t' in text
        assert 'request.offsets[i]-request.offsets[i-1] <= 64' in text; count+=2
    print(f'PASS: {count} reviewed-ABI/artifact/oracle/MOCK-child/S64-group/debug-mode/repeat/guard/fail-stop checks. '
          'FAKE libraries/objects/binary and MOCK NPU results only; no C++ build or actual device validation.')


if __name__=='__main__': main()
