"""FAKE installation and MOCK child protocol checks; no C++ build, ACLNN or NPU execution."""
from contextlib import redirect_stdout
from array import array
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, query_score_fixture as fixture, query_score_edge_fixture as edges
from standalone import run_resident_chain as tool

TOP_DECLS = {
    'aclnnBm25TopKGetWorkspaceSize':'const aclTensor *scores, int64_t k, const aclTensor *topScoresOut, const aclTensor *topIdsOut, uint64_t *workspaceSize, aclOpExecutor **executor',
    'aclnnBm25TopK':'void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream'}
MERGE_DECLS = {
    'aclnnBm25TopKMergeGetWorkspaceSize':'const aclTensor *partialScores, const aclTensor *partialIds, int64_t k, int64_t docCount, const aclTensor *topScoresOut, const aclTensor *topIdsOut, uint64_t *workspaceSize, aclOpExecutor **executor',
    'aclnnBm25TopKMerge':'void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream'}


def fake_install(base,operator,stem,prototypes):
    vendor = base/'vendors'/('bm25_'+operator)
    project = base/('project_'+operator)
    src_root = ROOT/'npu_prototype'/operator
    source_names = (f'op_host/{stem}.cpp',f'op_host/{stem}_tiling.h',f'op_kernel/{stem}.cpp')
    header = vendor/f'op_api/include/aclnn_{stem}.h'
    library = vendor/'op_api/lib/libcust_opapi.so'
    obj = vendor/f'op_impl/ai_core/tbe/kernel/ascend950/{stem}/TEST_ONLY_FAKE.o'
    for path in (header,library,obj): path.parent.mkdir(parents=True,exist_ok=True)
    header.write_bytes(('\n'.join(f'aclnnStatus {name}({params});' for name,params in prototypes.items())+'\n').encode())
    library.write_bytes(b'TEST ONLY FAKE API -- NEVER LOADED')
    obj.write_bytes(b'TEST ONLY FAKE DEVICE OBJECT -- NEVER EXECUTED')
    build = project/'build_out'; build.mkdir(parents=True)
    for path in (header,library,obj): (build/path.name).write_bytes(path.read_bytes())
    hashes = {}
    for name in source_names:
        destination = project/name; destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((src_root/name).read_bytes()); hashes[name] = tool.digest(destination)
    r.write_json(project/'bm25_project.json',{'operator':operator,'vendor_name':vendor.name,
                 'compute_unit':'ascend950','source_sha256':hashes})
    return vendor,project


def mock_outputs(directory,full,n,q,k,s):
    """Test fixture only: construct deterministic arrays, never used by production run()."""
    directory.mkdir(parents=True)
    arrays = {'scores.bin':array('f',full),'partial_scores.bin':array('f'),
              'partial_ids.bin':array('i'),'top_scores.bin':array('f'),'top_ids.bin':array('i')}
    width = n//s
    for row in range(q):
        base = row*n
        for shard in range(s):
            start = base+shard*width
            chosen = sorted(range(width),key=lambda j:(-full[start+j],j))[:k]
            arrays['partial_ids.bin'].extend(chosen)
            arrays['partial_scores.bin'].extend(full[start+j] for j in chosen)
        chosen = sorted(range(n),key=lambda j:(-full[base+j],j))[:k]
        arrays['top_ids.bin'].extend(chosen)
        arrays['top_scores.bin'].extend(full[base+j] for j in chosen)
    for name,values in arrays.items(): (directory/name).write_bytes(r.encode(values))


def main():
    checks = 0
    def fails(call):
        nonlocal checks
        try: call()
        except (ValueError,OSError,subprocess.SubprocessError): checks += 1
        else: raise AssertionError('invalid state accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='resident_chain_MOCK_ONLY_',dir=ROOT/'results') as temp:
        base = Path(temp)
        vendors = base/'vendors'
        score_vendor,score_project = fake_install(base,'query_score','bm25_query_score',tool.score.PROTOTYPES)
        top_vendor,top_project = fake_install(base,'topk_radix_collect','bm25_top_k',TOP_DECLS)
        merge_vendor,merge_project = fake_install(base,'topk_merge','bm25_top_k_merge',MERGE_DECLS)
        acceptance = base/'prior_TEST_ONLY_MOCK_acceptance'
        accepted = {'directory':str(acceptance),'report_sha256':'TEST_ONLY_MOCK','workspace_bytes':16777216}
        args = (vendors,score_project,top_project,merge_project,acceptance)
        inputs = base/'input'
        with redirect_stdout(io.StringIO()): fixture.prepare(inputs)
        index,records,unused = tool.score.validate_inputs(inputs)
        unused,values = r.validate_index(inputs/'index')
        cpu = {}
        for name in records:
            unused,queries = r.validate_queries(inputs/name,index)
            cpu[name] = r.diagnostic_scores(index,values,queries)
        runner = base/'TEST_ONLY_FAKE_RUNNER'; runner.write_bytes(b'NOT EXECUTABLE')
        with patch.object(tool,'validate_radix_acceptance',return_value=accepted):
            tool.validate_install(*args); checks += 1
            for path in (top_vendor/'op_api/lib/libcust_opapi.so',
                         merge_vendor/'op_api/include/aclnn_bm25_top_k_merge.h',
                         merge_vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_top_k_merge/TEST_ONLY_FAKE.o'):
                raw = path.read_bytes(); path.write_bytes(raw+b'CORRUPTED')
                fails(lambda:tool.validate_install(*args)); path.write_bytes(raw)

            for mode in ('pass','child_failure','index_upload_twice','roundtrip','wrong_device','stale_host',
                         'extra_query_upload','wrong_workspace','no_poison','bool_counter','repeat_bytes',
                         'wrong_local_id','wrong_top_id','score_bits','nan_full','truncated'):
                output = base/('mock_'+mode)
                calls = []
                def child(command,logfile,env):
                    calls.append(command)
                    assert env['ASCEND_CUSTOM_OPP_PATH'] == ':'.join(str(p.resolve()) for p in (score_vendor,top_vendor,merge_vendor))
                    assert command[0] == str(runner) and command[-3:] == ['3','2','16777216']
                    assert len(Path(command[2]).read_text().splitlines()) == 5
                    logfile.write_bytes(b'TEST ONLY MOCK CHILD -- NO DEVICE EXECUTION\n')
                    if mode == 'child_failure': raise subprocess.CalledProcessError(7,command)
                    actual = Path(command[3]); actual.mkdir()
                    k,s = 3,2
                    poison = 4*6*6+8*6*s*k+8*6*k
                    session = {'status':'device_resident_chain_completed_reference_and_ranking_checked','device':2,
                        'docs':6,'k':k,'shards':s,'pipeline_stages':3,'intermediate_host_roundtrips':0,
                        'index_upload_batches':1,'index_tensor_h2d_calls':5,'index_upload_bytes':index['index_bytes'],
                        'index_readback_byte_identical':True,'index_device_allocations_reused':True,
                        'debug_index_d2h_calls':5,'debug_index_d2h_bytes':index['index_bytes'],
                        'host_source_sha256':{name:tool.digest(ROOT/name) for name in tool.HOST_SOURCES},'requests':[]}
                    for i,name in enumerate(tool.SEQUENCE):
                        record = records[name]; q = record['num_queries']; label = f'call{i}_{name}'
                        mock_outputs(actual/label,cpu[name],6,q,k,s)
                        logical = record['query_input_bytes']
                        session['requests'].append({'label':label,'queries':q,'logical_word_count':record['query_term_count'],
                            'logical_query_bytes':logical,'query_h2d_bytes':logical+(4 if record['query_term_count'] == 0 else 0),
                            'query_tensor_h2d_calls':2,'topk_d2h_bytes':8*q*k,'topk_d2h_calls':2,
                            'debug_score_and_candidate_d2h_bytes':4*q*6+8*q*s*k,'debug_score_and_candidate_d2h_calls':3,
                            'debug_poison_h2d_calls':5,'debug_poison_h2d_bytes':poison,'workspace_bytes':[16777216]*3,
                            'outputs_poisoned':True,'fresh_executors':3,'chain_synchronizations':1,'debug_tensor_cleanup_synchronizations':1,
                            'full_reference_checked':True,'exact_device_ranking_checked':True,'repeated_all_output_bits_checked':i in (2,4)})
                    if mode == 'index_upload_twice': session['index_upload_batches']=2
                    if mode == 'roundtrip': session['intermediate_host_roundtrips']=1
                    if mode == 'wrong_device': session['device']=4
                    if mode == 'stale_host': session['host_source_sha256'][tool.HOST_SOURCES[0]]='0'*64
                    if mode == 'extra_query_upload': session['requests'][0]['query_h2d_bytes']+=4
                    if mode == 'wrong_workspace': session['requests'][0]['workspace_bytes'][1]+=64
                    if mode == 'no_poison': session['requests'][0]['outputs_poisoned']=False
                    if mode == 'bool_counter': session['index_upload_batches']=True
                    if mode == 'repeat_bytes': (actual/'call2_A/scores.bin').write_bytes(bytes(6*6*4))
                    for which in ('local','top'):
                        if mode == 'wrong_'+which+'_id':
                            file = actual/'call3_empty'/('partial_ids.bin' if which == 'local' else 'top_ids.bin')
                            raw,ids = tool.read(file,'i',3*(s if which == 'local' else 1)*k)
                            ids[0]=1; file.write_bytes(r.encode(ids))
                    if mode == 'score_bits':
                        file = actual/'call3_empty/top_scores.bin'; raw = file.read_bytes(); file.write_bytes(b'\x01\0\0\0'+raw[4:])
                    if mode == 'nan_full':
                        file = actual/'call3_empty/scores.bin'; raw = file.read_bytes(); file.write_bytes(b'\0\0\xc0\x7f'+raw[4:])
                    if mode == 'truncated': (actual/'call3_empty/top_ids.bin').write_bytes(b'')
                    r.write_json(actual/'device_session.json',session)
                with patch.object(tool.score,'execute',child),patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':'PARENT_UNCHANGED'}),redirect_stdout(io.StringIO()):
                    invoke = lambda:tool.run(inputs,output,runner,*args,2,3,2)
                    if mode == 'pass': invoke()
                    else: fails(invoke)
                    assert os.environ['ASCEND_CUSTOM_OPP_PATH'] == 'PARENT_UNCHANGED'
                report = json.loads((output/'report.json').read_text())
                assert len(calls) == 1
                assert report['status'] == ('resident_chain_device_session_and_all_rankings_passed' if mode == 'pass' else 'failed')
                assert (output/'device').exists() == (mode != 'child_failure')
                checks += 4
            with patch.object(tool.score,'execute') as child:
                fails(lambda:tool.run(inputs,base/'mock_pass',runner,*args,2,3,2))
                fails(lambda:tool.run(inputs,base/'bad_dimension',runner,*args,2,3,4))
                fails(lambda:tool.run(inputs,base/'bad_device',runner,*args,True,3,2))
                assert not child.called; checks += 1

        # Independent full-sort check over larger CPU-prepared TEST-ONLY score arrays.
        # These are temporary checker unit fixtures, never production device outputs.
        edge = base/'edge'
        with redirect_stdout(io.StringIO()): edges.prepare_case(edge,2048)
        index,arrays = r.validate_index(edge/'index')
        for k,s in ((1,1),(10,8),(32,32)):
            unused,queries = r.validate_queries(edge/'A',index)
            full = r.diagnostic_scores(index,arrays,queries)
            output = base/f'TEST_ONLY_CPU_ARRAYS_k{k}_s{s}'
            mock_outputs(output,full,2048,65,k,s)
            with redirect_stdout(io.StringIO()): result = tool.check_outputs(edge/'index',edge/'A',output,k,s)
            assert result['all_local_and_global_ids_strict'] and result['same_input_cpu_fp32_ids_strict']
            checks += 1
    print(f'PASS: {checks} resident-chain installation/protocol/ranking/tamper/fail-stop checks. '
          'FAKE artifacts, MOCK child and CPU-prepared checker fixtures removed. No C++ build, ACLNN or NPU execution.')


if __name__ == '__main__': main()
