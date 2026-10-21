"""Host protocol/provenance/failure tests with FAKE binaries and MOCK child; no ACLNN/NPU execution."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, query_score_fixture as fixture, validate_query_score_range_boundaries as tool


def main():
    checks = 0
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='score_host_MOCK_',dir=ROOT/'results') as temp:
        root = Path(temp)
        inputs = root/'input'
        with redirect_stdout(io.StringIO()): tool.boundaries.prepare_case(inputs,'single_tail')
        vendor, project = root/'bm25_query_score_range',root/'project'
        header = vendor/'op_api/include/aclnn_bm25_query_score.h'
        header.parent.mkdir(parents=True)
        header.write_text('\n'.join('aclnnStatus '+name+'('+params+');' for name,params in tool.score.PROTOTYPES.items()),encoding='utf-8')
        library = vendor/'op_api/lib/libcust_opapi.so'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'FAKE API LIBRARY -- NEVER LOADED')
        obj = vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_query_score/MOCK.o'
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b'FAKE NPU OBJECT -- NEVER EXECUTED')
        build = project/'build_out'
        build.mkdir(parents=True)
        for path in (header,library,obj): (build/path.name).write_bytes(path.read_bytes())
        metadata = {'operator':'query_score_range','vendor_name':'bm25_query_score_range','compute_unit':'ascend950','source_sha256':{},'max_terms_per_query':64,
                    'formula':'same_FP32_query_time_formula_and_original_term_order'}
        for name in tool.score.SOURCES:
            target = project/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/'npu_prototype/query_score_range'/name).read_bytes())
            metadata['source_sha256'][name] = tool.digest(target)
        r.write_json(project/'bm25_project.json',metadata)
        runner = root/'MOCK_RUNNER_NEVER_EXECUTED'
        runner.write_bytes(b'TEST ONLY -- NO EXECUTABLE CODE')
        tool.validate_install(vendor,project)
        index, records, snapshot = tool.validate_inputs(inputs)
        checks += 2
        # A build-tree ELF may have different RPATH bytes; the exact packaged copy must match.
        intermediate = build/'intermediate/libcust_opapi.so'
        intermediate.parent.mkdir()
        intermediate.write_bytes(b'FAKE BUILD-TREE RPATH BEFORE INSTALL')
        assert tool.validate_install(vendor,project)['api_sha256'] == tool.digest(library)
        checks += 1

        def fails(call):
            nonlocal checks
            try: call()
            except (ValueError,OSError,subprocess.SubprocessError): checks += 1
            else: raise AssertionError('invalid state accepted')

        for path in (header,library,obj,project/tool.score.SOURCES[0]):
            original = path.read_bytes()
            path.write_bytes(original+b'CORRUPTION')
            fails(lambda:tool.validate_install(vendor,project))
            path.write_bytes(original)
        original = header.read_bytes()
        wrong = original.replace(b'const aclTensor *queryTermIds, ',b'')
        header.write_bytes(wrong); (build/header.name).write_bytes(wrong)
        fails(lambda:tool.validate_install(vendor,project))
        header.write_bytes(original); (build/header.name).write_bytes(original)
        bad = copy.deepcopy(metadata); bad['compute_unit']='ascend310'
        r.write_json(project/'bm25_project.json',bad)
        fails(lambda:tool.validate_install(vendor,project))
        r.write_json(project/'bm25_project.json',metadata)
        ref_file = inputs/'A/expected_scores.bin'
        original = ref_file.read_bytes()
        ref_file.write_bytes(bytes(len(original)))
        fails(lambda:tool.validate_inputs(inputs))
        ref_file.write_bytes(original)

        for mode in ('pass','process_failure','index_uploaded_twice','index_changed','wrong_device','stale_host',
                     'bad_query_bytes','bad_call_label','nan_scores','repeat_bits','wrong_scores',
                     'missing_output','truncated_output','bool_counter'):
            output = root/('run_'+mode)
            expected_plan = [f'call{i}_{name}' for i,name in enumerate(tool.SEQUENCE)]
            invocations = []

            def mock_execute(command, logfile, env):
                invocations.append(command)
                assert command[0] == str(runner.resolve()) and command[-1] == str(vendor.resolve())
                assert env['ASCEND_CUSTOM_OPP_PATH'] == str(vendor.resolve())
                plan = Path(command[2]).read_text(encoding='utf-8').splitlines()
                assert [line.split('\t')[0] for line in plan] == expected_plan
                logfile.write_text('MOCK CHILD ONLY -- NO DEVICE WORK\n',encoding='utf-8')
                if mode == 'process_failure': raise subprocess.CalledProcessError(9,command)
                actual_dir = Path(command[3]); actual_dir.mkdir()
                device_record = {'status':'device_score_session_completed_reference_checked','device':2,
                    'index_upload_batches':1,'index_tensor_h2d_calls':5,'index_upload_bytes':index['index_bytes'],
                    'index_readback_byte_identical':True,'index_device_allocations_reused':True,
                    'debug_index_d2h_calls':5,'debug_index_d2h_bytes':index['index_bytes'],
                    'host_source_sha256':{name:tool.digest(ROOT/name) for name in tool.HOST_SOURCES},'requests':[]}
                for i,name in enumerate(tool.SEQUENCE):
                    item = records[name]
                    raw = (inputs/name/'expected_scores.bin').read_bytes()
                    if mode == 'nan_scores' and i == 0: raw = struct.pack('<I',0x7fc00000)+raw[4:]
                    if mode == 'repeat_bits' and i == 2:
                        bits, = struct.unpack('<I',raw[:4]); raw = struct.pack('<I',bits+1)+raw[4:]
                    if mode == 'wrong_scores' and name == 'B': raw = struct.pack('<f',1000)+raw[4:]
                    if mode == 'truncated_output' and name == 'A': raw = raw[:-4]
                    (actual_dir/f'call{i}_{name}_scores.bin').write_bytes(raw)
                    device_record['requests'].append({'label':f'call{i}_{name}','queries':item['num_queries'],
                        'logical_word_count':item['query_term_count'],'logical_query_bytes':item['query_input_bytes'],
                        'query_h2d_bytes':item['query_input_bytes']+(4 if item['query_term_count'] == 0 else 0),
                        'query_tensor_h2d_calls':2,'score_d2h_bytes':item['num_queries']*index['num_docs']*4,
                        'score_d2h_calls':1,'debug_poison_h2d_calls':1,
                        'debug_poison_h2d_bytes':max(x['num_queries'] for x in records.values())*index['num_docs']*4,
                        'workspace_bytes':16777216,'output_poisoned':True,'fresh_executor':True,
                        'full_reference_checked':True,'repeated_query_bits_checked':i in (2,4)})
                if mode == 'index_uploaded_twice': device_record['index_upload_batches']=2
                elif mode == 'index_changed': device_record['index_readback_byte_identical']=False
                elif mode == 'wrong_device': device_record['device']=1
                elif mode == 'stale_host': device_record['host_source_sha256'][tool.HOST_SOURCES[0]]='0'*64
                elif mode == 'bad_query_bytes': device_record['requests'][0]['query_h2d_bytes']+=4
                elif mode == 'bad_call_label': device_record['requests'][2]['label']='call2_B'
                elif mode == 'bool_counter': device_record['index_upload_batches']=True
                elif mode == 'missing_output': (actual_dir/'call3_empty_scores.bin').unlink()
                r.write_json(actual_dir/'device_session.json',device_record)

            with patch.object(tool,'execute',mock_execute),patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':'UNCHANGED_PARENT'}),redirect_stdout(io.StringIO()):
                if mode == 'pass': tool.run(inputs,output,runner,vendor,project,2)
                else: fails(lambda:tool.run(inputs,output,runner,vendor,project,2))
                assert os.environ['ASCEND_CUSTOM_OPP_PATH']=='UNCHANGED_PARENT'
            report = json.loads((output/'report.json').read_text())
            assert len(invocations)==1
            assert report['status']==('range_boundary_device_score_session_and_independent_formula_checks_passed' if mode=='pass' else 'failed')
            assert set(report['tool_source_sha256']) == set(tool.TOOL_SOURCES)
            if mode == 'process_failure': assert not (output/'device').exists()
            checks += 4

        # Prepare/validate the full deterministic suite, but MOCK every device call.
        saved_inputs,saved_index,saved_records,saved_snapshot=inputs,index,records,snapshot
        suite_input=root/'suite_input'
        with redirect_stdout(io.StringIO()): tool.boundaries.prepare(suite_input)
        for name,(n,q) in tool.boundaries.CASES.items():
            inputs=suite_input/name
            index,records,snapshot=tool.validate_inputs(inputs)
            assert index['num_docs']==n and records['A']['num_queries']==q
            assert tool.enforce_query_limits(inputs,index,records)=={'A':64,'B':64,'empty':0}
            rejected=tool.boundaries.check_rejection(inputs,index)
            assert rejected['device_submitted'] is False
            checks += 3
        mode='pass'
        invocations=[]
        # The mock reads each real child command's input directory, not a global oracle.
        old_mock=mock_execute
        def suite_mock(command,logfile,env):
            nonlocal inputs,index,records,snapshot
            inputs=Path(command[1]).parent
            index,records,snapshot=tool.validate_inputs(inputs)
            old_mock(command,logfile,env)
        with patch.object(tool,'execute',suite_mock),redirect_stdout(io.StringIO()):
            tool.run_suite(suite_input,root/'suite_output',runner,vendor,project,2)
        suite_report=json.loads((root/'suite_output/report.json').read_text())
        assert suite_report['status']=='all_three_range_boundary_device_sessions_passed'
        assert len(invocations)==3 and len(suite_report['cases'])==3
        checks += 3
        for name in tool.boundaries.CASES:
            rec=json.loads((root/'suite_output'/name/'report.json').read_text())
            assert rec['actual_max_query_terms']['A']==64 and rec['host_rejection']['device_submitted'] is False
            checks += 1
        mode='process_failure'
        invocations=[]
        with patch.object(tool,'execute',suite_mock),redirect_stdout(io.StringIO()):
            fails(lambda:tool.run_suite(suite_input,root/'suite_failed',runner,vendor,project,2))
        assert len(invocations)==1
        assert json.loads((root/'suite_failed/report.json').read_text())['status']=='failed'
        assert not (root/'suite_failed/split_tail').exists()
        checks += 2
        inputs,index,records,snapshot=saved_inputs,saved_index,saved_records,saved_snapshot

        # Real query-file length guards, before any child can be launched. No device.
        for length in (0,64,65):
            limited = root/f'limit_{length}'
            record,arrays = r.make_queries(index,[[0]*length],3)
            r.save_arrays(limited/'A',record,arrays)
            if length <= 64:
                assert tool.enforce_query_limits(limited,index,{'A':record}) == {'A':length}
                checks += 1
            else:
                fails(lambda:tool.enforce_query_limits(limited,index,{'A':record}))
                with patch.object(tool,'validate_inputs',return_value=(index,{'A':record},{})),patch.object(tool,'execute') as child:
                    fails(lambda:tool.run(limited,root/'over_limit_rejected',runner,vendor,project,2))
                    assert not child.called and not (root/'over_limit_rejected').exists()
                    checks += 1
        for key,value in (('max_terms_per_query',65),('max_terms_per_query',True),
                          ('formula','precomputed'),('operator','query_score')):
            bad = copy.deepcopy(metadata); bad[key] = value
            r.write_json(project/'bm25_project.json',bad)
            fails(lambda:tool.validate_install(vendor,project))
        r.write_json(project/'bm25_project.json',metadata)

        existing = root/'existing'; existing.mkdir()
        sentinel = existing/'keep'; sentinel.write_bytes(b'UNCHANGED')
        with patch.object(tool,'execute') as child:
            fails(lambda:tool.run(inputs,existing,runner,vendor,project,2))
            fails(lambda:tool.run(inputs,root/'bad_card',runner,vendor,project,True))
            assert not child.called and sentinel.read_bytes() == b'UNCHANGED'
            checks += 1
        assert all(tool.digest(Path(name)) == value for name,value in snapshot.items())
        checks += 1
        # Exercise the real stdout/stderr streaming helper with ordinary Python subprocesses.
        # These are logging tests only; the fake runner/libraries above remain unexecuted.
        stream_log = root/'real_process_logging_only.log'
        captured = io.StringIO()
        with redirect_stdout(captured):
            tool.execute([sys.executable,'-u','-c',
                          "import sys; print('LOG_TEST_STDOUT'); print('LOG_TEST_STDERR',file=sys.stderr)"],
                         stream_log,os.environ.copy())
        assert 'LOG_TEST_STDOUT' in captured.getvalue() and 'LOG_TEST_STDERR' in captured.getvalue()
        assert 'LOG_TEST_STDOUT' in stream_log.read_text() and 'LOG_TEST_STDERR' in stream_log.read_text()
        checks += 2
        with redirect_stdout(io.StringIO()):
            try:
                tool.execute([sys.executable,'-u','-c',"print('LOG_TEST_FAILURE'); raise SystemExit(7)"],
                             stream_log,os.environ.copy())
            except subprocess.CalledProcessError as error: assert error.returncode == 7
            else: raise AssertionError('real subprocess error ignored')
        assert 'LOG_TEST_FAILURE' in stream_log.read_text()
        checks += 2
    print(f'PASS: {checks} range-boundary fixture/independent-oracle/host-provenance/64-term guard/confirmed-signature/transfer-record/independent-check/failure-protocol tests. '
          'FAKE libraries/objects/runner and MOCK device child removed; real Python child logging tested. '
          'No C++ build, ACLNN or NPU execution.')


if __name__ == '__main__': main()
