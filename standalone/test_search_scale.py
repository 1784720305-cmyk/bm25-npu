"""Real host fixture checks plus FAKE installation/CPU/NPU child tests. No C++ or device execution."""
from array import array
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import search_scale_fixture as fixture, run_search_scale as tool, scale_fixture, resident_index as r
from standalone.test_resident_chain import mock_outputs


def mock_session(is_cpu, output, index, records, k, cpu_shards, shards, threads, capq, capwords, full):
    output.mkdir()
    n = index['num_docs']
    if is_cpu:
        session = {'status':'cpu_index_search_calls_completed_independent_check_pending', 'docs':n, 'k':k,
            'document_shards':cpu_shards, 'requested_threads':threads, 'actual_threads':threads,
            'capacity_queries':capq, 'capacity_words':capwords, 'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory',
            'timer_policy':tool.cpu.TIMER_POLICY, 'index_storage':'host_memory', 'query_index_loads':1,
            'index_bytes':index['index_bytes'], 'host_index_byte_identical_to_source':True,
            'debug_source_reread_batches':1, 'cold_bootstrap_empty_queries':capq, 'cpu_needs_backend_word_packing':False,
            'diagnostic_poison_before_calls':True, 'diagnostic_files_outside_Search':True,
            'fp32_flags':'-O3 -march=native -ffp-contract=off -fno-fast-math', 'cold_index_load_threads_capacity_bootstrap_ms':1.,
            'host_source_sha256':{name:tool.digest(ROOT/name) for name in tool.cpu.HOST_SOURCES}, 'requests':[]}
    else:
        session = {'status':'device_search_calls_completed_independent_check_pending', 'device':2,
            'docs':n, 'k':k, 'shards':shards, 'capacity_queries':capq, 'capacity_words':capwords,
            'index_upload_batches':1, 'index_tensor_h2d_calls':5, 'index_upload_bytes':index['index_bytes'],
            'index_readback_byte_identical':True, 'pipeline_stages':3, 'intermediate_host_roundtrips':0,
            'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory', 'timer_policy':tool.npu.TIMER_POLICY,
            'debug_index_d2h_calls':5, 'debug_index_d2h_bytes':index['index_bytes'],
            'host_source_sha256':{name:tool.digest(ROOT/name) for name in tool.npu.HOST_SOURCES},
            'cold_index_load_runtime_upload_capacity_bootstrap_ms':1.,
            'cold_bootstrap':{'synthetic_empty_queries':capq, 'query_h2d_calls':2, 'query_h2d_bytes':8*(capq+1)+4,
                'topk_d2h_calls':2, 'topk_d2h_bytes':8*capq*k}, 'workspace_capacity_bytes':[16777216]*3, 'requests':[]}
    for i, name in enumerate(tool.chain.SEQUENCE):
        label = f'call{i}_{name}'; source = records[name]; q = source['num_queries']
        mock_outputs(output/label, full[name], n, q, k, shards)
        if is_cpu:
            for file in ('partial_scores.bin', 'partial_ids.bin'): (output/label/file).unlink()
            session['requests'].append({'label':label, 'queries':q, 'words':source['query_term_count'], 'search_call_ms':.1,
                'topk_output_bytes':8*q*k, 'debug_full_scores_bytes':4*q*n, 'index_lookup_inside_Search':True,
                'topk_ready_before_stop':True, 'debug_outside_Search':True})
        else:
            logical = source['query_input_bytes']
            session['requests'].append({'label':label, 'queries':q, 'words':source['query_term_count'],
                'logical_query_bytes':logical, 'query_h2d_calls':2,
                'query_h2d_bytes':logical+(4 if source['query_term_count']==0 else 0),
                'topk_d2h_calls':2, 'topk_d2h_bytes':8*q*k, 'debug_poison_h2d_calls':5,
                'debug_poison_h2d_bytes':4*capq*n+8*capq*shards*k+8*capq*k,
                'debug_score_candidate_d2h_calls':3, 'debug_score_candidate_d2h_bytes':4*q*n+8*q*shards*k,
                'packing_inside_Search':True, 'topk_received_before_stop':True, 'debug_outside_Search':True,
                'search_call_ms':.2})
    return session


def main():
    checks = 0
    def fails(call):
        nonlocal checks
        try: call()
        except (ValueError, OSError, subprocess.SubprocessError): checks += 1
        else: raise AssertionError('Invalid scale fixture/run accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='search_scale_HOST_MOCK_ONLY_', dir=ROOT/'results') as temporary, redirect_stdout(io.StringIO()):
        base = Path(temporary)
        # Actual source generation, full index build, original-corpus oracle and disk roundtrip.
        for n, q, k in ((64, 1, 3), (128, 8, 10)):
            source = base/f'source_{n}_{q}'; inputs = base/f'input_{n}_{q}'
            scale_fixture.generate(source, n, q, k, 128, 20261014, 0, 'performance', 3)
            before = fixture.source_hashes(source); fixture.prepare(source, inputs)
            index, records, snapshot = fixture.validate_inputs(inputs)
            assert fixture.source_hashes(source) == before
            assert records['A']['query_input_bytes'] == 8*(q+1)+4*3*q
            assert records['empty']['query_term_count'] == 0 and records['empty']['num_queries'] == q
            assert records['A']['topk_output_bytes'] == 8*q*k; checks += 4
            fails(lambda:fixture.prepare(source, inputs))
        # Mutations use last, genuinely validated fixture; no broad bypass of its validator.
        for path in (inputs/'source/corpus.json', inputs/'B/query_term_ids.bin', inputs/'A/expected_scores.bin',
                     inputs/'index/posting_tf.bin', inputs/'B/reference.json'):
            raw = path.read_bytes(); path.write_bytes(b'X'+raw[1:])
            fails(lambda:fixture.validate_inputs(inputs)); path.write_bytes(raw)
        report_path = inputs/'preparation_report.json'; original = json.loads(report_path.read_text())
        for key, value in (('status', 'device_passed'), ('source_is_real_corpus', True), ('index_id', 'wrong')):
            bad = dict(original); bad[key] = value; r.write_json(report_path, bad)
            fails(lambda:fixture.validate_inputs(inputs))
        r.write_json(report_path, original)
        # Restore definitions from the validated normal input.
        index, records, unused = fixture.validate_inputs(inputs); unused, values = r.validate_index(inputs/'index')
        full = {name:r.diagnostic_scores(index, values, r.validate_queries(inputs/name, index)[1]) for name in records}
        cpu_exe = base/'FAKE_CPU_NOT_EXECUTED'; cpu_exe.write_bytes(b'TEST FAKE CPU')
        npu_exe = base/'FAKE_NPU_NOT_EXECUTED'; npu_exe.write_bytes(b'TEST FAKE NPU')
        vendors = base/'vendors'
        for name in ('bm25_query_score', 'bm25_topk_radix_collect', 'bm25_topk_merge'): (vendors/name).mkdir(parents=True)
        installation = {'TEST_ONLY_FAKE_INSTALL':True, 'topk_acceptance':{'workspace_bytes':16777216}}
        for mode in ('pass', 'cpu_failure', 'cpu_bits', 'cpu_stale_source', 'npu_failure', 'npu_ids', 'npu_transfer', 'raw_changed'):
            output = base/('run_'+mode); commands = []
            def child(command, logfile, env):
                is_cpu = command[0] == str(cpu_exe); commands.append(command)
                assert command[1] == str(inputs/'index') and command[2] == str(output/'requests.tsv')
                assert env.get('ASCEND_CUSTOM_OPP_PATH') == ('PARENT' if is_cpu else ':'.join(str((vendors/name).resolve())
                    for name in ('bm25_query_score', 'bm25_topk_radix_collect', 'bm25_topk_merge')))
                assert command[4:] == (['8', '10', '8', '8', '24', 'raw_query_ids_v1'] if is_cpu else
                    ['2', str(vendors), '10', '8', '8', '24', '16777216'])
                logfile.write_bytes(b'TEST ONLY MOCK CHILD; NO CPU C++ OR NPU EXECUTION')
                if (is_cpu and mode == 'cpu_failure') or (not is_cpu and mode == 'npu_failure'):
                    raise subprocess.CalledProcessError(7, command)
                actual = Path(command[3]); session = mock_session(is_cpu, actual, index, records, 10, 8, 8, 8, 8, 24, full)
                if is_cpu and mode == 'cpu_bits':
                    for label in ('call0_A', 'call2_A'):
                        path = actual/label/'scores.bin'; raw = path.read_bytes(); path.write_bytes(b'\xff\xff\xff\x7f'+raw[4:])
                if is_cpu and mode == 'cpu_stale_source': session['host_source_sha256'] = {}
                if not is_cpu and mode == 'npu_ids':
                    for label in ('call1_B', 'call4_B'): (actual/label/'top_ids.bin').write_bytes(bytes(8*10*4))
                if not is_cpu and mode == 'npu_transfer': session['requests'][0]['query_h2d_bytes'] += 4
                if mode == 'raw_changed' and not is_cpu: Path(command[2]).write_bytes(b'changed')
                r.write_json(actual/('cpu_search_session.json' if is_cpu else 'search_session.json'), session)
            with patch.object(tool.chain, 'validate_install', return_value=installation), patch.object(tool.chain.score, 'execute', child), patch.dict(os.environ, {'ASCEND_CUSTOM_OPP_PATH':'PARENT'}):
                invoke = lambda:tool.run(inputs, output, cpu_exe, npu_exe, vendors, base/'unused_score', base/'unused_top',
                    base/'unused_merge', base/'unused_acceptance', 8, 8, 8, 2)
                if mode == 'pass': invoke()
                else: fails(invoke)
                assert os.environ['ASCEND_CUSTOM_OPP_PATH'] == 'PARENT'; checks += 1
            report = json.loads((output/'report.json').read_text())
            assert report['status'] == (tool.STATUS if mode == 'pass' else 'failed'); checks += 1
            assert len(commands) == (1 if mode.startswith('cpu_') else 2); checks += 1
            if mode == 'pass':
                assert len(report['checks']) == 5 and len(report['output_sha256']) == 40
                assert all(row['cross']['strict_ids_identical'] for row in report['checks']); checks += 2
        with patch.object(tool.chain.score, 'execute') as child, patch.object(tool.chain, 'validate_install', return_value=installation):
            fails(lambda:tool.run(inputs, base/'run_pass', cpu_exe, npu_exe, vendors, base, base, base, base))
            fails(lambda:tool.run(inputs, base/'bad_cap', cpu_exe, npu_exe, vendors, base, base, base, base, capq=1))
            fails(lambda:tool.run(inputs, base/'bad_cpu_shards', cpu_exe, npu_exe, vendors, base, base, base, base, cpu_shards=9))
            assert not child.called; checks += 1
        # Existing small/edge tools deliberately remain separate; no monkey-patched input validator.
        assert fixture.CONTRACT != 'bm25_query_ids_v1'; checks += 1
    print(f'PASS: {checks} normal-source/index/oracle/byte/two-backend-protocol/fail-stop checks. '
          'Actual host data preparation; FAKE installation/executables and MOCK children only; no C++/NPU or performance claim.')


if __name__ == '__main__': main()
