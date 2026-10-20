"""Host oracles and FAKE/MOCK measurement-protocol tests; no C++ compilation or device execution."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import benchmark_search as tool, search_benchmark_checks as checks
from standalone import search_scale_fixture as fixture, scale_fixture, resident_index as r, run_search_scale as scale
from standalone.test_search_scale import mock_session
from standalone.test_resident_chain import mock_outputs


def main():
    count = 0
    def fails(call):
        nonlocal count
        try: call()
        except (ValueError, OSError, subprocess.SubprocessError): count += 1
        else: raise AssertionError('Invalid benchmark accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='search_benchmark_HOST_MOCK_ONLY_', dir=ROOT/'results') as temporary, redirect_stdout(io.StringIO()):
        base = Path(temporary); source = base/'source'; inputs = base/'inputs'
        scale_fixture.generate(source, 128, 8, 10, 128, 20261014, 0, 'performance', 3)
        fixture.prepare(source, inputs); index, records, snapshot = fixture.validate_inputs(inputs)
        cache = checks.cache_inputs(inputs, index, records); full = {name:row['scores'] for name, row in cache.items()}
        cpu_old = base/'FAKE_OLD_CPU'; npu_old = base/'FAKE_OLD_NPU'
        cpu_exe = base/'FAKE_BENCH_CPU'; npu_exe = base/'FAKE_BENCH_NPU'
        for path in (cpu_old, npu_old, cpu_exe, npu_exe): path.write_bytes(b'TEST ONLY FAKE EXECUTABLE: '+path.name.encode())
        vendors = base/'vendors'
        for name in ('bm25_query_score', 'bm25_topk_radix_collect', 'bm25_topk_merge'): (vendors/name).mkdir(parents=True)
        installation = {'TEST_ONLY_FAKE':True, 'topk_acceptance':{'workspace_bytes':16777216}}
        accepted = base/'accepted'
        def accepted_child(command, logfile, env):
            actual = Path(command[3]); is_cpu = command[0] == str(cpu_old)
            data = mock_session(is_cpu, actual, index, records, 10, 8, 8, 8, 8, 24, full)
            r.write_json(actual/('cpu_search_session.json' if is_cpu else 'search_session.json'), data)
            logfile.write_bytes(b'TEST ONLY MOCK PRIOR CPU/NPU CHILD')
        with patch.object(scale.chain, 'validate_install', return_value=installation), patch.object(scale.chain.score, 'execute', accepted_child):
            scale.run(inputs, accepted, cpu_old, npu_old, vendors, base, base, base, base, 8, 8, 8, 2)
        checks.validate_acceptance(accepted, inputs, index, records, snapshot, cache); count += 1
        prior_path = accepted/'report.json'; prior = json.loads(prior_path.read_text())
        for key, value in (('status', 'old_resident_compute'), ('index_id', 'wrong'), ('source_is_real_corpus', True)):
            bad = dict(prior); bad[key] = value; r.write_json(prior_path, bad)
            fails(lambda:checks.validate_acceptance(accepted, inputs, index, records, snapshot, cache))
        r.write_json(prior_path, prior)
        for text in ('1', '8:9', '0:0', 'a:2', '8:2:1', '-8:2'):
            fails(lambda:tool.config(text))
        assert tool.config('16:8') == (16, 8); count += 1
        summary = checks.summarize([float(x) for x in range(1, 31)], 8)
        assert summary['median_batch_ms'] == 15.5 and summary['p95_batch_ms'] == 29.0; count += 1
        fails(lambda:checks.summarize([float('nan')], 1))
        for mode in ('pass', 'cpu_failure', 'npu_failure', 'bad_npu_ids', 'nan_time', 'stale_host'):
            output = base/('bench_'+mode); commands = []
            def child(command, logfile, env):
                is_cpu = command[0] == str(cpu_exe); backend = 'cpu' if is_cpu else 'npu'; commands.append(command)
                assert command[1] == str(inputs/'index') and command[2:4] == [str(output/'raw_queries/A.txt'), str(output/'raw_queries/B.txt')]
                assert env.get('ASCEND_CUSTOM_OPP_PATH') == ('PARENT' if is_cpu else ':'.join(str((vendors/name).resolve())
                    for name in ('bm25_query_score', 'bm25_topk_radix_collect', 'bm25_topk_merge')))
                logfile.write_bytes(b'TEST ONLY MOCK BENCH CHILD; NO C++ OR DEVICE EXECUTED')
                if (is_cpu and mode == 'cpu_failure') or (not is_cpu and mode == 'npu_failure'):
                    raise subprocess.CalledProcessError(7, command)
                actual = Path(command[4]); actual.mkdir()
                threads = int(command[5]) if is_cpu else 8
                partitions = int(command[7] if is_cpu else command[8]); warmup, repeats = map(int, command[-2:])
                data = {'status':'search_benchmark_calls_completed_checks_pending', 'backend':backend, 'docs':128,
                    'queries':8, 'k':10, 'partitions':partitions, 'capacity_queries':8, 'capacity_words':24,
                    'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory', 'timer_policy':checks.TIMER,
                    'index_load_batches':1, 'index_bytes':index['index_bytes'], 'cold_init_ms':1., 'cold_synthetic_empty_queries':8,
                    'checkpoint_search_calls':4, 'warmup_cycles':warmup, 'warmup_search_calls':2*warmup,
                    'warmup_topk_bit_checks':2*warmup, 'measured_cycles':repeats, 'measured_search_calls':2*repeats,
                    'measured_topk_bit_checks':2*repeats, 'measured_debug_download_calls':0, 'measured_poison_calls':0,
                    'measured_file_io_calls':0, 'diagnostic_checkpoints_outside_measurement':True,
                    'index_readback_byte_identical':True,
                    'host_source_sha256':checks.sources(checks.CPU_SOURCES if is_cpu else checks.NPU_SOURCES), 'samples':[]}
                if is_cpu:
                    data.update(requested_threads=threads, actual_threads=threads, debug_source_reread_batches=1,
                                fp32_flags='-O3 -march=native -ffp-contract=off -fno-fast-math')
                else:
                    data.update(device=2, index_tensor_h2d_calls=5, pipeline_stages=3, debug_index_d2h_calls=5,
                                intermediate_host_roundtrips=0, workspace_capacity_bytes=[16777216]*3)
                for cycle in range(repeats):
                    for slot in range(2):
                        case = 'B' if slot^(cycle&1) else 'A'
                        elapsed = (.3 if threads == 1 else .1) if is_cpu else .08
                        data['samples'].append({'cycle':cycle, 'case':case, 'call_wall_ms':elapsed, 'internal_search_ms':elapsed*.99,
                            'query_h2d_bytes':0 if is_cpu else 168, 'topk_d2h_bytes':0 if is_cpu else 640})
                for label in checks.CHECKPOINTS:
                    name = label[-1]; mock_outputs(actual/label, full[name], 128, 8, 10, partitions if not is_cpu else 8)
                    if is_cpu:
                        for file in ('partial_scores.bin', 'partial_ids.bin'): (actual/label/file).unlink()
                if not is_cpu and mode == 'bad_npu_ids':
                    for label in ('before_A', 'after_A'): (actual/label/'top_ids.bin').write_bytes(bytes(8*10*4))
                if not is_cpu and mode == 'nan_time': data['samples'][0]['call_wall_ms'] = float('nan')
                if not is_cpu and mode == 'stale_host': data['host_source_sha256'] = {}
                r.write_json(actual/'benchmark_session.json', data)
            with patch.object(scale.chain, 'validate_install', return_value=installation), patch.object(scale.chain.score, 'execute', child), patch.dict(os.environ, {'ASCEND_CUSTOM_OPP_PATH':'PARENT'}):
                invoke = lambda:tool.run(inputs, accepted, output, cpu_exe, npu_exe, vendors, base, base, base, base,
                    [(1, 1), (8, 8)], 2, 2, 1, 10, 2, 8)
                if mode == 'pass': invoke()
                else: fails(invoke)
                assert os.environ['ASCEND_CUSTOM_OPP_PATH'] == 'PARENT'; count += 1
            report = json.loads((output/'report.json').read_text())
            assert report['status'] == (tool.STATUS if mode == 'pass' else 'failed'); count += 1
            if mode == 'cpu_failure': assert len(commands) == 1; count += 1
            if mode == 'pass':
                assert report['selected_cpu_config']['threads'] == 8
                assert [row['process_order'] for row in report['per_round']] == [['cpu', 'npu'], ['npu', 'cpu']]
                assert report['paired_summary']['A']['npu_faster_rounds'] == 2
                assert len(report['scan']) == 4 and all(len(row['cross_checks']) == 4 for row in report['per_round']); count += 4
                first = Path(report['per_round'][0]['npu']['directory'])/'actual'
                path = first/'benchmark_session.json'; saved = json.loads(path.read_text())
                for key, value in (('measured_debug_download_calls', 1), ('measured_file_io_calls', 1),
                                   ('warmup_search_calls', 1), ('index_load_batches', 2), ('timer_policy', 'kernel_only'),
                                   ('measured_topk_bit_checks', 0), ('index_readback_byte_identical', False)):
                    bad = copy.deepcopy(saved); bad[key] = value; r.write_json(path, bad)
                    fails(lambda:checks.check_session(first, 'npu', index, records, 8, 8, 2, 8, 24, 1, 10, 16777216))
                for key, value in (('case', 'B'), ('cycle', True), ('internal_search_ms', 1.), ('call_wall_ms', 0.),
                                   ('query_h2d_bytes', 0), ('topk_d2h_bytes', 0)):
                    bad = copy.deepcopy(saved); bad['samples'][0][key] = value; r.write_json(path, bad)
                    fails(lambda:checks.check_session(first, 'npu', index, records, 8, 8, 2, 8, 24, 1, 10, 16777216))
                r.write_json(path, saved)
                raw_path = first/'after_A/scores.bin'; raw = raw_path.read_bytes(); raw_path.write_bytes(bytes(len(raw)))
                fails(lambda:checks.check_checkpoints(first, 'npu', cache, 8)); raw_path.write_bytes(raw)
        with patch.object(scale.chain.score, 'execute') as child:
            fails(lambda:tool.run(inputs, accepted, base/'bench_pass', cpu_exe, npu_exe, vendors, base, base, base, base, [(1, 1), (8, 8)]))
            fails(lambda:tool.run(inputs, accepted, base/'odd_rounds', cpu_exe, npu_exe, vendors, base, base, base, base, [(1, 1), (8, 8)], rounds=3))
            fails(lambda:tool.run(inputs, accepted, base/'single_config', cpu_exe, npu_exe, vendors, base, base, base, base, [(8, 8)]))
            assert not child.called; count += 1
        code = (ROOT/'npu_prototype/search_benchmark_host/benchmark_common.h').read_text()
        measured = code[code.index('for (int cycle=0;cycle<repeats;'):code.index('    std::cout << "stage=after_measurement')]
        assert measured.index('const auto start=Clock::now();') < measured.index('session.Search(') < measured.index('const double elapsed=') < measured.index('validate(which,result)')
        assert 'poison();' not in measured and 'save(' not in measured and 'std::cout' not in measured; count += 2
    print(f'PASS: {count} provenance/oracle/CPU-selection/paired-order/timing/strict-output/fail-stop checks. '
          'Real host data preparation, FAKE executables/install and MOCK CPU/NPU children only; no C++ build or device/performance evidence.')


if __name__ == '__main__': main()
