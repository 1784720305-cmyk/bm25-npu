"""Real small host oracles; FAKE proof/binaries/install and MOCK CPU/NPU children. No device proof."""
from contextlib import ExitStack, redirect_stdout
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
from standalone import benchmark_compact_search_cpu as tool, scale_fixture
from standalone.test_search_scale import mock_session
from standalone.test_resident_chain import mock_outputs


def measurement(actual, index, records, full, backend, warmup, repeats, elapsed):
    actual.mkdir()
    cpu = backend == 'cpu'
    data = {'status': 'search_benchmark_calls_completed_checks_pending', 'backend': backend,
        'docs': 128, 'queries': 8, 'k': 10, 'partitions': 8, 'capacity_queries': 8, 'capacity_words': 24,
        'input_contract': 'raw_grouped_INT32_word_IDs_in_CPU_memory', 'timer_policy': tool.checks.TIMER,
        'index_load_batches': 1, 'index_bytes': index['index_bytes'], 'cold_init_ms': 1., 'cold_synthetic_empty_queries': 8,
        'checkpoint_search_calls': 4, 'warmup_cycles': warmup, 'warmup_search_calls': 2*warmup, 'warmup_topk_bit_checks': 2*warmup,
        'measured_cycles': repeats, 'measured_search_calls': 2*repeats, 'measured_topk_bit_checks': 2*repeats,
        'measured_debug_download_calls': 0, 'measured_poison_calls': 0, 'measured_file_io_calls': 0,
        'diagnostic_checkpoints_outside_measurement': True, 'index_readback_byte_identical': True,
        'host_source_sha256': tool.checks.sources(tool.checks.CPU_SOURCES if cpu else tool.checks.NPU_SOURCES), 'samples': []}
    if cpu:
        data.update(requested_threads=8, actual_threads=8, debug_source_reread_batches=1,
                    fp32_flags='-O3 -march=native -ffp-contract=off -fno-fast-math')
    else:
        data.update(device=2, index_tensor_h2d_calls=5, pipeline_stages=3, debug_index_d2h_calls=5,
                    intermediate_host_roundtrips=0, workspace_capacity_bytes=[16777216]*3)
    for cycle in range(repeats):
        for slot in range(2):
            name = 'B' if slot ^ (cycle & 1) else 'A'
            data['samples'].append({'cycle': cycle, 'case': name, 'call_wall_ms': elapsed,
                'internal_search_ms': elapsed*.99, 'query_h2d_bytes': 0 if cpu else 168, 'topk_d2h_bytes': 0 if cpu else 640})
    for label in tool.checks.CHECKPOINTS:
        mock_outputs(actual/label, full[label[-1]], 128, 8, 10, 8)
        if cpu:
            for file in ('partial_scores.bin', 'partial_ids.bin'): (actual/label/file).unlink()
    tool.r.write_json(actual/'benchmark_session.json', data)
    return data


def main():
    count = 0
    def fails(call):
        nonlocal count
        try: call()
        except (ValueError, OSError, subprocess.SubprocessError): count += 1
        else: raise AssertionError('invalid proof or run accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='compact_cpu_MOCK_ONLY_', dir=ROOT/'results') as temp, redirect_stdout(io.StringIO()):
        base = Path(temp); inputs, source = base/'inputs', base/'source'
        scale_fixture.generate(source, 128, 8, 10, 128, 20261014, 0, 'performance', 3)
        tool.fixture.prepare(source, inputs)
        index, records, snapshot = tool.fixture.validate_inputs(inputs)
        cache = tool.checks.cache_inputs(inputs, index, records)
        full = {name: value['scores'] for name, value in cache.items()}
        runners = {name: base/('FAKE_'+name) for name in ('cpu', 'npu', 'diagnostic', 'old_npu_diagnostic')}
        for path in runners.values(): path.write_bytes(b'MOCK NOT EXECUTABLE')
        vendors = base/'vendors'; projects = {key: base/key for key in ('score', 'topk', 'merge')}
        targets = {'bm25_query_score': vendors/'bm25_query_score_range',
                   'bm25_topk_radix_collect': vendors/tool.selected.compact.VENDOR, 'bm25_topk_merge': vendors/'bm25_topk_merge'}
        for path in targets.values(): path.mkdir(parents=True)
        installation = {'score': {'FAKE': 'score'}, 'topk': {'FAKE': 'top', 'installed_device_object_sha256': {'MOCK.o': 'hash'}},
            'merge': {'FAKE': 'merge'}, 'merge_device_objects': {'MOCK_merge.o': 'hash'},
            'topk_device_objects': {'MOCK.o': 'hash'}, 'topk_acceptance': {'workspace_bytes': 16777216}}

        def fake_create(path, mapping):
            path.mkdir(parents=True)
            for slot, target in mapping.items():
                (path/slot).mkdir(); (path/slot/'target.txt').write_text(str(target), encoding='utf-8')
            return fake_validate(path, mapping)
        def fake_validate(path, mapping):
            assert mapping == targets
            for slot, target in mapping.items():
                if (path/slot/'target.txt').read_text(encoding='utf-8') != str(target):
                    raise ValueError('MOCK wrong alias target')
            return {slot: str(target) for slot, target in mapping.items()}
        def install_patches(stack):
            stack.enter_context(patch.object(tool.selected.score, 'validate_install', return_value=installation['score']))
            stack.enter_context(patch.object(tool.selected.compact, 'validate_install', return_value=installation['topk']))
            stack.enter_context(patch.object(tool.scale.chain.merge, 'validate_install', return_value=installation['merge']))
            stack.enter_context(patch.object(tool.scale.chain, 'artifacts', return_value=installation['merge_device_objects']))
            stack.enter_context(patch.object(tool.selected.previous, 'create_runtime', fake_create))
            stack.enter_context(patch.object(tool.selected.previous, 'validate_runtime', fake_validate))

        # Build an explicitly mocked successful prior complete-Search report, with real data/checkers.
        accepted = base/'accepted'; accepted.mkdir()
        runtime = accepted/'runtime_candidate/vendors'; mapping = fake_create(runtime, targets)
        raw = accepted/'raw_queries'; raw.mkdir()
        for name in records: tool.search.write_queries(raw/(name+'.txt'), cache[name]['queries'])
        independent = base/'FAKE_PRIOR_PROOF'; independent.write_bytes(b'FAKE PRIOR DEVICE PROOF')
        evidence = {str(independent): tool.digest(independent)}
        def old_child(command, log, env):
            actual = Path(command[3]); log.write_bytes(b'MOCK OLD SEARCH CHILD')
            session = mock_session(False, actual, index, records, 10, 8, 8, 8, 8, 24, full)
            tool.r.write_json(actual/'search_session.json', session)
        with patch.object(tool.selected.score, 'execute', old_child):
            pre = tool.selected.previous.preflight(accepted/'preflight_candidate', inputs, index, records,
                cache, raw, runners['old_npu_diagnostic'], runtime, installation, 2, 8, 8, 24)
        prior_rows = []
        def old_measure(command, log, env):
            log.write_bytes(b'MOCK OLD BENCHMARK')
            measurement(Path(command[4]), index, records, full, 'npu', 1, 10, 7.)
        with patch.object(tool.scale.chain.score, 'execute', old_measure):
            for ordinal in range(2):
                result = tool.bench.run_one(accepted/f'round{ordinal+1}/candidate', 'npu', inputs, index, records, cache,
                    raw, runners['npu'], runtime, installation, 1, 1, 8, 2, 8, 24, 1, 10)
                prior_rows.append({'round': ordinal+1, 'status': 'both_topk_search_versions_and_all_five_array_bytes_passed',
                    'process_order': ['baseline', 'candidate'] if ordinal % 2 == 0 else ['candidate', 'baseline'], 'candidate': result})
        prior_record = {'status': tool.selected.STATUS, 'index_id': index['index_id'], 'docs': 128, 'queries': 8, 'k': 10,
            'input_sha256': snapshot, 'source_is_real_corpus': False, 'device': 2, 'shards': 8, 'capacity_queries': 8, 'capacity_words': 24,
            'tool_source_sha256': tool.checks.sources(tool.selected.TOOL_SOURCES),
            'host_source_sha256': tool.checks.sources(tuple(sorted(set(tool.checks.NPU_SOURCES) | set(tool.search.HOST_SOURCES) | set(tool.selected.compact.HOST_SOURCES)))),
            'runner_sha256': tool.digest(runners['npu']), 'diagnostic_runner_sha256': tool.digest(runners['old_npu_diagnostic']),
            'candidate_installation': installation, 'boundary_acceptance': {'evidence_sha256': evidence},
            'compact_acceptance': {'evidence_sha256': evidence}, 'runtime_mapping': {'candidate': mapping},
            'raw_query_sha256': {str(path): tool.digest(path) for path in raw.iterdir()},
            'preflight': {'candidate': pre}, 'rounds': 2, 'warmup_cycles': 1, 'measured_cycles': 10, 'per_round': prior_rows}
        path_report = accepted/'report.json'; tool.r.write_json(path_report, prior_record)
        with ExitStack() as stack:
            install_patches(stack)
            invoke_proof = lambda: tool.validate_selected(accepted, inputs, index, records, snapshot, cache, runners['npu'], vendors, projects)
            proof = invoke_proof(); count += 1
            for key, value in (('status', 'not_run'), ('input_sha256', {}), ('tool_source_sha256', {}),
                               ('host_source_sha256', {}), ('candidate_installation', {}), ('per_round', [])):
                bad = copy.deepcopy(prior_record); bad[key] = value; tool.r.write_json(path_report, bad)
                try: fails(invoke_proof)
                except KeyError: count += 1
            tool.r.write_json(path_report, prior_record)
            for path in (independent, accepted/'raw_queries/A.txt', accepted/'preflight_candidate/requests.tsv',
                         accepted/'round1/candidate/actual/before_A/top_ids.bin', accepted/'round1/candidate/run.log', runners['npu']):
                saved = path.read_bytes(); path.write_bytes(saved+b'CORRUPTION')
                fails(invoke_proof); path.write_bytes(saved)
            child_report = accepted/'round1/candidate/report.json'; child_saved = json.loads(child_report.read_text())
            bad = copy.deepcopy(child_saved); bad['summary']['A']['median_batch_ms'] = .001
            tool.r.write_json(child_report, bad)
            bad_root = copy.deepcopy(prior_record); bad_root['per_round'][0]['candidate']['report_sha256'] = tool.digest(child_report)
            tool.r.write_json(path_report, bad_root); fails(invoke_proof)
            tool.r.write_json(child_report, child_saved); tool.r.write_json(path_report, prior_record)

        modes = ('cpu_wins', 'npu_wins', 'cpu_preflight_fails', 'cpu_preflight_wrong_ids', 'cpu_preflight_fewer_threads',
                 'cpu_fails', 'npu_fails', 'wrong_npu_ids', 'cpu_fewer_threads', 'kernel_only_timer',
                 'stale_host', 'nan_time', 'wrong_transfer', 'input_changes', 'installed_changes')
        for mode in modes:
            output = base/('run_'+mode); commands = []
            def child(command, log, env):
                commands.append(command); diagnostic = command[0] == str(runners['diagnostic'])
                log.write_bytes(b'MOCK CURRENT CHILD -- NO DEVICE')
                if diagnostic:
                    assert command[1] == str(inputs/'index') and command[-1] == 'raw_query_ids_v1'
                    if mode == 'cpu_preflight_fails': raise subprocess.CalledProcessError(7, command)
                    actual = Path(command[3]); session = mock_session(True, actual, index, records, 10, 8, 8, 8, 8, 24, full)
                    if mode == 'cpu_preflight_wrong_ids': (actual/'call0_A/top_ids.bin').write_bytes(bytes(320))
                    if mode == 'cpu_preflight_fewer_threads': session['actual_threads'] = 4
                    tool.r.write_json(actual/'cpu_search_session.json', session)
                    return
                cpu = command[0] == str(runners['cpu']); backend = 'cpu' if cpu else 'npu'
                assert env['ASCEND_CUSTOM_OPP_PATH'] == ('PARENT' if cpu else ':'.join(str((output/'runtime/vendors'/slot).resolve()) for slot in tool.selected.SLOTS))
                assert command[2:4] == [str(output/'raw_queries/A.txt'), str(output/'raw_queries/B.txt')]
                if mode == backend+'_fails': raise subprocess.CalledProcessError(7, command)
                elapsed = (.1 if cpu else .2) if mode != 'npu_wins' else (.2 if cpu else .1)
                actual = Path(command[4]); session = measurement(actual, index, records, full, backend, 1, 10, elapsed)
                if mode == 'wrong_npu_ids' and not cpu: (actual/'before_A/top_ids.bin').write_bytes(bytes(320))
                if mode == 'cpu_fewer_threads' and cpu: session['actual_threads'] = 4
                if mode == 'kernel_only_timer': session['timer_policy'] = 'kernel only'
                if mode == 'stale_host': session['host_source_sha256'] = {}
                if mode == 'nan_time': session['samples'][0]['call_wall_ms'] = float('nan')
                if mode == 'wrong_transfer' and not cpu: session['samples'][0]['query_h2d_bytes'] = 0
                if mode == 'input_changes': (output/'raw_queries/A.txt').write_bytes(b'CORRUPTION')
                tool.r.write_json(actual/'benchmark_session.json', session)
            with ExitStack() as stack:
                install_patches(stack)
                stack.enter_context(patch.object(tool, 'validate_selected', return_value=proof))
                stack.enter_context(patch.object(tool.selected.score, 'execute', child))
                stack.enter_context(patch.object(tool.scale.chain.score, 'execute', child))
                stack.enter_context(patch.dict(os.environ, {'ASCEND_CUSTOM_OPP_PATH': 'PARENT'}))
                if mode == 'installed_changes':
                    stack.enter_context(patch.object(tool.selected.compact, 'validate_install', return_value={'CHANGED': True}))
                invoke = lambda: tool.run(inputs, accepted, output, runners['cpu'], runners['npu'], runners['diagnostic'], vendors,
                    projects, 8, 8, 2, 2, 1, 10)
                if mode in ('cpu_wins', 'npu_wins'): invoke()
                else: fails(invoke)
                assert os.environ['ASCEND_CUSTOM_OPP_PATH'] == 'PARENT'; count += 1
            report = json.loads((output/'report.json').read_text())
            assert report['status'] == (tool.STATUS if mode in ('cpu_wins', 'npu_wins') else 'failed'); count += 1
            if mode.startswith('cpu_preflight_'):
                assert len(commands) == 1 and not report['per_round']; count += 1
            if mode in ('cpu_wins', 'npu_wins'):
                assert len(commands) == 5 and [row['process_order'] for row in report['per_round']] == [['cpu', 'npu'], ['npu', 'cpu']]
                item = report['paired_summary']['A']
                assert item['npu_faster_rounds'] == (2 if mode == 'npu_wins' else 0)
                assert item['paired_cpu_over_npu_ratios'] == ([2., 2.] if mode == 'npu_wins' else [.5, .5])
                assert item['cpu_run_medians_ms'] == ([.2, .2] if mode == 'npu_wins' else [.1, .1])
                assert report['metric'] == 'batch_latency_and_search_time_throughput'; count += 5
        with patch.object(tool.selected.score, 'execute') as child:
            for options, target in (({'rounds': 3}, 'bad_rounds'), ({'parts': 9, 'threads': 8}, 'bad_parts'), ({}, 'run_cpu_wins')):
                fails(lambda: tool.run(inputs, accepted, base/target, runners['cpu'], runners['npu'], runners['diagnostic'], vendors, projects, **options))
            assert not child.called; count += 1
    print(f'PASS: {count} selected-combination proof, current CPU/NPU order, honest win/loss, full-boundary and fail-stop checks. '
          'FAKE binaries/install/proof, synthetic links and MOCK children only; no C++ build, ACLNN or hardware speedup evidence.')


if __name__ == '__main__': main()
