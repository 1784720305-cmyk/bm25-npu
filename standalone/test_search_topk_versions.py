"""Real fixture/checker tests with FAKE artifacts and MOCK Search children; no NPU evidence."""
from contextlib import ExitStack, redirect_stdout
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
sys.path.insert(0, str(ROOT))
from standalone import compare_search_topk_versions as tool, scale_fixture
from standalone.test_search_scale import mock_session
from standalone.test_resident_chain import mock_outputs


def make_topk_proof(directory, runner, installation):
    """Deliberately fake device proof files, generated only inside this test's temporary tree."""
    directory.mkdir()
    (directory/'library.log').write_bytes(b'MOCK LIBRARY CHECK -- NO EXECUTION')
    c = tool.compact
    report = {'status': c.STATUS, 'installation': installation, 'tool_source_sha256': tool.checks.sources(c.TOOLS+c.HOST_SOURCES),
              'device': 2, 'calls_per_case': 5, 'runner_sha256': tool.digest(runner), 'cases': []}
    specs = [('regression', spec) for spec in c.old.SPECS]+[('capacity', spec) for spec in c.BOUNDARIES]
    for ordinal, (kind, (q, docs, k)) in enumerate(specs):
        case = directory/f'case{ordinal}_{kind}_q{q}_d{docs}_k{k}'; case.mkdir()
        (c.old.prepare_pair if kind == 'regression' else c.prepare_boundary)(case, q, docs, k)
        snapshot = {p.relative_to(case).as_posix(): tool.digest(p) for p in case.rglob('*') if p.is_file()}
        actual = case/'actual'; actual.mkdir()
        for v in ('a', 'b'):
            for name in ('ids', 'scores'):
                (actual/f'{v}_top_{name}.bin').write_bytes((case/f'input_{v}/expected_{name}.bin').read_bytes())
        record = {'status': 'positive_compact_device_calls_and_cpp_full_sort_passed', 'queries': q, 'docs': docs, 'k': k,
                  'device': 2, 'validated_calls': 5, 'workspace_bytes': 16777216, 'declared_user_gm_scratch_bytes': 0,
                  'declared_ub_algorithm_bytes': 16704, 'input_calls': [3, 2], 'every_call_poisoned': True,
                  'every_call_full_sort_checked': True, 'host_source_sha256': c.host_hashes()}
        tool.r.write_json(actual/'device_validation.json', record)
        for name in ('device.log', 'check.log'): (case/name).write_bytes(b'MOCK LOG ONLY')
        report['cases'].append({'kind': kind, 'queries': q, 'docs': docs, 'k': k, 'status': 'passed',
            'command': [str(runner), str(case), str(actual), str(q), str(docs), str(k), '2', installation['api_library'], '5'],
            'input_sha256': snapshot, 'validation': record,
            'output_sha256': {name: tool.digest(actual/name) for name in
                             ('a_top_scores.bin', 'a_top_ids.bin', 'b_top_scores.bin', 'b_top_ids.bin')},
            'device_log_sha256': tool.digest(case/'device.log'), 'check_log_sha256': tool.digest(case/'check.log')})
    tool.r.write_json(directory/'report.json', report)
    return report


def main():
    count = 0
    def fails(call):
        nonlocal count
        try: call()
        except (ValueError, OSError, subprocess.SubprocessError): count += 1
        else: raise AssertionError('invalid comparison accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='search_topk_MOCK_ONLY_', dir=ROOT/'results') as temp, redirect_stdout(io.StringIO()):
        base = Path(temp); vendors = base/'vendors'; vendors.mkdir()
        runner, diagnostic, top_runner = [base/name for name in ('FAKE_MEASURE', 'FAKE_DIAGNOSTIC', 'FAKE_TOPK')]
        for path in (runner, diagnostic, top_runner): path.write_bytes(b'FAKE EXECUTABLE NEVER RUN')
        for slot in tool.SLOTS+('bm25_query_score_range', tool.compact.VENDOR): (vendors/slot).mkdir()
        installation = {'api_library': str(vendors/tool.compact.VENDOR/'FAKE_API'),
                        'installed_device_object_sha256': {'FAKE.o': 'MOCK'}, 'TEST_ONLY_FAKE': True}
        accepted = base/'topk_proof'
        saved = make_topk_proof(accepted, top_runner, installation)
        proof = tool.validate_compact_acceptance(accepted, installation)
        assert proof['device_cases'] == 37 and proof['device_calls'] == 185 and proof['workspace_bytes'] == 16777216; count += 3
        root_report = accepted/'report.json'
        for key, value in (('status', 'compiled_only'), ('tool_source_sha256', {}), ('installation', {}),
                           ('runner_sha256', 'bad'), ('cases', saved['cases'][:-1]), ('calls_per_case', 3)):
            bad = copy.deepcopy(saved); bad[key] = value; tool.r.write_json(root_report, bad)
            fails(lambda: tool.validate_compact_acceptance(accepted, installation))
        tool.r.write_json(root_report, saved)
        first = accepted/'case0_regression_q6_d6_k1'
        for path in (first/'input_a/scores.bin', first/'actual/a_top_ids.bin', first/'device.log',
                     first/'check.log', first/'actual/device_validation.json', top_runner):
            raw = path.read_bytes(); path.write_bytes(raw+b'CORRUPTION')
            fails(lambda: tool.validate_compact_acceptance(accepted, installation)); path.write_bytes(raw)
        for field, value in (('command', []), ('status', 'not_run'), ('input_sha256', {}), ('output_sha256', {})):
            bad = copy.deepcopy(saved); bad['cases'][0][field] = value; tool.r.write_json(root_report, bad)
            fails(lambda: tool.validate_compact_acceptance(accepted, installation))
        tool.r.write_json(root_report, saved)
        # Even if a report hash is rewritten, wrong scores/IDs fail independent full sorting.
        target = first/'actual/a_top_ids.bin'; original = target.read_bytes()
        target.write_bytes(struct.pack('<i', -1)*6)
        bad = copy.deepcopy(saved); bad['cases'][0]['output_sha256']['a_top_ids.bin'] = tool.digest(target)
        tool.r.write_json(root_report, bad)
        fails(lambda: tool.validate_compact_acceptance(accepted, installation))
        target.write_bytes(original); tool.r.write_json(root_report, saved)

        source, inputs = base/'source', base/'inputs'
        scale_fixture.generate(source, 128, 8, 10, 128, 20261014, 0, 'performance', 3)
        tool.fixture.prepare(source, inputs)
        index, records, unused = tool.fixture.validate_inputs(inputs)
        cache = tool.checks.cache_inputs(inputs, index, records)
        full = {name: value['scores'] for name, value in cache.items()}
        old_install = {'TEST_ONLY_FAKE': True, 'score': {'variant': 'original'}, 'topk_acceptance': {'workspace_bytes': 16777216}}
        range_install = {'variant': 'range', 'TEST_ONLY_FAKE': True}
        prior_score = {'evidence_sha256': {}}

        def fake_runtime(path, targets):
            path.mkdir(parents=True)
            for slot, target in targets.items():
                (path/slot).mkdir(); (path/slot/'target.txt').write_text(str(target), encoding='utf-8')
            return fake_validate_runtime(path, targets)
        def fake_validate_runtime(path, targets):
            assert targets['bm25_query_score'] == vendors/'bm25_query_score_range'
            expected_top = tool.compact.VENDOR if path.parent.name == 'runtime_candidate' else 'bm25_topk_radix_collect'
            assert targets['bm25_topk_radix_collect'] == vendors/expected_top
            assert all((path/slot/'target.txt').read_text(encoding='utf-8') == str(target) for slot, target in targets.items())
            return {slot: str(target) for slot, target in targets.items()}

        modes = ('pass', 'preflight_failure', 'preflight_bad_ids', 'measure_failure', 'bad_local_candidate',
                 'nan_time', 'stale_host', 'wrong_transfer', 'library_swap', 'input_mutation')
        for mode in modes:
            output = base/('run_'+mode); commands = []
            def child(command, logfile, env):
                commands.append(command)
                diagnostic_call = command[0] == str(diagnostic)
                runtime = Path(command[5] if diagnostic_call else command[6])
                assert env['ASCEND_CUSTOM_OPP_PATH'] == ':'.join(str((runtime/slot).resolve()) for slot in tool.SLOTS)
                assert command[1] == str(inputs/'index')
                candidate = runtime.parent.name == 'runtime_candidate'
                logfile.write_bytes(b'MOCK CHILD ONLY -- NO DEVICE')
                if (diagnostic_call and mode == 'preflight_failure') or (not diagnostic_call and mode == 'measure_failure'):
                    raise subprocess.CalledProcessError(7, command)
                actual = Path(command[3] if diagnostic_call else command[4])
                if diagnostic_call:
                    session = mock_session(False, actual, index, records, 10, 8, 8, 1, 8, 24, full)
                    if mode == 'preflight_bad_ids': (actual/'call0_A/top_ids.bin').write_bytes(bytes(8*10*4))
                    tool.r.write_json(actual/'search_session.json', session)
                    return
                actual.mkdir(); warmup, repeats = map(int, command[-2:])
                session = {'status': 'search_benchmark_calls_completed_checks_pending', 'backend': 'npu', 'docs': 128,
                    'queries': 8, 'k': 10, 'partitions': 8, 'capacity_queries': 8, 'capacity_words': 24,
                    'input_contract': 'raw_grouped_INT32_word_IDs_in_CPU_memory', 'timer_policy': tool.checks.TIMER,
                    'index_load_batches': 1, 'index_bytes': index['index_bytes'], 'cold_init_ms': 1., 'cold_synthetic_empty_queries': 8,
                    'checkpoint_search_calls': 4, 'warmup_cycles': warmup, 'warmup_search_calls': 2*warmup, 'warmup_topk_bit_checks': 2*warmup,
                    'measured_cycles': repeats, 'measured_search_calls': 2*repeats, 'measured_topk_bit_checks': 2*repeats,
                    'measured_debug_download_calls': 0, 'measured_poison_calls': 0, 'measured_file_io_calls': 0,
                    'diagnostic_checkpoints_outside_measurement': True, 'index_readback_byte_identical': True,
                    'host_source_sha256': tool.checks.sources(tool.checks.NPU_SOURCES), 'device': 2, 'index_tensor_h2d_calls': 5,
                    'pipeline_stages': 3, 'debug_index_d2h_calls': 5, 'intermediate_host_roundtrips': 0,
                    'workspace_capacity_bytes': [16777216]*3, 'samples': []}
                for cycle in range(repeats):
                    for slot in range(2):
                        case = 'B' if slot ^ (cycle & 1) else 'A'; elapsed = .1 if candidate else .2
                        session['samples'].append({'cycle': cycle, 'case': case, 'call_wall_ms': elapsed,
                            'internal_search_ms': elapsed*.99, 'query_h2d_bytes': 168, 'topk_d2h_bytes': 640})
                for label in tool.checks.CHECKPOINTS: mock_outputs(actual/label, full[label[-1]], 128, 8, 10, 8)
                if mode == 'bad_local_candidate': (actual/'before_A/partial_ids.bin').write_bytes(bytes(8*8*10*4))
                if mode == 'nan_time': session['samples'][0]['call_wall_ms'] = float('nan')
                if mode == 'stale_host': session['host_source_sha256'] = {}
                if mode == 'wrong_transfer': session['samples'][0]['query_h2d_bytes'] = 0
                if mode == 'input_mutation': (output/'raw_queries/A.txt').write_text('CORRUPTION', encoding='utf-8')
                tool.r.write_json(actual/'benchmark_session.json', session)
            with ExitStack() as stack:
                for owner, name, value in ((tool.scale.chain, 'validate_install', old_install),
                    (tool.score, 'validate_install', range_install), (tool.compact, 'validate_install', installation),
                    (tool.previous, 'validate_boundaries', prior_score), (tool, 'validate_compact_acceptance', proof)):
                    stack.enter_context(patch.object(owner, name, return_value=value))
                stack.enter_context(patch.object(tool.previous, 'create_runtime', fake_runtime))
                stack.enter_context(patch.object(tool.previous, 'validate_runtime',
                    side_effect=ValueError('MOCK replaced library mapping') if mode == 'library_swap' else fake_validate_runtime))
                stack.enter_context(patch.object(tool.score, 'execute', child))
                stack.enter_context(patch.object(tool.scale.chain.score, 'execute', child))
                stack.enter_context(patch.dict(os.environ, {'ASCEND_CUSTOM_OPP_PATH': 'PARENT'}))
                invoke = lambda: tool.run(inputs, output, runner, diagnostic, vendors,
                    base, base, base, base, base, base, accepted, base, 2, 8, 2, 1, 10)
                if mode == 'pass': invoke()
                else: fails(invoke)
                assert os.environ['ASCEND_CUSTOM_OPP_PATH'] == 'PARENT'; count += 1
            report = json.loads((output/'report.json').read_text(encoding='utf-8'))
            assert report['status'] == (tool.STATUS if mode == 'pass' else 'failed'); count += 1
            if mode.startswith('preflight_'):
                assert len(commands) == 1 and not report['per_round']; count += 1
            if mode == 'pass':
                assert len(commands) == 6 and report['metric'] == 'batch_latency_and_search_time_throughput'
                assert [row['process_order'] for row in report['per_round']] == [['baseline', 'candidate'], ['candidate', 'baseline']]
                assert report['paired_summary']['A']['paired_baseline_over_candidate_ratios'] == [2., 2.]
                assert all(len(row['cross_checks']) == 4 for row in report['per_round'])
                assert report['baseline_installation']['score'] == report['candidate_installation']['score'] == range_install
                count += 5
        with patch.object(tool.score, 'execute') as child:
            fails(lambda: tool.run(inputs, base/'run_pass', runner, diagnostic, vendors, base, base, base, base, base, base, accepted, base))
            fails(lambda: tool.run(inputs, base/'odd', runner, diagnostic, vendors, base, base, base, base, base, base, accepted, base, rounds=3))
            big_source, big = base/'big_source', base/'big'
            scale_fixture.generate(big_source, 128, 32, 10, 128, 20261014, 0, 'performance', 3)
            tool.fixture.prepare(big_source, big)
            fails(lambda: tool.run(big, base/'over64', runner, diagnostic, vendors, base, base, base, base, base, base, accepted, base))
            assert not child.called; count += 1
    print(f'PASS: {count} existing-proof/byte-check/paired-Search/timing/failure tests. '
          'FAKE binaries/artifacts, synthetic links and MOCK children; no CANN, OS symlink or NPU/performance proof.')


if __name__ == '__main__': main()
