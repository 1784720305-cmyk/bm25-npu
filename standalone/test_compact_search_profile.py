"""Real small host oracles, FAKE installation/runner/prior, MOCK child and synthetic CSV; no NPU evidence."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import profile_compact_search as app, analyze_compact_search_profile as tool, scale_fixture
from standalone.test_compact_search_cpu import measurement
from standalone.test_search_profile import make_csv, write_csv


def main():
    count = 0
    def fails(call):
        nonlocal count
        try: call()
        except (ValueError, OSError, subprocess.SubprocessError): count += 1
        else: raise AssertionError('Invalid current-combination profile accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='compact_profile_MOCK_ONLY_', dir=ROOT/'results') as temporary, redirect_stdout(io.StringIO()):
        base = Path(temporary); inputs, source = base/'inputs', base/'source'
        scale_fixture.generate(source, 128, 8, 10, 128, 20261014, 0, 'performance', 3)
        app.fixture.prepare(source, inputs)
        index, records, snapshot = app.fixture.validate_inputs(inputs)
        cache = app.checks.cache_inputs(inputs, index, records); full = {key: value['scores'] for key, value in cache.items()}
        runner = base/'FAKE_NOT_EXECUTABLE'; runner.write_bytes(b'MOCK NO DEVICE')
        evidence = base/'FAKE_PRIOR_DEVICE_PROOF'; evidence.write_bytes(b'MOCK PRIOR')
        vendors = base/'vendors'; projects = {key: base/key for key in ('score', 'topk', 'merge')}
        targets = {'bm25_query_score': vendors/'bm25_query_score_range',
            'bm25_topk_radix_collect': vendors/app.winner.selected.compact.VENDOR, 'bm25_topk_merge': vendors/'bm25_topk_merge'}
        for path in targets.values(): path.mkdir(parents=True)
        installation = {'score': {'MOCK': 'score'}, 'topk': {'MOCK': 'top'}, 'merge': {'MOCK': 'merge'},
            'merge_device_objects': {'MOCK.o': 'hash'}, 'topk_acceptance': {'workspace_bytes': 16777216}}
        prior = {'installation': installation, 'shards': 8, 'targets': targets,
            'evidence_sha256': {str(evidence): app.digest(evidence)}}
        def fake_validate(path, mapping):
            assert mapping == targets
            for slot, target in mapping.items():
                if (path/slot/'target.txt').read_text(encoding='utf-8') != str(target): raise ValueError('MOCK alias changed')
            return {slot: str(target) for slot, target in mapping.items()}
        def fake_create(path, mapping):
            path.mkdir(parents=True)
            for slot, target in mapping.items():
                (path/slot).mkdir(); (path/slot/'target.txt').write_text(str(target), encoding='utf-8')
            return fake_validate(path, mapping)
        with ExitStack() as stack:
            stack.enter_context(patch.object(app.winner, 'validate_selected', return_value=prior))
            stack.enter_context(patch.object(app.winner.selected.previous, 'create_runtime', fake_create))
            stack.enter_context(patch.object(app.winner.selected.previous, 'validate_runtime', fake_validate))
            stack.enter_context(patch.object(app.winner.selected.score, 'validate_install', return_value=installation['score']))
            stack.enter_context(patch.object(app.winner.selected.compact, 'validate_install', return_value=installation['topk']))
            stack.enter_context(patch.object(app.scale.chain.merge, 'validate_install', return_value=installation['merge']))
            stack.enter_context(patch.object(app.scale.chain, 'artifacts', return_value=installation['merge_device_objects']))
            app.validate_install(installation, vendors, projects); count += 1
            for key in ('score', 'topk', 'merge', 'merge_device_objects'):
                changed = copy.deepcopy(installation); changed[key] = {}
                fails(lambda: app.validate_install(changed, vendors, projects))
            for mode in ('pass', 'child_fails', 'wrong_ids'):
                output = base/('run_'+mode)
                def child(command, log, env):
                    log.write_bytes(b'MOCK CURRENT NPU CHILD -- NO DEVICE')
                    assert len(command) == 14 and command[5:] == ['2', str(output/'runtime/vendors'), '10', '8', '8', '24', '16777216', '5', '10']
                    assert env['ASCEND_CUSTOM_OPP_PATH'] == ':'.join(str((output/'runtime/vendors'/slot).resolve()) for slot in app.winner.selected.SLOTS)
                    if mode == 'child_fails': raise subprocess.CalledProcessError(7, command)
                    actual = Path(command[4]); measurement(actual, index, records, full, 'npu', 5, 10, .1)
                    if mode == 'wrong_ids': (actual/'before_A/top_ids.bin').write_bytes(bytes(320))
                with patch.object(app.scale.chain.score, 'execute', child):
                    invoke = lambda: app.run(inputs, base/'accepted', output, runner, vendors, projects)
                    if mode == 'pass': invoke()
                    else: fails(invoke)
                report = json.loads((output/'report.json').read_text())
                assert report['status'] == (app.STATUS if mode == 'pass' else 'failed'); count += 1
            run = base/'run_pass'; report, session = app.validate_record(run); count += 1
            csv_root = base/'SYNTHETIC_CSV'; rows, tasks = make_csv(csv_root, report, session)
            # Optional metrics are retained as raw strings, never converted into occupancy.
            for row in rows: row.update(aiv_vec_ratio='0.25', aiv_scalar_time='N/A')
            write_csv(csv_root/'op_summary_TEST_ONLY.csv', rows)
            analysis = tool.analyze(csv_root, run)
            assert analysis['own_operator_rows'] == 105 and analysis['measured_search_calls'] == 20
            assert analysis['score_variant'] == 'query_score_range' and analysis['topk_variant'] == 'topk_positive_compact'
            assert len(analysis['optional_raw_counters_by_task']) == 60
            assert analysis['optional_raw_counters_by_task'][0]['counters'] == {'aiv_vec_ratio': '0.25', 'aiv_scalar_time': 'N/A'}
            for case in ('A', 'B'):
                data = analysis['cases'][case]
                assert data['operators']['Bm25QueryScore']['count'] == 10
                assert data['kernel_sum_us']['median_us'] == 6
                assert abs(data['device_chain_span_us']['median_us']-6.2) < 1e-9
                assert abs(data['outside_device_chain_elapsed_us']['median_us']-93.8) < 1e-9
            assert analysis['whole_application_api_totals_not_steady_state'][0]['whole_application_total_us'] == 1000000
            count += 13
            for field, value in (('OP Type', 'Bm25Accumulate'), ('Device_id', '4'), ('Stream ID', '99'),
                ('Task Duration(us)', '0'), ('Block Num', '0'), ('Input Shapes', '1;1;1')):
                bad = copy.deepcopy(rows); bad[0][field] = value; write_csv(csv_root/'op_summary_TEST_ONLY.csv', bad)
                fails(lambda: tool.analyze(csv_root, run))
            write_csv(csv_root/'op_summary_TEST_ONLY.csv', rows[:-1]); fails(lambda: tool.analyze(csv_root, run))
            bad = copy.deepcopy(rows); bad[1]['Task ID'] = bad[0]['Task ID']
            write_csv(csv_root/'op_summary_TEST_ONLY.csv', bad); fails(lambda: tool.analyze(csv_root, run))
            write_csv(csv_root/'op_summary_TEST_ONLY.csv', rows)
            bad = copy.deepcopy(tasks); bad[0]['task_time(us)'] = '9'; write_csv(csv_root/'task_time_TEST_ONLY.csv', bad)
            fails(lambda: tool.analyze(csv_root, run)); write_csv(csv_root/'task_time_TEST_ONLY.csv', tasks)
            duplicate = csv_root/'op_summary_duplicate.csv'; write_csv(duplicate, rows)
            fails(lambda: tool.analyze(csv_root, run)); duplicate.unlink()
            shortened = copy.deepcopy(session)
            for sample in shortened['samples']: sample.update(call_wall_ms=.001, internal_search_ms=.0009)
            with patch.object(tool.application, 'validate_record', return_value=(report, shortened)):
                fails(lambda: tool.analyze(csv_root, run))
            root_report = run/'report.json'
            for key, value in (('status', 'old_profile_passed'), ('topk_variant', 'radix_collect'), ('tool_source_sha256', {})):
                bad = copy.deepcopy(report); bad[key] = value; app.r.write_json(root_report, bad)
                fails(lambda: app.validate_record(run))
            app.r.write_json(root_report, report)
            for file in (evidence, runner, run/'measurement/run.log', run/'measurement/actual/after_A/scores.bin',
                         run/'runtime/vendors/bm25_query_score/target.txt'):
                original = file.read_bytes(); file.write_bytes(original+b'CORRUPTION')
                fails(lambda: app.validate_record(run)); file.write_bytes(original)
            with patch.object(app.scale.chain.score, 'execute') as child:
                fails(lambda: app.run(inputs, base/'accepted', run, runner, vendors, projects))
                assert not child.called; count += 1
    print(f'PASS: {count} current-combination provenance/oracle/MOCK-child/synthetic-CSV/phase/clock/fail-stop checks. '
          'FAKE binaries/install/prior and synthetic aliases only; no compilation, msprof or device execution.')


if __name__ == '__main__': main()
