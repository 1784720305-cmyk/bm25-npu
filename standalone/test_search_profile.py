"""Host preparation, FAKE NPU execution and synthetic msprof CSV tests only; no device profiling."""
from contextlib import redirect_stdout
import copy
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import profile_search as app, analyze_search_profile as tool, resident_index as r
from standalone import search_scale_fixture as fixture, scale_fixture, search_benchmark_checks as checks
from standalone.test_resident_chain import mock_outputs


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def make_csv(root, report, session):
    root.mkdir(); summary, tasks = [], []
    index, records = report['index'], report['records']
    n, v, m, s, k = (index['num_docs'], index['vocab_size'], index['posting_count'], report['shards'], records['A']['k'])
    epoch = Decimal('1792081509347637.533')
    for call, (phase, case) in enumerate(tool.phases(session)):
        q = report['capacity_queries'] if phase == 'cold' else records[case]['num_queries']
        words = 1 if phase == 'cold' else records[case]['query_term_count']
        shapes = (f'{v+1};{m};{m};{n};{v};{q+1};{words}', f'{q*s},{n//s}', f'{q},{s},{k};{q},{s},{k}')
        start = epoch+call*100
        for pos, (op, duration) in enumerate(zip(tool.OPERATORS, (Decimal(2), Decimal(3), Decimal(1)))):
            tid = str(call*3+pos+5); stop = start+duration
            summary.append({'Device_id':'2', 'Task ID':tid, 'Stream ID':'56', 'OP Type':op,
                'Task Type':'AI_VECTOR_CORE', 'Task Start Time(us)':str(start)+'\t', 'Task Duration(us)':str(duration),
                'Block Num':str(1+pos), 'Input Shapes':'"'+shapes[pos]+'"'})
            tasks.append({'Device_id':'2', 'kernel_name':op+'_TEST_ONLY', 'kernel_type':'AI_VECTOR_CORE', 'stream_id':'56',
                'task_id':tid, 'task_time(us)':str(duration), 'task_start(us)':str(start)+'\t', 'task_stop(us)':str(stop)+'\t'})
            start = stop+Decimal('.1')
    write_csv(root/'op_summary_TEST_ONLY.csv', summary); write_csv(root/'task_time_TEST_ONLY.csv', tasks)
    write_csv(root/'api_statistic_TEST_ONLY.csv', [{'Device_id':'host', 'Level':'runtime', 'API Name':'DeviceReset',
        'Time(us)':'1000000', 'Count':'1', 'Avg(us)':'1000000', 'Min(us)':'1000000', 'Max(us)':'1000000'}])
    return summary, tasks


def main():
    count = 0
    def fails(call):
        nonlocal count
        try: call()
        except (ValueError, OSError, subprocess.SubprocessError): count += 1
        else: raise AssertionError('Invalid profile execution/CSV accepted')
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='search_profile_FAKE_CSV_ONLY_', dir=ROOT/'results') as temporary, redirect_stdout(io.StringIO()):
        base = Path(temporary); source = base/'source'; inputs = base/'inputs'
        scale_fixture.generate(source, 128, 2, 3, 128, 20261014, 0, 'performance', 3)
        fixture.prepare(source, inputs); index, records, snapshot = fixture.validate_inputs(inputs)
        cache = checks.cache_inputs(inputs, index, records)
        executable = base/'TEST_ONLY_FAKE_NPU_NEVER_EXECUTED'; executable.write_bytes(b'TEST ONLY FAKE NPU BINARY')
        baseline = base/'FAKE_FORMAL_REFERENCE'; baseline.mkdir()
        single = baseline/'FAKE_MEASUREMENT'; single.mkdir()
        r.write_json(single/'report.json', {'status':'actual_search_measurement_and_checkpoint_checks_passed', 'index_id':index['index_id']})
        linked = {'directory':str(single), 'report_sha256':app.digest(single/'report.json')}
        prior = {'status':app.bench.STATUS, 'index_id':index['index_id'], 'input_sha256':snapshot,
            'source_is_real_corpus':False, 'tool_source_sha256':checks.sources(app.bench.TOOL_SOURCES),
            'host_source_sha256':checks.sources(tuple(sorted(set(checks.CPU_SOURCES) | set(checks.NPU_SOURCES)))),
            'npu_runner_sha256':app.digest(executable), 'paired_rounds':2, 'per_round':[
                {'status':'both_actual_measurements_and_all_cross_checks_passed', 'cpu':linked, 'npu':linked} for unused in range(2)]}
        r.write_json(baseline/'report.json', prior)
        app.validate_baseline(baseline, index, snapshot, executable); count += 1
        for key, value in (('status', 'prepared_input_only'), ('index_id', 'wrong'), ('npu_runner_sha256', 'bad')):
            bad = dict(prior); bad[key] = value; r.write_json(baseline/'report.json', bad)
            fails(lambda:app.validate_baseline(baseline, index, snapshot, executable))
        r.write_json(baseline/'report.json', prior)
        vendors = base/'vendors'
        for name in ('bm25_query_score', 'bm25_topk_radix_collect', 'bm25_topk_merge'): (vendors/name).mkdir(parents=True)
        installation = {'TEST_ONLY_FAKE_INSTALL':True, 'topk_acceptance':{'workspace_bytes':16777216}}
        for mode in ('pass', 'child_failure', 'bad_ids'):
            output = base/('run_'+mode); commands = []
            def child(command, logfile, env):
                commands.append(command); logfile.write_bytes(b'TEST ONLY MOCK NPU CHILD; NO DEVICE')
                assert command[5:] == ['2', str(vendors), '3', '2', '2', '6', '16777216', '5', '10']
                if mode == 'child_failure': raise subprocess.CalledProcessError(7, command)
                actual = Path(command[4]); actual.mkdir()
                session = {'status':'search_benchmark_calls_completed_checks_pending', 'backend':'npu', 'docs':128,
                    'queries':2, 'k':3, 'partitions':2, 'capacity_queries':2, 'capacity_words':6,
                    'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory', 'timer_policy':checks.TIMER,
                    'index_load_batches':1, 'index_bytes':index['index_bytes'], 'cold_init_ms':1., 'cold_synthetic_empty_queries':2,
                    'checkpoint_search_calls':4, 'warmup_cycles':5, 'warmup_search_calls':10, 'warmup_topk_bit_checks':10,
                    'measured_cycles':10, 'measured_search_calls':20, 'measured_topk_bit_checks':20,
                    'measured_debug_download_calls':0, 'measured_poison_calls':0, 'measured_file_io_calls':0,
                    'diagnostic_checkpoints_outside_measurement':True, 'index_readback_byte_identical':True,
                    'host_source_sha256':checks.sources(checks.NPU_SOURCES), 'device':2, 'index_tensor_h2d_calls':5,
                    'pipeline_stages':3, 'debug_index_d2h_calls':5, 'intermediate_host_roundtrips':0,
                    'workspace_capacity_bytes':[16777216]*3, 'samples':[]}
                for cycle in range(10):
                    for slot in range(2):
                        session['samples'].append({'cycle':cycle, 'case':'B' if slot^(cycle&1) else 'A',
                            'call_wall_ms':.02, 'internal_search_ms':.019, 'query_h2d_bytes':48, 'topk_d2h_bytes':48})
                for name in checks.CHECKPOINTS:
                    mock_outputs(actual/name, cache[name[-1]]['scores'], 128, 2, 3, 2)
                if mode == 'bad_ids':
                    for label in ('before_A', 'after_A'): (actual/label/'top_ids.bin').write_bytes(bytes(2*3*4))
                r.write_json(actual/'benchmark_session.json', session)
            with patch.object(app.scale.chain, 'validate_install', return_value=installation), patch.object(app.scale.chain.score, 'execute', child):
                invoke = lambda:app.run(inputs, baseline, output, executable, vendors, base, base, base, base, 2, 2, 5, 10)
                if mode == 'pass': invoke()
                else: fails(invoke)
            assert len(commands) == 1; count += 1
            result = json.loads((output/'report.json').read_text())
            assert result['status'] == (app.STATUS if mode == 'pass' else 'failed'); count += 1
        run = base/'run_pass'; report, session = app.validate_record(run); count += 1
        csv_root = base/'SYNTHETIC_CSV'; summary, tasks = make_csv(csv_root, report, session)
        analysis = tool.analyze(csv_root, run)
        assert analysis['own_operator_rows'] == 105 and analysis['measured_search_calls'] == 20
        for case in ('A', 'B'):
            data = analysis['cases'][case]
            assert data['operators']['Bm25QueryScore']['count'] == 10
            assert data['kernel_sum_us']['median_us'] == 6 and abs(data['device_chain_span_us']['median_us']-6.2) < 1e-9
            assert abs(data['outside_device_chain_elapsed_us']['median_us']-13.8) < 1e-9
        assert analysis['whole_application_api_totals_not_steady_state'][0]['whole_application_total_us'] == 1000000; count += 8
        original_summary = copy.deepcopy(summary)
        for field, value in (('OP Type', 'Bm25Accumulate'), ('Device_id', '4'), ('Stream ID', '99'),
                             ('Task Duration(us)', '0'), ('Block Num', '0'), ('Input Shapes', '1;1;1')):
            bad = copy.deepcopy(original_summary); bad[0][field] = value; write_csv(csv_root/'op_summary_TEST_ONLY.csv', bad)
            fails(lambda:tool.analyze(csv_root, run))
        write_csv(csv_root/'op_summary_TEST_ONLY.csv', original_summary[:-1]); fails(lambda:tool.analyze(csv_root, run))
        write_csv(csv_root/'op_summary_TEST_ONLY.csv', original_summary)
        bad = copy.deepcopy(tasks); bad[0]['task_time(us)'] = '9'; write_csv(csv_root/'task_time_TEST_ONLY.csv', bad)
        fails(lambda:tool.analyze(csv_root, run))
        write_csv(csv_root/'task_time_TEST_ONLY.csv', tasks)
        extra = csv_root/'op_summary_duplicate.csv'; write_csv(extra, original_summary)
        fails(lambda:tool.analyze(csv_root, run)); extra.unlink()
        shortened = copy.deepcopy(session)
        for sample in shortened['samples']:
            sample['call_wall_ms'] = .001; sample['internal_search_ms'] = .0009
        with patch.object(tool.application, 'validate_record', return_value=(report, shortened)):
            fails(lambda:tool.analyze(csv_root, run))
        wrong = run/'measurement/actual/after_A/scores.bin'; original = wrong.read_bytes(); wrong.write_bytes(bytes(len(original)))
        fails(lambda:app.validate_record(run)); wrong.write_bytes(original)
        with patch.object(app.scale.chain.score, 'execute') as child:
            fails(lambda:app.run(inputs, baseline, run, executable, vendors, base, base, base, base))
            assert not child.called; count += 1
    print(f'PASS: {count} profile-provenance/real-host-oracle/MOCK-child/synthetic-CSV/window/shape/clock/fail-stop checks. '
          'Temporary fake artifacts removed; no C++ build, actual msprof/device execution or performance conclusion.')


if __name__ == '__main__': main()
