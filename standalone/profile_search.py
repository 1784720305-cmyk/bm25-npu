"""Run one accepted NPU Search binary for msprof diagnosis; never a new performance baseline."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import benchmark_search as bench, search_benchmark_checks as checks
from standalone import search_scale_fixture as fixture, resident_index as r, run_search_scale as scale, run_search as npu

STATUS = 'profile_search_execution_and_independent_checkpoints_passed'
TOOL_SOURCES = tuple(sorted(set(bench.TOOL_SOURCES) | {
    'standalone/profile_search.py', 'standalone/analyze_search_profile.py', 'standalone/analyze_pipeline_profile.py'}))
digest = scale.digest


def validate_baseline(path, index, snapshot, runner):
    record = json.loads((path/'report.json').read_text(encoding='utf-8'))
    r.require(record.get('status') == bench.STATUS and record.get('index_id') == index['index_id'] and
              record.get('input_sha256') == snapshot and record.get('source_is_real_corpus') is False,
              '需要同一正常输入已经完成的完整Search正式复测报告')
    r.require(record.get('tool_source_sha256') == checks.sources(bench.TOOL_SOURCES) and
              record.get('host_source_sha256') == checks.sources(tuple(sorted(set(checks.CPU_SOURCES) | set(checks.NPU_SOURCES)))) and
              record.get('npu_runner_sha256') == digest(runner), '正式复测源码或NPU程序已改变')
    rounds = record.get('per_round', [])
    r.require(r.integer(record.get('paired_rounds'), 2, 20) and len(rounds) == record['paired_rounds'] and
              all(row.get('status') == 'both_actual_measurements_and_all_cross_checks_passed' for row in rounds),
              '正式六轮/配对检查尚未全部完成')
    for row in rounds:
        for backend in ('cpu', 'npu'):
            run = row[backend]; path_run = Path(run['directory']); path_report = path_run/'report.json'
            r.require(digest(path_report) == run['report_sha256'], '正式单轮报告指纹改变')
            actual = json.loads(path_report.read_text(encoding='utf-8'))
            r.require(actual.get('status') == 'actual_search_measurement_and_checkpoint_checks_passed' and
                      actual.get('index_id') == index['index_id'], '正式单轮尚未通过或属于其他索引')
    return {'directory':str(path), 'report_sha256':digest(path/'report.json'),
            'scope':'accepted unprofiled reference only; its times are never subtracted from profiled kernel times'}


def run(inputs, baseline, output, runner, vendors, score_project, top_project, merge_project,
        acceptance, device=2, shards=32, warmup=5, repeats=10):
    inputs, baseline, output, runner, vendors = (path.resolve() for path in (inputs, baseline, output, runner, vendors))
    r.require(not output.exists(), 'Search采集运行目录已存在，请换新名字，不覆盖')
    r.require(runner.is_file() and r.integer(device, 0, 2147483647) and
              r.integer(warmup, 1, 10000) and r.integer(repeats, 10, 100000), '程序/设备/采集预热轮数非法')
    print('核对已完成的正常规模正式复测和原始输入；本次只采集诊断。', flush=True)
    index, records, snapshot = fixture.validate_inputs(inputs)
    prior = validate_baseline(baseline, index, snapshot, runner)
    k = records['A']['k']; scale.chain.dimensions(index['num_docs'], k, shards)
    cache = checks.cache_inputs(inputs, index, records)
    installation = scale.chain.validate_install(vendors, score_project, top_project, merge_project, acceptance)
    capq = max(row['num_queries'] for row in records.values()); capwords = max(row['query_term_count'] for row in records.values())
    output.mkdir(parents=True)
    report = {'status':'running', 'index':index, 'records':records, 'input_sha256':snapshot, 'baseline':prior,
        'device':device, 'shards':shards, 'warmup':warmup, 'repeats':repeats,
        'capacity_queries':capq, 'capacity_words':capwords, 'installation':installation,
        'host_source_sha256':checks.sources(checks.NPU_SOURCES), 'tool_source_sha256':checks.sources(TOOL_SOURCES),
        'scope':'Intended msprof diagnostic; verify profiler CSV separately. Profiled host times are not a new unprofiled benchmark or speedup claim.'}
    r.write_json(output/'report.json', report)
    try:
        raw = output/'raw_queries'; raw.mkdir(); raw_hashes = {}
        for name in ('A', 'B'):
            path = raw/(name+'.txt'); npu.write_queries(path, cache[name]['queries']); raw_hashes[str(path)] = digest(path)
        report['raw_query_sha256'] = raw_hashes; r.write_json(output/'report.json', report)
        # No replacement implementation: call the same actual compiled benchmark binary once.
        result = bench.run_one(output/'measurement', 'npu', inputs, index, records, cache, raw, runner, vendors,
            installation, 1, 1, shards, device, capq, capwords, warmup, repeats)
        scale.unchanged(snapshot, '采集期间原始输入/参考改变')
        scale.unchanged(raw_hashes, '采集期间原始查询文件改变')
        r.require(digest(baseline/'report.json') == prior['report_sha256'] and
                  report['host_source_sha256'] == checks.sources(checks.NPU_SOURCES) and
                  report['tool_source_sha256'] == checks.sources(TOOL_SOURCES) and
                  installation == scale.chain.validate_install(vendors, score_project, top_project, merge_project, acceptance),
                  '采集期间已有报告、源码或安装产物改变')
        report.update(status=STATUS, run=result)
        r.write_json(output/'report.json', report)
        calls = 1+4+2*warmup+2*repeats
        print(f'采集应用实际Search与前后完整结果核对通过。程序应提交{calls}次三段链，即{calls*3}条自研算子任务；须再与CSV交叉核对。', flush=True)
        print('此处调用时间开启采集，仅作诊断；不能替换此前普通正式基准。', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        r.write_json(output/'report.json', report)
        raise


def validate_record(output):
    output = output.resolve()
    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
    r.require(report.get('status') == STATUS and report.get('host_source_sha256') == checks.sources(checks.NPU_SOURCES) and
              report.get('tool_source_sha256') == checks.sources(TOOL_SOURCES), '需要当前源码已执行并核对通过的Search采集应用报告')
    scale.unchanged(report['input_sha256'], '采集应用原始输入/参考已改变')
    scale.unchanged(report['raw_query_sha256'], '采集应用原始查询已改变')
    measurement = output/'measurement'; linked = report['run']
    r.require(linked.get('directory') == str(measurement) and digest(measurement/'report.json') == linked.get('report_sha256'),
              '采集应用单轮报告来源/指纹不符')
    record = json.loads((measurement/'report.json').read_text(encoding='utf-8'))
    command = record.get('command', [])
    q, k = report['records']['A']['num_queries'], report['records']['A']['k']
    ws = report['installation']['topk_acceptance']['workspace_bytes']
    r.require(record.get('status') == 'actual_search_measurement_and_checkpoint_checks_passed' and
              record.get('backend') == 'npu' and record.get('index_id') == report['index']['index_id'] and len(command) == 14 and
              command[2:5] == [str(output/'raw_queries/A.txt'), str(output/'raw_queries/B.txt'), str(measurement/'actual')] and
              command[5] == str(report['device']) and command[7:] == [str(k), str(report['shards']), str(report['capacity_queries']),
                  str(report['capacity_words']), str(ws), str(report['warmup']), str(report['repeats'])] and
              digest(Path(command[0])) == record.get('runner_sha256') and digest(measurement/'run.log') == record.get('log_sha256'),
              '实际采集应用命令/程序/日志不符')
    expected_index = str(next(Path(path).parent for path in report['input_sha256'] if Path(path).name == 'metadata.json' and Path(path).parent.name == 'index'))
    r.require(command[1] == expected_index, '采集程序没有使用记录中的索引目录')
    files = scale.chain.FILES
    hashes = {name+'/'+file:digest(measurement/'actual'/name/file) for name in checks.CHECKPOINTS for file in files}
    r.require(hashes == record.get('output_sha256'), '采集前后完整分数/候选/Top-K文件改变')
    session, unused = checks.check_session(measurement/'actual', 'npu', report['index'], report['records'],
        report['shards'], 1, report['device'], report['capacity_queries'], report['capacity_words'], report['warmup'], report['repeats'], ws)
    r.require(session == record.get('session'), '采集单轮报告与实际会话记录不一致')
    for name in ('A', 'B'):
        r.require(all((measurement/'actual'/('before_'+name)/file).read_bytes() ==
                      (measurement/'actual'/('after_'+name)/file).read_bytes() for file in files), '采集前后重复结果字节不同')
    return report, session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runner', type=Path, default=ROOT/'results/search_benchmark_host_build/bm25_npu_search_benchmark')
    p.add_argument('--vendors', type=Path, default=ROOT/'results/custom_opp/vendors')
    p.add_argument('--score-project', type=Path, default=ROOT/'results/Bm25QueryScore')
    p.add_argument('--topk-project', type=Path, default=ROOT/'results/Bm25TopKRadixCollect')
    p.add_argument('--merge-project', type=Path, default=ROOT/'results/Bm25TopKMerge')
    p.add_argument('--topk-acceptance', type=Path, default=ROOT/'results/radix_collect_topk_validation1')
    p.add_argument('--device', type=int, default=2); p.add_argument('--shards', type=int, default=32)
    p.add_argument('--warmup', type=int, default=5); p.add_argument('--repeats', type=int, default=10)
    args = p.parse_args()
    try:
        run(args.input, args.benchmark, args.output, args.runner, args.vendors, args.score_project,
            args.topk_project, args.merge_project, args.topk_acceptance, args.device, args.shards, args.warmup, args.repeats)
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不产生CPU替代或成功采集结论。', file=sys.stderr); return 1
    return 0


if __name__ == '__main__': sys.exit(main())
