"""固定range评分＋正分压缩Top-K，与指定优化CPU配置同期比较完整Search；不改算子。"""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import compare_search_topk_versions as selected
from standalone import benchmark_search as bench, search_benchmark_checks as checks
from standalone import search_scale_fixture as fixture, run_search_scale as scale, run_search as search
from standalone import resident_index as r

STATUS = 'compact_search_cpu_npu_paired_and_all_checks_passed'
TOOL_SOURCES = tuple(sorted(set(selected.TOOL_SOURCES) | {'standalone/benchmark_compact_search_cpu.py'}))
digest = scale.digest


def validate_selected(path, inputs, index, records, snapshot, cache, runner, vendors, projects):
    """Recheck prior successful full-Search proof. Historical times are not current samples."""
    path = path.resolve(strict=True)
    report = json.loads((path/'report.json').read_text(encoding='utf-8'))
    q, k = records['A']['num_queries'], records['A']['k']
    hosts = tuple(sorted(set(checks.NPU_SOURCES) | set(search.HOST_SOURCES) | set(selected.compact.HOST_SOURCES)))
    r.require(report.get('status') == selected.STATUS and report.get('index_id') == index['index_id'] and
              report.get('input_sha256') == snapshot and report.get('queries') == q and report.get('k') == k and
              report.get('docs') == index['num_docs'] and report.get('source_is_real_corpus') is False and
              report.get('tool_source_sha256') == checks.sources(selected.TOOL_SOURCES) and
              report.get('host_source_sha256') == checks.sources(hosts) and
              report.get('runner_sha256') == digest(runner), '需要同输入/当前源码和程序已通过的Top-K完整Search对照报告')
    installation = report['candidate_installation']
    score_install = selected.score.validate_install(vendors/'bm25_query_score_range', projects['score'])
    top_install = selected.compact.validate_install(vendors/selected.compact.VENDOR, projects['topk'])
    merge_vendor = vendors/'bm25_topk_merge'
    merge_install = scale.chain.merge.validate_install(merge_vendor, projects['merge'])
    objects = scale.chain.artifacts(merge_vendor, projects['merge'], 'bm25_top_k_merge', 'aclnn_bm25_top_k_merge.h')
    r.require(installation.get('score') == score_install and installation.get('topk') == top_install and
              installation.get('merge') == merge_install and installation.get('merge_device_objects') == objects and
              installation.get('topk_device_objects') == top_install['installed_device_object_sha256'], '已通过组合与当前安装产物不同')
    evidence = {str(path/'report.json'): digest(path/'report.json')}
    for key in ('boundary_acceptance', 'compact_acceptance'):
        prior = report[key]
        r.require(prior.get('evidence_sha256'), '缺少评分或37组Top-K设备验收文件指纹')
        scale.unchanged(prior['evidence_sha256'], '已有独立设备验收文件改变')
        evidence.update(prior['evidence_sha256'])
    targets = {'bm25_query_score': (vendors/'bm25_query_score_range').resolve(strict=True),
               'bm25_topk_radix_collect': (vendors/selected.compact.VENDOR).resolve(strict=True),
               'bm25_topk_merge': merge_vendor.resolve(strict=True)}
    runtime = path/'runtime_candidate/vendors'
    r.require(selected.previous.validate_runtime(runtime, targets) == report['runtime_mapping']['candidate'], '已有运行目录映射改变')
    capq, capwords = max(v['num_queries'] for v in records.values()), max(v['query_term_count'] for v in records.values())
    device, shards = report.get('device'), report.get('shards')
    r.require(r.integer(device, 0, 2147483647) and r.integer(shards, 1, 32) and
              report.get('capacity_queries') == capq and report.get('capacity_words') == capwords and 1 <= capwords <= 64,
              '已有Search设备/容量记录不符')
    scale.chain.dimensions(index['num_docs'], k, shards)
    scale.unchanged(report['raw_query_sha256'], '已有原始查询文件改变')
    evidence.update(report['raw_query_sha256'])
    workspace = installation['topk_acceptance']['workspace_bytes']
    # Existing five-call full Search, checked from files rather than re-submitted to the device.
    pre = report['preflight']['candidate']; directory = path/'preflight_candidate'
    command = pre.get('command', [])
    r.require(pre.get('status') == 'five_full_search_calls_and_independent_rankings_passed' and
              json.loads((directory/'report.json').read_text(encoding='utf-8')) == pre and
              len(command) == 11 and command[1:] == [str(inputs/'index'), str(directory/'requests.tsv'),
              str(directory/'actual'), str(device), str(runtime), str(k), str(shards), str(capq), str(capwords), str(workspace)] and
              digest(Path(command[0])) == pre.get('runner_sha256') == report.get('diagnostic_runner_sha256') and
              digest(directory/'run.log') == pre.get('log_sha256'), '已有完整Search预验收程序/命令/日志改变')
    expected_plan = ('\n'.join(f'call{i}_{name}\t{path/"raw_queries"/(name+".txt")}'
        for i, name in enumerate(selected.score.SEQUENCE))+'\n').encode('utf-8')
    r.require((directory/'requests.tsv').read_bytes() == expected_plan, '已有完整Search查询计划改变')
    session = search.check_session(directory/'actual', index, records, device, k, shards, capq, capwords, workspace)
    r.require(session == pre.get('session'), '已有完整Search预验收会话记录不同')
    hashes = {}
    with redirect_stdout(io.StringIO()):
        for ordinal, name in enumerate(selected.score.SEQUENCE):
            label = f'call{ordinal}_{name}'; actual = directory/'actual'/label
            checks.check_output(actual, cache[name], 'npu', shards)
            hashes.update({label+'/'+file: digest(actual/file) for file in scale.chain.FILES})
    r.require(hashes == pre.get('output_sha256'), '已有完整Search预验收输出改变')
    for name in hashes: evidence[str(directory/'actual'/name)] = hashes[name]
    for file in (directory/'report.json', directory/'requests.tsv', directory/'run.log', directory/'actual/search_session.json', Path(command[0])):
        evidence[str(file)] = digest(file)
    rows = report.get('per_round', [])
    r.require(r.integer(report.get('rounds'), 2, 20) and report['rounds'] % 2 == 0 and len(rows) == report['rounds'], '旧对照配对轮数不完整')
    for ordinal, row in enumerate(rows):
        r.require(row.get('round') == ordinal+1 and row.get('status') == 'both_topk_search_versions_and_all_five_array_bytes_passed' and
                  row.get('process_order') == (['baseline', 'candidate'] if ordinal % 2 == 0 else ['candidate', 'baseline']), '旧对照轮次未通过')
        linked = row['candidate']; run = path/f'round{ordinal+1}/candidate'
        r.require(linked.get('directory') == str(run) and digest(run/'report.json') == linked.get('report_sha256'), '旧组合单轮报告改变')
        data = json.loads((run/'report.json').read_text(encoding='utf-8'))
        expected_command = [str(runner), str(inputs/'index'), str(path/'raw_queries/A.txt'), str(path/'raw_queries/B.txt'),
                            str(run/'actual'), str(device), str(runtime), str(k), str(shards), str(capq), str(capwords),
                            str(workspace), str(report['warmup_cycles']), str(report['measured_cycles'])]
        r.require(data.get('status') == 'actual_search_measurement_and_checkpoint_checks_passed' and
                  data.get('command') == expected_command and data.get('runner_sha256') == digest(runner) and
                  digest(run/'run.log') == data.get('log_sha256'), '旧组合实际执行命令/日志不符')
        actual_session, summary = checks.check_session(run/'actual', 'npu', index, records, shards, 1, device,
            capq, capwords, report['warmup_cycles'], report['measured_cycles'], workspace)
        r.require(actual_session == data.get('session') and summary == data.get('summary') == linked.get('summary'), '旧组合样本/汇总不符')
        with redirect_stdout(io.StringIO()):
            unused, outputs = checks.check_checkpoints(run/'actual', 'npu', cache, shards)
        r.require(outputs == data.get('output_sha256'), '旧组合全部检查点输出改变')
        for name, value in outputs.items(): evidence[str(run/'actual'/name)] = value
        for file in (run/'report.json', run/'run.log', run/'actual/benchmark_session.json'): evidence[str(file)] = digest(file)
    return {'directory': str(path), 'report_sha256': digest(path/'report.json'), 'evidence_sha256': evidence,
            'installation': installation, 'targets': targets, 'shards': shards,
            'preflight_actual': str(directory/'actual'), 'scope': 'prior correctness only; historical timing samples excluded'}


def cpu_preflight(directory, inputs, index, records, cache, raw, runner, prior, threads, parts, capq, capwords):
    directory.mkdir(parents=True)
    plan = directory/'requests.tsv'
    plan.write_bytes(('\n'.join(f'call{i}_{name}\t{raw/(name+".txt")}' for i, name in enumerate(selected.score.SEQUENCE))+'\n').encode('utf-8'))
    command = [str(runner), str(inputs/'index'), str(plan), str(directory/'actual'), str(threads),
               str(records['A']['k']), str(parts), str(capq), str(capwords), 'raw_query_ids_v1']
    selected.score.execute(command, directory/'run.log', os.environ.copy())
    session = scale.cpu.check_session(directory/'actual', index, records, threads, records['A']['k'], parts, capq, capwords)
    r.require(session['actual_threads'] == threads, 'CPU实际线程数不足，不将降配结果作为指定配置通过')
    cross = {}
    for ordinal, name in enumerate(selected.score.SEQUENCE):
        label = f'call{ordinal}_{name}'; actual = directory/'actual'/label
        checks.check_output(actual, cache[name], 'cpu', parts)
        cross[label] = scale.cpu.compare_npu(actual, Path(prior['preflight_actual'])/label,
            records[name]['num_queries'], index['num_docs'], records[name]['k'])
    result = {'command': command, 'runner_sha256': digest(runner), 'session': session, 'cross_checks': cross,
              'output_sha256': {label+'/'+file: digest(directory/'actual'/label/file)
                for label in cross for file in scale.cpu.FILES}, 'log_sha256': digest(directory/'run.log')}
    r.write_json(directory/'report.json', result)
    return result


def run(inputs, acceptance, output, cpu_runner, npu_runner, cpu_diagnostic, vendors, projects,
        threads=32, parts=32, device=2, rounds=6, warmup=5, repeats=30):
    inputs, acceptance, output, cpu_runner, npu_runner, cpu_diagnostic, vendors = (p.resolve() for p in
        (inputs, acceptance, output, cpu_runner, npu_runner, cpu_diagnostic, vendors))
    r.require(not output.exists(), '结果目录已存在，请换新名字，不覆盖')
    r.require(all(p.is_file() for p in (cpu_runner, npu_runner, cpu_diagnostic)) and r.integer(threads, 1, 65536) and
              r.integer(parts, 1, threads) and r.integer(device, 0, 2147483647) and r.integer(rounds, 2, 20) and rounds % 2 == 0 and
              r.integer(warmup, 1, 10000) and r.integer(repeats, 10, 100000), '程序缺失或配置/轮数参数非法')
    index, records, snapshot = fixture.validate_inputs(inputs)
    r.require(parts <= index['num_docs'], 'CPU文章分区超过文章数')
    cache = checks.cache_inputs(inputs, index, records)
    print('只读复核已通过的新组合完整Search；不重跑37组或采集。', flush=True)
    prior = validate_selected(acceptance, inputs, index, records, snapshot, cache, npu_runner, vendors, projects)
    shards, installation = prior['shards'], prior['installation']
    q, k = records['A']['num_queries'], records['A']['k']
    capq, capwords = max(v['num_queries'] for v in records.values()), max(v['query_term_count'] for v in records.values())
    tools = checks.sources(TOOL_SOURCES)
    hosts = checks.sources(tuple(sorted(set(checks.CPU_SOURCES) | set(checks.NPU_SOURCES) | set(scale.cpu.HOST_SOURCES))))
    output.mkdir(parents=True)
    report = {'status': 'running', 'index_id': index['index_id'], 'docs': index['num_docs'], 'queries': q, 'k': k,
              'threads': threads, 'cpu_document_shards': parts, 'device': device, 'npu_shards': shards,
              'metric': 'single_query_latency' if q == 1 else 'batch_latency_and_search_time_throughput',
              'cpu_selection_policy': 'explicit fixed optimized configuration; prior scan not repeated; not whole-machine optimum',
              'rounds': rounds, 'warmup_cycles': warmup, 'measured_cycles': repeats, 'input_sha256': snapshot,
              'prior_acceptance': {key: value for key, value in prior.items() if key != 'targets'},
              'cpu_runner_sha256': digest(cpu_runner), 'npu_runner_sha256': digest(npu_runner),
              'cpu_diagnostic_sha256': digest(cpu_diagnostic), 'tool_source_sha256': tools, 'host_source_sha256': hosts,
              'resources': bench.host_resources(), 'source_is_real_corpus': False, 'per_round': [],
              'scope': 'Matched complete raw word-ID to CPU Top-K wall time; resident raw-TF index and query-time FP32 formula on both. Actual CPU/NPU implementations differ; not pure hardware or BM25S speedup.'}
    r.write_json(output/'report.json', report)
    try:
        runtime = output/'runtime/vendors'
        mapping = selected.previous.create_runtime(runtime, prior['targets'])
        raw = output/'raw_queries'; raw.mkdir()
        for name in records: search.write_queries(raw/(name+'.txt'), cache[name]['queries'])
        raw_hashes = {str(p): digest(p) for p in raw.iterdir()}
        report.update(runtime_mapping=mapping, raw_query_sha256=raw_hashes)
        report['cpu_preflight'] = cpu_preflight(output/'cpu_preflight', inputs, index, records, cache, raw,
            cpu_diagnostic, prior, threads, parts, capq, capwords)
        r.write_json(output/'report.json', report)
        for ordinal in range(rounds):
            order = ['cpu', 'npu'] if ordinal % 2 == 0 else ['npu', 'cpu']
            row = {'round': ordinal+1, 'process_order': order, 'status': 'running'}
            report['per_round'].append(row); r.write_json(output/'report.json', report)
            print(f'同期CPU/NPU第{ordinal+1}/{rounds}轮：{order}，CPU={threads}线程/{parts}分区。', flush=True)
            for backend in order:
                r.require(selected.previous.validate_runtime(runtime, prior['targets']) == mapping, '运行库映射改变')
                row[backend] = bench.run_one(output/f'round{ordinal+1}'/backend, backend, inputs, index, records, cache, raw,
                    cpu_runner if backend == 'cpu' else npu_runner, runtime, installation,
                    threads, parts, shards, device, capq, capwords, warmup, repeats)
                if backend == 'cpu': r.require(row['cpu']['actual_threads'] == threads, 'CPU实际线程数不足')
                r.write_json(output/'report.json', report)
            row['cross_checks'] = {label: scale.cpu.compare_npu(Path(row['cpu']['directory'])/'actual'/label,
                Path(row['npu']['directory'])/'actual'/label, q, index['num_docs'], k) for label in checks.CHECKPOINTS}
            row['status'] = 'both_actual_search_measurements_and_all_cross_checks_passed'
            r.write_json(output/'report.json', report)
        for proof in (snapshot, raw_hashes, prior['evidence_sha256']): scale.unchanged(proof, '输入或既有验收文件改变')
        r.require(checks.sources(TOOL_SOURCES) == tools and checks.sources(tuple(hosts)) == hosts and
                  digest(cpu_runner) == report['cpu_runner_sha256'] and digest(npu_runner) == report['npu_runner_sha256'] and
                  digest(cpu_diagnostic) == report['cpu_diagnostic_sha256'], '源码或可执行程序改变')
        r.require(installation['score'] == selected.score.validate_install(vendors/'bm25_query_score_range', projects['score']) and
                  installation['topk'] == selected.compact.validate_install(vendors/selected.compact.VENDOR, projects['topk']) and
                  installation['merge'] == scale.chain.merge.validate_install(vendors/'bm25_topk_merge', projects['merge']) and
                  installation['merge_device_objects'] == scale.chain.artifacts(vendors/'bm25_topk_merge', projects['merge'],
                      'bm25_top_k_merge', 'aclnn_bm25_top_k_merge.h') and
                  selected.previous.validate_runtime(runtime, prior['targets']) == mapping, '安装产物或映射改变')
        summary = {}
        for name in ('A', 'B'):
            cm = [row['cpu']['summary'][name]['median_batch_ms'] for row in report['per_round']]
            nm = [row['npu']['summary'][name]['median_batch_ms'] for row in report['per_round']]
            summary[name] = {'cpu_run_medians_ms': cm, 'npu_run_medians_ms': nm,
                'cpu_median_of_run_medians_ms': statistics.median(cm), 'npu_median_of_run_medians_ms': statistics.median(nm),
                'paired_cpu_over_npu_ratios': [c/n for c, n in zip(cm, nm)], 'npu_faster_rounds': sum(c > n for c, n in zip(cm, nm)),
                'rounds': rounds, 'scope': 'current paired run medians only; ratio >1 means NPU faster'}
        report.update(status=STATUS, paired_summary=summary)
        r.write_json(output/'report.json', report)
        for name, item in summary.items():
            print(f'{name}同期完整Search：CPU各轮ms={item["cpu_run_medians_ms"]}；NPU={item["npu_run_medians_ms"]}；NPU更快{item["npu_faster_rounds"]}/{rounds}轮。', flush=True)
        print('全部实际结果及配对检查通过；是否达到加速目标取决于实测。已保存样本/p95；不是BM25S或纯硬件结论。', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        r.write_json(output/'report.json', report); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output'): p.add_argument('--'+name, type=Path, required=True)
    defaults = {'acceptance': 'results/search_topk_compact_256k_q8_pair1',
        'cpu-runner': 'results/search_benchmark_host_build/bm25_cpu_search_benchmark',
        'npu-runner': 'results/search_benchmark_host_build/bm25_npu_search_benchmark',
        'cpu-diagnostic': 'results/cpu_index_search_build/bm25_cpu_index_search_runner',
        'vendors': 'results/custom_opp/vendors', 'score-project': 'results/Bm25QueryScoreRange',
        'topk-project': 'results/Bm25TopKPositiveCompact', 'merge-project': 'results/Bm25TopKMerge'}
    for name, value in defaults.items(): p.add_argument('--'+name, type=Path, default=ROOT/value)
    p.add_argument('--cpu-config', default='32:32', help='线程数:文章分区数；当前Q8沿用此前已扫描配置32:32')
    for name, value in (('device', 2), ('rounds', 6), ('warmup', 5), ('repeats', 30)): p.add_argument('--'+name, type=int, default=value)
    a = p.parse_args()
    try:
        r.require(sys.platform == 'linux', '实际CPU/NPU同期对照请在A5 Linux执行')
        threads, parts = bench.config(a.cpu_config)
        run(a.input, a.acceptance, a.output, a.cpu_runner, a.npu_runner, a.cpu_diagnostic, a.vendors,
            {'score': a.score_project, 'topk': a.topk_project, 'merge': a.merge_project},
            threads, parts, a.device, a.rounds, a.warmup, a.repeats)
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不用CPU替代NPU，不生成成功比较。', file=sys.stderr); return 1
    return 0


if __name__ == '__main__': sys.exit(main())
