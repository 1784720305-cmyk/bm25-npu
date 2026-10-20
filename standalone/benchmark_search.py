"""Complete raw-ID Search: CPU configuration scan, then alternating independent CPU/NPU rounds."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import search_benchmark_checks as checks, search_scale_fixture as fixture
from standalone import resident_index as r, run_search_scale as scale, run_search as npu

TOOL_SOURCES = tuple(sorted(set(scale.TOOL_SOURCES) | {
    'standalone/search_benchmark_checks.py', 'standalone/benchmark_search.py'}))
STATUS = 'complete_search_cpu_scan_and_paired_rounds_passed'
digest = scale.digest


def config(text):
    parts = text.split(':')
    r.require(len(parts) == 2 and all(part.isdecimal() for part in parts), 'CPU配置格式是线程数:文章分区数，例如8:8')
    threads, shards = map(int, parts)
    r.require(r.integer(threads, 1, 65536) and r.integer(shards, 1, threads), 'CPU配置要求1<=文章分区<=线程数<=65536')
    return threads, shards


def host_resources():
    record = {'cpu_count_logical':os.cpu_count(), 'environment':{name:os.environ.get(name) for name in
        ('OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES', 'OMP_THREAD_LIMIT', 'GOMP_CPU_AFFINITY',
         'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')}}
    if hasattr(os, 'sched_getaffinity'):
        record['parent_affinity_inherited_by_children'] = sorted(os.sched_getaffinity(0))
    for name in ('/proc/self/status', '/proc/self/cgroup', '/sys/fs/cgroup/cpu.max',
                 '/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us'):
        path = Path(name)
        if path.is_file():
            try:
                text = path.read_text(encoding='utf-8')
                if name == '/proc/self/status':
                    text = '\n'.join(line for line in text.splitlines() if line.startswith(('Cpus_allowed', 'Mems_allowed')))
                record[name] = text.strip()
            except OSError as error:
                record[name] = {'unavailable':type(error).__name__}
    record['scope'] = 'Parent environment/affinity and visible quota files only; does not prove exclusive CPUs, no NUMA interference or idle NPU.'
    return record


def run_one(directory, backend, inputs, index, records, cache, raw, runner, vendors, installation,
            threads, cpu_shards, shards, device, capq, capwords, warmup, repeats):
    r.require(not directory.exists(), '单轮目录已存在，不覆盖')
    directory.mkdir(parents=True)
    k = records['A']['k']; workspace = installation['topk_acceptance']['workspace_bytes']
    command = [str(runner), str(inputs/'index'), str(raw/'A.txt'), str(raw/'B.txt'), str(directory/'actual')]
    if backend == 'cpu':
        command += [str(threads), str(k), str(cpu_shards), str(capq), str(capwords), str(warmup), str(repeats)]
    else:
        command += [str(device), str(vendors), str(k), str(shards), str(capq), str(capwords), str(workspace), str(warmup), str(repeats)]
    env = os.environ.copy()
    if backend == 'npu':
        env['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(str((vendors/name).resolve(strict=True)) for name in
            ('bm25_query_score', 'bm25_topk_radix_collect', 'bm25_topk_merge'))
    report = {'status':'running', 'backend':backend, 'command':command, 'runner_sha256':digest(runner),
              'index_id':index['index_id'], 'resources':host_resources(),
              'scope':'Complete public Search caller wall time; no kernel-only timing or debug I/O inside measured calls.'}
    r.write_json(directory/'report.json', report)
    try:
        print('实际计时命令：'+json.dumps(command, ensure_ascii=False), flush=True)
        scale.chain.score.execute(command, directory/'run.log', env)
        session, summary = checks.check_session(directory/'actual', backend, index, records,
            cpu_shards if backend == 'cpu' else shards, threads, device, capq, capwords, warmup, repeats, workspace)
        print('核对计时前后完整分数、候选、严格排名和重复字节：'+backend, flush=True)
        validation, hashes = checks.check_checkpoints(directory/'actual', backend, cache, cpu_shards if backend == 'cpu' else shards)
        r.require(digest(runner) == report['runner_sha256'], '单轮运行中计时程序改变')
        report.update(status='actual_search_measurement_and_checkpoint_checks_passed', session=session, summary=summary,
                      checks=validation, output_sha256=hashes, log_sha256=digest(directory/'run.log'))
        r.write_json(directory/'report.json', report)
        print(f'{backend}原查询A：整批中位数={summary["A"]["median_batch_ms"]:.6g}ms，p95={summary["A"]["p95_batch_ms"]:.6g}ms；B另列。', flush=True)
        return {'directory':str(directory), 'report_sha256':digest(directory/'report.json'), 'summary':summary,
                'actual_threads':session.get('actual_threads')}
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        r.write_json(directory/'report.json', report)
        raise


def run(inputs, acceptance, output, cpu_runner, npu_runner, vendors, score_project, top_project, merge_project,
        top_acceptance, configs, scan_rounds=3, rounds=6, warmup=5, repeats=30, device=2, shards=32):
    inputs, acceptance, output, cpu_runner, npu_runner, vendors = (p.resolve() for p in
        (inputs, acceptance, output, cpu_runner, npu_runner, vendors))
    r.require(not output.exists(), '正式Search结果目录已存在，请换新名字，不覆盖')
    r.require(cpu_runner.is_file() and npu_runner.is_file() and r.integer(device, 0, 2147483647) and
              r.integer(scan_rounds, 2, 20) and r.integer(rounds, 2, 20) and rounds%2 == 0 and
              r.integer(warmup, 1, 10000) and r.integer(repeats, 10, 100000), '调用器缺失或扫描/轮数/预热/测量参数非法；配对轮数必须为偶数')
    r.require(isinstance(configs, list) and len(configs) >= 2 and len(set(configs)) == len(configs) and
              all(isinstance(item, tuple) and len(item) == 2 and r.integer(item[0], 1, 65536) and
                  r.integer(item[1], 1, item[0]) for item in configs), '需要至少两个不重复、合法的CPU线程/分区候选')
    print('核对已验收正常语料和历史完整Search报告；不将历史诊断时间加入正式样本。', flush=True)
    index, records, snapshot = fixture.validate_inputs(inputs)
    k, q = records['A']['k'], records['A']['num_queries']
    scale.chain.dimensions(index['num_docs'], k, shards)
    r.require(all(item[1] <= index['num_docs'] for item in configs), 'CPU文章分区超过文章数')
    cache = checks.cache_inputs(inputs, index, records)
    prior = checks.validate_acceptance(acceptance, inputs, index, records, snapshot, cache)
    installation = scale.chain.validate_install(vendors, score_project, top_project, merge_project, top_acceptance)
    capq = max(x['num_queries'] for x in records.values()); capwords = max(x['query_term_count'] for x in records.values())
    output.mkdir(parents=True)
    report = {'status':'running', 'index_id':index['index_id'], 'queries':q, 'docs':index['num_docs'], 'k':k,
              'primary_case':'A_original_queries', 'secondary_case':'B_same_lengths_rotated_word_IDs',
              'metric':'single_query_latency' if q == 1 else 'batch_latency_and_search_time_throughput',
              'acceptance':prior, 'device':device, 'shards':shards, 'capacity_queries':capq, 'capacity_words':capwords,
              'cpu_candidates':[{'threads':t, 'document_shards':s} for t, s in configs],
              'scan_rounds':scan_rounds, 'paired_rounds':rounds, 'warmup_cycles':warmup, 'measured_cycles':repeats,
              'input_sha256':snapshot, 'cpu_runner_sha256':digest(cpu_runner), 'npu_runner_sha256':digest(npu_runner),
              'installation':installation, 'tool_source_sha256':checks.sources(TOOL_SOURCES),
              'host_source_sha256':checks.sources(tuple(sorted(set(checks.CPU_SOURCES) | set(checks.NPU_SOURCES)))),
              'scan':[], 'per_round':[], 'source_is_real_corpus':False, 'resources':host_resources(),
              'scope':'Same logical raw-ID to CPU Top-K boundary; caller host wall time includes every required NPU query transfer; cold index/init separate. Fastest tested CPU config on A, not whole-machine optimality. No BM25S comparison in this report.'}
    r.write_json(output/'report.json', report)
    try:
        raw = output/'raw_queries'; raw.mkdir(); raw_hashes = {}
        for name in ('A', 'B'):
            path = raw/(name+'.txt'); npu.write_queries(path, cache[name]['queries']); raw_hashes[str(path)] = digest(path)
        report['raw_query_sha256'] = raw_hashes
        for scan in range(scan_rounds):
            order = configs[scan%len(configs):]+configs[:scan%len(configs)]
            for threads, parts in order:
                print(f'CPU配置扫描{scan+1}/{scan_rounds}：{threads}线程/{parts}文章分区。', flush=True)
                result = run_one(output/'cpu_scan'/f'round{scan+1}_t{threads}_s{parts}', 'cpu', inputs, index, records, cache,
                    raw, cpu_runner, vendors, installation, threads, parts, shards, device, capq, capwords, warmup, repeats)
                report['scan'].append({'round':scan+1, 'threads':threads, 'document_shards':parts, 'run':result})
                r.write_json(output/'report.json', report)
        ranking = []
        for threads, parts in configs:
            runs = [row['run'] for row in report['scan'] if row['threads'] == threads and row['document_shards'] == parts]
            ranking.append({'threads':threads, 'document_shards':parts,
                'A_run_medians_ms':[row['summary']['A']['median_batch_ms'] for row in runs],
                'A_median_of_run_medians_ms':statistics.median(row['summary']['A']['median_batch_ms'] for row in runs),
                'B_median_of_run_medians_ms':statistics.median(row['summary']['B']['median_batch_ms'] for row in runs)})
        ranking.sort(key=lambda row:(row['A_median_of_run_medians_ms'], row['threads'], row['document_shards']))
        best = ranking[0]; threads, parts = best['threads'], best['document_shards']
        report.update(cpu_scan_ranking=ranking, selected_cpu_config=best,
                      selection_policy='Minimum median of scan-run A medians; A is unchanged original workload. B reported independently, not claimed optimal.')
        r.write_json(output/'report.json', report)
        print(f'选择已扫描A最佳配置：{threads}线程/{parts}文章分区；开始独立进程交替配对。', flush=True)
        for round_id in range(rounds):
            order = ['cpu', 'npu'] if round_id%2 == 0 else ['npu', 'cpu']
            row = {'round':round_id+1, 'process_order':order, 'status':'running'}
            report['per_round'].append(row); r.write_json(output/'report.json', report)
            print(f'配对第{round_id+1}/{rounds}轮，进程顺序={order}。', flush=True)
            for backend in order:
                row[backend] = run_one(output/f'paired_round{round_id+1}'/backend, backend, inputs, index, records, cache, raw,
                    cpu_runner if backend == 'cpu' else npu_runner, vendors, installation, threads, parts,
                    shards, device, capq, capwords, warmup, repeats)
                r.write_json(output/'report.json', report)
            cross = {}
            for checkpoint in checks.CHECKPOINTS:
                cross[checkpoint] = scale.cpu.compare_npu(Path(row['cpu']['directory'])/'actual'/checkpoint,
                    Path(row['npu']['directory'])/'actual'/checkpoint, q, index['num_docs'], k)
            row.update(status='both_actual_measurements_and_all_cross_checks_passed', cross_checks=cross)
            r.write_json(output/'report.json', report)
        scale.unchanged(snapshot, '正式实验期间语料/索引/查询/参考改变')
        scale.unchanged(raw_hashes, '正式实验期间原始词编号改变')
        r.require(digest(acceptance/'report.json') == prior['report_sha256'] and
                  digest(cpu_runner) == report['cpu_runner_sha256'] and digest(npu_runner) == report['npu_runner_sha256'] and
                  all(digest(ROOT/name) == value for key in ('tool_source_sha256', 'host_source_sha256') for name, value in report[key].items()) and
                  scale.chain.validate_install(vendors, score_project, top_project, merge_project, top_acceptance) == installation,
                  '正式实验期间已有验收/可执行程序/源码/安装产物改变')
        paired = {}
        for name in ('A', 'B'):
            cm = [row['cpu']['summary'][name]['median_batch_ms'] for row in report['per_round']]
            nm = [row['npu']['summary'][name]['median_batch_ms'] for row in report['per_round']]
            ratios = [c/n for c, n in zip(cm, nm)]
            paired[name] = {'cpu_run_medians_ms':cm, 'npu_run_medians_ms':nm,
                'cpu_median_of_run_medians_ms':statistics.median(cm), 'npu_median_of_run_medians_ms':statistics.median(nm),
                'paired_cpu_over_npu_median_ratios':ratios, 'npu_faster_rounds':sum(x > 1 for x in ratios), 'rounds':rounds,
                'scope':'Paired run-median ratios, not a statistical significance test or BM25S/hardware-only comparison.'}
        report.update(status=STATUS, paired_summary=paired)
        r.write_json(output/'report.json', report)
        for name in ('A', 'B'):
            item = paired[name]
            print(f'{name}：CPU各轮中位数(ms)={item["cpu_run_medians_ms"]}；NPU={item["npu_run_medians_ms"]}；NPU较快{item["npu_faster_rounds"]}/{rounds}轮。', flush=True)
        print('完整Search配置扫描/独立进程配对/全部结果检查通过。报告保存每次计时及p95；不把批量时间除以Q当单条延迟，不作BM25S或整机最优结论。', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        r.write_json(output/'report.json', report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--acceptance', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-runner', type=Path, default=ROOT/'results/search_benchmark_host_build/bm25_cpu_search_benchmark')
    p.add_argument('--npu-runner', type=Path, default=ROOT/'results/search_benchmark_host_build/bm25_npu_search_benchmark')
    p.add_argument('--vendors', type=Path, default=ROOT/'results/custom_opp/vendors')
    p.add_argument('--score-project', type=Path, default=ROOT/'results/Bm25QueryScore')
    p.add_argument('--topk-project', type=Path, default=ROOT/'results/Bm25TopKRadixCollect')
    p.add_argument('--merge-project', type=Path, default=ROOT/'results/Bm25TopKMerge')
    p.add_argument('--topk-acceptance', type=Path, default=ROOT/'results/radix_collect_topk_validation1')
    p.add_argument('--cpu-config', nargs='+', default=['1:1', '8:8', '16:16', '32:32', '64:64'])
    p.add_argument('--scan-rounds', type=int, default=3); p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--warmup', type=int, default=5); p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--device', type=int, default=2); p.add_argument('--shards', type=int, default=32)
    args = p.parse_args()
    try:
        run(args.input, args.acceptance, args.output, args.cpu_runner, args.npu_runner, args.vendors,
            args.score_project, args.topk_project, args.merge_project, args.topk_acceptance,
            [config(value) for value in args.cpu_config], args.scan_rounds, args.rounds, args.warmup,
            args.repeats, args.device, args.shards)
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不用CPU代替NPU，不生成成功比较。', file=sys.stderr); return 1
    return 0


if __name__ == '__main__': sys.exit(main())
