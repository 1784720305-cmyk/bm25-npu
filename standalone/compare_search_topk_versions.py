"""完整Search对照：固定已验收range评分/合并，仅替换局部Top-K；不采集、不重写内核。"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import compare_search_score_versions as previous
from standalone import validate_topk_positive_compact as compact
from standalone import resident_index as r, benchmark_search as bench, search_benchmark_checks as checks
from standalone import search_scale_fixture as fixture, run_search_scale as scale, run_search as search
from standalone import run_query_score_range as score

STATUS = 'complete_search_topk_versions_paired_and_all_checks_passed'
TOOL_SOURCES = tuple(sorted(set(previous.TOOL_SOURCES) | set(compact.TOOLS) | {
    'standalone/compare_search_topk_versions.py'}))
SLOTS = previous.SLOTS
digest = score.digest


def validate_compact_acceptance(directory, installation):
    """Read/check the existing 37-case proof; never submit device work or reuse its timings."""
    directory = directory.resolve(strict=True)
    report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    specs = [('regression', dims) for dims in compact.old.SPECS]+[('capacity', dims) for dims in compact.BOUNDARIES]
    r.require(report.get('status') == compact.STATUS and report.get('installation') == installation and
              report.get('tool_source_sha256') == checks.sources(compact.TOOLS+compact.HOST_SOURCES) and
              len(report.get('cases', [])) == len(specs), '需要当前安装/源码对应的37组压缩Top-K设备验收')
    device, calls = report.get('device'), report.get('calls_per_case')
    r.require(r.integer(device, 0, 2147483647) and r.integer(calls, 5, 100), '既有验收设备/调用次数非法')
    evidence = {str(directory/'report.json'): digest(directory/'report.json')}
    library_log = directory/'library.log'
    r.require(library_log.is_file() and library_log.stat().st_size > 0, '缺少实际API加载日志')
    evidence[str(library_log)] = digest(library_log)
    workspaces = set()
    runner = None
    for ordinal, (item, (kind, (q, docs, k))) in enumerate(zip(report['cases'], specs)):
        case = directory/f'case{ordinal}_{kind}_q{q}_d{docs}_k{k}'
        actual = case/'actual'
        r.require(item.get('status') == 'passed' and item.get('kind') == kind and
                  (item.get('queries'), item.get('docs'), item.get('k')) == (q, docs, k), '设备用例缺失或尺寸/顺序不符')
        command = item.get('command', [])
        r.require(len(command) == 9, '既有Top-K实际命令缺失')
        current_runner = Path(command[0]).resolve(strict=True)
        r.require(command == [str(current_runner), str(case), str(actual), str(q), str(docs), str(k),
                             str(device), installation['api_library'], str(calls)] and
                  digest(current_runner) == report.get('runner_sha256'), 'Top-K验收调用器/命令改变')
        r.require(runner is None or runner == current_runner, '验收用例混用调用器')
        runner = current_runner
        evidence[str(runner)] = digest(runner)
        validation = json.loads((actual/'device_validation.json').read_text(encoding='utf-8'))
        compact.validate_record(validation, q, docs, k, device, calls)
        r.require(validation == item.get('validation'), 'Top-K设备逐次核对记录改变')
        workspaces.add(validation['workspace_bytes'])
        expected_inputs = {f'input_{v}/{name}' for v in ('a', 'b') for name in
                           ('scores.bin', 'expected_ids.bin', 'expected_scores.bin', 'metadata.json')}
        if kind == 'capacity': expected_inputs.add('row_cases.json')
        snapshot = item.get('input_sha256', {})
        r.require(set(snapshot) == expected_inputs, 'Top-K验收输入文件清单不符')
        for name, value in snapshot.items():
            r.require(digest(case/name) == value, 'Top-K验收输入发生改变：'+name)
            evidence[str(case/name)] = value
        outputs = {name: digest(actual/name) for name in
                   ('a_top_scores.bin', 'a_top_ids.bin', 'b_top_scores.bin', 'b_top_ids.bin')}
        r.require(outputs == item.get('output_sha256') and digest(case/'device.log') == item.get('device_log_sha256') and
                  digest(case/'check.log') == item.get('check_log_sha256'), '既有Top-K输出或日志改变')
        with redirect_stdout(io.StringIO()):
            for variant in ('a', 'b'):
                compact.check(case/f'input_{variant}', actual/f'{variant}_top_scores.bin', actual/f'{variant}_top_ids.bin')
        for path in [actual/name for name in outputs]+[actual/'device_validation.json', case/'device.log', case/'check.log']:
            evidence[str(path)] = digest(path)
    r.require(len(workspaces) == 1, '既有Top-K工作区不固定，不能复用当前Search冷分配入口')
    return {'directory': str(directory), 'report_sha256': digest(directory/'report.json'),
            'workspace_bytes': next(iter(workspaces)), 'device_cases': len(specs), 'device_calls': len(specs)*calls,
            'evidence_sha256': evidence, 'scope': 'existing isolated correctness proof only; no timing samples reused'}


def run(inputs, output, runner, diagnostic_runner, vendors, score_project, range_project, top_project,
        compact_project, merge_project, top_acceptance, compact_acceptance, boundary_acceptance,
        device=2, shards=32, rounds=6, warmup=5, repeats=30):
    inputs, output, runner, diagnostic_runner, vendors = (p.resolve() for p in
        (inputs, output, runner, diagnostic_runner, vendors))
    r.require(not output.exists(), '结果目录已存在，请换新名字，不覆盖')
    r.require(runner.is_file() and diagnostic_runner.is_file() and r.integer(device, 0, 2147483647) and
              r.integer(shards, 1, 32) and r.integer(rounds, 2, 20) and rounds % 2 == 0 and
              r.integer(warmup, 1, 10000) and r.integer(repeats, 10, 100000), '程序缺失或卡号/配对轮数/预热参数非法')
    index, records, snapshot = fixture.validate_inputs(inputs)
    k, q = records['A']['k'], records['A']['num_queries']
    scale.chain.dimensions(index['num_docs'], k, shards)
    capq = max(item['num_queries'] for item in records.values())
    capwords = max(item['query_term_count'] for item in records.values())
    # Same accepted public binary: its aggregate capacity check is INSIDE Search.
    r.require(1 <= capwords <= 64, '本对照限整批最多64词；不能绕过range评分每查询64词限制')
    cache = checks.cache_inputs(inputs, index, records)
    old_install = scale.chain.validate_install(vendors, score_project, top_project, merge_project, top_acceptance)
    range_install = score.validate_install(vendors/'bm25_query_score_range', range_project)
    prior_score = previous.validate_boundaries(boundary_acceptance, range_install)
    compact_install = compact.validate_install(vendors/compact.VENDOR, compact_project)
    print('只读复核已有37组Top-K设备验收文件；不重新执行这些用例。', flush=True)
    prior_top = validate_compact_acceptance(compact_acceptance, compact_install)
    baseline = dict(old_install, score=range_install, scope='range score + accepted radix_collect + same merge')
    candidate = dict(baseline, topk=compact_install, topk_acceptance=prior_top,
                     topk_device_objects=compact_install['installed_device_object_sha256'],
                     scope='same range score + positive compaction Top-K + same merge')
    targets = {name: (vendors/name).resolve(strict=True) for name in SLOTS}
    targets['bm25_query_score'] = (vendors/'bm25_query_score_range').resolve(strict=True)
    candidate_targets = dict(targets, bm25_topk_radix_collect=(vendors/compact.VENDOR).resolve(strict=True))
    sources = checks.sources(TOOL_SOURCES)
    hosts = checks.sources(tuple(sorted(set(checks.NPU_SOURCES) | set(search.HOST_SOURCES) | set(compact.HOST_SOURCES))))
    output.mkdir(parents=True)
    report = {'status': 'running', 'index_id': index['index_id'], 'docs': index['num_docs'], 'queries': q, 'k': k,
              'device': device, 'shards': shards, 'capacity_queries': capq, 'capacity_words': capwords,
              'supported_request_scope': 'whole batch <=64 words; unchanged timed PackQueries capacity guard',
              'metric': 'single_query_latency' if q == 1 else 'batch_latency_and_search_time_throughput',
              'rounds': rounds, 'warmup_cycles': warmup, 'measured_cycles': repeats, 'input_sha256': snapshot,
              'boundary_acceptance': prior_score, 'compact_acceptance': prior_top,
              'baseline_installation': baseline, 'candidate_installation': candidate,
              'runner_sha256': digest(runner), 'diagnostic_runner_sha256': digest(diagnostic_runner),
              'tool_source_sha256': sources, 'host_source_sha256': hosts, 'resources': bench.host_resources(),
              'source_is_real_corpus': False, 'per_round': [],
              'scope': 'Same NPU, Search binaries, range score, merge, index, queries and shards; only local Top-K differs. Complete raw-ID to CPU Top-K wall time. No CPU/BM25S speedup claim.'}
    r.write_json(output/'report.json', report)
    try:
        runtime, mappings = {}, {}
        for role, mapped in (('baseline', targets), ('candidate', candidate_targets)):
            runtime[role] = output/('runtime_'+role)/'vendors'
            mappings[role] = previous.create_runtime(runtime[role], mapped)
        report['runtime_mapping'] = mappings
        raw = output/'raw_queries'; raw.mkdir()
        for name in records: search.write_queries(raw/(name+'.txt'), cache[name]['queries'])
        raw_hashes = {str(p): digest(p) for p in raw.iterdir()}
        report['raw_query_sha256'], report['preflight'] = raw_hashes, {}
        for role, installation in (('baseline', baseline), ('candidate', candidate)):
            report['preflight'][role] = previous.preflight(output/('preflight_'+role), inputs, index, records,
                cache, raw, diagnostic_runner, runtime[role], installation, device, shards, capq, capwords)
            r.write_json(output/'report.json', report)
        # Identical scoring means all five arrays must agree byte-for-byte, including local candidates.
        for ordinal, name in enumerate(score.SEQUENCE):
            label = f'call{ordinal}_{name}'
            a = report['preflight']['baseline']['output_sha256']
            b = report['preflight']['candidate']['output_sha256']
            r.require(all(a[label+'/'+file] == b[label+'/'+file] for file in scale.chain.FILES),
                      '完整Search预验收新旧Top-K输出字节不同：'+label)
        for ordinal in range(rounds):
            order = ['baseline', 'candidate'] if ordinal % 2 == 0 else ['candidate', 'baseline']
            row = {'round': ordinal+1, 'process_order': order, 'status': 'running'}
            report['per_round'].append(row); r.write_json(output/'report.json', report)
            print(f'完整Search局部Top-K配对{ordinal+1}/{rounds}：{order}', flush=True)
            for role in order:
                mapped = targets if role == 'baseline' else candidate_targets
                r.require(previous.validate_runtime(runtime[role], mapped) == mappings[role], '运行库映射改变')
                row[role] = bench.run_one(output/f'round{ordinal+1}'/role, 'npu', inputs, index, records, cache, raw,
                    runner, runtime[role], baseline if role == 'baseline' else candidate,
                    1, 1, shards, device, capq, capwords, warmup, repeats)
                r.write_json(output/'report.json', report)
            row['cross_checks'] = {}
            for label in checks.CHECKPOINTS:
                a = Path(row['baseline']['directory'])/'actual'/label
                b = Path(row['candidate']['directory'])/'actual'/label
                r.require(all(digest(a/file) == digest(b/file) for file in scale.chain.FILES),
                          '新旧完整Search的分数/局部候选/最终输出字节不同：'+label)
                row['cross_checks'][label] = {'all_five_array_bytes_equal': True}
            row['status'] = 'both_topk_search_versions_and_all_five_array_bytes_passed'
            r.write_json(output/'report.json', report)
        for proof in (snapshot, raw_hashes, prior_score['evidence_sha256'], prior_top['evidence_sha256']):
            scale.unchanged(proof, '输入或既有设备验收文件改变')
        r.require(old_install == scale.chain.validate_install(vendors, score_project, top_project, merge_project, top_acceptance) and
                  range_install == score.validate_install(vendors/'bm25_query_score_range', range_project) and
                  compact_install == compact.validate_install(vendors/compact.VENDOR, compact_project), '安装产物改变')
        r.require(checks.sources(TOOL_SOURCES) == sources and checks.sources(tuple(hosts)) == hosts and
                  digest(runner) == report['runner_sha256'] and digest(diagnostic_runner) == report['diagnostic_runner_sha256'],
                  '源码或实际调用器改变')
        for role, mapped in (('baseline', targets), ('candidate', candidate_targets)):
            r.require(previous.validate_runtime(runtime[role], mapped) == mappings[role], '运行库映射改变')
        report.update(status=STATUS, paired_summary=previous.pair_summary(report['per_round']))
        r.write_json(output/'report.json', report)
        for name, summary in report['paired_summary'].items():
            print(f'{name}完整Search：原Top-K各轮ms={summary["baseline_run_medians_ms"]}；压缩版={summary["candidate_run_medians_ms"]}；压缩版更快{summary["candidate_faster_rounds"]}/{rounds}轮。', flush=True)
        print('正确性和配对检查全部通过；报告保存全部样本及p95。是否保留看完整Search实测；不据此宣称超过CPU/BM25S。', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        r.write_json(output/'report.json', report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output'): p.add_argument('--'+name, type=Path, required=True)
    defaults = {'runner': 'results/search_benchmark_host_build/bm25_npu_search_benchmark',
                'diagnostic-runner': 'results/search_host_build/bm25_search_runner',
                'vendors': 'results/custom_opp/vendors', 'score-project': 'results/Bm25QueryScore',
                'range-project': 'results/Bm25QueryScoreRange', 'topk-project': 'results/Bm25TopKRadixCollect',
                'compact-project': 'results/Bm25TopKPositiveCompact', 'merge-project': 'results/Bm25TopKMerge',
                'topk-acceptance': 'results/radix_collect_topk_validation1',
                'compact-acceptance': 'results/topk_positive_compact_validation1',
                'boundary-acceptance': 'results/query_score_range_boundaries_check1'}
    for name, value in defaults.items(): p.add_argument('--'+name, type=Path, default=ROOT/value)
    for name, value in (('device', 2), ('shards', 32), ('rounds', 6), ('warmup', 5), ('repeats', 30)):
        p.add_argument('--'+name, type=int, default=value)
    a = p.parse_args()
    try:
        r.require(sys.platform == 'linux', '完整Search对照请在A5 Linux执行')
        run(a.input, a.output, a.runner, a.diagnostic_runner, a.vendors, a.score_project, a.range_project,
            a.topk_project, a.compact_project, a.merge_project, a.topk_acceptance, a.compact_acceptance,
            a.boundary_acceptance, a.device, a.shards, a.rounds, a.warmup, a.repeats)
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不回退CPU或旧Top-K，不生成成功汇总。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__': sys.exit(main())
