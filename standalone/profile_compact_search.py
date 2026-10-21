"""仅采集已验收的range评分＋正分压缩Top-K完整Search；不修改算子/计时程序。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import benchmark_compact_search_cpu as winner
from standalone import benchmark_search as bench, search_benchmark_checks as checks
from standalone import search_scale_fixture as fixture, run_search_scale as scale, resident_index as r

STATUS = 'compact_search_profile_execution_and_checkpoints_passed'
TOOLS = tuple(sorted(set(winner.TOOL_SOURCES) | {
    'standalone/profile_compact_search.py', 'standalone/analyze_compact_search_profile.py',
    'standalone/analyze_pipeline_profile.py'}))
digest = scale.digest


def validate_install(installation, vendors, projects):
    r.require(installation['score'] == winner.selected.score.validate_install(vendors/'bm25_query_score_range', projects['score']) and
        installation['topk'] == winner.selected.compact.validate_install(vendors/winner.selected.compact.VENDOR, projects['topk']) and
        installation['merge'] == scale.chain.merge.validate_install(vendors/'bm25_topk_merge', projects['merge']) and
        installation['merge_device_objects'] == scale.chain.artifacts(vendors/'bm25_topk_merge', projects['merge'],
            'bm25_top_k_merge', 'aclnn_bm25_top_k_merge.h'), '当前采集组合的实际安装产物改变')


def run(inputs, acceptance, output, runner, vendors, projects, device=2, warmup=5, repeats=10):
    inputs, acceptance, output, runner, vendors = (p.resolve() for p in (inputs, acceptance, output, runner, vendors))
    projects = {key: value.resolve() for key, value in projects.items()}
    r.require(not output.exists() and runner.is_file() and r.integer(device, 0, 2147483647) and
        r.integer(warmup, 1, 10000) and r.integer(repeats, 10, 100000), '目录已存在或程序/采集配置非法')
    print('核对已验收的新组合；采集本次实际代码，不使用旧266/271us代替。', flush=True)
    index, records, snapshot = fixture.validate_inputs(inputs)
    cache = checks.cache_inputs(inputs, index, records)
    prior = winner.validate_selected(acceptance, inputs, index, records, snapshot, cache, runner, vendors, projects)
    shards, installation = prior['shards'], prior['installation']
    capq, capwords = max(v['num_queries'] for v in records.values()), max(v['query_term_count'] for v in records.values())
    output.mkdir(parents=True)
    report = {'status': 'running', 'input_directory': str(inputs), 'index': index, 'records': records,
        'device': device, 'shards': shards, 'capacity_queries': capq, 'capacity_words': capwords,
        'warmup': warmup, 'repeats': repeats, 'score_variant': 'query_score_range', 'topk_variant': 'topk_positive_compact',
        'installation': installation, 'projects': {key: str(path) for key, path in projects.items()}, 'vendors': str(vendors),
        'input_sha256': snapshot, 'prior': {key: value for key, value in prior.items() if key != 'targets'},
        'runner_sha256': digest(runner), 'tool_source_sha256': checks.sources(TOOLS),
        'host_source_sha256': checks.sources(checks.NPU_SOURCES), 'resources': bench.host_resources(),
        'scope': 'Current-combination msprof diagnosis only; unchanged Search caller boundary. Not a new ordinary performance baseline or whole-card utilization measurement.'}
    r.write_json(output/'report.json', report)
    try:
        runtime = output/'runtime/vendors'
        report['runtime_mapping'] = winner.selected.previous.create_runtime(runtime, prior['targets'])
        raw = output/'raw_queries'; raw.mkdir()
        for name in records: winner.search.write_queries(raw/(name+'.txt'), cache[name]['queries'])
        report['raw_query_sha256'] = {str(path): digest(path) for path in raw.iterdir()}
        r.write_json(output/'report.json', report)
        report['run'] = bench.run_one(output/'measurement', 'npu', inputs, index, records, cache, raw, runner,
            runtime, installation, 1, 1, shards, device, capq, capwords, warmup, repeats)
        for proof in (snapshot, report['raw_query_sha256'], prior['evidence_sha256']): scale.unchanged(proof, '采集期间输入/既有验收改变')
        validate_install(installation, vendors, projects)
        r.require(winner.selected.previous.validate_runtime(runtime, prior['targets']) == report['runtime_mapping'] and
            checks.sources(TOOLS) == report['tool_source_sha256'] and
            checks.sources(checks.NPU_SOURCES) == report['host_source_sha256'] and digest(runner) == report['runner_sha256'],
            '采集期间运行映射、源码或程序改变')
        report['status'] = STATUS; r.write_json(output/'report.json', report)
        calls = 1+4+2*warmup+2*repeats
        print(f'新组合实际采集应用及完整结果核对通过：应有{calls}次Search、{calls*3}条自研任务；仍须设备表交叉核对。', flush=True)
        print('接下来分析评分、Top-K、合并、同次设备链及链外时间；采集耗时不替代此前普通0.333ms基准。', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        r.write_json(output/'report.json', report); raise


def validate_record(output):
    output = output.resolve(strict=True)
    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
    r.require(report.get('status') == STATUS and report.get('score_variant') == 'query_score_range' and
        report.get('topk_variant') == 'topk_positive_compact' and report.get('tool_source_sha256') == checks.sources(TOOLS) and
        report.get('host_source_sha256') == checks.sources(checks.NPU_SOURCES), '需要当前新组合已实际执行的采集应用报告')
    installation = report['installation']; vendors = Path(report['vendors'])
    projects = {key: Path(value) for key, value in report['projects'].items()}
    validate_install(installation, vendors, projects)
    for proof in (report['input_sha256'], report['raw_query_sha256'], report['prior']['evidence_sha256']):
        scale.unchanged(proof, '采集应用输入/原始查询/既有验收文件改变')
    targets = {'bm25_query_score': (vendors/'bm25_query_score_range').resolve(strict=True),
        'bm25_topk_radix_collect': (vendors/winner.selected.compact.VENDOR).resolve(strict=True),
        'bm25_topk_merge': (vendors/'bm25_topk_merge').resolve(strict=True)}
    runtime = output/'runtime/vendors'
    r.require(winner.selected.previous.validate_runtime(runtime, targets) == report.get('runtime_mapping'), '采集运行目录被替换')
    linked = report['run']; directory = output/'measurement'
    r.require(linked.get('directory') == str(directory) and digest(directory/'report.json') == linked.get('report_sha256'), '采集单轮报告来源改变')
    data = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    command = data.get('command', []); inputs = Path(report['input_directory'])
    q, k = report['records']['A']['num_queries'], report['records']['A']['k']
    ws = installation['topk_acceptance']['workspace_bytes']
    r.require(len(command) == 14 and command[1:] == [str(inputs/'index'), str(output/'raw_queries/A.txt'),
        str(output/'raw_queries/B.txt'), str(directory/'actual'), str(report['device']), str(runtime), str(k),
        str(report['shards']), str(report['capacity_queries']), str(report['capacity_words']), str(ws),
        str(report['warmup']), str(report['repeats'])] and data.get('status') == 'actual_search_measurement_and_checkpoint_checks_passed' and
        data.get('backend') == 'npu' and data.get('index_id') == report['index']['index_id'] and
        digest(Path(command[0])) == data.get('runner_sha256') == report.get('runner_sha256') and
        digest(directory/'run.log') == data.get('log_sha256'), '采集实际程序/命令/日志不符')
    r.require(str(inputs/'index/metadata.json') in report['input_sha256'], '采集实际索引不在输入证据中')
    hashes = {name+'/'+file: digest(directory/'actual'/name/file) for name in checks.CHECKPOINTS for file in scale.chain.FILES}
    r.require(hashes == data.get('output_sha256'), '采集前后完整分数/候选/Top-K改变')
    session, summary = checks.check_session(directory/'actual', 'npu', report['index'], report['records'], report['shards'],
        1, report['device'], report['capacity_queries'], report['capacity_words'], report['warmup'], report['repeats'], ws)
    r.require(session == data.get('session') and summary == data.get('summary') == linked.get('summary'), '采集会话/原始样本/汇总不符')
    for name in ('A', 'B'):
        r.require(all((directory/'actual'/('before_'+name)/file).read_bytes() ==
            (directory/'actual'/('after_'+name)/file).read_bytes() for file in scale.chain.FILES), '采集前后结果不逐字节一致')
    return report, session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output'): p.add_argument('--'+name, type=Path, required=True)
    for name, value in {'acceptance': 'results/search_topk_compact_256k_q8_pair1',
        'runner': 'results/search_benchmark_host_build/bm25_npu_search_benchmark', 'vendors': 'results/custom_opp/vendors',
        'score-project': 'results/Bm25QueryScoreRange', 'topk-project': 'results/Bm25TopKPositiveCompact',
        'merge-project': 'results/Bm25TopKMerge'}.items(): p.add_argument('--'+name, type=Path, default=ROOT/value)
    for name, value in (('device', 2), ('warmup', 5), ('repeats', 10)): p.add_argument('--'+name, type=int, default=value)
    a = p.parse_args()
    try:
        r.require(sys.platform == 'linux', '实际采集请在A5 Linux执行')
        run(a.input, a.acceptance, a.output, a.runner, a.vendors,
            {'score': a.score_project, 'topk': a.topk_project, 'merge': a.merge_project}, a.device, a.warmup, a.repeats)
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不产生成功采集或CPU替代结果。', file=sys.stderr); return 1
    return 0


if __name__ == '__main__': sys.exit(main())
