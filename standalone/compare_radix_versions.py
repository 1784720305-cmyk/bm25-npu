"""同一卡/输入/调用器轮换比较原Radix与零分汇总版；逐轮核对六数组，不自动宣称加速。"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.benchmark_case import require
from standalone.confirm_sharded import run_logged, read_npu_result, time_record, summarize
from standalone.pipeline_fixture import validate_input, check
from standalone.prepare_sharded_runtime import validate_runtime, RADIX_VARIANT, RADIX_ZERO_VARIANT
from standalone.run_sharded_pipeline import FILES
from standalone.validate_topk_merge import digest, write_json

TOOL_FILES = ('standalone/compare_radix_versions.py', 'standalone/run_sharded_pipeline.py',
              'standalone/prepare_sharded_runtime.py', 'standalone/confirm_sharded.py',
              'standalone/benchmark_case.py', 'standalone/run_pipeline.py',
              'standalone/run_pipeline_benchmark.py', 'standalone/pipeline_fixture.py',
              'standalone/compare_compute_outputs.py', 'standalone/topk_fixture.py',
              'standalone/validate_topk_merge.py', 'standalone/validate_topk_radix.py',
              'standalone/validate_topk_radix_zero.py')
LABELS = ('baseline', 'candidate')


def validate_pair(vendors, info, shards):
    records = {}
    for label, variant in zip(LABELS, (RADIX_VARIANT, RADIX_ZERO_VARIANT)):
        installation, scratch, _ = validate_runtime(vendors[label], info, shards)
        require(installation.get('variant') == variant and
                installation.get('topk_implementation') == 'radix' and scratch == 0,
                f'{label}运行目录版本错误，必须分别是原Radix与零分汇总版')
        records[label] = installation
    # Prevent a different contribution/accumulation/merge installation from masquerading
    # as a change isolated to the local Top-K histogram.
    for alias in ('bm25', 'bm25_accumulate', 'bm25_topk_merge'):
        require(records['baseline']['api_libraries'][alias] == records['candidate']['api_libraries'][alias],
                '新旧必须使用相同的贡献/累加/合并安装：' + alias)
    require(records['baseline']['merge_installation'] == records['candidate']['merge_installation'],
            '新旧合并算子来源不一致')
    for name in ('op_host/bm25_top_k.cpp', 'op_host/bm25_top_k_tiling.h'):
        require(records['baseline']['radix_installation']['project_source_sha256'][name] ==
                records['candidate']['radix_installation']['project_source_sha256'][name],
                '本实验只比较计数改动，Top-K注册/分核源码须相同：' + name)
    return records


def run(directory, output, runner, baseline_vendors, candidate_vendors, device, shards, rounds, warmup, repeats):
    directory, output, runner, baseline_vendors, candidate_vendors = (
        path.resolve() for path in (directory, output, runner, baseline_vendors, candidate_vendors))
    require(not output.exists(), '结果目录已存在，请选新目录；不覆盖已有报告')
    require(runner.is_file(), '实际四段调用器缺失')
    require(type(device) is int and 0 <= device <= 2147483647, '设备号非法')
    require(type(rounds) is int and 2 <= rounds <= 20 and rounds % 2 == 0,
            '轮数须为2..20的偶数，让新旧先跑次数相同')
    require(type(warmup) is int and 1 <= warmup <= 10000 and
            type(repeats) is int and 1 <= repeats <= 10000, '预热/测量数非法')
    for name in ('PROFILING_MODE', 'PROFILING_OPTIONS'):
        require(os.environ.get(name, '').lower() in ('', '0', 'off', 'false'),
                '新旧普通性能对比须关闭采集：' + name)
    info, _ = validate_input(directory)
    require(info.get('dataset', {}).get('case') == 'performance_uniform_short_v1',
            '本入口只接受正常performance输入，不将正确性压力样本当性能成绩')
    vendors = dict(zip(LABELS, (baseline_vendors, candidate_vendors)))
    installations = validate_pair(vendors, info, shards)
    runner_sha = digest(runner)
    tools = {name: digest(ROOT/name) for name in TOOL_FILES}
    output.mkdir(parents=True)
    report = {'status': 'running', 'input_directory': str(directory), 'input_sha256': info['input_sha256'],
              'dataset': info['dataset'], 'queries': info['num_queries'], 'docs': info['num_docs'], 'k': info['k'],
              'device': device, 'shards': shards, 'rounds': rounds, 'warmup': warmup, 'repeats': repeats,
              'mode': 'resident', 'order_policy': 'baseline first on odd rounds; candidate first on even rounds',
              'runner': str(runner), 'runner_sha256': runner_sha, 'tool_source_sha256': tools,
              'runtime_installations': installations, 'per_round': [], 'platform': platform.platform(),
              'shared_load': 'not_measured',
              'scope': 'Same prepared resident input, same device and C++ runner, same three other operators; '
                       'fresh process for each version each round. Child timers include API preparation, four '
                       'submissions and final synchronization; initialization/allocation/file IO/input packing/'
                       'transfers/debug validation excluded. Not pure-kernel timing, CPU/BM25S or complete '
                       'retrieval acceptance. No confidence/significance or automatic speedup claim.'}
    report_path = output/'report.json'
    write_json(report_path, report)
    expected_outputs = None
    try:
        for index in range(rounds):
            order = LABELS if index % 2 == 0 else tuple(reversed(LABELS))
            record = {'round': index+1, 'status': 'running', 'process_order': list(order), 'runs': {}}
            report['per_round'].append(record)
            write_json(report_path, report)
            round_dir = output/f'round{index+1:02d}'
            round_dir.mkdir()
            print(f'开始新旧第{index+1}/{rounds}轮，顺序={list(order)}，设备={device}。', flush=True)
            fingerprints = {}
            for label in order:
                destination, log = round_dir/label, round_dir/(label+'.log')
                command = [sys.executable, '-B', '-u', str(ROOT/'standalone/run_sharded_pipeline.py'),
                           '--input', str(directory), '--output', str(destination), '--runner', str(runner),
                           '--vendor-dir', str(vendors[label]), '--device', str(device), '--shards', str(shards),
                           '--warmup', str(warmup), '--repeats', str(repeats), '--mode', 'resident']
                run_logged(command, log)
                metadata, timing = read_npu_result(destination, info, runner, installations[label],
                                                 device, shards, warmup, repeats, 0)
                check(directory, destination)  # Independent formula/accumulation check outside all timers.
                # read_npu_result rechecks actual file hashes, all local/final ranking and
                # first/last equality, rather than trusting only a child's printed message.
                fingerprints[label] = {name: metadata['output_sha256'][name] for name in FILES}
                record['runs'][label] = time_record(destination, 'npu_timing.json', timing['median_batch_path_ms'],
                                                    timing['p95_batch_path_ms'], log)
                write_json(report_path, report)
            require(fingerprints['baseline'] == fingerprints['candidate'], '同一轮新旧六数组字节不一致，停止比较性能')
            if expected_outputs is None:
                expected_outputs = fingerprints['baseline']
            require(fingerprints['baseline'] == expected_outputs, '不同轮次的设备结果字节改变，停止')
            current_info, _ = validate_input(directory)
            require(current_info == info and digest(runner) == runner_sha, '对比过程中输入或调用器改变')
            require(validate_pair(vendors, info, shards) == installations, '对比过程中算子安装/验收引用改变')
            require(all(digest(ROOT/name) == sha for name, sha in tools.items()), '对比工具源码在运行中改变')
            baseline_ms = record['runs']['baseline']['median_batch_ms']
            candidate_ms = record['runs']['candidate']['median_batch_ms']
            record.update(status='both_device_runs_and_all_six_array_checks_passed',
                          baseline_over_candidate_ratio=baseline_ms/candidate_ms,
                          six_output_sha256=fingerprints['baseline'])
            write_json(report_path, report)
        report['baseline'] = summarize([row['runs']['baseline']['median_batch_ms'] for row in report['per_round']])
        report['candidate'] = summarize([row['runs']['candidate']['median_batch_ms'] for row in report['per_round']])
        for label in LABELS:
            report[label]['p95_batch_ms_per_run'] = [row['runs'][label]['p95_batch_ms'] for row in report['per_round']]
        ratios = [row['baseline_over_candidate_ratio'] for row in report['per_round']]
        report.update(status='all_paired_radix_runs_and_six_array_checks_passed',
                      paired_baseline_over_candidate_ratios=ratios,
                      median_paired_baseline_over_candidate_ratio=statistics.median(ratios),
                      candidate_faster_rounds=sum(value > 1 for value in ratios),
                      ratio_rule='baseline time / candidate time; >1 candidate faster, <1 baseline faster; paired per round')
        write_json(report_path, report)
        summary = {key: report[key] for key in ('status', 'queries', 'docs', 'k', 'device', 'shards',
                   'rounds', 'warmup', 'repeats', 'mode', 'baseline', 'candidate',
                   'paired_baseline_over_candidate_ratios', 'median_paired_baseline_over_candidate_ratio',
                   'candidate_faster_rounds', 'ratio_rule', 'scope')}
        summary['report_sha256'] = digest(report_path)
        write_json(output/'summary.json', summary)
        print('新旧各轮完整计算、候选及六数组字节核对全部通过。', flush=True)
        print('原Radix各轮中位数(ms)：' + str(report['baseline']['all_run_medians_ms']), flush=True)
        print('零分汇总各轮中位数(ms)：' + str(report['candidate']['all_run_medians_ms']), flush=True)
        print('逐轮原版时间/新版时间：' + str(ratios), flush=True)
        print(f'新版更快轮数={report["candidate_faster_rounds"]}/{rounds}；配对比值中位数='
              f'{report["median_paired_baseline_over_candidate_ratio"]:.6g}。报告：{report_path}', flush=True)
        print('这是新旧NPU驻留计算对比；点估计或接近1的比值不表示稳定加速，不代表CPU/BM25S/完整查询达标。', flush=True)
        return report
    except (Exception, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error) or type(error).__name__
        if report['per_round'] and report['per_round'][-1]['status'] == 'running':
            report['per_round'][-1]['status'] = report['status']
        write_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runner', default=str(ROOT/'results/radix_sharded_host_build/bm25_sharded_benchmark'))
    parser.add_argument('--baseline-vendors', default=str(ROOT/'results/radix_sharded_runtime/vendors'))
    parser.add_argument('--candidate-vendors', default=str(ROOT/'results/radix_zero_sharded_runtime/vendors'))
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--shards', type=int, default=32)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.output), Path(args.runner), Path(args.baseline_vendors),
            Path(args.candidate_vendors), args.device, args.shards, args.rounds, args.warmup, args.repeats)
    except KeyboardInterrupt:
        print('对比已中断，保留部分日志；不生成成功汇总。', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f'FAILED：{error}；停止，不生成CPU替代输出或成功汇总。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
