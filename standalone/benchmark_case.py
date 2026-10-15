"""一个正常查询工作负载的实际CPU线程扫描与NPU对照；失败停止，无CPU替代NPU。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import run_cpu as cpu_entry
from standalone import run_pipeline_benchmark as npu_entry
from standalone.compare_compute_outputs import cross_check, validate_producer, CPU_STATUSES
from standalone.pipeline_fixture import check
from standalone.run_pipeline import prepare_command

OUTPUTS = ('contributions.bin', 'scores.bin', 'top_scores.bin', 'top_ids.bin')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(values, fraction):
    return sorted(values)[max(1, math.ceil(len(values) * fraction)) - 1]


def read_cpu_result(directory, info, threads, warmup, repeats):
    record = validate_producer(directory, info, CPU_STATUSES)
    timing = json.loads((directory / 'cpu_timing.json').read_text(encoding='utf-8'))
    expected = {'status': 'cpu_computed_correctness_not_yet_checked',
        'requested_threads': threads, 'actual_threads': threads, 'warmup': warmup, 'repeats': repeats,
        'queries': info['num_queries'], 'docs': info['num_docs'], 'k': info['k']}
    require(all(timing.get(key) == value for key, value in expected.items()), 'CPU参数或实际线程数不匹配')
    samples = timing['samples_batch_ms']
    require(len(samples) == repeats and all(type(value) in (float, int) and math.isfinite(value) and value > 0
                                           for value in samples), 'CPU计时样本非法')
    for key, fraction in (('median_batch_ms', .5), ('p95_batch_ms', .95)):
        value = timing[key]
        require(type(value) in (float, int) and math.isfinite(value) and
                abs(value - percentile(samples, fraction)) <= 1e-9, 'CPU汇总与原始样本不一致')
    return record, timing


def run_case(directory, output, cpu_runner, npu_runner, vendors, threads, device, warmup, repeats):
    directory, output, cpu_runner, npu_runner, vendors = (path.resolve() for path in
        (directory, output, cpu_runner, npu_runner, vendors))
    require(not output.exists(), '实验输出目录已存在，请选新目录')
    require(cpu_runner.is_file(), 'C++ CPU程序不存在')
    require(threads and len(set(threads)) == len(threads) and
            all(type(value) is int and 1 <= value <= 65536 for value in threads), 'CPU线程列表非法或重复')
    require(type(warmup) is int and type(repeats) is int and 1 <= warmup <= 10000 and 1 <= repeats <= 10000,
            '预热/重复数非法')
    for key in ('PROFILING_MODE', 'PROFILING_OPTIONS'):
        require(os.environ.get(key, '').lower() in ('', '0', 'off', 'false'),
                '发现性能采集环境变量，本实验应在未开启采集的终端运行：' + key)
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    require(allowed is None or max(threads) <= len(allowed), '线程数超过当前进程CPU亲和性范围')
    _, info = prepare_command(directory, output / 'npu', npu_runner, vendors, device)
    require(info.get('dataset', {}).get('case') == 'performance_uniform_short_v1',
            '本入口只接受显式performance模式输入，不把正确性压力样本当作正常查询')
    output.mkdir(parents=True)
    cpu_results = []
    baseline = None
    for count in threads:
        destination = output / ('cpu_t' + str(count))
        print(f'开始实际C++ CPU：{count}线程，预热{warmup}，测量{repeats}。', flush=True)
        cpu_entry.run(directory, destination, cpu_runner, count, warmup, repeats)
        producer, timing = read_cpu_result(destination, info, count, warmup, repeats)
        if baseline is None:
            check(directory, destination)
            baseline = producer['output_sha256']
        else:
            require(producer['output_sha256'] == baseline, '不同CPU线程数的四数组不一致，停止')
        cpu_results.append({'threads': count, 'actual_threads': timing['actual_threads'],
            'median_batch_ms': timing['median_batch_ms'], 'p95_batch_ms': timing['p95_batch_ms'],
            'directory': str(destination),
            'timing_sha256': hashlib.sha256((destination / 'cpu_timing.json').read_bytes()).hexdigest()})
    print('开始实际NPU驻留计算；不复用CPU输出。', flush=True)
    npu_dir = output / 'npu'
    npu_entry.run(directory, npu_dir, npu_runner, vendors, device, warmup, repeats, 'resident')
    npu_timing = json.loads((npu_dir / 'npu_timing.json').read_text(encoding='utf-8'))
    npu_entry.validate_timing(info, npu_timing, 'resident', device, warmup, repeats)
    npu_ms = npu_timing['median_batch_path_ms']
    require(npu_ms > 0, 'NPU耗时必须为正')
    best = min(cpu_results, key=lambda item: (item['median_batch_ms'], item['threads']))
    cross = cross_check(directory, Path(best['directory']), npu_dir)
    # All CPU thread settings have identical checked output fingerprints.
    require(cross['all_top_ids_byte_identical'], 'CPU/NPU全部排名未通过')
    for item in cpu_results:
        item['cpu_over_npu_ratio'] = item['median_batch_ms'] / npu_ms
    result = {'status': 'actual_cpu_thread_scan_and_npu_case_outputs_checked',
        'scope': 'Prepared-input resident repeated computation only; synthetic uniform corpus; '
                 'not complete retrieval, not a BM25S result, not single-query latency when queries>1. '
                 'This is one process per setting with repeated identical arrays, not the final independent-run/cache-rotation study.',
        'input_directory': str(directory), 'input_sha256': info['input_sha256'],
        'queries': info['num_queries'], 'docs': info['num_docs'], 'k': info['k'],
        'dataset': info['dataset'], 'warmup': warmup, 'repeats': repeats, 'device': device,
        'cpu_threads': cpu_results, 'best_tested_cpu_threads': best['threads'],
        'best_tested_cpu_median_ms': best['median_batch_ms'], 'npu_median_batch_ms': npu_ms,
        'npu_p95_batch_ms': npu_timing['p95_batch_path_ms'],
        'best_tested_cpu_over_npu_ratio': best['median_batch_ms'] / npu_ms,
        'cpu_allowed_logical_ids': allowed, 'os_logical_cpu_count': os.cpu_count(),
        'cpu_quota': 'not_checked', 'platform': platform.platform(),
        'openmp_environment': {key: os.environ.get(key) for key in
            ('OMP_PROC_BIND', 'OMP_PLACES', 'OMP_WAIT_POLICY', 'GOMP_CPU_AFFINITY')},
        'cpu_npu_cross_check': cross,
        'npu_timing_sha256': hashlib.sha256((npu_dir / 'npu_timing.json').read_bytes()).hexdigest()}
    with (output / 'report.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('\n本工作负载检查通过，整批中位数：')
    for item in cpu_results:
        print(f'CPU {item["threads"]}线程：{item["median_batch_ms"]:.6g}ms')
    print(f'NPU设备{device}：{npu_ms:.6g}ms')
    print(f'已测试CPU中最快为{best["threads"]}线程；CPU时间/NPU时间={best["median_batch_ms"] / npu_ms:.6g}。')
    print('比值>1表示本范围NPU更快；<1表示CPU更快。批量时间不可当单条响应时间。')
    print('报告：', output / 'report.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--cpu-runner', default=str(ROOT / 'results/cpu_host_build/bm25_cpu_runner'))
    parser.add_argument('--npu-runner', default=str(ROOT / 'results/topk_parallel_pipeline_host_build/bm25_pipeline_benchmark'))
    parser.add_argument('--vendor-dir', default=str(ROOT / 'results/topk_parallel_runtime/vendors'))
    args = parser.parse_args()
    run_case(Path(args.input), Path(args.output), Path(args.cpu_runner), Path(args.npu_runner),
             Path(args.vendor_dir), args.threads, args.device, args.warmup, args.repeats)


if __name__ == '__main__':
    main()
