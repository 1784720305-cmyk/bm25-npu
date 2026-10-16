"""独立进程复测CPU和四段NPU配置；交替顺序、逐次核对、保留全部计时。"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.benchmark_case import read_cpu_result, require
from standalone.compare_compute_outputs import cross_check, validate_producer, NPU_STATUSES
from standalone.pipeline_fixture import validate_input, check
from standalone.prepare_sharded_runtime import validate_runtime, VARIANT, WARP_VARIANT
from standalone.run_pipeline_benchmark import validate_timing
from standalone.run_sharded_pipeline import check_candidates, FILES
from standalone.validate_topk_merge import digest, write_json


def run_logged(command, log_path):
    """Stream the actual child output to console and a per-process log; failures stop the suite."""
    print('实际命令：' + json.dumps(command, ensure_ascii=False), flush=True)
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    with log_path.open('x', encoding='utf-8', newline='\n') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1,
                                   env=environment, start_new_session=(os.name == 'posix'))
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if process.poll() is None:
                # Only this newly-created child session, including its C++ caller, is terminated.
                try:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def read_npu_result(destination, info, runner, installation, device, shards, warmup, repeats, scratch):
    metadata = validate_producer(destination, info, NPU_STATUSES)
    expected = {'status': 'device_repeated_call_completed_file_checks_passed',
                'topk_variant': installation['variant'], 'pipeline_stages': 4, 'shards': shards,
                'device': device, 'mode': 'resident', 'warmup': warmup, 'repeats': repeats,
                'runner_sha256': digest(runner), 'runtime_installation': installation,
                'parallel_user_scratch_bytes': scratch}
    require(all(metadata.get(key) == value for key, value in expected.items()), '四段运行记录与本轮配置不一致')
    for name in FILES:
        require(digest(destination / name) == metadata['output_sha256'][name], '六数组输出指纹不一致：' + name)
        require((destination / name).read_bytes() == (destination / 'bootstrap' / name).read_bytes(), '首末输出不一致：' + name)
    timing = json.loads((destination / 'npu_timing.json').read_text(encoding='utf-8'))
    validate_timing(info, timing, 'resident', device, warmup, repeats)
    require(timing.get('pipeline_stages') == 4 and timing.get('shards') == shards and
            timing.get('synchronization_policy') == 'synchronize after the entire four-operator chain',
            '不是本轮四段单次同步路径')
    capacities = timing['workspace_capacity_bytes']
    require(len(capacities) == 4 and all(type(value) is int and value >= 0 for value in capacities) and
            capacities[2] >= capacities[0] + scratch, '分块Top-K工作区不匹配')
    if installation['variant'] == WARP_VARIANT:
        require(timing.get('topk_implementation') == 'warp' and
                timing.get('local_topk_user_scratch_bytes') == scratch and capacities[2] == capacities[0] + scratch,
                'warp版本未使用新调用器及精确工作区规则')
    require(timing['median_batch_path_ms'] > 0, 'NPU耗时必须为正')
    check_candidates(info, destination, shards)
    return metadata, timing


def time_record(destination, timing_name, median, p95, log):
    return {'directory': str(destination), 'median_batch_ms': median, 'p95_batch_ms': p95,
            'timing_sha256': digest(destination / timing_name),
            'run_metadata_sha256': digest(destination / 'run_metadata.json'),
            'log_path': str(log), 'log_sha256': digest(log)}


def summarize(values):
    require(values and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values),
            '汇总计时值非法')
    return {'all_run_medians_ms': values, 'median_of_run_medians_ms': statistics.median(values),
            'min_run_median_ms': min(values), 'max_run_median_ms': max(values)}


def run(directory, output, cpu_runner, npu_runner, vendors, device, threads, shards, rounds, warmup, repeats):
    directory, output, cpu_runner, npu_runner, vendors = (
        path.resolve() for path in (directory, output, cpu_runner, npu_runner, vendors))
    require(not output.exists(), '输出目录已存在，请选择新目录')
    require(cpu_runner.is_file() and npu_runner.is_file(), '实际CPU/NPU可执行程序缺失')
    require(type(device) is int and 0 <= device <= 2147483647, '设备号非法')
    require(type(threads) is int and 1 <= threads <= 65536, 'CPU线程数非法')
    require(type(rounds) is int and 3 <= rounds <= 20 and
            type(warmup) is int and 1 <= warmup <= 10000 and
            type(repeats) is int and 1 <= repeats <= 10000, '轮数/预热数/测量数非法')
    require(shards and len(shards) == len(set(shards)), '分块列表为空或重复')
    for name in ('PROFILING_MODE', 'PROFILING_OPTIONS'):
        require(os.environ.get(name, '').lower() in ('', '0', 'off', 'false'),
                '普通复测请先关闭性能采集：' + name)
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    require(allowed is None or threads <= len(allowed), 'CPU线程数超过进程亲和性范围')
    info, _ = validate_input(directory)
    require(info.get('dataset', {}).get('case') == 'performance_uniform_short_v1',
            '本入口接受正常performance输入，不将正确性压力样本作性能工作负载')
    installation, scratches = None, {}
    for count in shards:
        current, scratch, _ = validate_runtime(vendors, info, count)
        require(installation is None or installation == current, '分块配置使用了不同运行库')
        installation = current
        scratches[count] = scratch
    executable_hashes = {'cpu': digest(cpu_runner), 'npu': digest(npu_runner)}
    output.mkdir(parents=True)
    completed = []
    baseline_cpu = None
    for index in range(rounds):
        order = list(shards) if index % 2 == 0 else list(reversed(shards))
        print(f'开始第{index+1}/{rounds}轮：CPU {threads}线程，然后NPU分块顺序{order}。', flush=True)
        round_dir = output / f'round{index+1:02d}'
        round_dir.mkdir()
        cpu = round_dir / 'cpu'
        cpu_log = round_dir / 'cpu.log'
        run_logged([sys.executable, '-B', '-u', str(ROOT / 'standalone/run_cpu.py'), '--input', str(directory),
                    '--output', str(cpu), '--runner', str(cpu_runner), '--threads', str(threads),
                    '--warmup', str(warmup), '--repeats', str(repeats)], cpu_log)
        cpu_metadata, cpu_timing = read_cpu_result(cpu, info, threads, warmup, repeats)
        check(directory, cpu)
        require(baseline_cpu is None or baseline_cpu == cpu_metadata['output_sha256'], '各轮CPU输出字节不同')
        baseline_cpu = cpu_metadata['output_sha256']
        cpu_record = time_record(cpu, 'cpu_timing.json', cpu_timing['median_batch_ms'], cpu_timing['p95_batch_ms'], cpu_log)
        npu_records = []
        for count in order:
            destination = round_dir / f'npu_s{count}'
            log = round_dir / f'npu_s{count}.log'
            run_logged([sys.executable, '-B', '-u', str(ROOT / 'standalone/run_sharded_pipeline.py'),
                        '--input', str(directory), '--output', str(destination), '--runner', str(npu_runner),
                        '--vendor-dir', str(vendors), '--device', str(device), '--shards', str(count),
                        '--warmup', str(warmup), '--repeats', str(repeats), '--mode', 'resident'], log)
            _, timing = read_npu_result(destination, info, npu_runner, installation, device, count, warmup, repeats, scratches[count])
            cross = cross_check(directory, cpu, destination)
            write_json(round_dir / f'cross_s{count}.json', cross)
            record = time_record(destination, 'npu_timing.json', timing['median_batch_path_ms'], timing['p95_batch_path_ms'], log)
            record.update(shards=count, cpu_over_npu_ratio=cpu_timing['median_batch_ms'] / timing['median_batch_path_ms'],
                          prepare_median_ms=timing['median_prepare_ms'], launch_sync_median_ms=timing['median_launch_sync_ms'],
                          cross_report_sha256=digest(round_dir / f'cross_s{count}.json'))
            npu_records.append(record)
        # Detect changed binaries, original input arrays or installed libraries during the suite.
        require(executable_hashes == {'cpu': digest(cpu_runner), 'npu': digest(npu_runner)}, '复测过程中可执行程序改变')
        current_info, _ = validate_input(directory)
        require(current_info == info, '复测过程中输入元信息或指纹改变')
        for count in shards:
            current, scratch, _ = validate_runtime(vendors, info, count)
            require(current == installation and scratch == scratches[count], '复测过程中运行库改变')
        record = {'round': index+1, 'npu_order': order, 'cpu': cpu_record, 'npu': npu_records}
        write_json(round_dir / 'checked.json', record)
        completed.append(record)
    cpu_summary = summarize([item['cpu']['median_batch_ms'] for item in completed])
    npu_summary = []
    for count in shards:
        records = [next(record for record in item['npu'] if record['shards'] == count) for item in completed]
        summary = summarize([record['median_batch_ms'] for record in records])
        summary.update(shards=count, p95_batch_ms_per_run=[record['p95_batch_ms'] for record in records],
                       paired_cpu_over_npu_ratios=[record['cpu_over_npu_ratio'] for record in records])
        npu_summary.append(summary)
    lowest = min(npu_summary, key=lambda item: (item['median_of_run_medians_ms'], item['shards']))
    report = {'status': 'independent_process_retests_and_all_cpu_npu_checks_passed',
              'input_directory': str(directory), 'input_sha256': info['input_sha256'], 'dataset': info['dataset'],
              'queries': info['num_queries'], 'docs': info['num_docs'], 'k': info['k'],
              'rounds': rounds, 'warmup': warmup, 'repeats_per_process': repeats,
              'device': device, 'cpu_threads': threads, 'shards': shards,
              'order_policy': 'CPU first every round, NPU list forward on odd rounds/reversed on even rounds',
              'cpu': cpu_summary, 'npu': npu_summary, 'per_round': completed,
              'lowest_point_estimate_shards': lowest['shards'], 'executable_sha256': executable_hashes,
              'runtime_installation': installation, 'cpu_allowed_logical_ids': allowed,
              'platform': platform.platform(), 'cpu_quota_and_shared_load': 'not_measured',
              'openmp_environment': {name: os.environ.get(name) for name in ('OMP_PROC_BIND', 'OMP_PLACES', 'OMP_WAIT_POLICY', 'GOMP_CPU_AFFINITY')},
              'scope': 'Same prepared resident arrays, separate CPU/NPU processes each round; initialization/file IO/debug checks excluded from child timers. CPU thread setting chosen from earlier scan, not globally optimal. No confidence/significance claim or complete-retrieval/BM25S/transfer result; point estimate alone is not performance acceptance.'}
    write_json(output / 'report.json', report)
    write_json(output / 'summary.json', {
        'status': report['status'], 'queries': info['num_queries'], 'docs': info['num_docs'], 'k': info['k'],
        'device': device, 'cpu_threads': threads, 'rounds': rounds, 'warmup': warmup, 'repeats_per_process': repeats,
        'cpu': cpu_summary, 'npu': npu_summary, 'lowest_point_estimate_shards': lowest['shards'],
        'ratio_rule': 'CPU time / NPU time: >1 NPU faster, <1 CPU faster; paired per round',
        'scope': report['scope'], 'full_report_sha256': digest(output / 'report.json')})
    print('独立进程复测及全部结果检查通过。CPU各轮中位数(ms)：' + str(cpu_summary['all_run_medians_ms']), flush=True)
    for item in npu_summary:
        print(f'S={item["shards"]}各轮中位数(ms)：{item["all_run_medians_ms"]}；这些中位数的中位数={item["median_of_run_medians_ms"]:.6g}；逐轮CPU时间/NPU时间={item["paired_cpu_over_npu_ratios"]}', flush=True)
    print(f'最低点估计为S={lowest["shards"]}，不能据此单独认定差异稳定或性能达标。报告：{output / "report.json"}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpu-runner', default=str(ROOT / 'results/cpu_host_build/bm25_cpu_runner'))
    parser.add_argument('--npu-runner', default=str(ROOT / 'results/sharded_host_build/bm25_sharded_benchmark'))
    parser.add_argument('--vendor-dir', default=str(ROOT / 'results/sharded_runtime/vendors'))
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--shards', nargs='+', type=int, default=[16, 32])
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.output), Path(args.cpu_runner), Path(args.npu_runner), Path(args.vendor_dir),
            args.device, args.threads, args.shards, args.rounds, args.warmup, args.repeats)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f'FAILED：{error}；停止，不生成CPU替代输出或总成功报告。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
