"""启动实际NPU同进程预热/重复调用，核对首末输出并记录计时范围。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.pipeline_fixture import check
from standalone.run_pipeline import prepare_command

OUTPUT_FILES = ('contributions.bin', 'scores.bin', 'top_scores.bin', 'top_ids.bin')


def validate_timing(info, timing, mode, device, warmup, repeats):
    expected = {'mode': mode, 'device': device, 'queries': info['num_queries'],
                'docs': info['num_docs'], 'k': info['k'], 'warmup': warmup, 'repeats': repeats,
                'additional_plain_warmup': 1,
                'status': 'device_repeated_completed_independent_correctness_not_yet_checked'}
    if any(timing.get(key) != value for key, value in expected.items()):
        raise ValueError('NPU计时报告参数/状态不匹配')
    samples = timing['samples']
    if len(samples) != repeats:
        raise ValueError('NPU计时样本数量不匹配')
    for sample in samples:
        values = [sample[key] for key in ('batch_path_ms', 'prepare_ms', 'launch_sync_ms')]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('NPU计时值非法')
        if values[0] + 1e-8 < values[1] + values[2]:
            raise ValueError('NPU总计时小于所包含阶段')
    # Do not trust summaries without checking against the raw timing samples.
    for field, summary, fraction in (('batch_path_ms', 'median_batch_path_ms', .5),
                                      ('batch_path_ms', 'p95_batch_path_ms', .95),
                                      ('prepare_ms', 'median_prepare_ms', .5),
                                      ('launch_sync_ms', 'median_launch_sync_ms', .5)):
        expected_value = sorted(s[field] for s in samples)[max(1, math.ceil(fraction * repeats)) - 1]
        actual_value = timing[summary]
        if type(actual_value) not in (int, float) or not math.isfinite(actual_value) or abs(actual_value - expected_value) > 1e-9:
            raise ValueError(f'NPU计时汇总不匹配：{summary}')


def run(directory, output, runner, vendors, device, warmup, repeats, mode):
    if type(warmup) is not int or type(repeats) is not int or not (1 <= warmup <= 10000 and 1 <= repeats <= 10000):
        raise ValueError('预热/重复次数必须在1..10000')
    if mode not in ('resident', 'with_transfers'):
        raise ValueError('计时模式非法')
    command, info = prepare_command(directory, output, runner, vendors, device)
    command.extend([str(warmup), str(repeats), mode])
    print('将运行实际NPU同进程预热/重复调用：', flush=True)
    print(json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)
    directory, output = directory.resolve(), output.resolve()
    timing = json.loads((output / 'npu_timing.json').read_text(encoding='utf-8'))
    validate_timing(info, timing, mode, device, warmup, repeats)
    # Both are actual NPU outputs; no expected file is used to manufacture either one.
    check(directory, output / 'bootstrap')
    check(directory, output)
    for name in OUTPUT_FILES:
        if (output / name).read_bytes() != (output / 'bootstrap' / name).read_bytes():
            raise ValueError(f'首末实际NPU输出字节不一致：{name}')
    record = {'status': 'device_repeated_call_completed_file_checks_passed', 'device': device,
              'command': command, 'mode': mode, 'queries': info['num_queries'], 'docs': info['num_docs'],
              'k': info['k'], 'warmup': warmup, 'repeats': repeats,
              'input_sha256': info['input_sha256'],
              'output_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in OUTPUT_FILES},
              'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(), 'platform': platform.platform(),
              'scope': 'prepared-input repeated experiment; includes per-iteration API preparation; not complete retrieval or an automatic speedup verdict',
              'validation_policy': timing['full_repeat_validation']}
    (output / 'run_metadata.json').write_bytes((json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(f'实际调用完成，首末独立公式与排名检查通过；模式={mode}，批次中位数={timing["median_batch_path_ms"]:.6g}ms。', flush=True)
    print('launch/sync包含主机提交和同步，不是纯内核时间；报告保存全部样本。不作完整检索或加速达标结论。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runner', default=str(ROOT / 'results/pipeline_host_build/bm25_pipeline_benchmark'))
    parser.add_argument('--vendor-dir', default=str(ROOT / 'results/custom_opp/vendors'))
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--mode', choices=('resident', 'with_transfers'), default='resident')
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.output), Path(args.runner), Path(args.vendor_dir),
            args.device, args.warmup, args.repeats, args.mode)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'FAILED：{error}；停止，不生成CPU替代输出。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
