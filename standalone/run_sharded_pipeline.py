"""执行实际四段NPU计算，独立核对各块候选和最终排名；失败不补CPU输出。"""
import argparse
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_result import read_floats
from standalone.pipeline_fixture import check
from standalone.run_pipeline import prepare_command
from standalone.run_pipeline_benchmark import OUTPUT_FILES, validate_timing
from standalone.prepare_sharded_runtime import validate_runtime, VARIANT
from standalone.validate_topk_merge import digest, write_json

FILES = OUTPUT_FILES + ('partial_scores.bin', 'partial_ids.bin')


def check_candidates(info, actual, shards):
    q, docs, k = (info[key] for key in ('num_queries', 'num_docs', 'k'))
    width = docs // shards
    scores = read_floats(actual / 'scores.bin')
    score_bytes = (actual / 'scores.bin').read_bytes()
    partial = (actual / 'partial_scores.bin').read_bytes()
    local = (actual / 'partial_ids.bin').read_bytes()
    count = q * shards * k
    if len(partial) != count * 4 or len(local) != count * 4 or len(scores) != q * docs:
        raise ValueError('设备分块候选/完整分数尺寸不匹配')
    ids = struct.unpack('<' + 'i' * count, local)
    for query in range(q):
        for shard in range(shards):
            base = query * docs + shard * width
            selected = sorted(range(width), key=lambda i: (-scores[base + i], i))[:k]
            offset = (query * shards + shard) * k
            if list(ids[offset:offset + k]) != selected:
                raise ValueError(f'NPU局部候选排名不一致：查询{query}、块{shard}')
            expected = b''.join(score_bytes[(base + i) * 4:(base + i + 1) * 4] for i in selected)
            if partial[offset * 4:(offset + k) * 4] != expected:
                raise ValueError('局部候选分数与完整文章总分的原始字节不同')
    # Additionally validate the final merge against the actual full FP32 device totals,
    # not just the independent-formula tolerance check in pipeline_fixture.check.
    final_ids = (actual / 'top_ids.bin').read_bytes()
    final_scores = (actual / 'top_scores.bin').read_bytes()
    expected_ids, expected_scores = [], []
    for query in range(q):
        base = query * docs
        selected = sorted(range(docs), key=lambda i: (-scores[base + i], i))[:k]
        expected_ids.extend(selected)
        expected_scores.extend(score_bytes[(base + i) * 4:(base + i + 1) * 4] for i in selected)
    if final_ids != struct.pack('<' + 'i' * len(expected_ids), *expected_ids) or final_scores != b''.join(expected_scores):
        raise ValueError('两级Top-K与全部实际FP32文章总分的完整排序不同')


def run(directory, output, runner, vendors, device, warmup, repeats, mode, shards):
    if type(warmup) is not int or type(repeats) is not int or not (1 <= warmup <= 10000 and 1 <= repeats <= 10000):
        raise ValueError('预热/重复次数须在1..10000')
    if mode not in ('resident', 'with_transfers'):
        raise ValueError('仅支持正常四段单次同步的resident/with_transfers模式')
    command, info = prepare_command(directory, output, runner, vendors, device)
    record, scratch, custom_path = validate_runtime(vendors.resolve(), info, shards)
    command.extend([str(warmup), str(repeats), mode, str(shards)])
    environment = os.environ.copy()
    environment['ASCEND_CUSTOM_OPP_PATH'] = custom_path
    environment['BM25_EXPECT_TOPK_SCRATCH_BYTES'] = str(scratch)
    print(f'四段运行库/工程指纹通过；S={shards}，局部Top-K额外工作区={scratch}字节。', flush=True)
    print('实际NPU命令：' + json.dumps(command, ensure_ascii=False), flush=True)
    # The C++ executable consumes only original input arrays; no expected/candidate CPU file is passed.
    subprocess.run(command, check=True, env=environment)
    output = output.resolve()
    timing = json.loads((output / 'npu_timing.json').read_text(encoding='utf-8'))
    validate_timing(info, timing, mode, device, warmup, repeats)
    if timing.get('shards') != shards or timing.get('pipeline_stages') != 4 or timing.get('synchronization_policy') != 'synchronize after the entire four-operator chain':
        raise ValueError('调用器未报告本次四段单次同步路径')
    capacity = timing['workspace_capacity_bytes']
    if len(capacity) != 4 or any(type(v) is not int or v < 0 for v in capacity) or capacity[2] < capacity[0] + scratch:
        raise ValueError('四段工作区或分块Top-K额外工作区不匹配')
    if validate_runtime(vendors.resolve(), info, shards) != (record, scratch, custom_path):
        raise ValueError('运行过程中库/头文件/工程改变')
    for actual in (output / 'bootstrap', output):
        check(directory, actual)
        check_candidates(info, actual, shards)
    for name in FILES:
        if (output / name).read_bytes() != (output / 'bootstrap' / name).read_bytes():
            raise ValueError(f'首末实际设备输出不一致：{name}')
    metadata = {'status': 'device_repeated_call_completed_file_checks_passed', 'topk_variant': VARIANT,
                'pipeline_stages': 4, 'shards': shards, 'device': device, 'mode': mode,
                'queries': info['num_queries'], 'docs': info['num_docs'], 'k': info['k'],
                'warmup': warmup, 'repeats': repeats, 'command': command,
                'input_sha256': info['input_sha256'], 'output_sha256': {name: digest(output / name) for name in FILES},
                'runner_sha256': digest(runner), 'runtime_installation': record,
                'parallel_user_scratch_bytes': scratch, 'platform': platform.platform(),
                'validation_policy': timing['full_repeat_validation'],
                'scope': 'prepared-input four-stage device chain; candidate generation and ID remapping on NPU, no intermediate host roundtrips; debug checks excluded; not complete retrieval or an automatic speedup verdict'}
    write_json(output / 'run_metadata.json', metadata)
    print('四段实际调用完成，首末贡献/总分/局部候选/最终排名及重复字节核对通过。', flush=True)
    print(f'模式={mode}，批次中位数={timing["median_batch_path_ms"]:.6g}ms；含每轮接口准备、四次提交及同步，非纯内核时间；仍需CPU直接交叉和性能对照。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runner', default=str(ROOT / 'results/sharded_host_build/bm25_sharded_benchmark'))
    parser.add_argument('--vendor-dir', default=str(ROOT / 'results/sharded_runtime/vendors'))
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--mode', choices=('resident', 'with_transfers'), default='resident')
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.output), Path(args.runner), Path(args.vendor_dir), args.device,
            args.warmup, args.repeats, args.mode, args.shards)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f'FAILED：{error}；停止，不产生CPU替代输出或通过记录。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
