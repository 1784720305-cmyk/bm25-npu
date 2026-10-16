"""核对相同输入后启动优化C++ CPU计算程序；不读取期望值生成输出。"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.pipeline_fixture import validate_input


def run(directory, output, runner, threads, warmup, repeats, topk_document_shards=0):
    directory, output, runner = (p.resolve() for p in (directory, output, runner))
    if output.exists() or not runner.is_file():
        raise ValueError('输出目录已存在或C++ CPU调用器不存在')
    if any(type(n) is not int for n in (threads, warmup, repeats)) or not (
            1 <= threads <= 65536 and 0 <= warmup <= 10000 and 1 <= repeats <= 10000):
        raise ValueError('线程数/预热数/重复数非法')
    info, _ = validate_input(directory)
    if type(topk_document_shards) is not int or not 0 <= topk_document_shards <= min(threads, info['num_docs']):
        raise ValueError('CPU Top-K文章分块须为0（原版）或不超过线程数、文章数的正整数')
    command = [str(runner), str(directory), str(output), str(info['num_queries']), str(info['num_docs']),
               str(info['k']), str(threads), str(warmup), str(repeats)]
    if topk_document_shards:
        command.append(str(topk_document_shards))
    print('输入检查通过，将执行C++ CPU贡献→累加→Top-K：', flush=True)
    print(json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)
    hashes = {}
    counts = {'contributions.bin': info['value_count'], 'scores.bin': info['group_count'],
              'top_scores.bin': info['num_queries'] * info['k'], 'top_ids.bin': info['num_queries'] * info['k']}
    for name, count in counts.items():
        data = (output / name).read_bytes()
        if len(data) != count * 4:
            raise ValueError(f'CPU输出尺寸不匹配：{name}')
        hashes[name] = hashlib.sha256(data).hexdigest()
    timing = json.loads((output / 'cpu_timing.json').read_text(encoding='utf-8'))
    if timing['status'] != 'cpu_computed_correctness_not_yet_checked':
        raise ValueError('CPU运行报告状态不匹配')
    if type(timing.get('topk_document_shards', 0)) is not int or timing.get('topk_document_shards', 0) != topk_document_shards:
        raise ValueError('CPU调用器未执行指定的Top-K文章分块版本')
    record = {'status': 'cpu_computed_correctness_not_yet_checked', 'command': command,
              'topk_document_shards': topk_document_shards,
              'input_sha256': info['input_sha256'], 'output_sha256': hashes, 'platform': platform.platform(),
              'scope': 'CPU computation reference only; not complete retrieval or a CPU/NPU speedup comparison'}
    (output / 'run_metadata.json').write_bytes((json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print('CPU调用完成；继续用pipeline_fixture.py check核对。当前时间不能与此前NPU首次诊断时间计算加速比。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runner', default=str(ROOT / 'results/cpu_host_build/bm25_cpu_runner'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--topk-document-shards', type=int, default=0,
                        help='0保留按查询并行；正整数按文章分块选K并合并，不超过线程数及文章数')
    args = parser.parse_args()
    run(Path(args.input), Path(args.output), Path(args.runner), args.threads, args.warmup, args.repeats, args.topk_document_shards)


if __name__ == '__main__':
    main()
