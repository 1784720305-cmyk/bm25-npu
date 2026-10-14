"""读取已准备的Top-K尺寸/指纹，再启动实际C++ ACLNN调用器；无CPU替代输出。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_result import read_floats
from standalone.topk_fixture import validate


def prepare_command(directory, output, runner, device):
    directory, output, runner = directory.resolve(), output.resolve(), runner.resolve()
    if output.exists():
        raise ValueError('输出目录已存在，请选择新目录，避免混入旧结果')
    if not runner.is_file():
        raise ValueError(f'调用器不存在，请先在黄区编译：{runner}')
    if type(device) is not int or not 0 <= device <= 2**31 - 1:
        raise ValueError('设备编号应为非负int32整数')
    info = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    queries, docs, k = info['num_queries'], info['num_docs'], info['k']
    scores = read_floats(directory / 'scores.bin')
    validate(scores, queries, docs, k)
    if info['input_shape'] != [queries, docs] or info['output_shape'] != [queries, k]:
        raise ValueError('元信息形状不一致')
    if info['ties'] != 'exact_FP32_score_desc_doc_id_asc':
        raise ValueError('同分规则与当前算子不一致')
    if hashlib.sha256((directory / 'scores.bin').read_bytes()).hexdigest() != info['scores_sha256']:
        raise ValueError('输入总分指纹改变，停止')
    command = [str(runner), str(directory), str(output), str(queries), str(docs), str(k), str(device)]
    return command, info


def run(directory, output, runner, device):
    command, info = prepare_command(directory, output, runner, device)
    print('Top-K输入尺寸/指纹检查通过；将执行实际NPU调用器：', flush=True)
    print(json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)  # Streams actual stages/errors; failure is never replaced with CPU output.
    output = output.resolve()
    expected_bytes = info['num_queries'] * info['k'] * 4
    fingerprints = {}
    for name in ('top_scores.bin', 'top_ids.bin'):
        data = (output / name).read_bytes()
        if len(data) != expected_bytes:
            raise ValueError(f'调用器返回成功但输出尺寸错误：{name}')
        fingerprints[name] = hashlib.sha256(data).hexdigest()
    record = {
        'status': 'device_call_completed_ranking_not_yet_verified',
        'device': device, 'num_queries': info['num_queries'], 'num_docs': info['num_docs'], 'k': info['k'],
        'command': command, 'input_sha256': info['scores_sha256'],
        'input_origin': info['input_origin'], 'output_sha256': fingerprints,
    }
    (output / 'run_metadata.json').write_bytes((json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print('调用器执行完成，输出尺寸检查通过；下一步仍需topk_fixture.py check核对排名。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(ROOT / 'results/topk_input'))
    parser.add_argument('--output', required=True, help='新的输出目录，保存top_scores.bin和top_ids.bin')
    parser.add_argument('--runner', default=str(ROOT / 'results/aclnn_topk_host_build/bm25_topk_runner'))
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.output), Path(args.runner), args.device)
    except subprocess.CalledProcessError as error:
        print(f'FAILED：NPU调用器退出码={error.returncode}，停止，不生成CPU替代输出。', file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as error:
        print(f'FAILED：{error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
