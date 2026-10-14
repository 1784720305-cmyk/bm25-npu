"""检查串联输入并启动实际C++三算子调用程序；CPU不生成设备替代结果。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.pipeline_fixture import validate_input


def prepare_command(directory, output, runner, vendors, device):
    directory, output, runner, vendors = (p.resolve() for p in (directory, output, runner, vendors))
    if output.exists():
        raise ValueError('输出目录已存在，请选新目录')
    if not runner.is_file():
        raise ValueError('串联调用器不存在，请先编译')
    if type(device) is not int or not 0 <= device <= 2**31-1:
        raise ValueError('设备编号非法')
    for vendor in ('bm25', 'bm25_accumulate', 'bm25_topk'):
        if not (vendors / vendor / 'op_api/lib/libcust_opapi.so').is_file():
            raise ValueError(f'实际已安装接口库缺失：{vendor}')
    info, _ = validate_input(directory)
    command = [str(runner), str(directory), str(output), str(info['num_queries']), str(info['num_docs']),
               str(info['k']), str(device), str(vendors)]
    return command, info


def run(directory, output, runner, vendors, device):
    command, info = prepare_command(directory, output, runner, vendors, device)
    print('串联输入指纹/尺寸检查通过；将启动同进程实际NPU调用程序：', flush=True)
    print(json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)
    output = output.resolve()
    counts = {'contributions.bin': info['value_count'], 'scores.bin': info['group_count'],
              'top_scores.bin': info['num_queries'] * info['k'], 'top_ids.bin': info['num_queries'] * info['k']}
    hashes = {}
    for name, count in counts.items():
        data = (output / name).read_bytes()
        if len(data) != count * 4:
            raise ValueError(f'调用器输出尺寸不匹配：{name}')
        hashes[name] = hashlib.sha256(data).hexdigest()
    record = {
        'status': 'device_pipeline_call_completed_correctness_not_yet_checked',
        'device': device, 'command': command, 'queries': info['num_queries'], 'docs': info['num_docs'], 'k': info['k'],
        'input_sha256': info['input_sha256'], 'output_sha256': hashes,
        'scope': 'one process, one stream; no intermediate host roundtrips; final debug downloads included; not a benchmark',
    }
    (output / 'run_metadata.json').write_bytes((json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print('串联调用器完成且输出尺寸通过；请用pipeline_fixture.py check核对三段实际结果。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(ROOT / 'results/pipeline_input'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--runner', default=str(ROOT / 'results/pipeline_host_build/bm25_pipeline_runner'))
    parser.add_argument('--vendor-dir', default=str(ROOT / 'results/custom_opp/vendors'))
    parser.add_argument('--device', type=int, default=2)
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.output), Path(args.runner), Path(args.vendor_dir), args.device)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'FAILED：{error}；停止，不生成CPU替代输出。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
