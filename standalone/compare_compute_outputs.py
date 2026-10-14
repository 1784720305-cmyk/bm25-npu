"""直接比较相同输入的CPU/NPU完整结果；只检查文件，不执行设备或判断加速。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_result import compare, read_floats
from standalone.pipeline_fixture import check, validate_input

CPU_STATUSES = {'cpu_computed_correctness_not_yet_checked'}
NPU_STATUSES = {'device_pipeline_call_completed_correctness_not_yet_checked',
                'device_repeated_call_completed_file_checks_passed'}
OUTPUT_FILES = ('contributions.bin', 'scores.bin', 'top_scores.bin', 'top_ids.bin')


def validate_producer(directory, info, statuses):
    record = json.loads((directory / 'run_metadata.json').read_text(encoding='utf-8'))
    if record['status'] not in statuses or record['input_sha256'] != info['input_sha256']:
        raise ValueError(f'生产者记录类型或输入指纹不一致：{directory}')
    for name in OUTPUT_FILES:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != record['output_sha256'][name]:
            raise ValueError(f'输出指纹与运行记录不一致：{directory / name}')
    return record


def cross_check(directory, cpu, npu):
    info, _ = validate_input(directory)
    cpu_record = validate_producer(cpu, info, CPU_STATUSES)
    npu_record = validate_producer(npu, info, NPU_STATUSES)
    check(directory, cpu)
    check(directory, npu)
    errors = {}
    for name in OUTPUT_FILES[:3]:
        errors[name] = compare(read_floats(npu / name), read_floats(cpu / name))
    if (cpu / 'top_ids.bin').read_bytes() != (npu / 'top_ids.bin').read_bytes():
        import struct
        count = info['num_queries'] * info['k']
        cpu_ids = struct.unpack('<' + 'i' * count, (cpu / 'top_ids.bin').read_bytes())
        npu_ids = struct.unpack('<' + 'i' * count, (npu / 'top_ids.bin').read_bytes())
        index = next(i for i, (c, n) in enumerate(zip(cpu_ids, npu_ids)) if c != n)
        raise ValueError(f'CPU/NPU排名不一致：查询{index // info["k"]}、位置{index % info["k"]}，'
                         f'CPU编号={cpu_ids[index]}，NPU编号={npu_ids[index]}；停止，不放宽同分规则')
    record = {'status': 'same_input_cpu_npu_file_correctness_passed',
              'scope': 'file comparison only; producer records are checked, actual device execution needs corresponding logs; no timing/speedup conclusion',
              'num_queries': info['num_queries'], 'num_docs': info['num_docs'], 'k': info['k'],
              'input_sha256': info['input_sha256'], 'cpu_directory': str(cpu.resolve()),
              'npu_directory': str(npu.resolve()), 'max_cpu_npu_absolute_errors': errors,
              'all_top_ids_byte_identical': True,
              'top_scores_byte_identical': (cpu / 'top_scores.bin').read_bytes() == (npu / 'top_scores.bin').read_bytes(),
              'cpu_output_sha256': cpu_record['output_sha256'], 'npu_output_sha256': npu_record['output_sha256']}
    print(f'CPU/NPU直接核对通过：全部{info["num_queries"]}条查询、每条{info["k"]}个编号及顺序严格一致。')
    print('贡献、完整总分、Top-K分数在原容差内一致：' + json.dumps(errors, ensure_ascii=False))
    print('此结论只比较结果，不证明加速；设备执行依据对应运行日志。')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--cpu', required=True)
    parser.add_argument('--npu', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = Path(args.report)
    if report.exists():
        raise ValueError('报告已存在，请选择新文件')
    record = cross_check(Path(args.input), Path(args.cpu), Path(args.npu))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes((json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
