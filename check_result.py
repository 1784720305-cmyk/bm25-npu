"""无需第三方包：检查打包样本，或核对实际设备输出。"""
import argparse
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent


def read_floats(path):
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"不是非空FP32数组：{path}")
    values = struct.unpack('<' + 'f' * (len(raw) // 4), raw)
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"包含NaN或无穷大：{path}")
    return values


def compare(actual, expected):
    if len(actual) != len(expected):
        raise ValueError("结果记录数与期望值不同")
    for i, (a, e) in enumerate(zip(actual, expected)):
        if abs(a - e) > 1e-6 + 2e-6 * abs(e):
            raise ValueError(f"记录{i}不一致：实际={a}，期望={e}")
    return max(abs(a - e) for a, e in zip(actual, expected))


def validate_fixture(directory):
    tf = read_floats(directory / 'tf.bin')
    norm = read_floats(directory / 'doc_norm.bin')
    idf = read_floats(directory / 'idf.bin')
    expected = read_floats(directory / 'expected.bin')
    spec = json.loads((directory / 'singleop.json').read_text(encoding='utf-8'))
    records = json.loads((directory / 'records.json').read_text(encoding='utf-8'))
    if len({len(tf), len(norm), len(idf), len(expected), records['record_count']}) != 1:
        raise ValueError('样本长度不一致')
    if spec[0]['op'] != 'Bm25Contribution':
        raise ValueError('算子名不匹配')
    if any(d['shape'] != [len(tf)] or d['type'] != 'float32' or d['format'] != 'ND'
           for d in spec[0]['input_desc'] + spec[0]['output_desc']):
        raise ValueError('模型描述与样本不匹配')
    for t, n, w in zip(tf, norm, idf):
        if t <= 0 or t > 2**24 or int(t) != t or n < 0 or w < 0:
            raise ValueError('输入值不符合计算约定')
    formula = [w * (t * 2.2) / (t + n) for t, n, w in zip(tf, norm, idf)]
    compare(expected, formula)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', default=str(ROOT / 'fixtures/contribution'))
    parser.add_argument('--actual', help='真实运行产生的FP32二进制输出')
    args = parser.parse_args()
    expected = validate_fixture(Path(args.fixture))
    print(f'样本检查通过：{len(expected)}条记录，公式和模型尺寸匹配。')
    if args.actual:
        error = compare(read_floats(Path(args.actual)), expected)
        print(f'输出数值通过：最大绝对误差={error:.8g}')
        print('设备执行成功还需对应的A5日志；检查器本身不执行NPU。')
    else:
        print('仅检查打包样本，没有执行NPU或验证设备输出。')


if __name__ == '__main__':
    main()
