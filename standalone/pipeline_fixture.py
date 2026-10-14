"""准备同进程BM25计算输入，核对最终下载结果；不执行或替代NPU。"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_result import read_floats, compare, validate_fixture
from standalone.accumulate_fixture import fp32, independent_scores
from standalone.topk_fixture import reference, validate

INPUT_FILES = ('tf.bin', 'doc_norm.bin', 'idf.bin', 'offsets.bin')


def prepare(output, k):
    if output.exists():
        raise ValueError('输出目录已存在，请选择新目录')
    fixture = ROOT / 'fixtures/contribution'
    expected = validate_fixture(fixture)
    records = json.loads((fixture / 'records.json').read_text(encoding='utf-8'))
    contract = json.loads((ROOT / 'fixtures/compute_contract_small.json').read_text(encoding='utf-8'))
    queries, docs = len(contract['queries']), len(contract['documents'])
    cpu_scores = independent_scores(contract['documents'], contract['queries'])
    validate(cpu_scores, queries, docs, k)
    fields = [read_floats(fixture / name) for name in INPUT_FILES[:3]]
    rows = records['rows']
    if len(rows) != len(expected):
        raise ValueError('记录说明与贡献样本长度不匹配')
    groups = []
    for i, row in enumerate(rows):
        q, d = row['query'], row['doc']
        if type(q) is not int or type(d) is not int or not (0 <= q < queries and 0 <= d < docs):
            raise ValueError('查询或文章编号非法')
        if any(fields[j][i] != row[key] for j, key in enumerate(('tf', 'doc_norm', 'idf'))):
            raise ValueError('记录说明与二进制输入不同')
        groups.append(q * docs + d)
    # Stable ordering keeps each query/document's original query-term contribution order.
    order = sorted(range(len(rows)), key=groups.__getitem__)
    counts = [0] * (queries * docs)
    for group in groups:
        counts[group] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    sorted_expected = [expected[index] for index in order]
    sums = []
    for begin, end in zip(offsets, offsets[1:]):
        total = 0.0
        for value in sorted_expected[begin:end]:
            total = fp32(total + value)
        sums.append(total)
    compare(sums, cpu_scores)
    payload = {}
    for name, field in zip(INPUT_FILES[:3], fields):
        values = [field[index] for index in order]
        payload[name] = struct.pack('<' + 'f' * len(values), *values)
    payload['offsets.bin'] = struct.pack('<' + 'q' * len(offsets), *offsets)
    payload['expected_contributions.bin'] = struct.pack('<' + 'f' * len(sorted_expected), *sorted_expected)
    payload['expected_scores.bin'] = struct.pack('<' + 'f' * len(cpu_scores), *cpu_scores)
    info = {
        'contract': 'bm25_device_pipeline_v1', 'num_queries': queries, 'num_docs': docs, 'k': k,
        'value_count': len(order), 'group_count': queries * docs, 'offset_count': len(offsets),
        'k1': 1.2, 'ties': 'exact_FP32_score_desc_doc_id_asc', 'source_order': order,
        'layout': 'contributions grouped by query*num_docs+doc before upload',
        'input_origin': 'packaged_TF_norm_IDF_reordered_before_NPU_compute',
        'input_sha256': {name: hashlib.sha256(payload[name]).hexdigest() for name in INPUT_FILES},
        'reference_sha256': {name: hashlib.sha256(payload[name]).hexdigest()
                             for name in ('expected_contributions.bin', 'expected_scores.bin')},
        'device_pipeline_verified': False,
    }
    output.mkdir(parents=True)
    for name, data in payload.items():
        (output / name).write_bytes(data)
    (output / 'metadata.json').write_bytes((json.dumps(info, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(f'串联输入准备通过：{len(order)}条TF/长度项/IDF，{queries*docs}组，K={k}。')
    print('先整理输入顺序；贡献和总分仍由NPU计算。CPU期望文件只给核对使用。尚未执行串联程序。')


def validate_input(directory):
    info = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if info['contract'] != 'bm25_device_pipeline_v1' or info['k1'] != 1.2:
        raise ValueError('计算合同或k1不匹配')
    if info['ties'] != 'exact_FP32_score_desc_doc_id_asc':
        raise ValueError('同分规则不匹配')
    q, d, k = info['num_queries'], info['num_docs'], info['k']
    if any(type(n) is not int for n in (q, d, k)) or not (q > 0 and d > 0 and q*d <= 2**31-1):
        raise ValueError('查询/文章数量非法，第一版总组数须<=INT32_MAX')
    if not 1 <= k <= min(d, 32):
        raise ValueError('K非法')
    for name in INPUT_FILES:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != info['input_sha256'][name]:
            raise ValueError(f'输入指纹改变：{name}')
    tf, norm, idf = [read_floats(directory / name) for name in INPUT_FILES[:3]]
    if len(tf) != info['value_count'] or not (0 < len(tf) <= 2**31-1) or len(tf) != len(norm) or len(tf) != len(idf):
        raise ValueError('TF/长度项/IDF数量不匹配')
    for t, n, w in zip(tf, norm, idf):
        if t <= 0 or t > 2**24 or int(t) != t or n < 0 or w < 0:
            raise ValueError('TF/长度项/IDF不符合合同')
    raw = (directory / 'offsets.bin').read_bytes()
    if len(raw) != (q*d+1)*8:
        raise ValueError('分组边界尺寸不匹配')
    offsets = struct.unpack('<' + 'q' * (q*d+1), raw)
    if offsets[0] != 0 or offsets[-1] != len(tf) or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError('分组边界非法')
    if info['group_count'] != q*d or info['offset_count'] != len(offsets):
        raise ValueError('元信息组数不匹配')
    return info, offsets


def check(directory, actual):
    info, offsets = validate_input(directory)
    for name in ('expected_contributions.bin', 'expected_scores.bin'):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != info['reference_sha256'][name]:
            raise ValueError(f'参考文件指纹改变：{name}')
    contributions = read_floats(actual / 'contributions.bin')
    error_c = compare(contributions, read_floats(directory / 'expected_contributions.bin'))
    scores = read_floats(actual / 'scores.bin')
    error_s = compare(scores, read_floats(directory / 'expected_scores.bin'))
    if any(v < 0 for v in contributions + scores):
        raise ValueError('设备贡献或总分为负')
    sums = []
    for begin, end in zip(offsets, offsets[1:]):
        total = 0.0
        for value in contributions[begin:end]:
            total = fp32(total + value)
        sums.append(total)
    compare(scores, sums)  # Also verify accumulation from this very device's contributions.
    ids, selected = reference(scores, info['num_queries'], info['num_docs'], info['k'])
    if (actual / 'top_ids.bin').read_bytes() != struct.pack('<' + 'i' * len(ids), *ids):
        raise ValueError('Top-K编号或顺序错误')
    if (actual / 'top_scores.bin').read_bytes() != struct.pack('<' + 'f' * len(selected), *selected):
        raise ValueError('Top-K分数与设备总分的字节不一致')
    print(f'串联文件核对通过：{len(contributions)}条贡献、{len(scores)}组总分、{len(ids)}个Top-K编号和分数。')
    print(f'贡献最大绝对误差={error_c:.8g}，总分最大绝对误差={error_s:.8g}，Top-K编号和分数字节一致。')
    for query in range(min(info['num_queries'], 6)):
        k = info['k']
        print(f'查询{query}文章编号：{ids[query*k:(query+1)*k]}')
    print('还需实际串联日志证明设备执行；检查器不执行NPU，不作性能达标结论。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', default=str(ROOT / 'results/pipeline_input'))
    p.add_argument('--k', type=int, default=3)
    c = sub.add_parser('check')
    c.add_argument('--input', default=str(ROOT / 'results/pipeline_input'))
    c.add_argument('--actual', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(Path(args.output), args.k)
    else:
        check(Path(args.input), Path(args.actual))


if __name__ == '__main__':
    main()
