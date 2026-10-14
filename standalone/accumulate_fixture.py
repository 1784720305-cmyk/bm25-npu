"""用贡献输出准备分组累加输入，或核对累加输出。标准库，不运行NPU。"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_result import read_floats, compare


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def group(values, rows, num_queries, num_docs):
    if len(values) != len(rows):
        raise ValueError('贡献记录数不匹配')
    buckets = [[] for _ in range(num_queries * num_docs)]
    for value, row in zip(values, rows):
        query, doc = row['query'], row['doc']
        if not (0 <= query < num_queries and 0 <= doc < num_docs):
            raise ValueError('查询/文章编号越界')
        if not math.isfinite(value) or value < 0:
            raise ValueError('贡献必须有限且非负')
        buckets[query * num_docs + doc].append(value)
    packed, offsets, sums = [], [0], []
    for bucket in buckets:
        total = 0.0
        for value in bucket:
            total = fp32(total + value)
        packed.extend(bucket)
        offsets.append(len(packed))
        sums.append(total)
    return packed, offsets, sums


def independent_scores(documents, queries):
    counts = [Counter(doc) for doc in documents]
    avgdl = sum(map(len, documents)) / len(documents)
    result = []
    for query in queries:
        for doc, counts_doc in zip(documents, counts):
            total = 0.0
            for term in query:  # Repeated query terms count repeatedly.
                tf = counts_doc[term]
                if not tf:
                    continue
                df = sum(term in c for c in counts)
                idf = math.log1p((len(documents) - df + .5) / (df + .5))
                norm = 1.2 * (1 - .75 + .75 * len(doc) / avgdl)
                total += idf * (tf * 2.2) / (tf + norm)
            result.append(fp32(total))
    return result


def prepare(contributions, output, cpu_reference=False):
    if output.exists():
        raise ValueError('输出目录已存在，请指定新目录，避免混入旧结果')
    fixture = ROOT / 'fixtures/contribution'
    contract = json.loads((ROOT / 'fixtures/compute_contract_small.json').read_text(encoding='utf-8'))
    records = json.loads((fixture / 'records.json').read_text(encoding='utf-8'))
    values = read_floats(contributions)
    compare(values, read_floats(fixture / 'expected.bin'))
    num_queries, num_docs = len(contract['queries']), len(contract['documents'])
    packed, offsets, sums = group(values, records['rows'], num_queries, num_docs)
    expected = independent_scores(contract['documents'], contract['queries'])
    compare(sums, expected)
    output.mkdir(parents=True)
    (output / 'values.bin').write_bytes(struct.pack('<' + 'f' * len(packed), *packed))
    (output / 'offsets.bin').write_bytes(struct.pack('<' + 'q' * len(offsets), *offsets))
    (output / 'expected.bin').write_bytes(struct.pack('<' + 'f' * len(expected), *expected))
    info = {
        'num_queries': num_queries, 'num_docs': num_docs, 'group_count': len(sums),
        'value_count': len(values), 'offset_count': len(offsets),
        'layout': 'scores[query * num_docs + doc]',
        'input_origin': 'CPU_REFERENCE_ONLY' if cpu_reference else 'user_supplied_contribution_output',
        'contribution_sha256': hashlib.sha256(contributions.read_bytes()).hexdigest(),
        'contribution_path': str(contributions),
        'npu_accumulation_verified': False,
    }
    (output / 'metadata.json').write_bytes((json.dumps(info, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    print(f'输入准备通过：{len(values)}条贡献→{len(sums)}组总分，空组也保留。')
    print('CPU分组核对独立公式通过；尚未执行NPU累加。')
    if cpu_reference:
        print('本次输入来自CPU参考，只用于本机检查，不算设备串联验证。')


def check(directory, actual):
    info = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    values = read_floats(actual)
    if len(values) != info['group_count']:
        raise ValueError('输出总分数量错误')
    error = compare(values, read_floats(directory / 'expected.bin'))
    print(f'文章总分数值核对通过：{len(values)}组，最大绝对误差={error:.8g}')
    for query in range(info['num_queries']):
        n = info['num_docs']
        print(f'查询{query}：{list(values[query*n:(query+1)*n])}')
    print('此检查只核对文件；必须结合实机运行日志确认NPU累加执行成功。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--contributions', required=True, help='第一次NPU实际贡献输出文件')
    p.add_argument('--output', default=str(ROOT/'results/accumulate_input'))
    p.add_argument('--cpu-reference', action='store_true', help='仅用于本机工具检查，明确标记CPU参考输入')
    c = sub.add_parser('check')
    c.add_argument('--directory', default=str(ROOT/'results/accumulate_input'))
    c.add_argument('--actual', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(Path(args.contributions), Path(args.output), args.cpu_reference)
    else:
        check(Path(args.directory), Path(args.actual))


if __name__ == '__main__':
    main()
