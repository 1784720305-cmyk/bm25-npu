"""准备自研Top-K输入/核对输出。Python只做数据与验证，不执行或替代NPU。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_result import read_floats, compare
from standalone.accumulate_fixture import independent_scores

MAX_K = 32


def validate(scores, queries, docs, k):
    if any(type(n) is not int for n in (queries, docs, k)):
        raise ValueError('查询数、文章数、K必须是整数')
    if not (1 <= queries <= 2**31 - 1 and 1 <= docs <= 2**31 - 1):
        raise ValueError('查询数、文章数必须在1..INT32_MAX')
    if not (1 <= k <= min(docs, MAX_K)):
        raise ValueError('第一版要求1 <= K <= min(文章数, 32)，不自动截断K')
    if len(scores) != queries * docs:
        raise ValueError('总分数量与查询数×文章数不匹配')
    if any(not math.isfinite(score) or score < 0 for score in scores):
        raise ValueError('BM25总分必须有限且非负')


def reference(scores, queries, docs, k):
    validate(scores, queries, docs, k)
    ids, selected = [], []
    for query in range(queries):
        row = scores[query * docs:(query + 1) * docs]
        # Independent full-sort oracle; device uses a bounded insertion list instead.
        chosen = sorted(range(docs), key=lambda doc: (-row[doc], doc))[:k]
        ids.extend(chosen)
        selected.extend(row[doc] for doc in chosen)
    return ids, selected


def write_input(scores, queries, docs, k, output, origin, source):
    if output.exists():
        raise ValueError('输出目录已存在，请指定新目录，避免混入旧结果')
    validate(scores, queries, docs, k)
    packed = struct.pack('<' + 'f' * len(scores), *scores)
    # Check in the same FP32 domain that the NPU will rank, including rounding-induced ties.
    fp32_scores = struct.unpack('<' + 'f' * len(scores), packed)
    expected_ids, expected_scores = reference(fp32_scores, queries, docs, k)
    info = {
        'num_queries': queries, 'num_docs': docs, 'k': k,
        'input_shape': [queries, docs], 'output_shape': [queries, k],
        'ties': 'exact_FP32_score_desc_doc_id_asc',
        'input_origin': origin, 'source': source,
        'scores_sha256': hashlib.sha256(packed).hexdigest(),
        'npu_topk_verified': False,
    }
    output.mkdir(parents=True)
    (output / 'scores.bin').write_bytes(packed)
    (output / 'expected_ids.bin').write_bytes(struct.pack('<' + 'i' * len(expected_ids), *expected_ids))
    (output / 'expected_scores.bin').write_bytes(struct.pack('<' + 'f' * len(expected_scores), *expected_scores))
    (output / 'metadata.json').write_bytes((json.dumps(info, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(f'Top-K输入准备通过：{queries}条查询×{docs}篇文章，K={k}，共{queries*k}个输出。')
    print('仅准备输入和CPU期望值，尚未执行或验证NPU Top-K。')
    return expected_ids


def prepare(scores_path, k, output):
    contract = json.loads((ROOT / 'fixtures/compute_contract_small.json').read_text(encoding='utf-8'))
    scores = read_floats(scores_path)
    # Confirm the user-supplied accumulation result before ranking that exact FP32 input.
    compare(scores, independent_scores(contract['documents'], contract['queries']))
    ids = write_input(scores, len(contract['queries']), len(contract['documents']), k, output,
                      'user_supplied_accumulation_output', str(scores_path))
    for query in range(len(contract['queries'])):
        print(f'查询{query}期望文章编号：{ids[query*k:(query+1)*k]}')


def edge_fixture(k, output):
    # Synthetic selection inputs only; they are not BM25 benchmark data.
    from random import Random
    random = Random(20261014)
    docs, queries = 40, 513  # >256 rows exercises grid-stride/multiple blocks when run on device.
    rows = []
    for query in range(queries):
        if query % 5 == 0:
            row = [0.0] * docs
        elif query % 5 == 1:
            row = [1.0] * docs
        elif query % 5 == 2:
            row = [float(doc) for doc in range(docs)]
        elif query % 5 == 3:
            row = [1.0] * docs
            row[-1] = struct.unpack('<f', struct.pack('<I', 0x3f800001))[0]  # One FP32 step above 1.
        else:
            row = [float(random.randrange(9)) for _ in range(docs)]
        rows.extend(row)
    write_input(rows, queries, docs, k, output, 'SYNTHETIC_TOPK_EDGE_ONLY', 'deterministic edge fixture')


def check(directory, scores_path, ids_path):
    info = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    packed = (directory / 'scores.bin').read_bytes()
    if hashlib.sha256(packed).hexdigest() != info['scores_sha256']:
        raise ValueError('输入总分指纹改变')
    scores = read_floats(directory / 'scores.bin')
    queries, docs, k = info['num_queries'], info['num_docs'], info['k']
    expected_ids, expected_scores = reference(scores, queries, docs, k)
    raw_ids = ids_path.read_bytes()
    if len(raw_ids) != queries * k * 4:
        raise ValueError('输出文章编号字节数不匹配，应为int32[查询数,K]')
    actual_ids = list(struct.unpack('<' + 'i' * (queries * k), raw_ids))
    if actual_ids != expected_ids:
        index = next(i for i, (a, e) in enumerate(zip(actual_ids, expected_ids)) if a != e)
        raise ValueError(f'查询{index//k}第{index%k}名文章编号错误：实际={actual_ids[index]}，期望={expected_ids[index]}')
    actual_scores = read_floats(scores_path)
    if len(actual_scores) != len(expected_scores):
        raise ValueError('输出分数数量不匹配')
    # Selection copies input bits; it performs no arithmetic, so no tolerant score comparison.
    expected_bytes = struct.pack('<' + 'f' * len(expected_scores), *expected_scores)
    if scores_path.read_bytes() != expected_bytes:
        raise ValueError('选中分数与对应输入FP32字节不一致')
    print(f'Top-K文件核对通过：{queries}条查询，每条{k}篇，编号与分数完全一致。')
    for query in range(min(queries, 6)):
        print(f'查询{query}文章编号：{actual_ids[query*k:(query+1)*k]}')
    print('需结合实机执行日志确认NPU Top-K运行成功；本检查器不执行NPU。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--scores', required=True, help='NPU实际累加输出')
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--output', default=str(ROOT / 'results/topk_input'))
    e = sub.add_parser('edge')
    e.add_argument('--k', type=int, required=True)
    e.add_argument('--output', required=True)
    c = sub.add_parser('check')
    c.add_argument('--directory', default=str(ROOT / 'results/topk_input'))
    c.add_argument('--actual-scores', required=True)
    c.add_argument('--actual-ids', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(Path(args.scores), args.k, Path(args.output))
    elif args.command == 'edge':
        edge_fixture(args.k, Path(args.output))
    else:
        check(Path(args.directory), Path(args.actual_scores), Path(args.actual_ids))


if __name__ == '__main__':
    main()
