"""生成更大规模的同输入BM25计算验证数据，保留原始合成语料；不运行CPU/NPU。"""
import argparse
from array import array
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from random import Random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_result import compare
from standalone.accumulate_fixture import fp32


def float_array(values=()):
    result = array('f', values)
    if result.itemsize != 4:
        raise ValueError('本机array.float不是4字节')
    return result


def write_array(path, values):
    if sys.byteorder != 'little':
        values = array(values.typecode, values)
        values.byteswap()
    path.write_bytes(values.tobytes())


def generate(output, docs, queries, k, vocab, seed, long_terms):
    if output.exists():
        raise ValueError('输出目录已存在，请指定新目录')
    if docs < 4 or queries < 5 or vocab < 8 or not 1 <= k <= min(docs, 32):
        raise ValueError('要求文章>=4、查询>=5、词表>=8、K在1..min(文章数,32)')
    if docs * queries > 2**31-1 or long_terms < 0:
        raise ValueError('总组数超过当前算子上限或长查询长度非法')
    start = time.perf_counter()
    random = Random(seed)
    documents = []
    for doc in range(docs):
        tokens = [] if doc % 31 == 0 else [random.randrange(vocab) for _ in range(random.randrange(16, 49))]
        documents.append(tokens)
    documents[2] = list(documents[1])  # Exact-document ties.
    if long_terms:
        documents[3] = list(range(vocab))  # Every long-query term can match this document.
    counts = [Counter(doc) for doc in documents]
    df = [0] * vocab
    for count in counts:
        for term in count:
            df[term] += 1
    avgdl = sum(map(len, documents)) / docs
    norms_raw = [1.2 * (1 - .75 + .75 * len(doc) / avgdl) for doc in documents]
    norms = [fp32(n) for n in norms_raw]
    idfs_raw = [math.log1p((docs - value + .5) / (value + .5)) for value in df]
    idfs = [fp32(value) for value in idfs_raw]
    query_tokens = [[], [vocab], [0, 0], [1, vocab, 2]]
    query_tokens.append([i % vocab for i in range(long_terms)] if long_terms else [2, 3, 4])
    while len(query_tokens) < queries:
        query_tokens.append([random.randrange(vocab) for _ in range(3)])
    tf_values, norm_values, idf_values = (float_array() for _ in range(3))
    expected_contributions, expected_scores, same_flow_scores = (float_array() for _ in range(3))
    offsets = array('q', [0])
    if offsets.itemsize != 8:
        raise ValueError('本机array.int64不是8字节')
    numerator_scale = fp32(2.2)
    max_group = 0
    for query in query_tokens:
        for doc, count in enumerate(counts):
            precise_total, fp_total = 0.0, 0.0
            begin = len(tf_values)
            for term in query:
                frequency = count.get(term, 0)
                if not frequency:
                    continue
                tf_values.append(float(frequency))
                norm_values.append(norms[doc])
                idf_values.append(idfs[term])
                numerator = fp32(float(frequency) * numerator_scale)
                denominator = fp32(float(frequency) + norms[doc])
                contribution = fp32(fp32(idfs[term] * numerator) / denominator)
                expected_contributions.append(contribution)
                fp_total = fp32(fp_total + contribution)
                # Separate high-precision formula from original corpus statistics.
                precise_total += idfs_raw[term] * (frequency * 2.2) / (frequency + norms_raw[doc])
            offsets.append(len(tf_values))
            max_group = max(max_group, len(tf_values) - begin)
            expected_scores.append(fp32(precise_total))
            same_flow_scores.append(fp_total)
    if not 0 < len(tf_values) <= 2**31-1:
        raise ValueError('物理贡献数量为空或超过当前算子上限')
    compare(same_flow_scores, expected_scores)  # Original strict tolerance; never silently widen it.
    output.mkdir(parents=True)
    payload = {'tf.bin': tf_values, 'doc_norm.bin': norm_values, 'idf.bin': idf_values, 'offsets.bin': offsets,
               'expected_contributions.bin': expected_contributions, 'expected_scores.bin': expected_scores}
    hashes = {}
    for name, values in payload.items():
        destination = output / name
        write_array(destination, values)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    corpus = {'kind': 'synthetic_numeric_corpus_v1', 'seed': seed, 'documents': documents,
              'queries': query_tokens, 'vocab_size': vocab, 'k1': 1.2, 'b': .75,
              'idf_formula': 'log1p((N-df+0.5)/(df+0.5))'}
    (output / 'corpus.json').write_bytes((json.dumps(corpus, separators=(',', ':')) + '\n').encode('utf-8'))
    info = {
        'contract': 'bm25_device_pipeline_v1', 'num_queries': queries, 'num_docs': docs, 'k': k,
        'value_count': len(tf_values), 'group_count': docs*queries, 'offset_count': len(offsets),
        'k1': 1.2, 'ties': 'exact_FP32_score_desc_doc_id_asc',
        'input_origin': 'synthetic_corpus_TF_norm_IDF_grouped_before_NPU_compute',
        'layout': 'contributions grouped by query*num_docs+doc before upload',
        'input_sha256': {name: hashes[name] for name in ('tf.bin', 'doc_norm.bin', 'idf.bin', 'offsets.bin')},
        'reference_sha256': {name: hashes[name] for name in ('expected_contributions.bin', 'expected_scores.bin')},
        'corpus_sha256': hashlib.sha256((output / 'corpus.json').read_bytes()).hexdigest(),
        'dataset': {'kind': 'synthetic', 'seed': seed, 'vocab_size': vocab,
                    'query_lengths': list(map(len, query_tokens)), 'max_contributions_per_group': max_group,
                    'empty_groups': sum(a == b for a, b in zip(offsets, offsets[1:]))},
        'device_pipeline_verified': False,
    }
    info['preparation_seconds'] = time.perf_counter() - start
    (output / 'metadata.json').write_bytes((json.dumps(info, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(f'规模输入准备通过：{docs}篇文章、{queries}条查询、K={k}，{len(tf_values)}条贡献、{docs*queries}组。')
    print(f'最长组={max_group}条贡献；空组={info["dataset"]["empty_groups"]}；保留corpus.json供独立BM25S对手后续使用。')
    print('同输入FP32参考与独立原始语料公式在原容差内一致；未执行或验证CPU/NPU程序，不作性能结论。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'results/pipeline_medium'))
    parser.add_argument('--docs', type=int, default=4096)
    parser.add_argument('--queries', type=int, default=17)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--vocab', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--long-query-terms', type=int, default=384,
                        help='默认正确性压力测试含384词查询；设0才不加入该压力查询，须在报告说明数据区别')
    args = parser.parse_args()
    generate(Path(args.output), args.docs, args.queries, args.k, args.vocab, args.seed, args.long_query_terms)


if __name__ == '__main__':
    main()
