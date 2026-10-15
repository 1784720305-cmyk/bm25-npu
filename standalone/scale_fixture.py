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


def generate(output, docs, queries, k, vocab, seed, long_terms,
             case='correctness', query_terms=3, max_groups=16777216):
    if output.exists():
        raise ValueError('输出目录已存在，请指定新目录')
    if case not in ('correctness', 'performance'):
        raise ValueError('不认识的数据用途')
    minimum_queries = 5 if case == 'correctness' else 1
    if docs < 4 or queries < minimum_queries or vocab < 8 or not 1 <= k <= min(docs, 32):
        raise ValueError(f'要求文章>=4、查询>={minimum_queries}、词表>=8、K在1..min(文章数,32)')
    if docs * queries > 2**31-1 or long_terms < 0:
        raise ValueError('总组数超过当前算子上限或长查询长度非法')
    if case == 'performance':
        if not 1 <= query_terms <= 64 or not 1 <= max_groups <= 2**31-1 or docs * queries > max_groups:
            raise ValueError('性能模式要求查询长度1..64且总组数不超过max-groups；先核对内存预算')
        long_terms = 0  # 正常短查询不混入正确性压力查询。
    start = time.perf_counter()
    random = Random(seed)
    documents = []
    for doc in range(docs):
        tokens = [] if case == 'correctness' and doc % 31 == 0 else [random.randrange(vocab) for _ in range(random.randrange(16, 49))]
        documents.append(tokens)
    if case == 'correctness':
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
    if case == 'correctness':
        query_tokens = [[], [vocab], [0, 0], [1, vocab, 2]]
        query_tokens.append([i % vocab for i in range(long_terms)] if long_terms else [2, 3, 4])
        while len(query_tokens) < queries:
            query_tokens.append([random.randrange(vocab) for _ in range(3)])
    else:
        query_tokens = [[random.randrange(vocab) for _ in range(query_terms)] for _ in range(queries)]
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
    if case == 'performance':
        info['dataset'].update({'case': 'performance_uniform_short_v1', 'query_terms': query_terms,
            'stress_queries_injected': False, 'source_is_real_corpus': False,
            'max_groups_budget': max_groups,
            'physical_input_bytes': sum((output / name).stat().st_size for name in info['input_sha256']),
            'score_buffer_bytes': 4 * docs * queries,
            'memory_note': 'max-groups limits Q*D only; Python corpus/Counter memory also grows with docs and is not a measured peak memory figure'})
    (output / 'metadata.json').write_bytes((json.dumps(info, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(f'规模输入准备通过：{docs}篇文章、{queries}条查询、K={k}，{len(tf_values)}条贡献、{docs*queries}组。')
    if case == 'performance':
        print(f'用途：均匀合成语料/每查询{query_terms}词，无额外空查询或384词压力查询；不是现实语料。')
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
                        help='仅正确性模式：默认含384词压力查询；性能模式不插入压力查询')
    parser.add_argument('--case', choices=('correctness', 'performance'), default='correctness')
    parser.add_argument('--query-terms', type=int, default=3, help='性能模式每条查询的词数')
    parser.add_argument('--max-groups', type=int, default=16777216,
                        help='性能模式Q*D预算；不是Python峰值内存限制，大规模需另核对内存')
    args = parser.parse_args()
    generate(Path(args.output), args.docs, args.queries, args.k, args.vocab, args.seed, args.long_query_terms,
             args.case, args.query_terms, args.max_groups)


if __name__ == '__main__':
    main()
