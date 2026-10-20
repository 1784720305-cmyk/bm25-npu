"""整库倒排索引与独立查询包：仅准备/核对数据，不调用或替代NPU，不作性能基准。"""
import argparse
from array import array
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.accumulate_fixture import fp32
from check_result import compare

LIMIT = 2**31-1
INDEX_CONTRACT = 'bm25_resident_postings_fp32_v1'
QUERY_CONTRACT = 'bm25_query_ids_v1'
TIES = 'exact_FP32_score_desc_doc_id_asc'
INDEX_TYPES = {'term_offsets.bin': 'q', 'posting_doc_ids.bin': 'i',
               'posting_tf.bin': 'f', 'doc_norm.bin': 'f', 'idf.bin': 'f'}
QUERY_TYPES = {'query_offsets.bin': 'q', 'query_term_ids.bin': 'i'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, record):
    path.write_bytes((json.dumps(record, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def encode(values):
    require(values.itemsize == {'q':8, 'i':4, 'f':4}[values.typecode], '本机array类型宽度不符')
    if sys.byteorder != 'little':
        values = array(values.typecode, values)
        values.byteswap()
    return values.tobytes()


def identity(record):
    # Source file path, queries, K and deployment/device state are deliberately excluded.
    keys = ('contract', 'num_docs', 'vocab_size', 'posting_count', 'k1', 'b', 'idf_formula', 'files')
    content = {key:record[key] for key in keys}
    return digest(json.dumps(content, sort_keys=True, separators=(',', ':')).encode('utf-8'))


def build_index(documents, vocab):
    require(isinstance(documents, list) and integer(len(documents), 1, LIMIT) and
            integer(vocab, 1, LIMIT-1), '文章数/词表尺寸非法')
    lengths = array('q')
    df = array('q', [0])*vocab
    for doc in documents:
        require(isinstance(doc, list) and len(doc) <= 2**24 and
                all(integer(t, 0, vocab-1) for t in doc), '文章须为词表范围内的INT32词编号；文章长度<=2^24')
        lengths.append(len(doc))
        for term in set(doc):
            df[term] += 1
    total = sum(lengths)
    require(total > 0, '第一版不接受全部文章都为空的语料')
    avgdl = total/len(documents)
    offsets = array('q', [0])
    for count in df:
        offsets.append(offsets[-1]+count)
    require(offsets[-1] <= LIMIT, '第一版倒排记录总数须<=INT32_MAX')
    docs = array('i', [0])*offsets[-1]
    tf = array('f', [0])*offsets[-1]
    cursor = array('q', offsets[:-1])
    # Visiting documents in ID order makes each term's posting IDs strictly increasing.
    for doc_id, doc in enumerate(documents):
        for term, count in Counter(doc).items():
            position = cursor[term]
            docs[position], tf[position] = doc_id, count
            cursor[term] += 1
    arrays = {'term_offsets.bin':offsets, 'posting_doc_ids.bin':docs, 'posting_tf.bin':tf,
              'doc_norm.bin':array('f', (1.2*(1-.75+.75*n/avgdl) for n in lengths)),
              'idf.bin':array('f', (math.log1p((len(documents)-n+.5)/(n+.5)) for n in df))}
    record = {'contract':INDEX_CONTRACT, 'num_docs':len(documents), 'vocab_size':vocab,
              'posting_count':offsets[-1], 'k1':1.2, 'b':.75,
              'idf_formula':'log1p((N-df+0.5)/(df+0.5))',
              'contribution_policy':'FP32_query_time_formula_no_precomputed_contributions',
              'files':{name:digest(encode(values)) for name, values in arrays.items()},
              'index_bytes':sum(len(encode(v)) for v in arrays.values()),
              'device_verified':False}
    record['index_id'] = identity(record)
    return record, arrays


def save_arrays(directory, record, arrays):
    require(not directory.exists(), '输出目录已存在，不覆盖：'+str(directory))
    directory.mkdir(parents=True)
    for name, values in arrays.items():
        (directory/name).write_bytes(encode(values))
    write_json(directory/'metadata.json', record)


def read_arrays(directory, types):
    record = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    require(set(record.get('files', {})) == set(types), '数组清单不符')
    arrays = {}
    for name, code in types.items():
        raw = (directory/name).read_bytes()
        require(digest(raw) == record['files'][name], '文件指纹不符：'+name)
        values = array(code)
        require(values.itemsize == {'q':8, 'i':4, 'f':4}[code] and len(raw)%values.itemsize == 0, '数组字节尺寸不符：'+name)
        values.frombytes(raw)
        if sys.byteorder != 'little':
            values.byteswap()
        arrays[name] = values
    return record, arrays


def validate_index(directory):
    record, values = read_arrays(directory, INDEX_TYPES)
    require(record.get('contract') == INDEX_CONTRACT and record.get('index_id') == identity(record), '索引合同/身份不符')
    n, v, m = (record[key] for key in ('num_docs', 'vocab_size', 'posting_count'))
    require(integer(n, 1, LIMIT) and integer(v, 1, LIMIT-1) and integer(m, 1, LIMIT), '索引尺寸非法')
    require(record['k1'] == 1.2 and record['b'] == .75 and
            record['idf_formula'] == 'log1p((N-df+0.5)/(df+0.5))' and
            record.get('contribution_policy') == 'FP32_query_time_formula_no_precomputed_contributions', '索引公式不符')
    offsets, docs, tf, norms, idfs = (values[key] for key in INDEX_TYPES)
    require(len(offsets) == v+1 and len(docs) == len(tf) == m and len(norms) == n and len(idfs) == v,
            '索引数组长度不符')
    require(offsets[0] == 0 and offsets[-1] == m and all(0 <= a <= b <= m for a,b in zip(offsets, offsets[1:])), '倒排边界非法')
    require(all(0 <= d < n for d in docs) and all(math.isfinite(f) and 1 <= f <= 2**24 and f == int(f) for f in tf), '倒排文章编号/TF非法')
    require(all(math.isfinite(x) and x > 0 for x in norms) and all(math.isfinite(x) and x >= 0 for x in idfs), '长度项/IDF非法')
    lengths = array('q', [0])*n
    for term in range(v):
        start, end = offsets[term:term+2]
        require(all(docs[i-1] < docs[i] for i in range(start+1, end)), '同词文章编号未严格递增或重复')
        require(idfs[term] == fp32(math.log1p((n-(end-start)+.5)/((end-start)+.5))), 'IDF与DF不符')
        for pos in range(start, end):
            lengths[docs[pos]] += int(tf[pos])
    require(all(x <= 2**24 for x in lengths), '重建文章长度超过第一版上限')
    avgdl = sum(lengths)/n
    require(all(x == fp32(1.2*(1-.75+.75*length/avgdl)) for x,length in zip(norms, lengths)), '长度项与TF重建长度不符')
    require(record.get('index_bytes') == sum(len(encode(v)) for v in values.values()), '索引字节统计不符')
    return record, values


def make_queries(index, queries, k):
    n = index['num_docs']
    require(isinstance(queries, list) and integer(len(queries), 1, LIMIT) and
            len(queries)*n <= LIMIT and integer(k, 1, min(32, n)), '查询数/Q*D/K超出第一版上限')
    terms, offsets = array('i'), array('q', [0])
    for query in queries:
        require(isinstance(query, list) and all(integer(t, 0, LIMIT-1) for t in query), '查询词编号必须为非负INT32，不能用负数哨兵')
        terms.extend(query)  # Preserve duplicates, order and unknown IDs; never deduplicate.
        offsets.append(len(terms))
    require(len(terms) <= LIMIT, '查询词总数超过INT32_MAX')
    arrays = {'query_offsets.bin':offsets, 'query_term_ids.bin':terms}
    record = {'contract':QUERY_CONTRACT, 'index_id':index['index_id'], 'num_queries':len(queries),
              'k':k, 'ties':TIES, 'query_term_count':len(terms),
              'query_input_bytes':sum(len(encode(v)) for v in arrays.values()),
              'topk_output_bytes':len(queries)*k*8, 'device_verified':False,
              'files':{name:digest(encode(v)) for name, v in arrays.items()}}
    return record, arrays


def validate_queries(directory, index):
    record, values = read_arrays(directory, QUERY_TYPES)
    require(record.get('contract') == QUERY_CONTRACT and record.get('index_id') == index['index_id'], '查询包引用了其他索引或合同')
    offsets, terms = (values[key] for key in QUERY_TYPES)
    q = record.get('num_queries')
    require(integer(q, 1, LIMIT) and q*index['num_docs'] <= LIMIT and len(offsets) == q+1,
            '查询尺寸不符')
    require(offsets[0] == 0 and offsets[-1] == len(terms) <= LIMIT and
            all(0 <= a <= b <= len(terms) for a,b in zip(offsets, offsets[1:])), '查询边界非法')
    queries = [list(terms[a:b]) for a,b in zip(offsets, offsets[1:])]
    expected, unused = make_queries(index, queries, record.get('k'))
    require(all(record.get(key) == value for key,value in expected.items() if key != 'device_verified'), '查询规则/指纹/字节统计不符')
    return record, queries


def diagnostic_scores(index, values, queries):
    """Slow host diagnostic only; never measured as a CPU opponent or saved as NPU output."""
    n = index['num_docs']
    scores = array('f', [0])*(n*len(queries))
    offsets, docs, tf, norms, idfs = (values[key] for key in INDEX_TYPES)
    for q, terms in enumerate(queries):
        for term in terms:
            if term >= index['vocab_size']:
                continue
            for pos in range(offsets[term], offsets[term+1]):
                doc = docs[pos]
                contribution = fp32(fp32(idfs[term]*fp32(tf[pos]*fp32(2.2)))/fp32(tf[pos]+norms[doc]))
                target = q*n+doc
                scores[target] = fp32(scores[target]+contribution)
    return scores


def read_source(directory):
    info = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    raw = (directory/'corpus.json').read_bytes()
    require(digest(raw) == info.get('corpus_sha256'), '源语料指纹不符')
    corpus = json.loads(raw)
    require(corpus.get('kind') == 'synthetic_numeric_corpus_v1' and corpus.get('k1') == 1.2 and
            corpus.get('b') == .75 and corpus.get('idf_formula') == 'log1p((N-df+0.5)/(df+0.5))', '源语料公式不符')
    require(len(corpus['documents']) == info['num_docs'] and len(corpus['queries']) == info['num_queries'], '源语料数量不符')
    return info, corpus


def check_source_scores(directory, index, values, query_record, queries):
    info, corpus = read_source(directory)
    require(corpus['queries'] == queries and len(corpus['documents']) == index['num_docs'] and
            corpus['vocab_size'] == index['vocab_size'] and info['k'] == query_record['k'], '参考语料/查询/K不匹配')
    # Verify every TF posting against source corpus, including terms absent from queries.
    cursor = list(values['term_offsets.bin'][:-1])
    for doc_id, doc in enumerate(corpus['documents']):
        for term, count in Counter(doc).items():
            require(integer(term, 0, index['vocab_size']-1), '源语料词编号非法')
            pos = cursor[term]
            require(pos < values['term_offsets.bin'][term+1] and
                    values['posting_doc_ids.bin'][pos] == doc_id and values['posting_tf.bin'][pos] == count,
                    '倒排记录与原始语料TF不符')
            cursor[term] += 1
    require(cursor == list(values['term_offsets.bin'][1:]), '索引含额外倒排记录')
    raw = (directory/'expected_scores.bin').read_bytes()
    require(digest(raw) == info['reference_sha256']['expected_scores.bin'] and
            len(raw) == len(queries)*index['num_docs']*4, '独立公式期望文件指纹/长度不符')
    expected = struct.unpack('<'+'f'*(len(raw)//4), raw)
    require(all(math.isfinite(x) and x >= 0 for x in expected), '独立公式期望值非法')
    error = compare(diagnostic_scores(index, values, queries), expected)
    print(f'主机格式往返及原始语料全倒排记录核对通过；独立公式总分最大绝对误差={error:.8g}。')
    return error


def prepare(source, output):
    require(not output.exists(), '输出目录已存在，不覆盖历史记录')
    info, corpus = read_source(source)
    index, values = build_index(corpus['documents'], corpus['vocab_size'])
    query_record, query_arrays = make_queries(index, corpus['queries'], info['k'])
    save_arrays(output/'index', index, values)
    save_arrays(output/'queries', query_record, query_arrays)
    # Check the bytes read back from disk, not just pre-serialization Python objects.
    index, values = validate_index(output/'index')
    query_record, queries = validate_queries(output/'queries', index)
    error = check_source_scores(source, index, values, query_record, queries)
    write_json(output/'preparation_report.json', {
        'status':'host_index_and_query_roundtrip_checked_device_not_run', 'index_id':index['index_id'],
        'index_bytes':index['index_bytes'], 'query_input_bytes':query_record['query_input_bytes'],
        'topk_output_bytes':query_record['topk_output_bytes'], 'max_abs_score_error':error,
        'source_corpus_sha256':info['corpus_sha256'], 'device_verified':False,
        'scope':'Byte/format/formula checks only; no NPU implementation, timings or speedup claim.'})
    print(f'整库索引={index["index_bytes"]}字节；本批查询输入={query_record["query_input_bytes"]}字节；最终Top-K输出={query_record["topk_output_bytes"]}字节。')
    print('仅完成主机数据准备和核对，尚未上传索引、运行新NPU查询链或验证性能。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('queries', help='复用现有索引，仅生成另一批查询；JSON为二维词编号列表')
    p.add_argument('--index', required=True, type=Path)
    p.add_argument('--input-json', required=True, type=Path)
    p.add_argument('--k', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('check')
    p.add_argument('--index', required=True, type=Path)
    p.add_argument('--queries', required=True, type=Path)
    p.add_argument('--reference-input', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args.input, args.output)
        else:
            index, values = validate_index(args.index)
            if args.command == 'queries':
                queries = json.loads(args.input_json.read_text(encoding='utf-8'))
                record, arrays = make_queries(index, queries, args.k)
                save_arrays(args.output, record, arrays)
                validate_queries(args.output, index)
                print(f'新查询包通过；引用同一索引，输入={record["query_input_bytes"]}字节。索引未改，未运行NPU。')
            else:
                record, queries = validate_queries(args.queries, index)
                if args.reference_input:
                    check_source_scores(args.reference_input, index, values, record, queries)
                print('索引/查询文件结构及指纹通过；未提交任何设备计算，不代表NPU正确性或性能。')
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不生成NPU替代结果。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
