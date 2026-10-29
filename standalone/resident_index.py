"""Final index/query byte contracts; no experimental operator dependency."""
from array import array
import hashlib,json,math,sys
from pathlib import Path
LIMIT=2**31-1
INDEX_CONTRACT='bm25_resident_postings_fp32_v1'
QUERY_CONTRACT='bm25_query_ids_v1'
TIES='exact_FP32_score_desc_doc_id_asc'
INDEX_TYPES={'term_offsets.bin':'q','posting_doc_ids.bin':'i','posting_tf.bin':'f','doc_norm.bin':'f','idf.bin':'f'}
QUERY_TYPES={'query_offsets.bin':'q','query_term_ids.bin':'i'}
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

def compare(actual, expected):
    if len(actual) != len(expected):
        raise ValueError("结果记录数与期望值不同")
    for i, (a, e) in enumerate(zip(actual, expected)):
        if abs(a - e) > 1e-6 + 2e-6 * abs(e):
            raise ValueError(f"记录{i}不一致：实际={a}，期望={e}")
    return max(abs(a - e) for a, e in zip(actual, expected))

def write_queries(path,queries):
    require(isinstance(queries,list) and 1<=len(queries)<=10000 and
        all(isinstance(row,list) and all(integer(w,0,2147483646) for w in row) for row in queries),
        '原始查询须为非空二维INT32词编号列表；每条可为空，保留重复及未知词')
    lines=['BM25_QUERY_IDS_V1',str(len(queries))]
    lines.extend(' '.join([str(len(row))]+[str(w) for w in row]) for row in queries)
    path.write_bytes(('\n'.join(lines)+'\n').encode('ascii'))
