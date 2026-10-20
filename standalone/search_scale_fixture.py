"""Reuse a normal synthetic corpus for resident-index A/B/empty Search checks; no device execution."""
import argparse
from array import array
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import resident_index as r, run_query_score as score

CONTRACT = 'normal_corpus_resident_search_checks_v1'
STATUS = 'host_normal_search_fixtures_checked_device_not_run'
SOURCE_FILES = ('metadata.json', 'corpus.json', 'expected_scores.bin')


def definition(source):
    info, corpus = r.read_source(source)
    n, q, v = info['num_docs'], info['num_queries'], corpus['vocab_size']
    r.require(r.integer(n, 4, 1048576) and r.integer(q, 1, 128) and
              n*q <= 16777216 and r.integer(v, 8, 65536),
              '正常Search输入超过本轮尺寸预算；Q*N<=16777216，N<=1048576')
    r.require(info.get('dataset', {}).get('case') == 'performance_uniform_short_v1' and
              info['dataset'].get('stress_queries_injected') is False and
              info.get('contract') == 'bm25_device_pipeline_v1' and
              info.get('ties') == r.TIES and r.integer(info.get('k'), 1, min(32, n)),
              '只接受已有正常短查询performance语料，不混用正确性压力语料')
    words = info['dataset'].get('query_terms')
    r.require(r.integer(words, 1, 64) and all(isinstance(row, list) and len(row) == words and
              all(r.integer(t, 0, v-1) for t in row) for row in corpus['queries']) and
              all(isinstance(row, list) and 16 <= len(row) <= 48 and
                  all(r.integer(t, 0, v-1) for t in row) for row in corpus['documents']),
              '原始文章/短查询词编号或长度与正常语料声明不符')
    old = (source/'expected_scores.bin').read_bytes()
    r.require(len(old) == q*n*4 and r.digest(old) == info.get('reference_sha256', {}).get('expected_scores.bin'),
              '原始独立公式总分文件长度/指纹不符')
    a = corpus['queries']
    # A is exactly the original workload. B changes only word IDs and keeps Q/length.
    b = [[(word+1) % v for word in row] for row in a]
    return info, corpus, {'A':a, 'B':b, 'empty':[[] for unused in a]}


def independent_scores(documents, cases):
    """Oracle from original document TF/DF, double formula then final FP32; no postings used."""
    n = len(documents)
    df = Counter(term for doc in documents for term in set(doc))
    average = sum(map(len, documents))/n
    idfs = {term:math.log1p((n-count+.5)/(count+.5)) for term, count in df.items()}
    results = {name:array('f', [0])*(n*len(queries)) for name, queries in cases.items()}
    for doc_id, doc in enumerate(documents):
        count = Counter(doc)
        norm = 1.2*(1-.75+.75*len(doc)/average)
        for name, queries in cases.items():
            for q, words in enumerate(queries):
                total = 0.0
                for word in words:
                    tf = count[word]
                    if tf:
                        total += idfs[word]*(tf*2.2)/(tf+norm)
                results[name][q*n+doc_id] = total
    return results


def source_hashes(source):
    return {name:score.digest(source/name) for name in SOURCE_FILES}


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    r.require(not output.exists(), '规模Search输入目录已存在，请换新名字，不覆盖')
    print('读取原正常合成语料，保留原查询为A；不读取旧TF展开数组。', flush=True)
    info, corpus, cases = definition(source)
    before = source_hashes(source)
    index, values = r.build_index(corpus['documents'], corpus['vocab_size'])
    print('整库倒排索引已建立；从原始文章独立计算A/B/空查询期望。', flush=True)
    expected = independent_scores(corpus['documents'], cases)
    r.require(r.encode(expected['A']) == (source/'expected_scores.bin').read_bytes(),
              '重新独立统计的A分数与原始语料参考不逐字节一致')
    errors = {name:r.compare(r.diagnostic_scores(index, values, queries), expected[name])
              for name, queries in cases.items()}
    r.require(r.encode(expected['A']) != r.encode(expected['B']), 'A/B分数全部相同，无法验证换查询')
    r.require(before == source_hashes(source), '准备期间源语料/期望改变')
    r.save_arrays(output/'index', index, values)
    saved = output/'source'; saved.mkdir()
    for name in SOURCE_FILES:
        (saved/name).write_bytes((source/name).read_bytes())
    report = {'contract':CONTRACT, 'status':STATUS, 'index_id':index['index_id'],
              'source_sha256':before, 'source_is_real_corpus':False, 'device_verified':False,
              'cases':{}, 'scope':'Normal synthetic corpus correctness fixtures; no execution, timings or speedup claim.'}
    for name, queries in cases.items():
        record, arrays = r.make_queries(index, queries, info['k'])
        directory = output/name
        r.save_arrays(directory, record, arrays)
        raw = r.encode(expected[name]); (directory/'expected_scores.bin').write_bytes(raw)
        r.write_json(directory/'reference.json', {'index_id':index['index_id'], 'query_files':record['files'],
            'num_queries':len(queries), 'num_docs':index['num_docs'], 'expected_sha256':r.digest(raw),
            'reference':'independent_original_corpus_formula', 'device_verified':False})
        report['cases'][name] = {'query_files':record['files'], 'expected_sha256':r.digest(raw),
                                'max_abs_host_error':errors[name], 'query_input_bytes':record['query_input_bytes']}
    r.write_json(output/'preparation_report.json', report)
    validate_inputs(output)
    r.require(before == source_hashes(source), '保存/回读期间源语料改变')
    q = info['num_queries']; k = info['k']
    print(f'主机准备通过：N={index["num_docs"]}，A/B各Q={q}，每条{info["dataset"]["query_terms"]}词，K={k}。', flush=True)
    print(f'索引一次加载={index["index_bytes"]}字节；A/B每批逻辑查询={report["cases"]["A"]["query_input_bytes"]}字节；Top-K={8*q*k}字节。', flush=True)
    print('空查询仅用于正确性；未执行CPU/NPU Search，不是性能成绩。', flush=True)


def validate_inputs(directory):
    directory = directory.resolve()
    report = json.loads((directory/'preparation_report.json').read_text(encoding='utf-8'))
    r.require(report.get('contract') == CONTRACT and report.get('status') == STATUS and
              report.get('source_is_real_corpus') is False and report.get('device_verified') is False,
              '正常规模Search准备合同/状态不符')
    r.require(report.get('source_sha256') == source_hashes(directory/'source'), '复制的原始语料/期望指纹改变')
    info, corpus, cases = definition(directory/'source')
    index, values = r.validate_index(directory/'index')
    rebuilt, unused = r.build_index(corpus['documents'], corpus['vocab_size'])
    r.require(index['index_id'] == rebuilt['index_id'] == report.get('index_id') and
              set(report.get('cases', {})) == set(cases), '整库索引不是该原始语料的全量TF索引')
    del unused
    expected = independent_scores(corpus['documents'], cases)
    r.require(r.encode(expected['A']) == (directory/'source/expected_scores.bin').read_bytes(),
              '原始A期望与独立原始文章公式不符')
    records = {}
    for name, queries in cases.items():
        path = directory/name
        record, original = r.validate_queries(path, index)
        raw = r.encode(expected[name]); saved = report['cases'][name]
        r.require(original == queries and record['k'] == info['k'] and saved.get('query_files') == record['files'] and
                  (path/'expected_scores.bin').read_bytes() == raw and saved.get('expected_sha256') == r.digest(raw) and
                  saved.get('query_input_bytes') == record['query_input_bytes'], '查询/原始公式/来源不符：'+name)
        reference = json.loads((path/'reference.json').read_text(encoding='utf-8'))
        wanted = {'index_id':index['index_id'], 'query_files':record['files'], 'num_queries':len(queries),
                  'num_docs':index['num_docs'], 'expected_sha256':r.digest(raw),
                  'reference':'independent_original_corpus_formula', 'device_verified':False}
        r.require(reference == wanted, '独立参考说明不符：'+name)
        error = r.compare(r.diagnostic_scores(index, values, queries), expected[name])
        r.require(type(saved.get('max_abs_host_error')) in (int, float) and saved['max_abs_host_error'] == error,
                  '主机格式/独立公式误差记录不符：'+name)
        records[name] = record
    r.require(records['A']['files'] != records['B']['files'] and
              r.encode(expected['A']) != r.encode(expected['B']), 'A/B应有不同输入及不同总分')
    snapshot = {str(path.resolve()):score.digest(path) for path in sorted(directory.rglob('*')) if path.is_file()}
    return index, records, snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--input', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('check'); a.add_argument('--input', type=Path, required=True)
    args = p.parse_args()
    try:
        if args.command == 'prepare': prepare(args.input, args.output)
        else:
            index, records, unused = validate_inputs(args.input)
            print(f'规模Search索引/原始语料/三批查询/独立公式核对通过：N={index["num_docs"]}，Q={records["A"]["num_queries"]}；未执行设备。')
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；准备/检查停止，不生成CPU或NPU实际结果。', file=sys.stderr); return 1
    return 0


if __name__ == '__main__': sys.exit(main())
