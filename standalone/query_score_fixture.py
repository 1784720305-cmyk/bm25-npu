"""准备新评分算子A/B/全空查询；或只核对设备分数文件。不运行或替代NPU。"""
import argparse
from array import array
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import resident_index as r
from standalone.accumulate_fixture import independent_scores


def prepare(output):
    r.require(not output.exists(), '输出目录已存在，请换新名字，不覆盖')
    source = json.loads((ROOT/'fixtures/compute_contract_small.json').read_text(encoding='utf-8'))
    vocab = source['vocabulary']
    r.require(sorted(vocab.values()) == list(range(len(vocab))), '小语料词编号不连续')
    documents = [[vocab[t] for t in row] for row in source['documents']]
    index, arrays = r.build_index(documents, len(vocab))
    r.save_arrays(output/'index', index, arrays)
    index, arrays = r.validate_index(output/'index')
    cases = {
        'A':source['queries'],
        'B':[['GPU'], ['编程'], [], ['不存在的词'], ['GPU', 'GPU', 'NPU'], ['NPU', '实践']],
        'empty':[[], [], []],
    }
    report = {'status':'host_query_score_fixtures_checked_device_not_run', 'index_id':index['index_id'],
              'source_contract_sha256':r.digest((ROOT/'fixtures/compute_contract_small.json').read_bytes()),
              'cases':{}, 'device_verified':False}
    for name, original in cases.items():
        queries = [[vocab.get(t, len(vocab)) for t in row] for row in original]
        record, words = r.make_queries(index, queries, 3)
        directory = output/name
        r.save_arrays(directory, record, words)
        r.validate_queries(directory, index)
        expected = array('f', independent_scores(source['documents'], original))
        actual = r.diagnostic_scores(index, arrays, queries)
        error = r.compare(actual, expected)
        raw = r.encode(expected)
        (directory/'expected_scores.bin').write_bytes(raw)
        reference = {'index_id':index['index_id'], 'query_files':record['files'],
                     'num_queries':len(queries), 'num_docs':len(documents),
                     'expected_sha256':r.digest(raw), 'reference':'independent_original_corpus_formula',
                     'device_verified':False}
        r.write_json(directory/'reference.json', reference)
        report['cases'][name] = {'query_files':record['files'], 'expected_sha256':r.digest(raw),
                                'max_abs_host_error':error, 'query_input_bytes':record['query_input_bytes']}
    r.write_json(output/'preparation_report.json',report)
    print('新评分小样本准备通过：同一整库索引，A/B两批不同查询及全空查询，独立公式核对通过。')
    print('仅准备数据，未编译/执行NPU；empty逻辑词数为0，未来主机调用需提供1个不被读取的物理INT32元素。')


def check(index_dir, query_dir, actual_path):
    index, unused = r.validate_index(index_dir)
    record, queries = r.validate_queries(query_dir, index)
    ref = json.loads((query_dir/'reference.json').read_text(encoding='utf-8'))
    expected_raw = (query_dir/'expected_scores.bin').read_bytes()
    r.require(ref['index_id'] == index['index_id'] and ref['query_files'] == record['files'] and
              ref['num_queries'] == record['num_queries'] and ref['num_docs'] == index['num_docs'] and
              r.digest(expected_raw) == ref['expected_sha256'], '期望分数与索引/查询身份不符')
    raw = actual_path.read_bytes()
    size = len(queries)*index['num_docs']*4
    r.require(len(raw) == len(expected_raw) == size, '实际/期望分数尺寸不符')
    values, expected = array('f'), array('f')
    values.frombytes(raw)
    expected.frombytes(expected_raw)
    if sys.byteorder != 'little':
        values.byteswap()
        expected.byteswap()
    r.require(all(math.isfinite(x) and x >= 0 for x in values), '实际分数含负值/NaN/Inf，不能通过')
    r.require(all(math.isfinite(x) and x >= 0 for x in expected), '期望分数非法')
    error = r.compare(values, expected)
    print(f'评分文件核对通过：{len(queries)}条查询×{index["num_docs"]}篇文章，最大绝对误差={error:.8g}。')
    print('只核对已有文件，设备执行仍须真实运行日志；不是search或NPU性能验收。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output',type=Path,default=ROOT/'results/query_score_small_input1')
    p = sub.add_parser('check')
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--actual',type=Path,required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare': prepare(args.output)
        else: check(args.index,args.queries,args.actual)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不生成CPU替代设备输出。',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
