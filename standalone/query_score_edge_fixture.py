"""准备确定性评分边界语料及独立公式期望；不执行NPU，不作性能测试。"""
import argparse
from array import array
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r

VERSION = 'query_score_tile_edges_v1'
SIZES = (129,2047,2048,2049,4097)
VOCAB_SIZE = 9


def profile(n):
    return {'kind':VERSION,'num_docs':n}


def definition(spec):
    r.require(isinstance(spec,dict) and set(spec) == {'kind','num_docs'} and
              spec['kind'] == VERSION and type(spec['num_docs']) is int and spec['num_docs'] in SIZES,
              '只接受明确的评分边界版本及文章尺寸')
    n = spec['num_docs']
    rng = random.Random(20261020+n)
    documents = []
    markers = {0,127,128,2046,2047,2048,2049,4095,4096,n-1}
    for i in range(n):
        doc = []
        if i % 11:
            doc = [0]*(1+i%3) # frequent postings, also leave empty documents
            if i%3 == 0: doc += [1,1]
            if i%17 == 0: doc += [2]*5
            doc += [rng.randrange(3,6) for unused in range(rng.randrange(4))]
        if i in markers: doc += [6]*(1+i%4) # matches on both sides of a tile/tail boundary
        documents.append(doc)
    # 7 and 8 have no postings; 9 and INT32_MAX-1 are unknown IDs.
    patterns = [[],[9],[7,8],[6],[0],[1,1,6],[2,3,2],
                [9,6,2147483646],[5,4,3,0],[0,6,0]]
    a = [list(patterns[i%len(patterns)]) for i in range(65)]
    a[-1] = [2,3,0]*17 # ordered accumulation across a longer query
    b = [[6,6],[],[2147483646],[8],[1,0],[5,4,5],[0,0,6]]
    return documents, {'A':a,'B':b,'empty':[[],[],[]]}


def source_identity(documents,cases):
    return r.digest(json.dumps({'documents':documents,'queries':cases},sort_keys=True,
                               separators=(',',':')).encode('utf-8'))


def independent_scores(documents,queries):
    # Oracle scans original per-document word counts, never the index/postings or device results.
    # DF is computed once to avoid the small-fixture oracle's quadratic scan. Arithmetic
    # remains double-precision BM25 followed by one FP32 cast per completed document score.
    counts = [Counter(doc) for doc in documents]
    df = Counter(term for doc in counts for term in doc)
    n = len(documents)
    average = sum(map(len,documents))/n
    idf = {t:math.log1p((n-f+.5)/(f+.5)) for t,f in df.items()}
    norms = [1.2*(1-.75+.75*len(doc)/average) for doc in documents]
    values = array('f')
    for query in queries:
        for counter,norm in zip(counts,norms):
            total = 0.0
            for term in query:
                tf = counter[term]
                if tf: total += idf[term]*(tf*2.2)/(tf+norm)
            values.append(total)
    return values


def prepare_case(directory,n):
    r.require(not directory.exists(),'边界输入目录已存在，不覆盖')
    spec = profile(n)
    documents,cases = definition(spec)
    index,arrays = r.build_index(documents,VOCAB_SIZE)
    r.save_arrays(directory/'index',index,arrays)
    report = {'status':'host_query_score_fixtures_checked_device_not_run','index_id':index['index_id'],
              'source_contract_sha256':source_identity(documents,cases),'fixture_profile':spec,
              'cases':{},'device_verified':False}
    for name,queries in cases.items():
        record,words = r.make_queries(index,queries,min(10,n))
        query_dir = directory/name
        r.save_arrays(query_dir,record,words)
        expected = independent_scores(documents,queries)
        error = r.compare(r.diagnostic_scores(index,arrays,queries),expected)
        raw = r.encode(expected)
        (query_dir/'expected_scores.bin').write_bytes(raw)
        r.write_json(query_dir/'reference.json',{
            'index_id':index['index_id'],'query_files':record['files'],
            'num_queries':len(queries),'num_docs':n,'expected_sha256':r.digest(raw),
            'reference':'independent_original_corpus_formula','device_verified':False})
        report['cases'][name] = {'query_files':record['files'],'expected_sha256':r.digest(raw),
                                'max_abs_host_error':error,'query_input_bytes':record['query_input_bytes']}
    r.write_json(directory/'preparation_report.json',report)
    print(f'边界输入通过：N={n}，A=65/B=7/空=3条查询；独立原始语料公式核对通过，未执行NPU。',flush=True)


def prepare(directory):
    r.require(not directory.exists(),'输入套件目录已存在，请换新名字，不覆盖')
    directory.mkdir(parents=True)
    for n in SIZES: prepare_case(directory/f'n{n}',n)
    r.write_json(directory/'suite.json',{'kind':VERSION,'sizes':list(SIZES),
                 'status':'host_edge_suite_prepared_device_not_run','device_verified':False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/query_score_edges_input1')
    args = p.parse_args()
    try: prepare(args.output)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；输入准备停止，不生成NPU输出。',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__': sys.exit(main())
