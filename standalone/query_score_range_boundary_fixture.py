"""Deterministic range-score boundary inputs/oracles; no device or performance result."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, query_score_edge_fixture as edge
from standalone import run_query_score_range as score

VERSION='range_score_64_terms_contiguous_boundaries_v1'
CASES={'single_tail':(4097,1),'split_tail':(16385,8),'over_core_queries':(2049,57)}
VOCAB=9


def definition(name):
    r.require(name in CASES,'不是指定的连续分区评分边界用例')
    n,q=CASES[name]
    # Cover endpoints of every possible <=56-way partition and each inner tile,
    # without asserting that this is the actual device core count/tiling choice.
    markers={0,n-1,127,128,2047,2048}
    for p in range(1,min(56,(n+2047)//2048)+1):
        for i in range(p):
            first,last=n*i//p,n*(i+1)//p
            for tile in range(first,last,2048): markers.update((tile-1,tile,tile+1))
            markers.update((first-1,first,last-1,last))
    documents=[]
    for i in range(n):
        words=[] if i%11==0 else [0]*(1+i%3)
        if i%3==0: words += [1,1]
        if i%17==0: words += [2]*5
        if i%7==0: words += [3,4,5]
        if i in markers: words += [6]*(1+i%4)
        documents.append(words)
    patterns=[[],[9,2147483646],[7,8],[6],[0,1,0],[2,3,2],[5,4,3,0]]
    a=[list(patterns[i%len(patterns)]) for i in range(q)]
    a[0]=[0,1,2,6]*16 # all 64 cursor slots contain known, nonempty posting lists
    return documents,{'A':a,'B':[[6]*64,[7,8,9,2147483646],[0,0,3,4]],'empty':[[],[],[]]}


def prepare_case(directory,name):
    r.require(not directory.exists(),'输出目录已存在，不覆盖')
    documents,cases=definition(name)
    index,arrays=r.build_index(documents,VOCAB)
    r.save_arrays(directory/'index',index,arrays)
    report={'kind':VERSION,'case':name,'status':'host_range_boundaries_prepared_device_not_run',
            'index_id':index['index_id'],'source_contract_sha256':edge.source_identity(documents,cases),
            'cases':{},'device_verified':False}
    for label,queries in cases.items():
        record,words=r.make_queries(index,queries,3)
        target=directory/label; r.save_arrays(target,record,words)
        expected=edge.independent_scores(documents,queries)
        error=r.compare(r.diagnostic_scores(index,arrays,queries),expected)
        raw=r.encode(expected); (target/'expected_scores.bin').write_bytes(raw)
        r.write_json(target/'reference.json',{'index_id':index['index_id'],'query_files':record['files'],
            'num_queries':len(queries),'num_docs':len(documents),'expected_sha256':r.digest(raw),
            'reference':'independent_original_corpus_formula','device_verified':False})
        report['cases'][label]={'query_files':record['files'],'expected_sha256':r.digest(raw),
                                'max_abs_host_error':error}
    # Unsupported request is only used for host rejection; never submitted to the NPU.
    record,words=r.make_queries(index,[[0]*65],3)
    r.save_arrays(directory/'reject65',record,words)
    r.write_json(directory/'preparation_report.json',report)
    print(f'准备{name}：{len(documents)}篇文章，A={len(cases["A"])}条，A/B含每条64词；独立公式通过，未执行设备。',flush=True)


def validate_case(directory):
    preparation=json.loads((directory/'preparation_report.json').read_text(encoding='utf-8'))
    r.require(preparation.get('kind')==VERSION and
              preparation.get('status')=='host_range_boundaries_prepared_device_not_run',
              '不是当前连续分区边界输入')
    documents,cases=definition(preparation.get('case'))
    index,unused=r.validate_index(directory/'index')
    expected_index,unused=r.build_index(documents,VOCAB)
    r.require(index['index_id']==expected_index['index_id']==preparation.get('index_id') and
              preparation.get('source_contract_sha256')==edge.source_identity(documents,cases) and
              set(preparation.get('cases',{}))==set(cases),'边界语料或索引来源不符')
    records={}
    for label,wanted in cases.items():
        target=directory/label
        record,queries=r.validate_queries(target,index)
        r.require(queries==wanted and record['files']==preparation['cases'][label]['query_files'],
                  '查询不是指定的64词/未知/重复/全空边界')
        raw=r.encode(edge.independent_scores(documents,wanted))
        r.require((target/'expected_scores.bin').read_bytes()==raw and
                  preparation['cases'][label]['expected_sha256']==r.digest(raw),'期望文件与独立原始语料公式不符')
        ref=json.loads((target/'reference.json').read_text(encoding='utf-8'))
        r.require(ref=={'index_id':index['index_id'],'query_files':record['files'],
            'num_queries':len(wanted),'num_docs':len(documents),'expected_sha256':r.digest(raw),
            'reference':'independent_original_corpus_formula','device_verified':False},'参考说明不匹配')
        records[label]=record
    rejected,words=r.validate_queries(directory/'reject65',index)
    r.require(words==[[0]*65],'65词拒绝用例内容不符')
    snapshot={str(p.resolve()):score.digest(p) for p in directory.rglob('*') if p.is_file()}
    return index,records,snapshot


def check_rejection(directory,index):
    record,unused=r.validate_queries(directory/'reject65',index)
    try: score.enforce_query_limits(directory,index,{'reject65':record})
    except ValueError as error:
        r.require('最多64词' in str(error) and '出现65词' in str(error),'不是预期的65词拒绝原因')
        return {'status':'host_65_terms_rejected_before_device_call','device_submitted':False,'error':str(error)}
    raise ValueError('65词请求未被主机保护拒绝')


def prepare(directory):
    r.require(not directory.exists(),'输入套件目录已存在，请换新名字')
    directory.mkdir(parents=True)
    for name in CASES: prepare_case(directory/name,name)
    r.write_json(directory/'suite.json',{'kind':VERSION,'cases':list(CASES),
        'status':'host_range_boundary_suite_prepared_device_not_run','device_verified':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/query_score_range_boundaries_input1')
    args=p.parse_args()
    try: prepare(args.output.resolve())
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不生成设备答案。',file=sys.stderr); return 1
    return 0


if __name__=='__main__': sys.exit(main())
