"""Two-pass, file-backed corpus/index/oracle builder; no whole corpus or expanded TF tensor."""
import argparse
from array import array
from collections import Counter,defaultdict
from contextlib import ExitStack
import hashlib
import heapq
import json
import math
import mmap
from pathlib import Path
from random import Random
import shutil
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r,run_search as search

CONTRACT='large_corpus_file_backed_fp32_reference_v1'
LIMIT=2**31-1


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def fp(value):return struct.unpack('<f',struct.pack('<f',value))[0]


def load_raw(path):
    parts=path.read_text(encoding='utf-8').split();r.require(parts and parts[0]=='BM25_QUERY_IDS_V1','Raw grouped query contract missing')
    count=int(parts[1]);r.require(1<=count<=10000,'Raw query count exceeds declared range');pos=2;rows=[]
    for _ in range(count):
        length=int(parts[pos]);pos+=1;r.require(0<=length<=64,'Query length exceeds64')
        rows.append([int(x) for x in parts[pos:pos+length]]);pos+=length
    r.require(pos==len(parts) and all(0<=word<LIMIT for row in rows for word in row),'Invalid query packet')
    return rows


def documents(n,vocab,seed,corpus=None):
    if corpus is None:
        random=Random(seed)
        for _ in range(n):yield [random.randrange(vocab) for unused in range(random.randrange(16,49))]
    else:
        with corpus.open(encoding='utf-8') as f:
            for i in range(n):
                line=f.readline();r.require(bool(line),f'JSONL corpus has fewer than{n} documents')
                row=json.loads(line)
                r.require(isinstance(row,list) and len(row)<=2**24 and all(r.integer(t,0,vocab-1) for t in row),'JSONL requires one token-ID array per article')
                yield row


def mapped(stack,path,size):
    f=stack.enter_context(path.open('w+b'));f.truncate(size)
    return stack.enter_context(mmap.mmap(f.fileno(),size,access=mmap.ACCESS_WRITE))


def distribution(values):
    ordered=sorted(values);n=len(ordered)
    return {'count':n,'min':ordered[0],'median':(ordered[(n-1)//2]+ordered[n//2])/2,
        'p95':ordered[max(0,math.ceil(.95*n)-1)],'max':ordered[-1],'mean':sum(ordered)/n}


def prepare(output,n,vocab,seed,cases,k=10,corpus=None):
    r.require(not output.exists() and sys.byteorder=='little','New output and little-endian host required')
    q=len(cases['A'])
    r.require(1<=n<=LIMIT and n>=k and 1<=q<=10000 and q*((n+8191)//8192*8192)<=LIMIT and 8<=vocab<=65536,
        'Explicit padded INT32 N/Q/index contract exceeded')
    r.require(set(cases)=={'A','B'} and cases['A']!=cases['B'] and all(len(rows)==q and all(len(row)==3 and
        all(0<=term<vocab for term in row) for row in rows) for rows in cases.values()),'Fixed Q,three-word A/B queries required')
    df=array('q',[0])*vocab;total=0;tf_hist=Counter();length_hist=Counter()
    source_sha=digest(corpus) if corpus else None
    for i,doc in enumerate(documents(n,vocab,seed,corpus)):
        counts=Counter(doc);total+=len(doc);length_hist[len(doc)]+=1;tf_hist.update(counts.values())
        for term in counts:df[term]+=1
        if (i+1)%1000000==0:print(f'Pass1 original corpus: {i+1}/{n}',flush=True)
    r.require(total>0,'All-empty corpus unsupported')
    offsets=array('q',[0])
    for count in df:offsets.append(offsets[-1]+count)
    m=offsets[-1];r.require(1<=m<=LIMIT,'Posting count exceeds retained INT32 operator/CPU contract')
    # Actual posting count, including external JSONL: the synthetic48-token
    # estimate in the orchestrator cannot bound an arbitrary supplied corpus.
    needed=8*m+4*n+4*vocab+8*(vocab+1)+8*q*n+16*q*k+512*1024**2
    parent=output.parent
    while not parent.exists():parent=parent.parent
    r.require(needed<shutil.disk_usage(parent).free,'Actual index/oracle disk budget exceeded before allocation')
    average=total/n;idfs=array('f',(math.log1p((n-count+.5)/(count+.5)) for count in df));cursor=array('q',offsets[:-1])
    output.mkdir(parents=True);index=output/'index';index.mkdir()
    (index/'term_offsets.bin').write_bytes(r.encode(offsets));(index/'idf.bin').write_bytes(r.encode(idfs))
    routes=defaultdict(list)
    for case,queries in cases.items():
        (output/case).mkdir();search.write_queries(output/(case+'.txt'),queries)
        for query,row in enumerate(queries):
            for slot,term in enumerate(row):routes[term].append((case,query,slot))
    leaves=(n+8191)//8192
    positives={case:[0]*q for case in cases};heaps={case:[[] for unused in range(q)] for case in cases}
    leaf_positives={case:array('I',[0])*(q*leaves) for case in cases}
    leaf_visits={case:array('I',[0])*(q*leaves) for case in cases}
    with ExitStack() as stack:
        ids=mapped(stack,index/'posting_doc_ids.bin',m*4);tfs=mapped(stack,index/'posting_tf.bin',m*4)
        norms=mapped(stack,index/'doc_norm.bin',n*4)
        refs={case:mapped(stack,output/case/'expected_scores.bin',n*q*4) for case in cases}
        for doc_id,doc in enumerate(documents(n,vocab,seed,corpus)):
            counts=Counter(doc);norm=fp(1.2*(1-.75+.75*len(doc)/average));struct.pack_into('<f',norms,doc_id*4,norm)
            contributions=defaultdict(list)
            for term,tf in counts.items():
                pos=cursor[term];struct.pack_into('<i',ids,pos*4,doc_id);struct.pack_into('<f',tfs,pos*4,tf);cursor[term]+=1
                if term in routes:
                    # Oracle from original documents, not from generated posting arrays.
                    value=fp(fp(idfs[term]*fp(tf*fp(2.2)))/fp(tf+norm))
                    for case,query,slot in routes[term]:
                        contributions[(case,query)].append((slot,value));leaf_visits[case][query*leaves+doc_id//8192]+=1
            for (case,query),values in contributions.items():
                score=0.0
                for slot,value in sorted(values):score=fp(score+value)
                struct.pack_into('<f',refs[case],(query*n+doc_id)*4,score);positives[case][query]+=1
                leaf_positives[case][query*leaves+doc_id//8192]+=1
            # First K article IDs seed zero fillers. Thereafter zero scores cannot
            # outrank them. Every real article enters each affected heap only once.
            for case in cases:
                queries=range(q) if doc_id<k else [query for label,query in contributions if label==case]
                for query in queries:
                    score=struct.unpack_from('<f',refs[case],(query*n+doc_id)*4)[0]
                    item=(score,-doc_id);heap=heaps[case][query]
                    if len(heap)<k:heapq.heappush(heap,item)
                    elif item>heap[0]:heapq.heapreplace(heap,item)
            if (doc_id+1)%1000000==0:print(f'Pass2 index and ordered original-document FP32 oracle: {doc_id+1}/{n}',flush=True)
        r.require(list(cursor)==list(offsets[1:]),'Two corpus passes differ')
        for value in [ids,tfs,norms,*refs.values()]:value.flush()
    if corpus:r.require(digest(corpus)==source_sha,'Source corpus changed during generation')
    hashes={name:digest(index/name) for name in r.INDEX_TYPES}
    record={'contract':r.INDEX_CONTRACT,'num_docs':n,'vocab_size':vocab,'posting_count':m,'k1':1.2,'b':.75,
        'idf_formula':'log1p((N-df+0.5)/(df+0.5))','contribution_policy':'FP32_query_time_formula_no_precomputed_contributions',
        'files':hashes,'index_bytes':sum((index/name).stat().st_size for name in hashes),'device_verified':False}
    record['index_id']=r.identity(record);r.write_json(index/'metadata.json',record)
    query_records={}
    for case,queries in cases.items():
        query,arrays=r.make_queries(record,queries,k)
        for name,values in arrays.items():(output/case/name).write_bytes(r.encode(values))
        r.write_json(output/case/'metadata.json',query)
        ordered=[sorted(heap,key=lambda item:(-item[0],-item[1])) for heap in heaps[case]]
        (output/case/'expected_top_ids.bin').write_bytes(r.encode(array('i',(-item[1] for row in ordered for item in row))))
        (output/case/'expected_top_scores.bin').write_bytes(r.encode(array('f',(item[0] for row in ordered for item in row))))
        query_records[case]={'terms':queries,'hit_records_per_query':[sum(df[t] for t in row) for row in queries],
            'unique_term_hit_records_per_query':[sum(df[t] for t in set(row)) for row in queries],
            'hit_documents_per_query':positives[case],'query_term_df':{str(t):df[t] for row in queries for t in row},
            'leaf_hit_records_distribution':distribution(leaf_visits[case]),'leaf_positive_distribution':distribution(leaf_positives[case]),
            'leaf_selection_path_counts':{'simd_eligible':sum(k<count<=1024 for count in leaf_positives[case]),
                'dense_fallback':sum(count>1024 for count in leaf_positives[case]),'few_or_zero_fallback':sum(count<=k for count in leaf_positives[case])},
            'reference_sha256':digest(output/case/'expected_scores.bin')}
    report={'contract':CONTRACT,'status':'host_file_backed_index_and_original_document_oracle_ready_device_not_run',
        'dataset_kind':'uniform_synthetic_not_MT_approved' if corpus is None else 'external_token_ID_jsonl_not_MT_approved',
        'generation':{'seed':seed,'vocab':vocab,'synthetic_length_range':[16,48],'prefix_stable':True,
            'source_jsonl':str(corpus) if corpus else None,'source_sha256':source_sha},
        'docs':n,'queries':q,'query_length':3,'k':k,'posting_records':m,'tokens':total,'index_bytes':record['index_bytes'],
        'document_length_histogram':dict(length_hist),'posting_tf_histogram':dict(tf_hist),'term_df_distribution':distribution(df),
        'cases':query_records,'index_id':record['index_id'],'source_reference':'ordered_FP32_formula_from_original_document_TF_DF_not_generated_postings'}
    r.write_json(output/'preparation_report.json',report)
    report['file_sha256']={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='preparation_report.json'}
    r.write_json(output/'preparation_report.json',report)
    print(f'Ready N={n},Q={q},K={k},M={m},index_bytes={record["index_bytes"]}; not device evidence.',flush=True)
    return report


def validate(path,rehash=True):
    report=json.loads((path/'preparation_report.json').read_text(encoding='utf-8'))
    r.require(report['contract']==CONTRACT and report['status']=='host_file_backed_index_and_original_document_oracle_ready_device_not_run','Large corpus contract mismatch')
    index=json.loads((path/'index/metadata.json').read_text(encoding='utf-8'))
    r.require(index['index_id']==r.identity(index)==report['index_id'],'Large index identity mismatch')
    if rehash:
        for name,sha in report['file_sha256'].items():
            p=(path/name).resolve();r.require(p.is_relative_to(path.resolve()) and digest(p)==sha,'Large source file changed: '+name)
    for case in ('A','B'):
        r.require(load_raw(path/(case+'.txt'))==report['cases'][case]['terms'],'Raw A/B differs from fixed queries')
        r.require((path/case/'expected_scores.bin').stat().st_size==report['queries']*report['docs']*4,'Oracle size mismatch')
    return report,index


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--docs',type=int,required=True)
    p.add_argument('--vocab',type=int,default=1024);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--queries-a',type=Path,required=True);p.add_argument('--queries-b',type=Path,required=True);p.add_argument('--corpus-jsonl',type=Path)
    a=p.parse_args();prepare(a.output.resolve(),a.docs,a.vocab,a.seed,{'A':load_raw(a.queries_a),'B':load_raw(a.queries_b)},corpus=a.corpus_jsonl)


if __name__=='__main__':main()
