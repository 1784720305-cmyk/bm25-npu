"""Final source lock and exact selection/scoring models; not a device simulator."""
from pathlib import Path
import json,random,re,struct,sys,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
THREADS=256;WARPS=8;CAPACITY=1024;INF=0x7fffffff
def range_coverage(begin,end):
    old={tid:list(range(begin+tid,end,THREADS)) for tid in range(THREADS)}
    new={tid:[] for tid in range(THREADS)}
    for warp in range(WARPS):
        warp_begin=begin+warp*32;span=end-warp_begin
        iterations=max(0,(span+THREADS-1)//THREADS)
        for iteration in range(iterations):
            for lane in range(32):
                tid=warp*32+lane;pos=warp_begin+lane+iteration*THREADS
                if pos<end:new[tid].append(pos)
    assert new==old
    return old

def append_ranges(ranges,positive,order,capacity=CAPACITY):
    """按term屏障、warp顺序和warp内迭代顺序模拟单次聚合原子分配。"""
    cursor=0;compact=[None]*capacity;atomics=0;expected=set()
    for begin,end in ranges:
        range_coverage(begin,end)
        expected.update(pos for pos in positive if begin<=pos<end)
        for warp in order:
            warp_begin=begin+warp*32;span=end-warp_begin
            iterations=max(0,(span+THREADS-1)//THREADS)
            for iteration in range(iterations):
                hits=[]
                for lane in range(32):
                    pos=warp_begin+lane+iteration*THREADS
                    hits.append(pos<end and pos in positive)
                if not any(hits):continue
                base=cursor;count=sum(hits);cursor+=count;atomics+=1
                for lane,hit in enumerate(hits):
                    if hit:
                        slot=base+sum(hits[:lane])
                        if slot<capacity:
                            assert compact[slot] is None
                            compact[slot]=warp_begin+lane+iteration*THREADS
    assert cursor==len(expected)
    assert atomics<=len(expected)
    if cursor<=capacity:
        assert set(compact[:cursor])==expected and len(set(compact[:cursor]))==cursor
    return compact,cursor,atomics

def collect_streams(streams,bits,k):
    """候选的无序流收集器；也用于debug=2保留的有序owner流。"""
    docs=[doc for stream in streams for doc in stream]
    assert len(docs)==len(set(docs)) and k<len(docs)
    key=lambda doc:(~bits[doc])&0xffffffff
    expected=sorted((key(doc),doc) for doc in docs)[:k]
    threshold=sorted(key(doc) for doc in docs)[k-1]
    strict=[[doc for doc in stream if key(doc)<threshold] for stream in streams]
    equals=[[doc for doc in stream if key(doc)==threshold] for stream in streams]
    strict_count=sum(map(len,strict));remaining=k-strict_count
    assert 0<remaining<=sum(map(len,equals))
    candidates=[(key(doc),doc) for rows in strict for doc in rows]
    heads=[min(rows,default=INF) for rows in equals]
    for rank in range(remaining):
        winner=min(heads);assert winner!=INF
        candidates.append((threshold,winner))
        if rank+1<remaining:
            for tid,rows in enumerate(equals):
                if heads[tid]==winner:
                    heads[tid]=min((doc for doc in rows if doc>winner),default=INF)
    assert sorted(candidates)==expected
    return remaining

def append_models():
    rng=random.Random(95020261028);rows=0;overflow=0;empty=0
    fixed=[(0,1),(0,32),(3,31),(7,263),(19,8192),(1,1025),(255,256),(256,257)]
    ranges=list(fixed)
    for _ in range(180):
        begin=rng.randrange(0,600);end=begin+rng.randrange(0,3500)
        ranges.append((begin,end))
    for index,(begin,end) in enumerate(ranges):
        probability=(0.09,0.02,0.5)[index%3]
        positive={pos for pos in range(begin,end) if rng.random()<probability}
        if index%11==0:positive=set(range(begin,end))
        for order in (list(range(WARPS)),list(reversed(range(WARPS)))):
            compact,total,atomics=append_ranges([(begin,end)],positive,order)
            assert total==len(positive);rows+=1;empty+=atomics==0
            if total>CAPACITY:
                overflow+=1
                assert all(doc is None or begin<=doc<end for doc in compact)
    # 多term保持游标；同一文档只由首次正分事件追加。
    positive={3,31,257,900,1200,4097,7000}
    compact,total,atomics=append_ranges([(0,1024),(1024,4096),(4096,8192)],positive,list(reversed(range(WARPS))))
    assert total==len(positive) and set(compact[:total])==positive
    assert empty and overflow
    return rows

def collector_models():
    rng=random.Random(95020261029);rows=0;unique=0;tied=0
    for k in (1,10,32):
        for count in (k+1,33,127,1024):
            if count<=k:continue
            for pattern in range(12):
                docs=list(range(count));rng.shuffle(docs)
                if pattern>=9:bits={doc:(1,2,0x007fffff,0x00800000,0x7f7fffff)[doc%5] for doc in docs}
                elif pattern%3==0:bits={doc:0x3f800000 for doc in docs}
                elif pattern%3==1:bits={doc:(0x3f800000,0x40000000,0x40400000)[doc%3] for doc in docs}
                else:bits={doc:0x00800000+doc for doc in docs}
                direct=[docs[tid::THREADS] for tid in range(THREADS)]
                remaining=collect_streams(direct,bits,k)
                unique+=remaining==1;tied+=remaining>1;rows+=1
                # 同一收集器对debug=2保序owner段也必须保持精确同分顺序。
                ordered=sorted(docs);owner=[ordered[tid::THREADS] for tid in range(THREADS)]
                streams=[]
                for stream in owner:
                    # owner模型用每线程连续段，跨线程任意乱序不影响全局精确性。
                    streams.append(stream)
                rng.shuffle(streams);collect_streams(streams,bits,k)
    assert unique and tied
    return rows,unique,tied

def boundary_models():
    checked=0
    for count in (0,1,9,10,11,31,32,33,1023,1024,1025,8192):
        positive=set(range(count))
        for order in (list(range(WARPS)),list(reversed(range(WARPS)))):
            compact,total,_=append_ranges([(0,8192)],positive,order)
            for k in (1,10,32):
                mode=int(k<total<=CAPACITY)
                assert mode==int(k<count<=CAPACITY)
                if mode:
                    docs=compact[:total]
                    for bits in ({d:0x3f800000 for d in docs},{d:0x3f800000+(d%3) for d in docs}):
                        collect_streams([docs[tid::THREADS] for tid in range(THREADS)],bits,k)
                checked+=1
    return checked

def ordered_score_models():
    """Overlapping/repeated terms and random warp interleaving; actual FP32 bits."""
    fp=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
    bit=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
    rng=random.Random(95020261030);checks=0;overflow=0;zero=0
    for width in (1,31,32,33,255,256,257,1024,1025,8192):
        for seed in range(6):
            norms=[fp(.5+rng.random()*2) for _ in range(width)]
            postings=[]
            for term in range(4):
                docs=sorted(rng.sample(range(width),int(width*(.09,.5,1,0)[term])))
                postings.append([(doc,float(1+doc%3)) for doc in docs])
            weights=[fp(.7),fp(1.3),0.0,fp(.4)]
            words=[2,0,1,0,3,1] if seed%2 else [2,3,2]
            reference=[0.0]*width;scores=[0.0]*width;compact=[]
            def contribution(term,doc,tf):
                return fp(fp(weights[term]*fp(tf*fp(2.2)))/fp(tf+norms[doc]))
            for term in words:
                rows=postings[term]
                for doc,tf in rows:reference[doc]=fp(reference[doc]+contribution(term,doc,tf))
                # One independent FIFO per warp; arbitrate among warps randomly.
                queues=[list(range(w*32,len(rows),THREADS)) for w in range(WARPS)]
                while any(queues):
                    w=rng.choice([w for w,q in enumerate(queues) if q]);begin=queues[w].pop(0)
                    events=[]
                    for pos in range(begin,min(begin+32,len(rows))):
                        doc,tf=rows[pos];old=scores[doc];total=fp(old+contribution(term,doc,tf));scores[doc]=total
                        if old==0.0 and total>0.0:events.append(doc)
                    compact.extend(events)
            assert list(map(bit,scores))==list(map(bit,reference))
            expected={doc for doc,value in enumerate(reference) if value>0.0}
            assert len(compact)==len(set(compact))==len(expected) and set(compact)==expected
            overflow+=len(compact)>CAPACITY;zero+=not compact
            for k in (1,10,32):
                if k<len(compact)<=CAPACITY:
                    collect_streams([compact[t::THREADS] for t in range(THREADS)],dict(enumerate(map(bit,scores))),k)
            checks+=1
    assert overflow and zero
    return checks

def main():
 import hashlib
 (ROOT/'results').mkdir(exist_ok=True)
 lock=json.loads((ROOT/'delivery_lock.json').read_text(encoding='utf-8'))
 for key in ('frozen_operators','retained_core_source_sha256'):
  for name,sha in lock[key].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 print('Models:',append_models(),collector_models(),boundary_models(),ordered_score_models())
 from standalone import large_corpus as c,final_support as f
 for n in (1024,8192,65536,262144,1048576,4194304,10000000):
  plan=f.layout(n,1024,1);assert plan['list_counts'][-1]==1 and all(a>=b for a,b in zip(plan['list_counts'],plan['list_counts'][1:]))
 with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
  target=Path(temp)/'data';cases={'A':[[i%8,(i+1)%8,i%8] for i in range(8)],'B':[[(i+2)%8,(i+3)%8,(i+4)%8] for i in range(8)]}
  d=c.prepare(target,128,8,20261014,cases);c.validate(target)
  assert d['queries']==8 and d['query_length']==3
  (target/'A.txt').write_text('bad',encoding='utf-8')
  try:c.validate(target)
  except ValueError:pass
  else:raise AssertionError('Corrupt query accepted')
 for script in ('large_corpus.py','build_final.py','benchmark_final.py','bm25s_benchmark.py'):
  p=subprocess.run([sys.executable,str(ROOT/'standalone'/script),'--help'],cwd=ROOT.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  assert p.returncode==0,p.stdout
 print('PASS: frozen kernels/core bytes, mathematical models, corpus/oracle corruption checks, plans and direct final CLIs. No A5 compile/performance claim.')
if __name__=='__main__':main()
