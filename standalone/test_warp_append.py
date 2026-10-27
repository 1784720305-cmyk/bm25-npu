"""950 warp聚合直接紧凑的整数/顺序模型与源码隔离检查；不是CANN或设备证据。"""
from pathlib import Path
import json
import random
import re
import struct
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import warp_append_sources as source

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


def source_contracts():
    original=ROOT/'npu_prototype/fused_score_topk'
    prior=source.base.candidate_sources(original);candidate=source.candidate_sources(original)
    assert prior.keys()==candidate.keys()
    assert all(prior[name]==raw for name,raw in candidate.items() if name!=source.KERNEL)
    old=prior[source.KERNEL].decode('utf-8');new=candidate[source.KERNEL].decode('utf-8')
    restored=source.simd.t.replace_once(new,source.DIRECT_PACK,source.pack_function(old))
    score_at=restored.index(source.SCORE_MARKER);collect_at=restored.index(source.COLLECT_MARKER)
    prefix,score,suffix=restored[:score_at],restored[score_at:collect_at],restored[collect_at:]
    for old_text,new_text in reversed(source.COMMON):score=source.simd.t.replace_once(score,new_text,old_text)
    for old_text,new_text in reversed(source.COLLECT):suffix=source.simd.t.replace_once(suffix,new_text,old_text)
    assert prefix+score+suffix==old
    # Reverse substitutions alone did not catch the old generated stray Pack tail.
    clean=re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"','',new)
    depth=0
    for char in clean:
        depth+=(char=='{')-(char=='}')
        assert depth>=0,'Generated C++ closes a scope before opening it'
    assert depth==0,'Generated C++ has unbalanced scopes'
    assert source.pack_function(new)==source.DIRECT_PACK
    after_pack=new.split(source.DIRECT_PACK,1)[1].split('RegisterThreshold(',1)[0]
    assert 'PrefixPair(' not in after_pack and 'scratch[' not in after_pack
    score=new[new.index(source.SCORE_MARKER):new.index(source.COLLECT_MARKER)]
    collect=new[new.index(source.COLLECT_MARKER):]
    normal_pack=source.DIRECT_PACK[source.DIRECT_PACK.index('if constexpr (BitmapReady)'):source.DIRECT_PACK.index('    } else {')]
    assert 'asc_atomic_or' not in score and 'asc_ballot' in score and 'atomicAdd(bitmap+kControl' in score
    assert 'asc_shfl(warpBase, 0, kWarp)' in score and 'iterations' in score
    scoring_loop=score.split(source.LOOP_HEADER_NEW,1)[1].split('const uint32_t hits = asc_ballot',1)[0]
    assert 'continue;' not in scoring_loop and 'return;' not in scoring_loop
    assert 'PrefixPair' not in normal_pack and 'lowMask' not in normal_pack and 'highMask' not in normal_pack
    assert 'scratch[kPositiveBits+pos] = scores[doc]' in normal_pack
    assert 'PrefixPair(positive, invalid' in source.DIRECT_PACK # debug=2原路径保留
    assert 'id > winner && id < equalHead' in collect and 'equalPosition' not in collect
    assert new.count('PackForSimd<true>')==1 and new.count('PackForSimd<false>')==1
    assert new.count('FindSimdThreshold(local, data.k);')==1 and new.count('VF_CALL<FusedCollectSimt>')==1
    assert source.VENDOR!=source.base.VENDOR


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


def prepare_contracts():
    original=ROOT/'npu_prototype/fused_score_topk'
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        target=Path(folder)/'project'
        def fake_prepare(_original,destination):
            destination.mkdir(parents=True)
            for name in source.candidate_sources(original):(destination/name).parent.mkdir(parents=True,exist_ok=True)
            preset={'configurePresets':[{'cacheVariables':{'vendor_name':{'value':source.base.VENDOR}}}]}
            (destination/'CMakePresets.json').write_text(json.dumps(preset),encoding='utf-8')
            return {'vendor_name':source.base.VENDOR}
        with patch.object(source.base,'prepare',side_effect=fake_prepare):metadata=source.prepare(original,target)
        assert metadata['vendor_name']==source.VENDOR and metadata['status']=='warp_append_source_prepared_not_device_verified'
        for name,raw in source.candidate_sources(original).items():assert (target/name).read_bytes()==raw
        preset=json.loads((target/'CMakePresets.json').read_text(encoding='utf-8'))
        values=[x['cacheVariables']['vendor_name']['value'] for x in preset['configurePresets']]
        assert set(values)=={source.VENDOR}


def driver_contracts():
    from standalone import sparse_pack as old_engine
    before=(old_engine.source,old_engine.OWN,old_engine.PASS,old_engine.MISS,old_engine.decision)
    from standalone import warp_append as driver
    assert before==(old_engine.source,old_engine.OWN,old_engine.PASS,old_engine.MISS,old_engine.decision)
    assert driver.source is source and driver.SPARSE_OWN==old_engine.OWN
    rows=[]
    for i in range(6):
        def result(median,p95):return {'summary':{case:{'median_batch_ms':median,'p95_batch_ms':p95} for case in ('A','B')}}
        rows.append({'round':i+1,'baseline':result(100,120),'candidate':result(80,90),
            'cpu':result(100,110),'cpu_directory':result(110,115)})
    verdict=driver.decision(rows,262144)
    assert verdict['retain_candidate'] and verdict['baseline'].startswith('accepted_sparse_pack')
    assert verdict['versus_verified_npu']['A']['wins']==6
    assert driver.command_for is old_engine.command_for
    raw={'calls':54,'status':'actual_simd_selector_guards_and_strict_bit_rankings_passed'}
    affected={'contract':driver.AFFECTED_CONTRACT,'normal_score_ks':[1,10,32],'raw_selector':raw}
    for suffix in ('','_k1','_k32'):
        for role in ('baseline','candidate'):affected[role+suffix]={'output_sha256':{'scores':'a','ids':'b'}}
    driver.check_affected(affected)
    affected['candidate_k32']={'output_sha256':{'scores':'a','ids':'wrong'}}
    try:driver.check_affected(affected)
    except (ValueError,RuntimeError):pass
    else:raise AssertionError('Candidate reuse accepted an incorrect normal K32 ranking')


def header_discovery():
    from standalone import warp_append as d
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        root=Path(folder);atomic=root/'device_atomic_functions.h';warp=root/'warp_intrinsics.h'
        atomic.write_text('void atomicAdd();\n',encoding='utf-8')
        warp.write_text('void asc_ballot();\nvoid asc_shfl();\n',encoding='utf-8')
        initial={'path':str(atomic),'files':{str(atomic):d.digest(atomic)},'scope':'fixture'}
        with patch.object(d.engine,'actual_atomic_headers',return_value=initial):result=d.actual_atomic_headers(root)
        assert set(result['warp_declarations'])=={'atomicAdd','__popc','asc_ballot','asc_shfl'}
        assert result['warp_declarations_not_textually_located']==['__popc']
        warp.write_text('// no declarations\n',encoding='utf-8')
        with patch.object(d.engine,'actual_atomic_headers',return_value=initial):missing=d.actual_atomic_headers(root)
        assert set(missing['warp_declarations_not_textually_located'])=={'__popc','asc_ballot','asc_shfl'}


def sweep_contracts():
    from standalone import warp_append as d
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        root=Path(folder)/'sweep';calls=[]
        def measure(a):
            calls.append((a.docs,a.candidate_from));a.output.mkdir()
            d.r.write_json(a.output/'report.json',{'status':d.MISS,'decision':{'retain_candidate':False,
                'versus_all_declared_cpu_configs':{'performance_target_met_at_this_scale':False}}})
            return 2
        with patch.object(d.sys,'platform','linux'),patch.object(d,'run',side_effect=measure),\
            patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)),patch('builtins.print'):
            assert d.sweep(SimpleNamespace(output=root,candidate_from=None))==0
        assert [n for n,_ in calls]==list(d.SIZES) and d.SIZES[0]==262144
        assert set(d.SIZES)==set(d.coverage.SIZES+(10000000,)) and len(d.SIZES)==7
        assert calls[0][1] is None and all(p==root/'N262144' for _,p in calls[1:])
        data=json.loads((root/'report.json').read_text(encoding='utf-8'))
        assert data['status']==d.CHECKED and not data['retained_candidate_docs']


def main():
    rows=append_models();collectors,unique,tied=collector_models();boundary=boundary_models();ordered=ordered_score_models()
    source_contracts();prepare_contracts();driver_contracts();header_discovery();sweep_contracts()
    print(f'PASS: {rows}组warp追加模型，{ordered}组重叠/重复词FP32及随机warp调度，{collectors}组无序收集（{unique}单名填充/{tied}多名同分），{boundary}组K及容量边界；完整生成C++结构、无跳过集合指令、源码逐字还原、独立vendor、头文件和七档调度通过。不是CANN/设备/性能证据。')


if __name__=='__main__':main()
