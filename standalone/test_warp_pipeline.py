"""NPU warp流水候选的整数并发模型与源码隔离检查；不是CANN或设备性能证据。"""
from pathlib import Path
import json
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import warp_pipeline_sources as source


def warp_layout(counts,order,threads=256,warps=8):
    """模拟lane31原子分配；空warp不推进游标，begin仅用于其空段。"""
    assert len(counts)==threads and threads==warps*32
    bases=[0]*warps;totals=[sum(counts[w*32:(w+1)*32]) for w in range(warps)]
    cursor=0
    for warp in order:
        if totals[warp]:bases[warp]=cursor;cursor+=totals[warp]
    begins=[]
    for thread,count in enumerate(counts):
        warp=thread//32;lane=thread%32
        begins.append(bases[warp]+sum(counts[warp*32:warp*32+lane]))
    assert cursor==sum(counts)
    return begins,cursor,bases


def pack_model(masks,order,threads=256,capacity=1024,k=10):
    counts=[(low|(high<<32)).bit_count() for low,high in masks]
    begins,total,_=warp_layout(counts,order,threads,threads//32)
    compact=[None]*capacity;writes=0
    for tid,((low,high),begin,count) in enumerate(zip(masks,begins,counts)):
        if begin+count>capacity:continue
        written=0
        for half,membership in enumerate((low,high)):
            remaining=membership
            while remaining:
                lowest=remaining&(-remaining)
                bit=lowest.bit_length()-1
                doc=tid+(half*32+bit)*threads
                slot=begin+written
                assert 0<=slot<capacity and compact[slot] is None
                compact[slot]=doc;written+=1;writes+=1
                remaining^=lowest
        assert written==count
    mode=int(total>k and total<=capacity)
    if mode:assert writes==total and all(doc is not None for doc in compact[:total])
    return compact,total,begins,counts,mode


def collect_model(compact,begins,counts,bits_by_doc,strict_order,k=10,threads=256):
    total=sum(counts);assert k<total<=len(compact)
    key=lambda doc:(~bits_by_doc[doc])&0xffffffff
    threshold=sorted(key(doc) for doc in compact[:total])[k-1]
    expected=sorted(((key(doc),doc) for doc in compact[:total]))[:k]
    strict=[];equals=[]
    for tid,(begin,count) in enumerate(zip(begins,counts)):
        stream=compact[begin:begin+count]
        assert stream==sorted(stream)
        strict.append([doc for doc in stream if key(doc)<threshold])
        equals.append([doc for doc in stream if key(doc)==threshold])
    strict_counts=[len(x) for x in strict]
    strict_begins,strict_total,_=warp_layout(strict_counts,strict_order,threads,threads//32)
    strict_count=k-(k-sum(strict_counts))
    assert strict_total==strict_count
    remaining=k-strict_count
    candidates=[(0xffffffff,0xffffffff) for _ in range(32)]
    for rows,begin in zip(strict,strict_begins):
        assert begin+len(rows)<=strict_count
        for slot,doc in zip(range(begin,begin+len(rows)),rows):candidates[slot]=(key(doc),doc)
    if remaining==1:
        winner=min((rows[0] for rows in equals if rows),default=0xffffffff)
        assert winner!=0xffffffff
        candidates[strict_count]=(threshold,winner)
    else:
        cursors=[0]*threads
        for rank in range(remaining):
            choices=[(rows[cursors[tid]],tid) for tid,rows in enumerate(equals) if cursors[tid]<len(rows)]
            assert choices
            winner,tid=min(choices);cursors[tid]+=1
            candidates[strict_count+rank]=(threshold,winner)
    actual=sorted(candidates)[:k]
    assert actual==expected
    return remaining


def concurrent_models():
    rng=random.Random(95020261028);rows=0;unique=0;tied=0
    fixed=[
        [(0,0) for _ in range(256)],
        [(0xffffffff,0xffffffff) for _ in range(256)],
        [(0x00000001,0)]+[(0,0) for _ in range(255)],
        [(0,0x80000000)]+[(0,0) for _ in range(255)],
        [(0x00000003,0x00000003) for _ in range(256)],
    ]
    cases=list(fixed)
    for _ in range(240):
        probability=(0.09,0.02,0.5)[rng.randrange(3)]
        masks=[]
        for _ in range(256):
            low=sum(1<<bit for bit in range(32) if rng.random()<probability)
            high=sum(1<<bit for bit in range(32) if rng.random()<probability)
            masks.append((low,high))
        cases.append(masks)
    for index,masks in enumerate(cases):
        orders=[list(range(8)),list(reversed(range(8)))]
        for _ in range(2):order=list(range(8));rng.shuffle(order);orders.append(order)
        for order in orders:
            compact,total,begins,counts,mode=pack_model(masks,order)
            docs=[doc for doc in compact[:total] if doc is not None]
            assert len(set(docs))==len(docs)
            rows+=1
            if not mode:continue
            if index%3==0:bits={doc:0x00800000+doc for doc in docs}
            elif index%3==1:bits={doc:(0x3f800000,0x40000000,0x40400000)[doc%3] for doc in docs}
            else:bits={doc:rng.randrange(1,0x7f800000) for doc in docs}
            strict_order=list(reversed(order)) if index%2 else order
            remaining=collect_model(compact,begins,counts,bits,strict_order)
            unique+=remaining==1;tied+=remaining>1
    assert unique and tied
    return rows,unique,tied


def source_contracts():
    original=ROOT/'npu_prototype/fused_score_topk'
    prior=source.base.candidate_sources(original);candidate=source.candidate_sources(original)
    assert prior.keys()==candidate.keys()
    assert all(prior[name]==raw for name,raw in candidate.items() if name!=source.KERNEL)
    old=prior[source.KERNEL].decode('utf-8');new=candidate[source.KERNEL].decode('utf-8')
    begin=new.index(source.COLLECT_MARKER);head,tail=new[:begin],new[begin:]
    for old_text,new_text in reversed(source.COMMON):head=source.replace(head,new_text,old_text)
    for old_text,new_text in reversed(source.COLLECT):tail=source.replace(tail,new_text,old_text)
    assert head+tail==old
    pack=new[new.index('__simt_callee__ __aicore__ inline uint32_t WarpAllocate('):new.index('/*',new.index('PackForSimd'))]
    collect=new[new.index(source.COLLECT_MARKER):]
    assert 'if constexpr (BitmapReady) {\n        begin = WarpAllocate(positive, scratch+kControl);' in pack
    assert 'PrefixPair(positive, invalid' in pack # debug=2及稠密回退仍保留
    assert 'WarpAllocate(privateStrict' in collect and 'PrefixPair(privateStrict' not in collect
    assert 'atomicMin(scratch+kCandidateIds+strictCount, equalHead)' in collect
    assert 'if (remaining == 1)' in collect and 'for (uint32_t rank = 0; rank < remaining;' in collect
    fast=collect.index('        if (remaining == 1) {',collect.index('const uint32_t strictTotal'))
    rank=collect.index('        for (uint32_t rank = 0; rank < remaining;')
    assert fast<rank and 'WriteSelected' in collect[fast:rank] and 'MinimumHead' not in collect[fast:rank]
    assert new.count('VF_CALL<FusedCollectSimt>')==1 and new.count('FindSimdThreshold(local, data.k);')==1
    assert 'kCapacity' in pack and 'scratch[kControl+6]' in collect
    assert source.VENDOR!=source.base.VENDOR
    assert 'if (thread == 0) bitmap[kControl] = 0;' in new
    assert 'scratch[kControl+4] = 0;' in pack and 'scratch[kControl+6] = 0;' in pack
    assert 'kControl+7' in new and 'kControl+7' not in source.WARP_ALLOCATE


def boundary_models():
    checked=0
    for count in (0,1,9,10,11,31,32,33,1023,1024,1025,8192):
        masks=[(0,0)]*256
        for index in range(count):
            doc=(19*index+7)%8192;tid=doc%256;bit=doc//256
            masks[tid]=(masks[tid][0]|(1<<bit),0)
        for order in (list(range(8)),list(reversed(range(8)))):
            for k in (1,10,32):
                compact,total,begins,counts,mode=pack_model(masks,order,k=k)
                assert total==count and mode==int(k<count<=1024)
                if mode:
                    docs=compact[:total]
                    for bits in ({d:0x3f800000+d for d in docs},{d:0x3f800000+(d%3) for d in docs}):
                        collect_model(compact,begins,counts,bits,list(reversed(order)),k=k)
                checked+=1
    return checked


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
        assert metadata['vendor_name']==source.VENDOR and metadata['status']=='warp_pipeline_source_prepared_not_device_verified'
        for name,raw in source.candidate_sources(original).items():assert (target/name).read_bytes()==raw
        preset=json.loads((target/'CMakePresets.json').read_text(encoding='utf-8'))
        values=[x['cacheVariables']['vendor_name']['value'] for x in preset['configurePresets']]
        assert set(values)=={source.VENDOR}


def driver_contracts():
    from standalone import sparse_pack as old_engine
    before=(old_engine.source,old_engine.OWN,old_engine.PASS,old_engine.MISS,old_engine.decision)
    from standalone import warp_pipeline as driver
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
    assert driver.command_for is old_engine.command_for # 冻结CPU命令合同直接复用，不重编译或调参


def header_discovery():
    from standalone import warp_pipeline as d
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        root=Path(folder);atomic=root/'device_atomic_functions.h';shuffle=root/'warp_intrinsics.h'
        atomic.write_text('void atomicAdd();\nvoid atomicMin();\n',encoding='utf-8')
        shuffle.write_text('void __shfl_up();\nvoid asc_shfl();\n',encoding='utf-8')
        initial={'path':str(atomic),'files':{str(atomic):d.digest(atomic)},'scope':'fixture'}
        with patch.object(d.engine,'actual_atomic_headers',return_value=initial):result=d.actual_atomic_headers(root)
        assert set(result['warp_declarations'])=={'atomicAdd','atomicMin','__shfl_up','asc_shfl'}
        assert str(shuffle.resolve()) in result['files']
        assert result['warp_declarations_not_textually_located']==[]
        shuffle.write_text('// fixture with no declarations\n',encoding='utf-8')
        with patch.object(d.engine,'actual_atomic_headers',return_value=initial):missing=d.actual_atomic_headers(root)
        assert set(missing['warp_declarations_not_textually_located'])=={'__shfl_up','asc_shfl'}


def baseline_contracts():
    from standalone import warp_pipeline as d
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        root=Path(folder);completed=root/'accepted';completed.mkdir();pointdir=completed/'N1024';pointdir.mkdir()
        binary=root/'runner';binary.write_bytes(b'fixture runner')
        inputs=root/'inputs';inputs.mkdir();(inputs/'preparation_report.json').write_text('{}',encoding='utf-8')
        vendors=root/'installed';trusted=vendors/d.engine.source.VENDOR;trusted.mkdir(parents=True)
        artifact=trusted/'kernel.o';artifact.write_bytes(b'fixture kernel')
        aliases=root/'runtime';aliases.mkdir();wrong=root/'old_bitmap';wrong.mkdir()
        desc={'docs':1024,'queries':8,'query_length':3,'k':10}
        rows=[]
        def summary(v):return {c:{'median_batch_ms':v,'p95_batch_ms':v+.01} for c in ('A','B')}
        for i in range(6):
            row={'round':i+1}
            for role in d.coverage.process_order(1024,i):
                path=pointdir/f'round{i+1}'/role;path.mkdir(parents=True)
                front=[str(binary),str(inputs/'index'),str(inputs/'A.txt'),str(inputs/'B.txt'),str(path/'actual')]
                args=['5',str(aliases),'10','8','24','5','30','refA','refB'] if role in ('baseline','candidate') else (
                    ['1','10','1','8','24','5','30'] if role.endswith('_single') else ['32','10','32','8','24','5','30'])
                data={'command':front+args,'summary':summary(1. if role=='candidate' else 2.),'runner_sha256':d.digest(binary)}
                d.r.write_json(path/'report.json',data)
                row[role]={'directory':str(path),'report_sha256':d.digest(path/'report.json'),'summary':data['summary']}
            rows.append(row)
        meta={'vendor_name':d.engine.source.VENDOR,'source_sha256':{n:d.simd.hashlib.sha256(b).hexdigest()
            for n,b in d.engine.source.candidate_sources(ROOT/'npu_prototype/fused_score_topk').items()}}
        point={'status':d.SPARSE_PASS,'docs':1024,'device':5,'workload':desc,'per_round':rows,'decision':d.sparse_decision(rows,1024),
            'source_sha256':d.g.checks.sources(d.SPARSE_OWN+d.retained.OWN+d.retained.HOST),
            'previous_sha256':{str(binary.resolve()):d.digest(binary)},'candidate_sha256':{str(artifact.resolve()):d.digest(artifact)},
            'candidate':{'vendor':str(trusted),'metadata':meta}}
        p=pointdir/'report.json';d.r.write_json(p,point)
        tier={'docs':1024,'report':str(p),'report_sha256':d.digest(p),'status':point['status'],'decision':point['decision']}
        d.r.write_json(completed/'report.json',{'status':'actual_sparse_pack_all_scale_coverage_checked','tiers':[tier]})
        a=SimpleNamespace(completed=completed,docs=1024,original=ROOT/'npu_prototype/fused_score_topk',vendors=vendors)
        original_resolve=Path.resolve;bad=False
        def resolved(path,*args,**kwargs):
            # Alias resolution only is injected; real report files, hashes,
            # generated source and command selection are checked locally.
            if path==aliases/'bm25_fused_score_topk':return original_resolve(wrong if bad else trusted)
            return original_resolve(path,*args,**kwargs)
        with patch.object(Path,'resolve',resolved),patch.object(d.retained.corpus,'validate',return_value=(desc,{})):
            result=d.baseline_evidence(a)
            assert result[1]['npu']['command'][6]==str(aliases)
            assert set(result[1])=={'npu',*d.coverage.cpu_roles(1024)}
            bad=True
            try:d.baseline_evidence(a)
            except Exception as error:assert '别名' in str(error)
            else:raise AssertionError('old bitmap alias accepted as sparse baseline')


def sweep_contracts():
    from standalone import warp_pipeline as d
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
        assert [n for n,_ in calls]==list(d.coverage.SIZES)+( [10000000])
        assert calls[0][1] is None and all(p==root/'N1024' for _,p in calls[1:])
        data=json.loads((root/'report.json').read_text(encoding='utf-8'))
        assert data['status']==d.CHECKED and not data['retained_candidate_docs'] and not data['stable_cpu_winning_docs']


def main():
    rows,unique,tied=concurrent_models();boundary=boundary_models();source_contracts();prepare_contracts();driver_contracts()
    header_discovery();baseline_contracts();sweep_contracts()
    print(f'PASS: {rows}组warp乱序/容量模型，{unique}组阈值只补1名、{tied}组阈值需补多名，{boundary}组K1/10/32及容量边界；源码还原、独立vendor、头文件查找、Sparse基线别名和七档调度通过。不是CANN/设备/性能证据。')


if __name__=='__main__':main()
