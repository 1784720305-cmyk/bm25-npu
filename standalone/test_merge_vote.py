"""Merge唯一赢家快速路径模型；不是CANN或设备证据。"""
import json,os,random,shutil,struct,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import merge_vote_sources as source,merge_vote_run as runner

NO=2**31-1;WARP=32

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def better(a,b):return a[0]>b[0] or (a[0]==b[0] and a[1]<b[1])

def old_reduce(values):
    values=list(values)
    for mask in (16,8,4,2,1):
        other=[values[lane^mask] for lane in range(WARP)]
        for lane in range(WARP):
            if better(other[lane],values[lane]):values[lane]=other[lane]
    return values[0]

def merge(lists,k,fast):
    heads=[(rows[0] if rows else (-1.0,NO)) for rows in lists]
    heads += [(-1.0,NO)]*(WARP-len(heads));used=[0]*WARP;out=[];paths=[0,0]
    for _ in range(k):
        if not fast:
            winner=old_reduce(heads);paths[1]+=1
        else:
            scores=[value[0] for value in heads]
            for mask in (16,8,4,2,1):
                other=[scores[lane^mask] for lane in range(WARP)]
                for lane in range(WARP):
                    if other[lane]>scores[lane]:scores[lane]=other[lane]
            mask=0
            for lane,value in enumerate(heads):
                if value[0]==scores[lane] and value[1]!=NO:mask|=1<<lane
            if mask and mask&(mask-1)==0:
                lane=mask.bit_length()-1;winner=heads[lane];paths[0]+=1
            else:
                winner=old_reduce(heads);paths[1]+=1
        out.append(winner)
        for lane,value in enumerate(heads):
            if value[1]==winner[1] and winner[1]!=NO:
                used[lane]+=1
                rows=lists[lane] if lane<len(lists) else []
                heads[lane]=rows[used[lane]] if used[lane]<len(rows) else (-1.0,NO)
    return out,paths

def models():
    rng=random.Random(95020261030);cases=fast=fallback=signed=0
    for shards in (1,2,8,31,32):
        for k in (1,3,10,32):
            for pattern in range(16):
                lists=[]
                for lane in range(shards):
                    rows=[]
                    for rank in range(k):
                        if pattern==0:score=f32(1.0)
                        elif pattern==1:score=f32(0.0 if (lane+rank)%2 else -0.0)
                        elif pattern<6:score=f32((rank*7+lane*3)%4+.25)
                        else:score=f32(rng.random()*100)
                        rows.append((score,lane*100003+rank))
                    rows.sort(key=lambda x:(-x[0],x[1]));lists.append(rows)
                old,p0=merge(lists,k,False);new,p1=merge(lists,k,True)
                assert [(bits(s),i) for s,i in old]==[(bits(s),i) for s,i in new]
                assert p0[1]==k and p1[0]+p1[1]==k
                fast+=p1[0];fallback+=p1[1];signed+=pattern==1;cases+=1
    assert fast and fallback and signed
    return cases,fast,fallback


def boundary_models():
    count=0
    tiny=f32(2**-149);huge=f32(3.4028234663852886e38)
    fixtures=[[],[[]],[[],[]],[[(-0.0,7)]],[[(-0.0,7)],[(0.0,3)]],
        [[(tiny,99)],[(huge,1)]], [[(1.0,31-i)] for i in range(32)],
        [[(float(32-i),i)] for i in range(32)],
        [[(0.0,100),(0.0,101)],[],[(-0.0,2),(-0.0,3)]]]
    for lists in fixtures:
        for k in (1,3,10,32):
            old,_=merge(lists,k,False);new,_=merge(lists,k,True)
            gold=sorted([v for rows in lists for v in rows],key=lambda x:(-x[0],x[1]))[:k]
            gold += [(-1.0,NO)]*(k-len(gold))
            encode=lambda values:[(bits(s),i) for s,i in values]
            assert encode(old)==encode(new)==encode(gold);count+=1
    return count

def contracts():
    original=ROOT/'npu_prototype/topk_merge'
    prior=source.base.merge_sources(original);candidate=source.candidate_sources(original)
    assert candidate.keys()==prior.keys()
    assert all(candidate[name]==raw for name,raw in prior.items() if name!=source.KERNEL)
    text=candidate[source.KERNEL].decode();old=prior[source.KERNEL].decode()
    restored=source.simd.t.replace_once(text,source.INCLUDE_NEW,source.INCLUDE_OLD)
    restored=source.simd.t.replace_once(restored,source.REDUCE_NEW,source.REDUCE_OLD)
    assert restored==old
    assert text.count('asc_ballot')==1 and '__popc(winners-1U)' in text
    assert text.count('asc_shfl_xor(winnerId')==1,'同分必须保留且只保留原ID归约'
    assert source.VENDOR!=source.base.MERGE and runner.source is source
    assert 'standalone/merge_vote_run.py' in runner.OWN
    assert runner.builder.source is source.base,'导入候选不得改已验收builder全局配置'
    assert runner.e.build_operator.__globals__['source'] is source
    assert runner.e.build_operator.__globals__ is not runner.builder.build_operator.__globals__
    # Direct-file launch must work without a repo-root PYTHONPATH.
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8'
    result=subprocess.run([sys.executable,str(ROOT/'standalone/merge_vote_run.py'),'--help'],
        cwd=ROOT.parent,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    assert result.returncode==0,result.stdout
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='merge_vote_LOCAL_',dir=ROOT/'results') as tmp:
        folder=Path(tmp);template=folder/'original';target=folder/'candidate'
        shutil.copytree(original,template)
        (template/'bm25_project.json').write_text(json.dumps({'compute_unit':'ascend950'}),encoding='utf-8')
        (template/'CMakePresets.json').write_text(json.dumps({'configurePresets':[{'cacheVariables':{
            'ASCEND_COMPUTE_UNIT':'ascend950','vendor_name':{'type':'STRING','value':'old_vendor'}}}]}),encoding='utf-8')
        metadata=source.prepare(template,target)
        assert metadata['vendor_name']==source.VENDOR
        assert all((target/name).read_bytes()==raw for name,raw in candidate.items())
        assert all((template/name).read_bytes()==(original/name).read_bytes() for name in prior)
        preset=json.loads((target/'CMakePresets.json').read_text(encoding='utf-8'))
        assert preset['configurePresets'][0]['cacheVariables']['vendor_name']['value']==source.VENDOR
        assert runner.builder.source is source.base


def gates():
    item=lambda x:{'summary':{c:{'median_batch_ms':x,'p95_batch_ms':x*1.1} for c in ('A','B')}}
    rows=[{'baseline':item(1),'candidate':item(.94),'cpu':item(2),'cpu_directory':item(2)} for _ in range(6)]
    assert runner.e.decision(rows,262144)['retain_candidate']
    rows[0]['candidate']=item(1.01)
    assert not runner.e.decision(rows,262144)['retain_candidate']
    rows=[{'baseline':item(1),'candidate':item(.98),'cpu':item(2),'cpu_directory':item(2)} for _ in range(6)]
    verdict=runner.e.decision(rows,262144)
    assert not verdict['retain_candidate'] and verdict['versus_all_declared_cpu_configs']['performance_target_met_at_this_scale']
    rows=[{'baseline':item(1),'candidate':item(.94),'cpu':item(.93),'cpu_directory':item(2)} for _ in range(6)]
    assert not runner.e.decision(rows,262144)['retain_candidate']

if __name__=='__main__':
    print('merge models',models());print('boundary models',boundary_models());contracts();gates()
    print('PASS: source-only change/restoration, unique/tie/empty/signed-zero/extreme-score models, isolated builder/vendor, direct CLI and independent incremental/CPU gates; device performance remains unverified')
