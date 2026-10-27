"""稀疏 Pack 整数模型与源码隔离检查；不是 CANN、设备或性能证据。"""
from pathlib import Path
import json
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import sparse_pack_sources as source,sparse_pack as driver


def old_emit(low,high,positive,tid=255,threads=256,begin=12345):
    output=[]
    for pos in range(64):
        if len(output)>=positive:break
        membership=low if pos<32 else high
        bit=pos if pos<32 else pos-32
        if membership&(1<<bit):
            output.append((begin+len(output),tid+pos*threads,0x3f800000+pos))
    return output


def new_emit(low,high,positive,tid=255,threads=256,begin=12345):
    output=[]
    for half in range(2):
        if len(output)>=positive:break
        remaining=low if half==0 else high
        while remaining and len(output)<positive:
            lowest=remaining&((-remaining)&0xffffffff)
            bit=(lowest-1).bit_count()
            pos=half*32+bit
            output.append((begin+len(output),tid+pos*threads,0x3f800000+pos))
            remaining^=lowest
    return output


def exact_models():
    rng=random.Random(95020261027);rows=0
    fixed=((0,0),(1,0),(0x80000000,0),(0,1),(0,0x80000000),
        (0xffffffff,0),(0,0xffffffff),(0xffffffff,0xffffffff),
        (0x55555555,0xaaaaaaaa),(0xaaaaaaaa,0x55555555),
        (0x80000001,0x80000001),(0x7fffffff,0xfffffffe))
    pairs=list(fixed)
    for _ in range(3000):
        if rng.randrange(3)==0:
            low=rng.getrandbits(32);high=rng.getrandbits(32)
        else:
            low=sum((1<<bit) for bit in range(32) if rng.random()<.09)
            high=sum((1<<bit) for bit in range(32) if rng.random()<.09)
        pairs.append((low,high))
    for low,high in pairs:
        count=low.bit_count()+high.bit_count()
        limits={count,0,count//2}
        if count:limits.add(rng.randrange(count+1))
        for positive in limits:
            expected=old_emit(low,high,positive)
            actual=new_emit(low,high,positive)
            assert actual==expected
            assert [item[0] for item in actual]==list(range(12345,12345+len(actual)))
            assert [item[1] for item in actual]==sorted(item[1] for item in actual)
            rows+=1
    return rows


def source_contracts():
    original=ROOT/'npu_prototype/fused_score_topk'
    prior=source.base.candidate_sources(original);candidate=source.candidate_sources(original)
    assert prior.keys()==candidate.keys()
    assert all(prior[name]==raw for name,raw in candidate.items() if name!=source.KERNEL)
    old=prior[source.KERNEL].decode('utf-8');new=candidate[source.KERNEL].decode('utf-8')
    assert source.EMIT_OLD in old and source.EMIT_OLD not in new
    assert new.count(source.EMIT_NEW)==1
    assert new.replace(source.EMIT_NEW,source.EMIT_OLD,1)==old
    assert '__popc(lowest-1U)' in new and '__builtin_ctz' not in new and '__ffs' not in new
    assert new.count('PackForSimd<true>')==1 and new.count('PackForSimd<false>')==1
    assert new.count('FindSimdThreshold(local, data.k);')==1
    assert new.count('VF_CALL<FusedCollectSimt>')==1
    assert source.VENDOR!=source.base.VENDOR


def prepare_contracts():
    original=ROOT/'npu_prototype/fused_score_topk'
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        target=Path(folder)/'project'
        def fake_base_prepare(_original,destination):
            destination.mkdir(parents=True)
            for name in source.candidate_sources(original):(destination/name).parent.mkdir(parents=True,exist_ok=True)
            preset={'configurePresets':[{'cacheVariables':{'vendor_name':{'value':source.base.VENDOR}}}]}
            (destination/'CMakePresets.json').write_text(json.dumps(preset),encoding='utf-8')
            return {'vendor_name':source.base.VENDOR}
        with patch.object(source.base,'prepare',side_effect=fake_base_prepare):metadata=source.prepare(original,target)
        assert metadata['vendor_name']==source.VENDOR
        assert metadata['status']=='sparse_pack_source_prepared_not_device_verified'
        candidate=source.candidate_sources(original)
        for name,raw in candidate.items():assert (target/name).read_bytes()==raw
        preset=json.loads((target/'CMakePresets.json').read_text(encoding='utf-8'))
        vendors=[]
        for item in preset.get('configurePresets',[]):
            value=item.get('cacheVariables',{}).get('vendor_name')
            if isinstance(value,dict):value=value.get('value')
            if value is not None:vendors.append(value)
        assert vendors and set(vendors)=={source.VENDOR}


def driver_contracts():
    assert driver.actual_atomic_headers is driver.bitmap.actual_atomic_headers
    assert {'standalone/sparse_pack.py','standalone/sparse_pack_sources.py','standalone/score_bitmap_scales.py',*driver.bitmap.OWN}==set(driver.OWN)
    assert driver.source is source


def summary(value):return {c:{'median_batch_ms':value,'p95_batch_ms':value+.01} for c in ('A','B')}


def cpu_policy():
    row={role:{'summary':summary(value)} for role,value in
        (('cpu',2.),('cpu_directory',2.1),('baseline',1.2),('candidate',1.),('cpu_single',.5),('cpu_directory_single',.6))}
    small=driver.decision([row]*6,1024)
    assert small['versus_frozen_cpus']['performance_target_met_at_this_scale']
    assert not small['versus_all_declared_cpu_configs']['performance_target_met_at_this_scale']
    assert not small['retain_candidate']
    assert driver.decision([row]*6,10000000)['retain_candidate']


def bitmap_baseline_selection():
    # Real local report files and hashes; hardware and already-accepted artifact
    # chain are injected. Ensure the incremental experiment selects BITMAP,
    # never the older pre-bitmap baseline vendor.
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        root=Path(folder);accepted=root/'accepted';accepted.mkdir();(accepted/'report.json').write_text('{}')
        scales=root/'scales';scales.mkdir();inputs=root/'inputs';inputs.mkdir()
        (inputs/'preparation_report.json').write_text('{}')
        binary=root/'runner';binary.write_bytes(b'test runner')
        bitmap_vendor=root/'bitmap_vendor';pre_bitmap=root/'old_vendor'
        generation={'seed':950,'vocab':1024};cases={'A':{'terms':[[0,1,2]]*8},'B':{'terms':[[3,4,5]]*8}}
        desc={'docs':1024,'queries':8,'query_length':3,'k':10,'generation':generation,'cases':cases}
        tenm={**desc,'docs':10000000}
        def cmd(role):
            front=[str(binary),str(inputs/'index'),str(inputs/'A.txt'),str(inputs/'B.txt'),'unused']
            if role in ('baseline','candidate'):return front+['5',str(bitmap_vendor if role=='candidate' else pre_bitmap),'10','8','24','5','30','refA','refB']
            return front+(['1','10','1','8','24','5','30'] if role.endswith('_single') else ['32','10','32','8','24','5','30'])
        roles=driver.coverage.process_order(1024,0);links=[];prior={}
        for round_index in range(6):
            row={}
            for role in roles:
                path=root/f'round{round_index}'/role;path.mkdir(parents=True)
                data={'command':cmd(role),'summary':summary(1. if role=='candidate' else 2.),'runner_sha256':driver.digest(binary)}
                driver.r.write_json(path/'report.json',data)
                row[role]={'directory':str(path),'summary':data['summary'],'report_sha256':driver.digest(path/'report.json')}
                if round_index==0:prior[role]=data
            links.append(row)
        old={'device':5,'docs':10000000,'workload':tenm}
        sweep={'status':driver.coverage.CHECKED,'source_sha256':driver.g.checks.sources(driver.coverage.OWN),
            'previous_sha256':{},'reused_10m':{'report':str(accepted/'report.json'),'report_sha256':driver.digest(accepted/'report.json')},
            'tiers':[{'docs':1024,'status':'six_rounds_checked','workload':desc,'per_round':links,'decision':driver.coverage.point_decision(links,1024)}]}
        driver.r.write_json(scales/'report.json',sweep)
        args=SimpleNamespace(accepted=accepted,scales=scales,docs=1024)
        with patch.object(driver.coverage,'accepted_evidence',return_value=(old,{},prior,{})),patch.object(driver.retained.corpus,'validate',return_value=(desc,{})):
            result=driver.baseline_evidence(args)
            assert result[2]['npu']['command'][6]==str(bitmap_vendor)
            assert set(result[2])=={'npu',*driver.coverage.cpu_roles(1024)}
            # A rehashed report pointing at the old vendor must still fail.
            path=Path(links[0]['candidate']['directory'])/'report.json'
            corrupt=json.loads(path.read_text());corrupt['command'][6]=str(pre_bitmap)
            driver.r.write_json(path,corrupt);links[0]['candidate']['report_sha256']=driver.digest(path)
            driver.r.write_json(scales/'report.json',sweep)
            try:driver.baseline_evidence(args)
            except Exception:pass
            else:raise AssertionError('pre-bitmap vendor incorrectly accepted')


def sweep_continues_losses():
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        output=Path(folder)/'sweep';calls=[]
        row={role:{'summary':summary(1. if role=='candidate' else (1.01 if role=='baseline' else 2.))} for role in driver.coverage.process_order(1024,0)}
        def measured(a):
            calls.append((a.docs,a.candidate_from));a.output.mkdir()
            driver.r.write_json(a.output/'report.json',{'status':driver.MISS,'decision':driver.decision([row]*6,a.docs)})
            return 2
        with patch.object(driver.sys,'platform','linux'),patch.object(driver,'run',side_effect=measured),\
            patch.object(driver.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)),patch('builtins.print'):
            assert driver.sweep(SimpleNamespace(output=output,candidate_from=None))==0
        assert [n for n,_ in calls]==list(driver.coverage.SIZES)+( [10000000])
        assert calls[0][1] is None and all(path==output/'N1024' for _,path in calls[1:])
        report=json.loads((output/'report.json').read_text())
        assert len(report['tiers'])==7 and report['status']=='actual_sparse_pack_all_scale_coverage_checked'
        assert report['retained_candidate_docs']==[]


def candidate_reuse():
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        root=Path(folder);original=ROOT/'npu_prototype/fused_score_topk';artifact=root/'kernel.o';artifact.write_bytes(b'fixture object')
        row={role:{'summary':summary(1. if role=='candidate' else 2.)} for role in driver.coverage.process_order(1024,0)}
        baseline_file=root/'baseline';baseline_file.write_bytes(b'fixture baseline')
        candidate={'vendor':str(root/'vendors'/source.VENDOR),'metadata':{'source_sha256':
            {name:source.base.simd.hashlib.sha256(raw).hexdigest() for name,raw in source.candidate_sources(original).items()}}}
        report={'status':driver.PASS,'docs':1024,'per_round':[row]*6,'decision':driver.decision([row]*6,1024),
            'source_sha256':driver.g.checks.sources(driver.OWN+driver.retained.OWN+driver.retained.HOST),
            'previous_sha256':{str(baseline_file):driver.digest(baseline_file)},'candidate_sha256':{str(artifact):driver.digest(artifact)},
            'candidate':candidate,'affected_checks':{'raw_selector':{'calls':54,'status':'actual_simd_selector_guards_and_strict_bit_rankings_passed'},
                'baseline':{'output_sha256':{'scores':'same'}},'candidate':{'output_sha256':{'scores':'same'}}}}
        driver.r.write_json(root/'report.json',report)
        a=SimpleNamespace(candidate_from=root,original=original,vendors=root/'vendors');proofs={}
        assert driver.reused_candidate(a,proofs)['candidate']==candidate
        assert str(artifact) in proofs and str(baseline_file) in proofs
        artifact.write_bytes(b'changed object')
        try:driver.reused_candidate(a,{})
        except Exception:pass
        else:raise AssertionError('changed candidate binary reused')


def main():
    rows=exact_models();source_contracts();prepare_contracts();driver_contracts()
    cpu_policy();bitmap_baseline_selection();sweep_continues_losses();candidate_reuse()
    print(f'PASS: {rows} 组置位发射等价模型；顺序、槽位、提前停止、工程 vendor、源码逐字还原、位图版基线选择、小库CPU门禁、七档覆盖及候选复用调度通过。不是 CANN/设备/性能证据。')


if __name__=='__main__':main()
