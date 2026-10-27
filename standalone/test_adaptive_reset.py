"""Persistent UB-reset bit models and source/driver contracts; no device evidence."""
from pathlib import Path
import inspect,json,random,struct,sys,tempfile
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from standalone import sparse_pack as accepted
BEFORE=(accepted.source,accepted.OWN,accepted.PASS,accepted.MISS,accepted.decision)
from standalone import adaptive_reset_sources as source,adaptive_reset as driver,warp_pipeline

NO_ID=0xffffffff
def fpbits(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def number(bits):return struct.unpack('<f',struct.pack('<I',bits))[0]


def reset_model():
    rng=random.Random(95020261027);tasks=0;sparse_tasks=0;full_tasks=0;stores=0;old_stores=0
    # Include odd tails, underflow-sized positives, repeated term contributions,
    # poisoned first launches, many blocks, invalid/few/dense predecessor modes.
    for width,k in ((w,k) for w in (33,65,1025,8192) for k in (1,10,32)):
        for blocks in (1,3,8):
            state=[{'bits':[0xdeadbeef]*width,'ids':[],'mode':2,'count':NO_ID} for _ in range(blocks)]
            for launch in range(3):
                for task in range(blocks*14+1):
                    s=state[task%blocks];count=s['count']
                    use_sparse=task>=blocks and s['mode']==1 and k<count<=1024
                    if use_sparse:
                        assert len(s['ids'])==count and len(set(s['ids']))==count
                        for doc in s['ids']:
                            assert 0<=doc<width;s['bits'][doc]=0
                        sparse_tasks+=1;stores+=count
                    else:
                        # Model masked64-word SIMD stores, protecting tail guard.
                        guarded=s['bits']+[0xa5a5a5a5]*64
                        for tile in range((width+63)//64):
                            for lane in range(min(64,width-tile*64)):guarded[tile*64+lane]=0
                        assert guarded[width:]==[0xa5a5a5a5]*64
                        s['bits']=guarded[:width];full_tasks+=1;stores+=width
                    assert s['bits']==[0]*width
                    oracle=[0]*width
                    wanted=min(width,(0,1,k-1,k,k+1,63,730,1023,1024,1025,width)[task%11])
                    docs=rng.sample(range(width),wanted)
                    for slot in range(3):
                        for doc in docs:
                            delta=(0,1,0x3f000000,0x3f800000,0x00800000)[(doc+slot)%5]
                            s['bits'][doc]=fpbits(number(s['bits'][doc])+number(delta))
                            oracle[doc]=fpbits(number(oracle[doc])+number(delta))
                    assert s['bits']==oracle
                    ids=[doc for tid in range(256) for doc in range(tid,width,256) if oracle[doc]!=0]
                    total=len(ids);compact=k<total<=1024
                    s.update(ids=ids if compact else [],count=total if compact else NO_ID,
                        mode=1 if compact else rng.choice((0,1,2)))
                    # Errors after Pack must force next task to a full clear even
                    # when a plausible compact count remains in the old metadata.
                    if task%29==28:s['mode']=2
                    tasks+=1;old_stores+=width
    assert sparse_tasks and full_tasks and stores<old_stores
    return tasks,sparse_tasks,full_tasks,stores,old_stores


def source_contracts():
    original=ROOT/'npu_prototype/fused_score_topk'
    old=source.base.candidate_sources(original);new=source.candidate_sources(original)
    assert old.keys()==new.keys()
    assert all(old[n]==raw for n,raw in new.items() if n!=source.KERNEL)
    text=new[source.KERNEL].decode('utf-8');restored=text
    for before,after in reversed(source.CHANGES):restored=source.simd.t.replace_once(restored,after,before)
    assert restored.encode()==old[source.KERNEL]
    # Prove the borrowed metadata word does not belong to any old selector path.
    before=old[source.KERNEL].decode('utf-8')
    assert 'kControl+4' not in before and 'kControl + 4' not in before
    assert text.count('scratch[kControl+4] =')==1
    assert 'oldMode == 1 && oldCount > data.k && oldCount <= kCapacity' in text
    assert 'if (task >= data.blockNum)' in text and 'if (data.debug != 2)' in text
    assert 'if (previousDoc < width)' in text
    assert source.ZERO_NEW.index('asc_syncthreads();')<source.ZERO_NEW.index('if (resetInvalid != 0)')
    assert 'uint16_t tile = 0' in text and 'V::StoreAlign<uint32_t>(cursor, zero, valid)' in text
    assert source.ZERO_OLD not in text
    assert text.count('VF_CALL<FusedScorePackSimt>')==1
    assert text.count('VF_CALL<FusedCollectSimt>')==1
    assert 'FindSimdThreshold(local, data.k);' in text
    assert 'WarpAllocate' not in text and source.VENDOR!=source.base.VENDOR


def prepare_contracts():
    original=ROOT/'npu_prototype/fused_score_topk'
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        target=Path(folder)/'project'
        def prepared(_original,destination):
            destination.mkdir()
            for name in source.candidate_sources(original):(destination/name).parent.mkdir(parents=True,exist_ok=True)
            (destination/'CMakePresets.json').write_text(json.dumps({'configurePresets':[{'cacheVariables':{'vendor_name':{'value':source.base.VENDOR}}}]}),encoding='utf-8')
            return {'vendor_name':source.base.VENDOR}
        with patch.object(source.base,'prepare',side_effect=prepared):metadata=source.prepare(original,target)
        assert metadata['vendor_name']==source.VENDOR
        assert metadata['status']=='adaptive_reset_source_prepared_not_device_verified'
        for name,raw in source.candidate_sources(original).items():assert (target/name).read_bytes()==raw
        preset=json.loads((target/'CMakePresets.json').read_text(encoding='utf-8'))
        assert preset['configurePresets'][0]['cacheVariables']['vendor_name']['value']==source.VENDOR


def driver_contracts():
    assert BEFORE==(accepted.source,accepted.OWN,accepted.PASS,accepted.MISS,accepted.decision)
    assert inspect.getsource(driver.baseline_evidence)==inspect.getsource(warp_pipeline.baseline_evidence)
    assert driver.SPARSE_OWN==accepted.OWN and driver.source is source
    assert driver.command_for is accepted.command_for
    assert '1048577' in inspect.getsource(driver.run)
    docs=driver.adversarial_documents()
    # First/final fixture leaves and three selection-mode boundaries tile across
    # the entire fixture rather than putting all dense cases in its first leaf.
    for i in range(8193):
        row=next(docs)
        if i in (0,8192):assert row==[4,0,0,1,2,3]
        if i==1023:assert 1 in row and 2 in row and 3 not in row
        if i==1024:assert 1 in row and 2 not in row
        if i==1025:assert 1 not in row
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
        output=Path(folder)/'sweep';calls=[]
        def measured(a):
            calls.append((a.docs,a.candidate_from));a.output.mkdir()
            driver.r.write_json(a.output/'report.json',{'status':driver.MISS,'decision':{'retain_candidate':False,
                'versus_all_declared_cpu_configs':{'performance_target_met_at_this_scale':False}}})
            return 2
        with patch.object(driver.sys,'platform','linux'),patch.object(driver,'run',side_effect=measured),\
            patch.object(driver.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)),patch('builtins.print'):
            assert driver.sweep(SimpleNamespace(output=output,candidate_from=None))==0
        assert [n for n,_ in calls]==list(driver.coverage.SIZES)+( [10000000])
        assert calls[0][1] is None and all(p==output/'N1024' for _,p in calls[1:])
        data=json.loads((output/'report.json').read_text(encoding='utf-8'))
        assert data['status']==driver.CHECKED and not data['retained_candidate_docs']


def main():
    values=reset_model();source_contracts();prepare_contracts();driver_contracts()
    print('PASS persistent reset tasks/sparse/full/new_stores/old_stores='+str(values))
    print('Exact FP32 bit model,masked tails,poison/relaunch,fallback/source restoration/vendor and7-scale scheduling passed. Not CANN/device/performance evidence.')
if __name__=='__main__':main()
