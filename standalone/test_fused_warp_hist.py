"""Affected per-warp first-pass histogram/UB/source/closure tests; Python models and MOCK processes only."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import fused_warp_hist as x
from standalone.test_fused_ub_contract import constants,audit
from standalone.test_fused_resource_report import ACTUAL


def key(bits):return (~(0 if bits&0x7fffffff==0 else bits))&0xffffffff


def threshold_model(words,k,layout):
    """Model exactly the changed counter addresses plus old three-pass rank, not NPU execution."""
    positive=[(doc,bits) for doc,bits in enumerate(words) if bits&0x7fffffff]
    compact=len(words)>1024 and len(words)<=256*64 and len(positive)<=1024
    if compact and len(positive)<=k:return sorted(range(len(words)),key=lambda i:(key(words[i]),i))[:k]
    source=positive if compact else list(enumerate(words))
    prefix=mask=0;remaining=k
    for pass_number,(shift,bins) in enumerate(((21,2048),(10,2048),(0,1024))):
        buffer=[None]*(layout['kSelectScratchBytes']//4)
        # Same unique-owner clear domains as actual C++; untouched metadata deliberately stays None.
        for thread in range(256):
            for bin in range(thread,bins,256):buffer[bin]=0
            if pass_number==0:
                for word in range(thread,layout['kPrivateHistWords'],256):buffer[layout['kPrivateHist']+word]=0
        expected=Counter()
        for doc,bits in source:
            current=key(bits)
            if current&mask!=prefix:continue
            bin=(current>>shift)&(bins-1);expected[bin]+=1
            warp=(doc%256)//32
            offset=layout['kPrivateHist']+(warp-1)*layout['kHistStride'] if pass_number==0 and warp else 0
            assert buffer[offset+bin] is not None;buffer[offset+bin]+=1
        # Every bin has exactly one reduction owner; read/write sets for different owners do not overlap.
        if pass_number==0:
            for thread in range(256):
                for bin in range(thread,bins,256):
                    buffer[bin]+=sum(buffer[layout['kPrivateHist']+(w-1)*layout['kHistStride']+bin] for w in range(1,8))
        assert buffer[:bins]==[expected[i] for i in range(bins)]
        before=0
        for bin,count in enumerate(buffer[:bins]):
            if before<remaining<=before+count:remaining-=before;prefix|=bin<<shift;mask|=(bins-1)<<shift;break
            before+=count
        else:raise AssertionError('Lost candidate rank')
    oracle=sorted(range(len(words)),key=lambda i:(key(words[i]),i))[:k]
    assert prefix==key(words[oracle[-1]])
    return [i for i in range(len(words)) if key(words[i])<prefix]+sorted(i for i in range(len(words)) if key(words[i])==prefix)[:remaining]


def main():
    count=0
    def reject(call):
        nonlocal count
        try:call()
        except (ValueError,OSError,subprocess.SubprocessError):count+=1
        else:raise AssertionError('Invalid warp histogram accepted')
    original=ROOT/'npu_prototype/fused_score_topk';old=x.t.candidate_sources(original);new=x.candidate_sources(original)
    kernel=new['op_kernel/bm25_fused_score_top_k.cpp'].decode();host=new['op_host/bm25_fused_score_top_k.cpp'].decode()
    prior=old['op_kernel/bm25_fused_score_top_k.cpp'].decode();c,h=constants(kernel),constants(host)
    assert (c['kThreads'],c['kWarps'],c['kHistStride'],c['kPrivateHistWords'])==(256,8,2056,14392)
    assert c['kPrivateHist']==c['kPositiveIds']+1024 and c['kSelectScratchBytes']==74304
    assert c['kBufferBytes']==h['kBufferBytes']==108416 and h['kDynamicUbBytes']==131072;count+=3
    assert h['kMinDCacheBytes']==32768 and h['kCompilerReservedBytes']==8192
    assert h['kDynamicUbBytes']+h['kMinDCacheBytes']+h['kCompilerReservedBytes']==172032<=221184
    assert 'ubBytes < kDynamicUbBytes+kMinDCacheBytes+kCompilerReservedBytes' in host;count+=3
    regions=[('scores',0,c['kBegins']*4),('begins',c['kBegins']*4,c['kEnds']*4),
        ('ends',c['kEnds']*4,c['kTermIds']*4),('terms',c['kTermIds']*4,c['kScoreControl']*4),
        ('control',c['kScoreControl']*4,c['kSelection']*4),
        ('selector',c['kSelection']*4,c['kSelection']*4+c['kSelectScratchBytes'])]
    assert audit(regions,h['kDynamicUbBytes'])==108416;count+=1
    intervals=[(0,2048)]+[(c['kPrivateHist']+(w-1)*c['kHistStride'],c['kPrivateHist']+(w-1)*c['kHistStride']+2048) for w in range(1,8)]
    assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])) and intervals[-1][1]<=c['kSelectScratchBytes']//4
    assert intervals[1][0]>=c['kPositiveIds']+1024;count+=2
    assert kernel[kernel.index('__simt_callee__'):kernel.index('            const uint32_t shift')]==prior[
        prior.index('__simt_callee__'):prior.index('            const uint32_t shift')]
    tail='        const uint32_t strictCount = k - remaining;';end='// Three disjoint regions:'
    assert kernel.split(tail)[1].split(end)[0]==prior.split(tail)[1].split(end)[0]
    marker='constexpr float kMaxFloat = 3.402823466e+38F;'
    assert kernel.split(marker)[1]==prior.split(marker)[1]
    assert 'uint64_t sourceBegin' in kernel and 'int64_t pos = begins' in kernel;count+=4
    rng=random.Random(950)
    for width in (1,10,31,32,255,256,257,1024,1025,4097,8192):
        patterns=[[0x80000000 if i%2 else 0 for i in range(width)], [0x3f800000]*width,
            [rng.choice((0,1,0x3f800000,0x40000000,0x7f7fffff)) for i in range(width)],
            [0x3f800000 if i in (0,width-1) else 0 for i in range(width)]]
        for words in patterns:
            for k in sorted({1,min(width,10),min(width,32)}):
                actual=threshold_model(words,k,c);expected=sorted(range(width),key=lambda i:(key(words[i]),i))[:k]
                assert sorted(actual,key=lambda i:(key(words[i]),i))==expected;count+=1
    # The count model must reflect all logical sources and unchanged producer ownership after compaction.
    fixture={'q':1,'n':32*8192,'k':10,'scores':[0.0]*(32*8192)}
    stats=x.first_pass_counts({'A':fixture,'B':fixture})
    assert stats['A']['blocks']==0 and stats['A']['early_return_blocks']==32;count+=1
    fixture['scores']=[1.0 if i%8==0 else 0.0 for i in range(32*8192)]
    stats=x.first_pass_counts({'A':fixture,'B':fixture})
    assert stats['A']['median_hottest_shared_counter_updates']==1024
    assert stats['A']['median_hottest_private_counter_updates']==128;count+=2
    with tempfile.TemporaryDirectory(prefix='warp_hist_MOCK_ONLY_',dir=ROOT/'results') as tmp,redirect_stdout(io.StringIO()):
        base=Path(tmp);project=base/'original';project.mkdir()
        for name in x.fused.generator.SOURCES:
            file=project/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes((original/name).read_bytes())
        (project/'op_kernel/CMakeLists.txt').write_bytes(ACTUAL.encode())
        x.r.write_json(project/'CMakePresets.json',{'configurePresets':[{'cacheVariables':{
            'ASCEND_COMPUTE_UNIT':{'type':'STRING','value':'ascend950'},'vendor_name':{'type':'STRING','value':'bm25_fused_score_topk'}}}]})
        x.r.write_json(project/'bm25_project.json',{'dynamic_ub_bytes':65536,'compute_unit':'ascend950'})
        (project/'build.sh').write_bytes(b'MOCK NO BUILD')
        tree=lambda path:{str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}
        before=tree(project);target=base/'candidate';data=x.prepare(project,target)
        assert tree(project)==before and data['dynamic_ub_bytes']==131072
        assert all((target/n).read_bytes()==v for n,v in new.items());count+=2
        reject(lambda:x.prepare(project,target));reject(lambda:x.prepare(project,project/'nested'))
        vendor=base/x.VENDOR;(vendor/'op_api/include').mkdir(parents=True);(vendor/'op_api/lib').mkdir()
        header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
        declaration='\n'.join('aclnnStatus '+n+'('+args+');' for n,args in x.fused.PROTOTYPES.items())
        header.write_bytes(declaration.encode());(vendor/'op_api/lib/libcust_opapi.so').write_bytes(b'MOCK NOT ELF')
        build=target/'build_out';build.mkdir();(build/header.name).write_bytes(header.read_bytes());(build/'libcust_opapi.so').write_bytes(b'MOCK NOT ELF')
        objects=vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_fused_score_top_k';objects.mkdir(parents=True)
        (objects/'FAKE.o').write_bytes(b'MOCK NOT DEVICE');(build/'FAKE.o').write_bytes(b'MOCK NOT DEVICE')
        assert x.candidate_install(vendor,target,project)['vendor']==str(vendor.resolve());count+=1
        header.write_bytes(declaration.replace('int64_t k','int32_t k').encode());reject(lambda:x.candidate_install(vendor,target,project))
        header.write_bytes(declaration.encode())
        failure=base/'compiler_failure';failure.mkdir()
        with patch.object(x.subprocess,'run',return_value=SimpleNamespace(returncode=7)) as child:
            reject(lambda:x.build_install(project,failure/'project',base/'vendors',failure));assert child.call_count==1;count+=1
        a=SimpleNamespace(input=base/'inputs',previous=base/'old',completed=base/'completed',project=project,
            vendors=base/'installed/vendors',acceptance=base/'accepted',profile=base/'profile',runner=base/'runner',
            diagnostic_runner=base/'diagnostic',cpu_runner=base/'cpu',device=5,output=base/'run_failure')
        fake_evidence=({}, {}, {}, {}, {},base/'runtime',{},[])
        with patch.object(x.profile,'evidence',return_value=fake_evidence),patch.object(x,'profile_evidence',return_value=({},{})), \
             patch.object(x.profile,'unchanged'),patch.object(x.close.checks,'cache_inputs',return_value={}), \
             patch.object(x,'first_pass_counts',return_value={}),patch.object(x,'build_install',side_effect=ValueError('MOCK compiler failure')), \
             patch.object(x.t,'diagnostic',side_effect=AssertionError('must not submit')), \
             patch.object(x,'paired',side_effect=AssertionError('must not time')):
            reject(lambda:x.run(a));assert json.loads((a.output/'report.json').read_text())['status']=='failed';count+=1
        records={n:{'num_queries':8,'query_term_count':24} for n in ('A','B','empty')}
        for mode in ('win','lose','first_error','bad_cross'):
            out=base/mode;out.mkdir();calls=[];baseline=base/'baseline_runtime';candidate=base/'candidate_runtime'
            def measure(directory,*args):
                role='candidate' if args[6]==candidate else 'baseline';calls.append(role)
                if mode=='first_error' and role=='candidate':raise ValueError('MOCK device failure')
                value=.20 if role=='baseline' else (.15 if mode=='win' else .21)
                return {'directory':str(directory),'summary':{n:{'median_batch_ms':value,'p95_batch_ms':value+.01} for n in ('A','B')}}
            def cpu(directory,*args):
                assert args[0]=='cpu' and args[9:]==(32,32,32,5,8,24,5,30);calls.append('cpu')
                return {'directory':str(directory),'actual_threads':32,'summary':{n:{'median_batch_ms':.1,'p95_batch_ms':.11} for n in ('A','B')}}
            with patch.object(x.accepted,'measure',measure),patch.object(x.close.bench,'run_one',cpu), \
                 patch.object(x.close,'cross',side_effect=ValueError('MOCK wrong IDs') if mode=='bad_cross' else None):
                action=lambda:x.paired(out,a,{},records,{},base,baseline,candidate,lambda:None)
                if mode in ('first_error','bad_cross'):
                    reject(action);assert calls==['baseline','candidate'];count+=1
                else:
                    result=action();assert result['decision']['keep_fused_candidate']==(mode=='win')
                    assert len(result['per_round'])==6 and calls[:4]==['baseline','candidate','candidate','baseline']
                    assert calls.count('cpu')==(6 if mode=='win' else 0);count+=1
    print(f'PASS: {count} affected histogram/rank/address/UB/provenance/first-error/gate checks; '
        'Python models and MOCK processes only, no CANN compilation or NPU correctness/performance evidence.')


if __name__=='__main__':main()
