"""Source-derived warp/UB checks and MOCK gate/command checks; no compiler or device evidence."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import fused_threads256 as t
from standalone.test_fused_ub_contract import constants,audit
from standalone.test_fused_resource_report import ACTUAL


def main():
    count=0
    def reject(call):
        nonlocal count
        try:call()
        except (ValueError,OSError):count+=1
        else:raise AssertionError('Invalid candidate accepted')
    baseline=ROOT/'npu_prototype/fused_score_topk';original={n:(baseline/n).read_bytes() for n in t.fused.generator.SOURCES}
    candidate=t.candidate_sources(baseline)
    old=constants(original['op_kernel/bm25_fused_score_top_k.cpp'].decode())
    new=constants(candidate['op_kernel/bm25_fused_score_top_k.cpp'].decode())
    host=constants(candidate['op_host/bm25_fused_score_top_k.cpp'].decode())
    assert old['kThreads']==128 and new['kThreads']==256 and new['kWarps']==8
    assert new['kControl']==new['kScan']+2*new['kWarps']
    assert new['kSelectScratchBytes']==16736 and new['kBufferBytes']==host['kBufferBytes']==50848
    assert host['kDynamicUbBytes']==65536
    assert candidate[t.fused.generator.SOURCES[1]]==original[t.fused.generator.SOURCES[1]];count+=5
    # All other algorithm/formula/branches/barriers are byte-identical after removing allowed replacements/comments.
    import re
    clean=lambda s:re.sub(r'//[^\n]*|/\*.*?\*/','',s,flags=re.S)
    expected=original['op_kernel/bm25_fused_score_top_k.cpp'].decode()
    for a,b in (('kThreads = 128','kThreads = 256'),('kControl = kScan + 8','kControl = kScan + 2*kWarps'),
                ('kSelectScratchBytes == 16704','kSelectScratchBytes == 16736'),('kBufferBytes == 50816','kBufferBytes == 50848')):
        expected=expected.replace(a,b)
    assert clean(expected)==clean(candidate['op_kernel/bm25_fused_score_top_k.cpp'].decode());count+=1
    regions=[('scores',0,4*new['kMaxDocs']),('begins',4*new['kBegins'],4*new['kEnds']),
        ('ends',4*new['kEnds'],4*new['kTermIds']),('terms',4*new['kTermIds'],4*new['kScoreControl']),
        ('score_control',4*new['kScoreControl'],4*new['kSelection']),
        ('selection',4*new['kSelection'],new['kBufferBytes'])]
    assert audit(regions,host['kDynamicUbBytes'])==50848
    assert new['kControl']+8==new['kCandidateKeys']
    assert new['kCandidateIds']+32==new['kPositiveBits']
    assert new['kPositiveIds']+1024==new['kSelectScratchBytes']//4;count+=4
    # PrefixPair writes sixteen totals; all disjoint from control/candidates, not the old eight-slot layout.
    rng=random.Random(20261023)
    for case in range(30):
        a=[rng.randrange(0,8193) for unused in range(256)];b=[rng.randrange(2) for unused in range(256)]
        totals_a=[sum(a[w*32:(w+1)*32]) for w in range(8)]
        totals_b=[sum(b[w*32:(w+1)*32]) for w in range(8)]
        positions={new['kScan']+w for w in range(8)}|{new['kScan']+8+w for w in range(8)}
        assert len(positions)==16 and max(positions)<new['kControl']
        for thread in range(256):
            warp,lane=divmod(thread,32)
            assert sum(a[warp*32:thread])+sum(totals_a[:warp])==sum(a[:thread])
            assert sum(b[warp*32:thread])+sum(totals_b[:warp])==sum(b[:thread])
        assert sum(totals_a)==sum(a) and sum(totals_b)==sum(b);count+=1
    # Membership masks and the producer-owned positive streams cover each ID exactly once at maximum width.
    for width in (10,1024,1025,4097,8192):
        docs=[]
        for thread in range(256):
            stream=list(range(thread,width,256));assert len(stream)<=64
            low=high=0
            for ordinal,doc in enumerate(stream):
                if doc%7==0:
                    if ordinal<32:low|=1<<ordinal
                    else:high|=1<<(ordinal-32)
            chosen=[thread+i*256 for i in range(64) if ((low if i<32 else high)>>(i%32))&1]
            assert chosen==[x for x in stream if x%7==0];docs+=stream
        assert sorted(docs)==list(range(width));count+=1
    with tempfile.TemporaryDirectory(prefix='fused_t256_MOCK_ONLY_',dir=ROOT/'results') as tmp,redirect_stdout(io.StringIO()):
        base=Path(tmp);project=base/'accepted';(project/'op_kernel').mkdir(parents=True)
        for name,data in original.items():
            path=project/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (project/'op_kernel/CMakeLists.txt').write_bytes(ACTUAL.encode())
        (project/'CMakePresets.json').write_bytes(json.dumps({'configurePresets':[{'cacheVariables':{
            'ASCEND_COMPUTE_UNIT':{'type':'STRING','value':'ascend950'},'vendor_name':{'type':'STRING','value':'bm25_fused_score_topk'}}}]}).encode())
        (project/'build.sh').write_bytes(b'MOCK NO BUILD\n')
        t.r.write_json(project/'bm25_project.json',{'dynamic_ub_bytes':65536,'compute_unit':'ascend950'})
        before={p.relative_to(project).as_posix():p.read_bytes() for p in project.rglob('*') if p.is_file()}
        metadata=t.prepare(project,base/'candidate')
        assert metadata['vendor_name']==t.VENDOR and metadata['simt_threads']==256
        assert all((base/'candidate'/name).read_bytes()==data for name,data in candidate.items())
        assert {p.relative_to(project).as_posix():p.read_bytes() for p in project.rglob('*') if p.is_file()}==before;count+=3
        reject(lambda:t.prepare(project,base/'candidate'))
        reject(lambda:t.prepare(project,project/'nested'))
        # Actual generated header is read before any device submission; reject a changed declaration.
        vendor=base/t.VENDOR;(vendor/'op_api/include').mkdir(parents=True);(vendor/'op_api/lib').mkdir()
        header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
        declaration='\n'.join('aclnnStatus '+name+'('+args+');' for name,args in t.fused.PROTOTYPES.items())
        header.write_bytes(declaration.encode());(vendor/'op_api/lib/libcust_opapi.so').write_bytes(b'MOCK NOT ELF')
        build=base/'candidate/build_out';build.mkdir()
        (build/header.name).write_bytes(header.read_bytes());(build/'libcust_opapi.so').write_bytes(b'MOCK NOT ELF')
        kernel=vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_fused_score_top_k';kernel.mkdir(parents=True)
        (kernel/'MOCK.o').write_bytes(b'MOCK NOT DEVICE');(build/'MOCK.o').write_bytes(b'MOCK NOT DEVICE')
        assert t.candidate_install(vendor,base/'candidate',project)['vendor']==str(vendor.resolve());count+=1
        header.write_bytes(declaration.replace('int64_t k','int32_t k').encode())
        reject(lambda:t.candidate_install(vendor,base/'candidate',project))
        runner=base/'FAKE_NOT_EXECUTABLE';runner.write_bytes(b'MOCK '+b' '.join(
            value.encode() for value in t.close.checks.sources(t.fused.HOST_SOURCES).values()))
        evidence=base/'resources';evidence.mkdir()
        names=t.fused.generator.SOURCES+('op_kernel/CMakeLists.txt','CMakePresets.json','build.sh')
        resource_record={'status':'resource_diagnostic_compile_completed','accepted_artifacts_unchanged':True,
            'accepted_runner_sha256':t.digest(runner),'accepted_source_sha256':{n:t.digest(project/n) for n in names},
            'fused_vf_reports':[{'function':'FusedScoreTopKSimt_MOCK','stack_size_bytes':68,'used_register_number':114}]*2}
        t.r.write_json(evidence/'report.json',resource_record)
        assert t.baseline_resources(evidence,project,runner)['attributed_functions']==1;count+=1
        resource_record['fused_vf_reports'][0]['stack_size_bytes']=0;t.r.write_json(evidence/'report.json',resource_record)
        reject(lambda:t.baseline_resources(evidence,project,runner))
        # Entire pipeline stops at the first compiler failure, without install/acceptance/measurement calls.
        root_vendors=base/'installed/vendors';root_vendors.mkdir(parents=True)
        a=SimpleNamespace(input=base,output=base/'compile_failure',project=project,vendors=root_vendors,
            merge_project=base,runner=runner,diagnostic_runner=runner,cpu_runner=runner,acceptance=base,
            baseline_resources=base,device=5)
        def failed_build(command,**kwargs):
            assert command==['bash','build.sh'] and kwargs['cwd']==a.output/'project'
            kwargs['stdout'].write(b'MOCK compiler error\n');return SimpleNamespace(returncode=7)
        with patch.object(t.prior,'evidence',return_value=({'num_docs':262144},{'A':{'num_queries':8,'k':10}}, {},{},{})), \
             patch.object(t,'baseline_resources',return_value={'proof_sha256':{}}), \
             patch.object(t.subprocess,'run',side_effect=failed_build) as compiler, \
             patch.object(t,'diagnostic',side_effect=AssertionError('must stop before device')), \
             patch.object(t,'paired',side_effect=AssertionError('must stop before timing')):
            reject(lambda:t.run(a));assert compiler.call_count==1
        assert json.loads((a.output/'report.json').read_text())['status']=='failed';count+=1
        # MOCK Search children: verify exactly six alternating pairs; no CPU execution unless both A/B retain.
        for mode in ('win','lose','first_error','bad_cross'):
            output=base/mode;output.mkdir();calls=[]
            def run(directory,*args):
                role='candidate' if args[6]==base/'candidate_runtime' else 'baseline'
                calls.append(role)
                if mode=='first_error' and role=='candidate':raise ValueError('MOCK device error')
                value=.20 if role=='baseline' else (.15 if mode=='win' else .21)
                return {'directory':str(directory),'summary':{x:{'median_batch_ms':value,'p95_batch_ms':value+.01} for x in ('A','B')}}
            def cpu(directory,*args):
                assert args[0]=='cpu' and args[8]=={'topk_acceptance':{'workspace_bytes':0}}
                assert args[9:]==(32,32,32,5,8,24,5,30)
                calls.append('cpu')
                return {'directory':str(directory),'actual_threads':32,
                    'summary':{x:{'median_batch_ms':.1,'p95_batch_ms':.11} for x in ('A','B')}}
            records={x:{'num_queries':8,'query_term_count':24} for x in ('A','B','empty')}
            action=lambda:t.paired(output,base,{},records,{},base,base,base,base/'baseline_runtime',
                base/'candidate_runtime',5,lambda:None)
            with patch.object(t.close,'run_fused',run),patch.object(t.close.bench,'run_one',cpu), \
                 patch.object(t.close,'cross',side_effect=ValueError('MOCK wrong rank') if mode=='bad_cross' else None):
                if mode in ('first_error','bad_cross'):reject(action)
                else:
                    record=action();assert record['status']==t.STATUS
                    assert len(record['per_round'])==6
                    assert record['decision']['keep_fused_candidate']==(mode=='win')
                    assert calls[:4]==['baseline','candidate','candidate','baseline']
                    assert calls.count('cpu')==(6 if mode=='win' else 0);count+=1
            if mode=='first_error':assert calls==['baseline','candidate'];count+=1
            if mode=='bad_cross':assert calls==['baseline','candidate'];count+=1
    assert {n:(baseline/n).read_bytes() for n in original}==original;count+=1
    print(f'PASS: {count} affected 256-thread layout/prefix/membership/immutable-copy/paired-gate checks. MOCK children and host models only; no CANN compilation, installation or NPU execution.')


if __name__=='__main__':main()
