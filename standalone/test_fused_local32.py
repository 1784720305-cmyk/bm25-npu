"""Affected local-integer-width/provenance/closure tests. MOCK children only; no CANN or device proof."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import fused_local32 as x
from standalone.test_fused_resource_report import ACTUAL


def main():
    count=0
    def reject(call):
        nonlocal count
        try:call()
        except (ValueError,OSError,subprocess.SubprocessError):count+=1
        else:raise AssertionError('Invalid local32 candidate/proof accepted')
    original=ROOT/'npu_prototype/fused_score_topk'
    wide=x.t.candidate_sources(original);narrow=x.candidate_sources(original)
    assert set(wide)==set(narrow)
    assert all(wide[n]==narrow[n] for n in wide if n!=x.KERNEL);count+=2
    a=wide[x.KERNEL].decode();b=narrow[x.KERNEL].decode()
    # Independent expected diff: only local integer types and one now-redundant
    # widening cast disappear. Preserve both global query-row parameters.
    expected=re.sub(r'\buint64_t\b','uint32_t',a.split(x.SPLIT)[0])
    expected=expected.replace('uint32_t query','uint64_t query')
    expected=expected.replace('static_cast<uint32_t>(winner)','winner')
    assert expected+x.SPLIT+a.split(x.SPLIT)[1]==b
    assert a.split(x.SPLIT)[1]==b.split(x.SPLIT)[1]
    assert re.findall(r'uint64_t\s+(\w+)',b.split(x.SPLIT)[0])==['query','query'];count+=3
    reject(lambda:x.narrow_kernel(wide[x.KERNEL].replace(b'uint64_t equalPosition',b'uint32_t equalPosition')))
    reject(lambda:x.narrow_kernel(wide[x.KERNEL].replace(x.SPLIT.encode(),b'wrong boundary')))
    # Host enforces width<=8192. Verify narrow loop additions (including exhausted sentinel
    # positions) and compact streams never wrap. This is a range model, not a Top-K test.
    rng=random.Random(950)
    for width in (1,10,31,32,255,256,257,1024,1025,4097,8191,8192):
        seen=[]
        for thread in range(256):
            stream=list(range(thread,width,256));seen.extend(stream)
            assert all(v+256<2**32 for v in stream)
            wrapped=[];p=thread
            while p<width:wrapped.append(p);p=(p+256)&0xffffffff
            assert stream==wrapped and p<=8192+255
        assert sorted(seen)==list(range(width));count+=1
    for unused in range(20):
        sizes=[0]*256
        for n in range(rng.randrange(1025)):sizes[rng.randrange(256)]+=1
        start=0
        for size in sizes:
            end=start+size
            assert end<=1024 and list(range(start,end))==[v&0xffffffff for v in range(start,end)]
            start=end
        count+=1
    # Large GLOBAL task row and int64 resident offsets stay wide even though local positions narrow.
    assert 'uint64_t query' in b and 'const uint64_t tasks' in b and 'int64_t pos = begins' in b
    assert 'const int64_t begin = termOffsets' in b;count+=1
    assert 'docs / (*shards) > 8192' in narrow['op_host/bm25_fused_score_top_k.cpp'].decode();count+=1
    with tempfile.TemporaryDirectory(prefix='local32_MOCK_ONLY_',dir=ROOT/'results') as tmp,redirect_stdout(io.StringIO()):
        base=Path(tmp);project=base/'original';(project/'op_kernel').mkdir(parents=True)
        for name in x.fused.generator.SOURCES:
            file=project/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes((original/name).read_bytes())
        (project/'op_kernel/CMakeLists.txt').write_bytes(ACTUAL.encode())
        x.r.write_json(project/'CMakePresets.json',{'configurePresets':[{'cacheVariables':{
            'ASCEND_COMPUTE_UNIT':{'type':'STRING','value':'ascend950'},'vendor_name':{'type':'STRING','value':'bm25_fused_score_topk'}}}]})
        x.r.write_json(project/'bm25_project.json',{'dynamic_ub_bytes':65536,'compute_unit':'ascend950'})
        (project/'build.sh').write_bytes(b'MOCK NO COMPILER\n')
        def tree(p):return {f.relative_to(p).as_posix():f.read_bytes() for f in p.rglob('*') if f.is_file()}
        before=tree(project);target=base/'candidate';meta=x.prepare(project,target)
        assert meta['vendor_name']==x.VENDOR and meta['local_selection_index_bits']==32 and meta['simt_threads']==256
        assert all((target/n).read_bytes()==v for n,v in narrow.items()) and tree(project)==before;count+=2
        reject(lambda:x.prepare(project,target))
        reject(lambda:x.prepare(project,project/'nested'))
        vendor=base/x.VENDOR;(vendor/'op_api/include').mkdir(parents=True);(vendor/'op_api/lib').mkdir()
        header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
        declaration='\n'.join('aclnnStatus '+n+'('+p+');' for n,p in x.fused.PROTOTYPES.items())
        header.write_bytes(declaration.encode());(vendor/'op_api/lib/libcust_opapi.so').write_bytes(b'MOCK NOT ELF')
        build=target/'build_out';build.mkdir();(build/header.name).write_bytes(header.read_bytes())
        (build/'libcust_opapi.so').write_bytes(b'MOCK NOT ELF')
        objects=vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_fused_score_top_k';objects.mkdir(parents=True)
        (objects/'FAKE.o').write_bytes(b'MOCK NOT DEVICE');(build/'FAKE.o').write_bytes(b'MOCK NOT DEVICE')
        assert x.candidate_install(vendor,target,project)['vendor']==str(vendor.resolve());count+=1
        header.write_bytes(declaration.replace('int64_t k','int32_t k').encode())
        reject(lambda:x.candidate_install(vendor,target,project));header.write_bytes(declaration.encode())
        (target/x.KERNEL).write_bytes(narrow[x.KERNEL]+b'// forbidden change\n')
        reject(lambda:x.candidate_install(vendor,target,project));(target/x.KERNEL).write_bytes(narrow[x.KERNEL])
        # Compiler failure must stop before package listing, install, device acceptance and timings.
        failure=base/'compile_failure';failure.mkdir()
        with patch.object(x.subprocess,'run',return_value=SimpleNamespace(returncode=7)) as child, \
             patch.object(x.t.resource,'compiler_reports',side_effect=AssertionError('must not parse success')):
            reject(lambda:x.build_install(project,failure/'project',base/'vendors',failure))
            assert child.call_count==1 and child.call_args.args[0]==['bash','build.sh'];count+=1
        records={n:{'num_queries':8,'query_term_count':24} for n in ('A','B','empty')}
        for mode in ('win','lose','first_error','wrong_result'):
            out=base/mode;out.mkdir();calls=[]
            baseline=base/'baseline_runtime';candidate=base/'candidate_runtime'
            def measured(directory,*args):
                role='candidate' if args[6]==candidate else 'baseline';calls.append(role)
                if mode=='first_error' and role=='candidate':raise ValueError('MOCK first device failure')
                value=.20 if role=='baseline' else (.15 if mode=='win' else .21)
                return {'directory':str(directory),'summary':{n:{'median_batch_ms':value,'p95_batch_ms':value+.01} for n in ('A','B')}}
            def cpu(directory,*args):
                assert args[0]=='cpu' and args[9:]==(32,32,32,5,8,24,5,30);calls.append('cpu')
                return {'directory':str(directory),'actual_threads':32,
                    'summary':{n:{'median_batch_ms':.1,'p95_batch_ms':.11} for n in ('A','B')}}
            action=lambda:x.paired(out,base,{},records,{},base,base,base,baseline,candidate,5,lambda:None)
            with patch.object(x.accepted,'measure',measured),patch.object(x.close.bench,'run_one',cpu), \
                 patch.object(x.close,'cross',side_effect=ValueError('MOCK ranking error') if mode=='wrong_result' else None):
                if mode in ('first_error','wrong_result'):
                    reject(action);assert calls==['baseline','candidate'];count+=1
                else:
                    data=action();assert data['status']==x.STATUS and len(data['per_round'])==6
                    assert data['decision']['keep_fused_candidate']==(mode=='win')
                    assert calls[:4]==['baseline','candidate','candidate','baseline']
                    assert calls.count('cpu')==(6 if mode=='win' else 0);count+=1
        # Finished baseline evidence is re-read/hash checked, never trusted only from its directory name.
        done=base/'completed_proof';done.mkdir();rows=[];cpu_rows=[]
        for i in range(6):
            row={'round':i+1,'process_order':['baseline','candidate']};cpu_row={'round':i+1,'process_order':['cpu','candidate']}
            for role,value in (('baseline',.24),('candidate',.18),('cpu',.12)):
                child=done/f'round{i}'/role;child.mkdir(parents=True)
                x.r.write_json(child/'report.json',{'MOCK':'not a device report'})
                item={'directory':str(child),'report_sha256':x.digest(child/'report.json'),
                    'summary':{n:{'median_batch_ms':value,'p95_batch_ms':value+.01} for n in ('A','B')}}
                if role in ('baseline','candidate'):row[role]=item
                if role in ('cpu','candidate'):cpu_row[role]=item
            rows.append(row);cpu_rows.append(cpu_row)
        comparison={'status':x.accepted.STATUS,'per_round':rows,'cpu_rounds':cpu_rows,
            'decision':x.close.decision(rows),'cpu_comparison':x.close.decision([dict(v,baseline=v['cpu']) for v in cpu_rows])}
        parent={'status':x.accepted.STATUS,'comparison':comparison,'candidate_installation':{'MOCK':1},
            'input_sha256':{'MOCK':2},'source_sha256':x.close.checks.sources(('standalone/fused_threads256.py',))}
        x.r.write_json(done/'report.json',parent);x.r.write_json(done/'comparison.json',comparison)
        assert x.completed_evidence(done,{'MOCK':1},{'MOCK':2});count+=1
        reject(lambda:x.completed_evidence(done,{'MOCK':'wrong install'},{'MOCK':2}))
        child=Path(rows[0]['candidate']['directory'])/'report.json';child.write_bytes(b'corrupt report')
        reject(lambda:x.completed_evidence(done,{'MOCK':1},{'MOCK':2}))
        # Parent writes a failed report and never submits acceptance/timing after build error.
        runner=base/'MOCK_runner';runner.write_bytes(b'MOCK')
        a=SimpleNamespace(input=base/'inputs',previous=base/'old',completed=base/'completed',output=base/'parent_failure',
            project=project,vendors=base/'installed/vendors',merge_project=base/'merge',runner=runner,
            diagnostic_runner=runner,cpu_runner=runner,acceptance=base,device=5)
        frozen=x.close.checks.sources(('standalone/fused_threads256.py',))
        with patch.object(x.t.prior,'evidence',return_value=({'num_docs':262144},{'A':{'num_queries':8,'k':10}},{},{},{})), \
             patch.object(x.accepted,'validate_previous',return_value=({'frozen_source_sha256':frozen},{},base/'runtime',{})), \
             patch.object(x,'completed_evidence',return_value={}),patch.object(x.fused,'installed',return_value={}), \
             patch.object(x.t,'candidate_install',return_value={}), \
             patch.object(x.accepted,'opp_paths',return_value=[str((a.vendors/x.t.VENDOR).resolve()),str((a.vendors/'bm25_topk_merge').resolve())]), \
             patch.object(x,'build_install',side_effect=ValueError('MOCK compile failure')), \
             patch.object(x.t,'diagnostic',side_effect=AssertionError('must not submit')), \
             patch.object(x,'paired',side_effect=AssertionError('must not time')):
            reject(lambda:x.run(a))
        assert json.loads((a.output/'report.json').read_text())['status']=='failed';count+=1
    print(f'PASS: {count} affected bounded-counter/source-immutability/actual-ABI/first-error/paired-gate checks; '
        'Python range models and MOCK children only. No CANN compilation or NPU correctness/performance conclusion.')


if __name__=='__main__':main()
