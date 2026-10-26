"""Changed mixed-selector models/contracts and fail-stop orchestration, NOT CANN/device tests."""
from collections import Counter
from pathlib import Path
import contextlib
import io
import json
import math
import random
import re
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import fused_simd_select as x
from standalone.test_fused_ub_contract import constants


def key(bits):return (~(0 if bits&0x7fffffff==0 else bits))&0xffffffff


def sync_namespace_contract():
    # Actual yellow 9.2.0 diagnostic: kernel_common.h:90 declares GLOBAL
    # GetTPipePtr() returning AscendC::TPipe*. The two waits keep their order.
    header=(ROOT/x.source.HEADER).read_text(encoding='utf-8')
    assert 'AscendC::GetTPipePtr' not in header and header.count('::GetTPipePtr()')==2
    for function,event in (('VectorToScalar','V_S'),('ScalarToVector','S_V')):
        body=re.search(r'void '+function+r'\(\)\s*\{([^}]+)\}',header).group(1)
        expected=(f'::GetTPipePtr()->FetchEventID(AscendC::HardEvent::{event})',
                  f'AscendC::SetFlag<AscendC::HardEvent::{event}>(e);',
                  f'AscendC::WaitFlag<AscendC::HardEvent::{event}>(e);')
        positions=[body.index(fragment) for fragment in expected]
        assert positions==sorted(positions)
    kernel=x.source.candidate_sources(ROOT/'npu_prototype/fused_score_topk')[
        'op_kernel/bm25_fused_score_top_k.cpp'].decode()
    assert 'AscendC::GetTPipePtr' not in kernel and kernel.count('::GetTPipePtr()')==2


def model(words,k):
    streams=[[(d,words[d]) for d in range(t,len(words),256) if words[d]&0x7fffffff] for t in range(256)]
    compact=[pair for stream in streams for pair in stream]
    if not k<len(compact)<=1024:
        return sorted(range(len(words)),key=lambda d:(key(words[d]),d))[:k]
    prefix=mask=0;rank=k
    for shift in (24,16,8,0):
        histogram=Counter((key(bits)>>shift)&255 for d,bits in compact if key(bits)&mask==prefix)
        cdf=[];total=0
        for b in range(256):total+=histogram[b];cdf.append(total)
        # Model the hardware's accumulated counts and bounded scalar lower bound.
        assert 0<rank<=total<=1024
        lo,hi=0,255
        while lo<hi:
            mid=(lo+hi)//2
            if cdf[mid]>=rank:hi=mid
            else:lo=mid+1
        rank-=cdf[lo-1] if lo else 0
        prefix|=lo<<shift;mask|=255<<shift
    strict=[d for d,bits in compact if key(bits)<prefix]
    assert len(strict)==k-rank
    # Accepted collector is a k-way merge of ASCENDING REAL ID producer streams.
    equal=[[d for d,bits in stream if key(bits)==prefix] for stream in streams]
    for _ in range(rank):
        heads=[stream[0] for stream in equal if stream]
        winner=min(heads);strict.append(winner)
        next(stream for stream in equal if stream and stream[0]==winner).pop(0)
    return sorted(strict,key=lambda d:(key(words[d]),d))


def main():
    original=ROOT/'npu_prototype/fused_score_topk'
    old=x.t.candidate_sources(original);new=x.source.candidate_sources(original)
    kernel=new['op_kernel/bm25_fused_score_top_k.cpp'].decode()
    prior=old['op_kernel/bm25_fused_score_top_k.cpp'].decode()
    host=new['op_host/bm25_fused_score_top_k.cpp'].decode()
    c,h=constants(kernel),constants(host)
    assert (c['kThreads'],c['kWarps'],c['kBufferBytes'],h['kDynamicUbBytes'])==(256,8,50848,65536)
    # Vector CDF, producer ranges, scan scratch, selected32 and compact1024 are disjoint.
    regions=[(0,256),(256,512),(512,768),(c['kScan'],c['kControl']),
             (c['kControl'],c['kCandidateKeys']),(c['kCandidateKeys'],c['kCandidateIds']),
             (c['kCandidateIds'],c['kPositiveBits']),(c['kPositiveBits'],c['kPositiveIds']),
             (c['kPositiveIds'],c['kPositiveIds']+1024)]
    assert all(a[1]<=b[0] for a,b in zip(regions,regions[1:]))
    assert regions[-1][1]*4==c['kSelectScratchBytes']
    for offset in (c['kSelection'],c['kSelection']+c['kPositiveBits']):assert offset*4%32==0
    # No modified score formula, no altered per-term barrier, no replacement fallback or tie collector.
    for begin,end in (('__simt_callee__ __aicore__ inline bool ValidBits','// Three disjoint regions:'),
                      ('        if (thread == 0) {\n            bounds[0]', '        const bool bad = control[0] != 0;')):
        assert prior[prior.index(begin):prior.index(end)]==kernel[kernel.index(begin):kernel.index(end)]
    tail='        const uint32_t strictCount = k - remaining;'
    assert prior[prior.index(tail):prior.index('// Three disjoint regions:')] in kernel
    assert 'static_cast<uint64_t>(length[1]) != static_cast<uint64_t>(queries)*docs' in host
    assert kernel.count('VF_CALL<FusedScorePackSimt>')==kernel.count('VF_CALL<FusedCollectSimt>')==1
    assert 'AscendC::TopK' not in kernel and 'local.SetValue(base+kControl+1, 2U)' in kernel
    rng=random.Random(950);rows=0
    for width in (33,65,1025,2049,4097,8192):
        assert math.gcd(19,width)==1
        for k in (1,10,32):
            counts={min(n,width) for n in (0,1,k-1,k,k+1,63,64,65,511,512,513,1023,1024,1025,width)}
            for n in counts:
                for kind in range(3):
                    words=[0x80000000 if i%2 else 0 for i in range(width)]
                    for j in range(n):
                        # Wide keys exercise all four bytes, byte-127/128 boundaries and exact ties.
                        bits=(0x3f800000 if kind==0 else rng.choice((1,2,0x7fffff,0x800000,
                              0x3f7fffff,0x3f800000,0x3f800001,0x40000000,0x7f7fffff)))
                        words[(j*19+13)%width]=bits
                    actual=model(words,k);expected=sorted(range(width),key=lambda d:(key(words[d]),d))[:k]
                    assert actual==expected
                    assert [words[d] for d in actual]==[words[d] for d in expected];rows+=1
    with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as temp:
        toolkit=Path(temp);head=toolkit/'kernel_reg_compute_histograms_intf.h'
        # Actual API declaration discovery fails closed, without launching tools/devices.
        head.write_text('void Histograms();\n')
        try:x.actual_headers(toolkit)
        except ValueError:pass
        else:raise AssertionError('Missing interfaces accepted')
        names=('LoadAlign','StoreAlign','DeInterleave','Interleave','Compares','Pack','ShiftRights',
               'Not','Duplicate','And','UpdateMask','CreateMask')
        (toolkit/'rest.h').write_text('\n'.join('void '+name+'();' for name in names))
        with contextlib.redirect_stdout(io.StringIO()):record=x.actual_headers(toolkit)
        assert len(record['files'])==2 and len(record['api_names'])==13
        # Independent vendor/ABI check rejects signature mismatch before artifact/device inspection.
        vendor=toolkit/x.VENDOR;header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
        header.parent.mkdir(parents=True);header.write_text('aclnnStatus aclnnBm25FusedScoreTopK(void *wrong);')
        project=toolkit/'project';project.mkdir()
        metadata={'vendor_name':x.VENDOR,'simt_threads':256,'explicit_ub_bytes':50848,'dynamic_ub_bytes':65536,
            'selection_scratch_bytes':16736,'radix_byte_passes':4,'radix_bins':256,'positive_capacity':1024,
            'source_sha256':{}}
        for name,raw in new.items():
            p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            metadata['source_sha256'][name]=x.hashlib.sha256(raw).hexdigest()
        (project/'bm25_project.json').write_text(json.dumps(metadata))
        with patch.object(x.fused.chain,'artifacts',side_effect=AssertionError('ABI gate bypassed')):
            try:x.candidate_install(vendor,project,original)
            except ValueError:pass
            else:raise AssertionError('Wrong actual ABI accepted')
    # Exercise the actual orchestration: fail the first process, no next round/CPU/success report.
    with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as temp:
        a=SimpleNamespace(device=5,input=Path('input'),runner=Path('runner'),cpu_runner=Path('cpu'))
        records={'A':{'num_queries':8,'query_term_count':24}}
        with patch.object(x.accepted,'measure',side_effect=ValueError('first_device_failure')) as measure:
            try:x.paired(Path(temp),a,None,records,{},None,None,None,lambda:None)
            except ValueError:pass
            else:raise AssertionError('First error ignored')
            assert measure.call_count==1
        data=json.loads((Path(temp)/'comparison.json').read_text())
        assert data['status']=='running' and data['per_round']==[] and data['cpu_rounds']==[]
    assert x.profile_evidence is x.old_trial.profile_evidence
    # Both branches of the actual six-pair workflow, without hardware/timing simulation claims.
    for improvement in (False,True):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as temp:
            calls=[];cpu=[]
            def result(ms):return {'summary':{label:{'median_batch_ms':ms,'p95_batch_ms':ms*1.1}
                for label in ('A','B')},'actual_threads':32}
            def measure(directory,*unused):
                calls.append(directory.name)
                return result(1 if directory.name=='baseline' else (0.9 if improvement else 1.01))
            def cpu_run(*args):cpu.append(args);return result(.5)
            with patch.object(x.accepted,'measure',side_effect=measure),patch.object(x.close,'cross'),\
                 patch.object(x.close.bench,'run_one',side_effect=cpu_run),contextlib.redirect_stdout(io.StringIO()):
                out=x.paired(Path(temp),a,None,records,{},None,None,None,lambda:None)
            assert calls[:12]==['baseline','candidate','candidate','baseline']*3
            assert out['status']==x.STATUS and out['decision']['keep_fused_candidate']==improvement
            assert len(cpu)==(6 if improvement else 0)
            assert [r['process_order'] for r in out['cpu_rounds']]==(
                [['cpu','candidate'],['candidate','cpu']]*3 if improvement else [])
            if improvement:
                assert not out['cpu_comparison']['keep_fused_candidate']
    # Parse separate score/collect compiler reports; absence is unknown, never zero usage.
    with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as temp:
        p=Path(temp);(p/'compiler_logs').mkdir()
        (p/'compiler_logs/success.log').write_text('SIMT VF Function properties for FusedScorePackSimt: '
            'Stack size: 68 bytes, Used register number: 100\nSIMT VF Function properties for FusedCollectSimt: '
            'Stack size: 4 bytes, Used register number: 90\n')
        logs,resources=x.compiler_resources(p)
        assert len(logs)==1 and len(resources)==2 and resources[0]['used_register_number']==100
        (p/'compiler_logs/success.log').write_text('No recognizable resource fields')
        assert x.compiler_resources(p)==([],[])
    print(f'Local mixed-selector contract/model checks passed: {rows} full-sort cases, UB/score/tie identity, '
          'actual-header/ABI rejection and first-error stop. No CANN compilation or NPU execution.')


if __name__=='__main__':
    sync_namespace_contract()
    if sys.argv[1:]==['--sync-namespace-only']:
        print('Global GetTPipePtr and unchanged V_S/S_V signal/wait order verified in source and generated kernel. '
              'Only the compile-namespace regression checked; no CANN/device or performance conclusion.')
    elif len(sys.argv)==1:main()
    else:raise SystemExit('Use no arguments or --sync-namespace-only')
