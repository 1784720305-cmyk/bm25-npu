"""Exact integer model and affected contracts. No compiler, device or timing simulation."""
from bisect import bisect_left
from collections import Counter
import contextlib
import io
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import register_select as x
from standalone.test_fused_simd_select import key


def threshold(words,k):
    prefix=mask=0;rank=k
    for shift in (24,16,8,0):
        frequencies=Counter((key(bits)>>shift)&255 for bits in words if key(bits)&mask==prefix)
        cdf=[];total=0
        for b in range(256):total+=frequencies[b];cdf.append(total)
        # Hardware BIN0 and BIN1 expose halves of the same cumulative distribution.
        lo,hi=cdf[:128],cdf[128:]
        candidates=[min(i if value>=rank else 65535,i+128 if other>=rank else 65535)
            for i,(value,other) in enumerate(zip(lo,hi))]
        bucket=min(candidates)
        assert bucket==bisect_left(cdf,rank) and bucket<=255
        before=max(max(a if i<bucket else 0,b if i+128<bucket else 0)
            for i,(a,b) in enumerate(zip(lo,hi)))
        rank-=before;prefix|=bucket<<shift;mask|=255<<shift
        assert 1<=rank<=k and 0<=before<1024
    return prefix,rank


def exact_models():
    rng=random.Random(950);rows=0
    fixed=(1,2,0x7f,0x80,0xff,0x100,0x007fffff,0x00800000,
        0x3f7f7f7f,0x3f808080,0x3fffffff,0x40000000,0x7f7fffff)
    for n in (2,11,33,63,64,65,127,128,129,511,512,513,1023,1024):
        for k in (1,10,32):
            if k>=n:continue
            for kind in range(12):
                words=[(0x3f800000 if kind==0 else rng.choice(fixed) if kind<6 else
                    rng.randrange(1,0x7f800000)) for _ in range(n)]
                prefix,rank=threshold(words,k)
                ordered=sorted(key(word) for word in words)
                assert prefix==ordered[k-1]
                assert rank==k-sum(word<prefix for word in ordered)
                # Unchanged exact collector retains all strict entries and smallest real IDs for ties.
                strict=[i for i,bits in enumerate(words) if key(bits)<prefix]
                equal=[i for i,bits in enumerate(words) if key(bits)==prefix]
                chosen=sorted(strict+equal[:rank],key=lambda i:(key(words[i]),i))
                assert chosen==sorted(range(n),key=lambda i:(key(words[i]),i))[:k]
                rows+=1
    return rows


def source_contracts():
    original=ROOT/'npu_prototype/fused_score_topk'
    prior=x.source.directory.candidate_sources(original);new=x.source.candidate_sources(original)
    name='op_kernel/bm25_fused_score_top_k.cpp'
    assert {k:v for k,v in prior.items() if k!=name}=={k:v for k,v in new.items() if k!=name}
    old_header=(ROOT/x.simd.source.HEADER).read_text(encoding='utf-8')
    begin=old_header.index('// Hardware ACCUMULATE returns a CDF')
    own_header=(ROOT/x.source.HEADER).read_text(encoding='utf-8')
    # Byte-identical score, compaction, collector, fallback, task dispatch and host tiling.
    replacement=old_header[:begin]+own_header
    assert new[name].decode().replace(replacement,old_header,1)==prior[name].decode()
    vector=own_header[:own_header.index('__aicore__ inline void VectorToScalar')]
    assert vector.count('__VEC_SCOPE__')==1
    assert 'GetValue' not in vector and 'SetValue' not in vector
    assert 'VectorToScalar()' not in vector and 'ScalarToVector()' not in vector
    assert 'V::Duplicate<uint16_t>(bucket,bucketReduced,all16)' in vector
    assert 'V::Duplicate<uint16_t>(before,beforeReduced,all16)' in vector
    assert 'static_cast<int16_t>(128)' in vector and 'V::And(selected,compared,valid,all32)' in vector
    assert 'result+64' in vector and 'result+128' not in vector
    # Scratch result words0..127 are disjoint from producer metadata at256..767.
    assert 128<=256 and 'kOwnerBegin = 256, kOwnerCount = 512' in old_header


def command_contracts():
    cpu={'command':['cpu','index','A','B','old','32','10','32','8','24','5','30']}
    npu={'command':['npu','index','A','B','old','5','oldvendor','10','32','8','24','5','30','refA','refB','1','1']}
    destination=Path('new');vendors=Path('newvendor')
    for role,old in (('cpu',cpu),('cpu_directory',cpu),('graph_directory',npu)):
        new=x.measurement_command(old,destination,role,vendors)
        changed=[i for i,(a,b) in enumerate(zip(old['command'],new)) if a!=b]
        assert changed==([4,6] if role=='graph_directory' else [4])
    bad={'command':cpu['command'][:5]+['1']+cpu['command'][6:]}
    try:x.measurement_command(bad,destination,'cpu',vendors)
    except ValueError:pass
    else:raise AssertionError('Changed CPU thread count accepted')
    # Exact six-round gate: one failed round must not be hidden by the aggregate median.
    def value(time):return {'summary':{c:{'median_batch_ms':time,'p95_batch_ms':time} for c in ('A','B')}}
    rows=[{'cpu':value(.066),'cpu_directory':value(.083),'graph_directory':value(.055)} for _ in range(6)]
    assert x.g.summary_gate(rows)['performance_target_met']
    rows[-1]['graph_directory']=value(.067)
    assert not x.g.summary_gate(rows)['performance_target_met']


def counter_contracts():
    with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as temp:
        path=Path(temp);(path/'profiler').mkdir()
        csvfile=path/'profiler/op_summary_1.csv'
        csvfile.write_text('OP Type,Task Start Time(us),vec_ratio,simt_occupancy\n'+
            '\n'.join(f'{op},{i*2+j},{20 if i==1 else 99},N/A' for i in range(3)
                for j,op in enumerate(x.profile.OPERATORS))+'\n',encoding='utf-8')
        analysis={'csv_sha256':{str(csvfile):x.digest(csvfile)},'cases':{}}
        x.r.write_json(path/'device_analysis.json',analysis)
        x.r.write_json(path/'report.json',{'status':x.profile.STATUS,'tool_sha256':x.g.checks.sources(x.profile.TOOLS),
            'device_analysis_sha256':x.digest(path/'device_analysis.json'),'session':{}})
        with patch.object(x.profile.rows,'phases',return_value=[('cold','A',0),('measured','A',0),('warmup','B',0)]):
            got=x.inspect_profile(path)
        for fields in got['counter_fields'].values():
            assert fields['vec_ratio']['raw_statistics']['median']==20
            assert fields['simt_occupancy']['available']==0
            assert fields['simt_occupancy']['raw_statistics'] is None
        csvfile.write_text('tampered',encoding='utf-8')
        try:x.inspect_profile(path)
        except ValueError:pass
        else:raise AssertionError('Modified counters accepted')
    # Optional management-tool failure records uncertainty and cannot invent device utilization.
    with patch('shutil.which',return_value=None):
        assert x.management_snapshot(Path('unused'),'test')=={'status':'npu_smi_unavailable'}


def first_error_stops():
    directory=ROOT/'results';directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        base=Path(temp)
        a=SimpleNamespace(output=base/'new',completed=base/'completed',profile=base/'profile',
            accepted=base/'accepted',original=base/'original',vendors=base/'vendors',toolkit=base/'toolkit')
        a.profile.mkdir();a.accepted.mkdir()
        selector=a.accepted/'selector_host_build/bm25_fused_simd_selector_check'
        selector.parent.mkdir();selector.write_bytes(b'existing verified checker')
        x.r.write_json(a.accepted/'report.json',{'selector_checks':{'runner_sha256':x.digest(selector)}})
        x.r.write_json(a.profile/'report.json',{'prior_sha256':{}})
        (a.profile/'device_analysis.json').write_bytes(b'{}')
        completed={'accepted_report_sha256':x.digest(a.accepted/'report.json'),'device':5,
            'directory_installation':{'metadata':{'source_sha256':{}}}}
        with patch.object(x.sys,'platform','linux'),patch.object(x.profile,'evidence',return_value=(completed,{},None,{})),\
             patch.object(x,'inspect_profile',return_value={'source_sha256':{},'counter_fields':{}}),\
             patch.object(x.g.source,'candidate_sources',return_value={}),\
             patch.object(x,'management_snapshot',return_value={}),patch.object(x,'actual_headers',return_value={}),\
             patch.object(x,'build_install',side_effect=ValueError('first_compile_failure')),\
             patch.object(x.g,'execute') as execute,contextlib.redirect_stdout(io.StringIO()):
            try:x.run(a)
            except ValueError as error:assert str(error)=='first_compile_failure'
            else:raise AssertionError('First failure ignored')
            execute.assert_not_called()
        report=x.simd.json.loads((a.output/'report.json').read_text(encoding='utf-8'))
        assert report['status']=='failed' and report['first_error']=='first_compile_failure' and report['per_round']==[]


def main():
    rows=exact_models();source_contracts();command_contracts();counter_contracts();first_error_stops()
    print(f'PASS: {rows} exact register-threshold model rows; source isolation, frozen CPU commands, six-round gate, counter provenance, first-error stop. Not CANN/device/performance evidence.')


if __name__=='__main__':main()
