"""Bounded scaling contracts, streaming independent oracle, hierarchical ranks and honest gates."""
from array import array
import contextlib
import io
import json
from pathlib import Path
import random
import re
import shutil
import struct
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import large_search as x,large_corpus as c,large_search_sources as s,resident_index as r
from standalone.test_fused_ub_contract import constants


def plan_contracts():
    wanted=((32,[32,1]),(128,[128,4,1]),(512,[512,16,1]),(1221,[1221,39,2,1]))
    for n,(leaves,lists) in zip(x.SIZES,wanted):
        p=x.layout(n,1024,n*32)
        assert p['leaves']==leaves and p['list_counts']==lists and p['padded_docs']-n<8192
        assert (p['packet_input_bytes'],p['packet_output_bytes'])==(192,640)
    assert x.layout(10000000,1024,320000000)['candidate_bytes']<1024**2
    for args in ((2**31-1,1024,100),(10000000,1024,2**31),(8192*65537,1024,100),
        (8192,1024,100,10000,0),(100,1024,100,8,10,-1),(100,1024,0)):
        try:x.layout(*args)
        except ValueError:pass
        else:raise AssertionError('Unsupported capacity accepted')
    header=(ROOT/'common/large_search_plan.h').read_text(encoding='utf-8')
    assert 'Leaf=8192,FanIn=32,MaxLeaves=65536,Limit=2147483647' in header
    assert 'Mul(leaves,Leaf)' in header and 'q<=Limit/padded' in header


def sources():
    original=ROOT/'npu_prototype/fused_score_topk'
    before=s.base.candidate_sources(original);after=s.fused_sources(original)
    kernel='op_kernel/bm25_fused_score_top_k.cpp';host='op_host/bm25_fused_score_top_k.cpp'
    prior=before[kernel].decode();new=after[kernel].decode()
    restored=new.replace('            topIds[query*k + lane] = valid ? static_cast<int32_t>(scratch[kControl+7]+id) : -1;',
        '            topIds[query*k + lane] = valid ? static_cast<int32_t>(id) : -1;').replace(
        '    if (thread==0) memory[kSelection+kControl+7]=(blockId%shards)*width;\n','')
    assert restored==prior
    prior_host=before[host].decode();new_host=after[host].decode()
    assert new_host.replace('*shards > 65536','(*shards > 32 && *shards != 64)')==prior_host
    const=constants(new);assert const['kBufferBytes']==50848 and const['kMaxDocs']==8192 and const['kCapacity']==1024
    assert '*debug != 2' in new_host and 'docs / (*shards) > 8192' in new_host
    # Scratch control+7 survives all accepted selection and fallback paths.
    assert not re.search(r'kControl\s*\+\s*7',prior)
    merge=s.merge_sources(ROOT/'npu_prototype/topk_merge')
    body=merge['op_kernel/bm25_top_k_merge.cpp'].decode();mh=merge['op_host/bm25_top_k_merge.cpp'].decode()
    assert 'inputLane < shards' in body and 'lane) * docsPerShard +' not in body
    assert 'task * k' in body and 'query * shards + inputLane' in body and 'uint64_t task=' in body
    assert 'outShape.GetDimNum() != 3' in mh and '(scores->GetDim(1)+31)/32' in mh
    cmake=(ROOT/'npu_prototype/large_search_host/CMakeLists.txt').read_text(encoding='utf-8')
    names=re.search(r'set\(SOURCES\s+(.*?)\)',cmake,re.S).group(1).split()
    assert tuple(names)==x.HOST
    code=(ROOT/'npu_prototype/large_search_host/large_search.h').read_text(encoding='utf-8')
    search=code[code.index('    PlannedResult Search('):code.index('    void PoisonForDiagnostic(')]
    assert 'Prepare(' not in search and 'reference' not in search and 'aclmdlRIExecuteAsync' in search
    assert 'actual free device memory' in code and 'paddedNorm.resize(plan.padded,1.0F)' in code


def oracle_tests():
    parent=ROOT.parent/'results';parent.mkdir(exist_ok=True)
    cases={'A':[[0,0,2],[3,1,2],[1,2,3],[7,2,4],[4,3,2],[1,1,1],[5,5,2],[6,7,0]],
        'B':[[1,2,3],[4,5,6],[2,3,4],[0,0,0],[3,2,1],[5,6,7],[7,7,7],[1,3,5]]}
    with tempfile.TemporaryDirectory(dir=parent) as temp:
        temp=Path(temp)
        for n in (10,97,8193):
            output=temp/str(n)
            with contextlib.redirect_stdout(io.StringIO()):info=c.prepare(output,n,16,950,cases)
            index,values=r.validate_index(output/'index');c.validate(output)
            assert sum(int(tf)*count for tf,count in info['posting_tf_histogram'].items())==info['tokens']
            assert sum(info['posting_tf_histogram'].values())==info['posting_records']
            for case,queries in cases.items():
                expected=r.diagnostic_scores(index,values,queries)
                assert (output/case/'expected_scores.bin').read_bytes()==r.encode(expected)
                selected=[sorted(range(n),key=lambda d:(-expected[q*n+d],d))[:10] for q in range(8)]
                assert (output/case/'expected_top_ids.bin').read_bytes()==r.encode(array('i',(d for row in selected for d in row)))
                assert (output/case/'expected_top_scores.bin').read_bytes()==r.encode(array('f',(expected[q*n+d] for q,row in enumerate(selected) for d in row)))
                assert info['cases'][case]['hit_records_per_query']==[sum(
                    values['term_offsets.bin'][t+1]-values['term_offsets.bin'][t] for t in row) for row in queries]
            if n==97:cpu_verifier_contract(output,info,index)
        # Unknown words/all-zero scores and repeated words on a streamed external corpus.
        docs=[[0,0,1],[],[1,1],[],[0]]*8;source=temp/'tokens.jsonl'
        source.write_text(''.join(json.dumps(row)+'\n' for row in docs),encoding='utf-8')
        other={'A':[[0,0,1]]*8,'B':[[14,15,14]]*8};output=temp/'external'
        with contextlib.redirect_stdout(io.StringIO()):info=c.prepare(output,len(docs),16,1,other,corpus=source)
        idx,values=r.validate_index(output/'index')
        assert info['dataset_kind']=='external_token_ID_jsonl_not_MT_approved'
        assert (output/'B/expected_top_ids.bin').read_bytes()==r.encode(array('i',list(range(10))*8))
        (output/'A.txt').write_text('tampered',encoding='utf-8')
        try:c.validate(output)
        except ValueError:pass
        else:raise AssertionError('Changed input accepted')


def cpu_verifier_contract(inputs,description,index):
    # Synthetic contract fixture only. These times are never written outside this
    # TemporaryDirectory, treated as hardware evidence, or fed to a device workflow.
    actual=inputs/'synthetic_cpu_contract';actual.mkdir()
    for checkpoint in x.g.checks.CHECKPOINTS:
        folder=actual/checkpoint;folder.mkdir();case=checkpoint[-1]
        for src,dst in (('expected_scores.bin','scores.bin'),('expected_top_ids.bin','top_ids.bin'),('expected_top_scores.bin','top_scores.bin')):
            shutil.copyfile(inputs/case/src,folder/dst)
    data={'status':'search_benchmark_calls_completed_checks_pending','backend':'cpu','docs':description['docs'],
        'queries':8,'k':10,'partitions':32,'capacity_queries':8,'capacity_words':24,'timer_policy':x.g.checks.TIMER,
        'warmup_cycles':5,'measured_cycles':30,'measured_search_calls':60,'measured_topk_bit_checks':60,
        'checkpoint_search_calls':4,'index_readback_byte_identical':True,'measured_file_io_calls':0,
        'measured_debug_download_calls':0,'measured_poison_calls':0,'diagnostic_checkpoints_outside_measurement':True,
        'index_load_batches':1,'requested_threads':32,'actual_threads':32,'index_bytes':index['index_bytes'],
        'fp32_flags':'-O3 -march=native -ffp-contract=off -fno-fast-math','host_source_sha256':x.g.checks.sources(x.g.checks.CPU_SOURCES),
        'samples':[{'cycle':cycle,'case':'B' if slot^(cycle&1) else 'A','call_wall_ms':.2,'internal_search_ms':.1,
            'query_h2d_bytes':0,'topk_d2h_bytes':0} for cycle in range(30) for slot in range(2)]}
    r.write_json(actual/'benchmark_session.json',data)
    x.verify_run(actual,inputs,description,index,'cpu',5)
    file=actual/'before_A/scores.bin';raw=bytearray(file.read_bytes());raw[-1]^=1;file.write_bytes(raw)
    try:x.verify_run(actual,inputs,description,index,'cpu',5)
    except ValueError:pass
    else:raise AssertionError('Changed CPU full score accepted')
    shutil.copyfile(inputs/'A/expected_scores.bin',file)
    data['actual_threads']=1;r.write_json(actual/'benchmark_session.json',data)
    try:x.verify_run(actual,inputs,description,index,'cpu',5)
    except ValueError:pass
    else:raise AssertionError('Weakened CPU opponent accepted')


def merge_model(lists,k):
    while True:
        result=[]
        for begin in range(0,len(lists),32):
            active=lists[begin:begin+32];heads=[0]*len(active);chosen=[]
            for unused in range(k):
                available=[(-row[heads[lane]][0],row[heads[lane]][1],lane) for lane,row in enumerate(active) if heads[lane]<len(row)]
                _,_,lane=min(available);chosen.append(active[lane][heads[lane]]);heads[lane]+=1
            assert chosen==sorted([x for row in active for x in row],key=lambda x:(-x[0],x[1]))[:k]
            result.append(chosen)
        if len(result)==1:return result[0]
        lists=result


def merge_tests():
    rng=random.Random(950);cases=0
    for count in (1,31,32,33,1023,1024,1025,1221):
        for k in (1,10,32):
            for kind in range(3):
                full=[[(0.0 if kind==0 else 1.0 if kind==1 else struct.unpack('<f',struct.pack('<I',0x3f800000+rng.randrange(65536)))[0],leaf*8192+d)
                    for d in range(41)] for leaf in range(count)]
                local=[sorted(row,key=lambda x:(-x[0],x[1]))[:k] for row in full]
                got=merge_model(local,k)
                assert got==sorted([x for row in full for x in row],key=lambda x:(-x[0],x[1]))[:k];cases+=1
    assert sum(len(x.layout(max(10,s*8192),1,1)['list_counts'])-1 for s in (1,31,32,33,1023,1024,1025))*3==36
    return cases


def decisions():
    def value(ms):return {'summary':{name:{'median_batch_ms':ms,'p95_batch_ms':ms} for name in ('A','B')}}
    def rows(ms):return [{'cpu':value(1),'cpu_directory':value(.9),'npu':value(ms)} for _ in range(6)]
    assert not x.gate(rows(.95))['performance_target_met_at_this_scale']
    assert x.gate(rows(.8))['performance_target_met_at_this_scale']
    bad=rows(.8);bad[-1]['npu']=value(1.1);assert not x.gate(bad)['performance_target_met_at_this_scale']
    tiers=[{'docs':n,'status':'six_rounds_checked','decision':x.gate(rows(.95 if i==0 else .8))} for i,n in enumerate(x.SIZES)]
    result=x.crossover(tiers)
    assert result['large_scale_hardware_target_met'] and result['first_tested_large_winning_docs']==1048576
    assert result['previous_tested_losing_docs']==262144
    tiers[2]['decision']=x.gate(rows(.95));assert x.crossover(tiers)['nonmonotonic_tested_results']
    for tier in tiers:tier['decision']=x.gate(rows(.95))
    assert not x.crossover(tiers)['large_scale_hardware_target_met']
    tiers[-1]['status']='not_measured_device_budget'
    assert not x.crossover(tiers)['all_planned_scales_measured']


if __name__=='__main__':
    plan_contracts();sources();oracle_tests();count=merge_tests();decisions()
    print(f'PASS: four scale plans, unchanged leaf FP32/UB, streamed97/8193/tie fixtures, {count} hierarchical rank models, fair six-round and large-only gates. NOT CANN/A5/performance evidence.')
