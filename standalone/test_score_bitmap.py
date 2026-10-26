"""评分位图整数/FP32模型与源码隔离检查；不是CANN编译、设备或性能证据。"""
from pathlib import Path
import contextlib
import io
import random
import struct
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import score_bitmap_sources as source, score_bitmap as driver
from standalone import resident_index as reference
from standalone.test_fused_ub_contract import constants


def fp32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]


def bits(value):
    return struct.unpack('<I',struct.pack('<f',value))[0]


def key(raw):
    return (~(0 if raw&0x7fffffff==0 else raw))&0xffffffff


def score_and_pack(width,threads,terms,wrong_owner=False):
    scores=[0.0]*width
    masks=[[0,0] for _ in range(threads)]
    for postings in terms:
        owned=set()
        for position,(doc,contribution) in enumerate(postings):
            assert 0<=doc<width and doc not in owned
            owned.add(doc)
            previous=scores[doc]
            total=fp32(previous+contribution)
            assert total>=0.0
            scores[doc]=total
            if previous==0.0 and total>0.0:
                # Actual kernel thread owns this posting position; it usually
                # differs from the residue owner used by compact packing.
                thread=position%threads if wrong_owner else doc%threads;part=doc//threads
                masks[thread][part//32]|=1<<(part%32)

    expected_masks=[[0,0] for _ in range(threads)]
    for doc,value in enumerate(scores):
        raw=bits(value)
        assert raw<0x80000000
        if raw:
            thread=doc%threads;part=doc//threads
            expected_masks[thread][part//32]|=1<<(part%32)
    assert masks==expected_masks

    packed=[]
    for thread in range(threads):
        for part in range((width-thread+threads-1)//threads):
            doc=thread+part*threads
            if masks[thread][part//32]&(1<<(part%32)):
                packed.append((doc,bits(scores[doc])))
    expected=[(doc,bits(scores[doc])) for thread in range(threads)
        for doc in range(thread,width,threads) if bits(scores[doc])]
    assert packed==expected
    assert sum(word.bit_count() for pair in masks for word in pair)==len(packed)
    return scores,packed


def exact_models():
    rng=random.Random(95020261026);rows=0
    subnormal=struct.unpack('<f',struct.pack('<I',1))[0]
    negative_zero=struct.unpack('<f',struct.pack('<I',0x80000000))[0]
    for threads in (128,256):
        for width in (1,6,31,32,33,63,64,65,127,128,129,255,256,257,1023,1024,1025,4097,8192):
            order=list(range(width));rng.shuffle(order)
            counts={0,min(1,width),min(9,width),min(10,width),min(11,width),
                min(1023,width),min(1024,width),min(1025,width),width}
            for count in sorted(counts):
                docs=order[:count]
                terms=[
                    [(doc,1.0 if doc&1 else subnormal) for doc in docs[::2]],
                    [(doc,negative_zero) for doc in docs[1::2]],
                    [],
                    [(doc,subnormal if doc%3 else 2.0) for doc in docs[1::2]],
                    [(doc,0.5) for doc in docs],
                ]
                scores,packed=score_and_pack(width,threads,terms)
                assert len(packed)==count
                for k in (1,10,32):
                    if k>width:continue
                    candidates=list(packed)
                    if len(candidates)<k:
                        zero_ids=(doc for doc,value in enumerate(scores) if bits(value)==0)
                        candidates.extend((doc,0) for doc in list(zero_ids)[:k-len(candidates)])
                    chosen=sorted(candidates,key=lambda item:(key(item[1]),item[0]))[:k]
                    expected=sorted(((doc,bits(value)) for doc,value in enumerate(scores)),
                        key=lambda item:(key(item[1]),item[0]))[:k]
                    assert [doc for doc,raw in chosen]==[doc for doc,raw in expected]
                    assert [raw for doc,raw in chosen]==[raw for doc,raw in expected]
                rows+=1
    return rows


def source_contracts():
    original=ROOT/'npu_prototype/fused_score_topk'
    prior=source.large.fused_sources(original)
    candidate=source.candidate_sources(original)
    assert prior.keys()==candidate.keys()
    assert all(prior[name]==raw for name,raw in candidate.items() if name!=source.KERNEL)
    text=candidate[source.KERNEL].decode('utf-8')
    restored=text
    for old,new in reversed(source.REPLACEMENTS):restored=restored.replace(new,old,1)
    restored=restored.replace('#include "simt_api/device_sync_functions.h"\n#include "simt_api/device_atomic_functions.h"\n#include "simt_api/device_functions.h"',
        '#include "simt_api/device_sync_functions.h"',1)
    assert restored==prior[source.KERNEL].decode('utf-8')
    assert text.count('const float previous = scores[local];')==1
    assert text.count('previous == 0.0F && total > 0.0F')==1
    assert text.count('bitmap[kOwnerBegin+thread]')==1
    assert text.count('bitmap[kOwnerCount+thread]')==1
    assert 'const uint32_t owner = local%kThreads;' in text
    assert 'asc_atomic_or(bitmap+kOwnerBegin+owner, 1U << part)' in text
    assert 'asc_atomic_or(bitmap+kOwnerCount+owner, 1U << (part-32))' in text
    assert constants(text)['kBufferBytes']==50848 and constants(text)['kMaxDocs']==8192
    assert 'scratch[kControl+7]+id' in text and '*shards > 65536' in candidate['op_host/bm25_fused_score_top_k.cpp'].decode()
    assert text.count('__popc(lowMask)+__popc(highMask)')==1
    assert text.count('PackForSimd<true>')==1 and text.count('PackForSimd<false>')==1
    assert 'if constexpr (BitmapReady)' in text
    # 只改变位图生产与计数；阈值、紧凑写出、严格同分收集及回退均可逐字还原。
    assert 'FindSimdThreshold(local, data.k);' in text
    assert 'VF_CALL<FusedCollectSimt>' in text
    assert source.VENDOR!=source.base.VENDOR


def driver_contracts():
    graph=['runner','index','A','B','old','5','vendors','10','8','24','5','30','refA','refB']
    previous={'command':graph};original=list(graph)
    changed=driver.command_for(previous,Path('destination'),'new_vendors')
    assert graph==original and changed[4]=='destination' and changed[6]=='new_vendors'
    assert all(changed[i]==original[i] for i in range(14) if i not in (4,6))
    cpu=['runner','input','index','output','old']+['32','10','32','8','24','5','30']
    previous={'command':cpu};original=list(cpu)
    changed=driver.command_for(previous,Path('destination'))
    assert cpu==original and changed[4]=='destination'
    assert all(changed[i]==original[i] for i in range(12) if i!=4)


def collision_regressions():
    # The previous kernel incorrectly chose position%threads. Reject that exact
    # mutation with sorted unique postings, not a doc-owned idealized model.
    terms=[[(doc,1.0) for doc in range(0,8192,256)],[(doc,.5) for doc in range(0,8192,256)]]
    score_and_pack(8192,256,terms)
    try:score_and_pack(8192,256,terms,wrong_owner=True)
    except AssertionError:pass
    else:raise AssertionError('Wrong scoring-thread owner escaped regression')
    # Owner0 receives32 distinct bits from32 posting threads. Plain |= can lose
    # one update if both read the old0 value; atomic OR is order independent.
    updates=[1<<bit for bit in range(32)]
    value=0
    for mask in reversed(updates):value |= mask
    assert value==0xffffffff and (0|updates[0])!=(updates[0]|updates[1])


def decision_contracts():
    def item(ms,p95=None):return {'summary':{c:{'median_batch_ms':ms,'p95_batch_ms':ms if p95 is None else p95} for c in ('A','B')}}
    def rows(candidate):return [{'baseline':item(1),'candidate':item(candidate),'cpu':item(2),'cpu_directory':item(1.8)} for _ in range(6)]
    assert driver.decision(rows(.90))['retain_candidate']
    assert not driver.decision(rows(.97))['retain_candidate']
    bad=rows(.90);bad[2]['candidate']=item(1.01);assert not driver.decision(bad)['retain_candidate']
    bad=rows(.90);bad[0]['cpu']=item(.80);assert not driver.decision(bad)['retain_candidate']
    bad=rows(.90)
    for row in bad:row['candidate']=item(.90,1.1)
    assert not driver.decision(bad)['retain_candidate']


def affected_fixture_and_headers():
    with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as folder:
        folder=Path(folder);atomic=folder/'simt_api';atomic.mkdir()
        (atomic/'device_atomic_functions.h').write_text('inline uint32_t asc_atomic_or(uint32_t*,uint32_t);\n',encoding='utf-8')
        (atomic/'device_functions.h').write_text('int __popc(unsigned int);\n',encoding='utf-8')
        assert driver.actual_atomic_headers(folder)['popc_header']['declarations']
        (atomic/'device_functions.h').write_text('missing',encoding='utf-8')
        try:driver.actual_atomic_headers(folder)
        except ValueError:pass
        else:raise AssertionError('Missing popc header declaration accepted')
        original=folder/'docs.jsonl'
        import json
        original.write_text(''.join(json.dumps(row)+'\n' for row in driver.adversarial_documents()),encoding='utf-8')
        cases={'A':[[0,0,7],[1,7,7],[2,7,7],[3,7,7]]*2,'B':[[7,7,7],[0,2,0],[2,3,2],[1,3,1]]*2}
        output=folder/'fixture'
        with contextlib.redirect_stdout(io.StringIO()):info=driver.retained.corpus.prepare(output,8193,8,950,cases,corpus=original)
        assert info['cases']['A']['hit_documents_per_query']==[33,1025,1024,1023]*2
        assert info['cases']['B']['hit_documents_per_query'][0]==0
        idx,values=reference.validate_index(output/'index')
        for case in cases:
            assert (output/case/'expected_scores.bin').read_bytes()==reference.encode(reference.diagnostic_scores(idx,values,cases[case]))


def main():
    rows=exact_models()
    source_contracts()
    driver_contracts()
    collision_regressions();decision_contracts();affected_fixture_and_headers()
    print(f'PASS: {rows} score/bitmap/pack models; actual posting-owner regression, atomic-word collisions, unchanged FP32/UB/global-ID tree,8193 boundary fixture, paired six-round retention gate. Not CANN/device/performance evidence.')


if __name__=='__main__':main()
