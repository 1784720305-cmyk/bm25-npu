"""Integer/thread model for positive compaction, exact radix and original-ID ties; no NPU execution."""
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import check_radix_model as old, check_radix_collect_model as collect

THREADS=128
CAPACITY=1024
MASK_DOCS=8192


def threshold(streams,k):
    prefix=mask=0; remaining=k
    for shift,bins in ((21,2048),(10,2048),(0,1024)):
        histogram=[0]*bins
        for stream in streams:
            for raw,doc in stream:
                key=old.key(raw)
                if key&mask==prefix: histogram[(key>>shift)&(bins-1)]+=1
        chunk=bins//THREADS
        counts=[sum(histogram[t*chunk:(t+1)*chunk]) for t in range(THREADS)]
        ranks,unused,total,unused=old.prefix_pair(counts,[0]*THREADS)
        assert 0<remaining<=total
        found=[]
        for t in range(THREADS):
            before=ranks[t]
            for j in range(chunk):
                bucket=t*chunk+j; count=histogram[bucket]
                if before<remaining<=before+count: found.append((bucket,remaining-before))
                before+=count
        assert len(found)==1
        bucket,remaining=found[0]
        prefix|=bucket<<shift; mask|=(bins-1)<<shift
    return prefix,remaining


def select(raw,k,thread_order=None,scratch=None):
    if type(k) is not int or not 1<=k<=min(32,len(raw)): raise ValueError('K非法')
    keys=[old.key(x) for x in raw] # Reject NaN/Inf/negative, permit signed zeros and subnormals.
    order=list(range(THREADS)) if thread_order is None else thread_order
    assert sorted(order)==list(range(THREADS))
    n=len(raw); streams=[[(raw[d],d) for d in range(t,n,THREADS)] for t in range(THREADS)]
    route='full'; positives=None
    if CAPACITY<n<=MASK_DOCS:
        masks=[[0,0] for unused in range(THREADS)]
        counts=[0]*THREADS; heads=[old.INVALID]*THREADS
        for t in order:
            for ordinal,(value,doc) in enumerate(streams[t]):
                assert ordinal<64
                if value&0x7fffffff:
                    masks[t][ordinal//32]|=1<<(ordinal%32); counts[t]+=1
                elif heads[t]==old.INVALID: heads[t]=doc
        starts,unused,positives,unused=old.prefix_pair(counts,[0]*THREADS)
        if positives<=CAPACITY:
            route='compact'
            if scratch is None: scratch=[None]*CAPACITY
            assert len(scratch)==CAPACITY
            occupied=set()
            for t in order:
                written=0
                for ordinal in range(64):
                    if written==counts[t]: break
                    if masks[t][ordinal//32]&(1<<(ordinal%32)):
                        doc=t+ordinal*THREADS; pos=starts[t]+written
                        assert doc<n and pos<CAPACITY and pos not in occupied
                        scratch[pos]=(raw[doc],doc); occupied.add(pos); written+=1
                assert written==counts[t]
            assert len(occupied)==positives and occupied==set(range(positives))
            streams=[list(scratch[starts[t]:starts[t]+counts[t]]) for t in range(THREADS)]
            for stream in streams: assert [x[1] for x in stream]==sorted(x[1] for x in stream)
            if positives<=k:
                chosen=[(old.key(value),doc) for stream in streams for value,doc in stream]
                for rank in range(k-positives):
                    winner=collect.minimum_head(heads)
                    assert winner!=old.INVALID
                    chosen.append((0xffffffff,winner)); owner=winner%THREADS
                    if rank+1<k-positives:
                        heads[owner]=next((d for d in range(winner+THREADS,n,THREADS)
                                           if (raw[d]&0x7fffffff)==0),old.INVALID)
                result=old.sort_candidates(chosen)
                ids=[entry[1] for entry in result]
                return ids,[raw[i] for i in ids],{'route':'compact_zero_fill','positive_count':positives,
                    'histogram_items_per_pass':0,'prepass_reads':n,'compaction_reads':positives}
    prefix,remaining=threshold(streams,k)
    strict_count=k-remaining
    counts=[sum(old.key(value)<prefix for value,doc in stream) for stream in streams]
    starts,unused,total,unused=old.prefix_pair(counts,[0]*THREADS)
    assert total==strict_count
    chosen=[None]*k; head_positions=[None]*THREADS; heads=[old.INVALID]*THREADS
    for t in order:
        written=0
        for pos,(value,doc) in enumerate(streams[t]):
            if old.key(value)<prefix:
                slot=starts[t]+written; assert chosen[slot] is None
                chosen[slot]=(old.key(value),doc); written+=1
            if old.key(value)==prefix and heads[t]==old.INVALID:
                heads[t]=doc; head_positions[t]=pos
        assert written==counts[t]
    for rank in range(remaining):
        winner=collect.minimum_head(heads); assert winner!=old.INVALID
        owners=[t for t in range(THREADS) if heads[t]==winner]; assert len(owners)==1
        owner=owners[0]; chosen[strict_count+rank]=(prefix,winner)
        if rank+1<remaining:
            heads[owner]=old.INVALID
            for pos in range(head_positions[owner]+1,len(streams[owner])):
                value,doc=streams[owner][pos]
                if old.key(value)==prefix:
                    heads[owner]=doc; head_positions[owner]=pos; break
    assert len({item[1] for item in chosen})==k
    ids=[entry[1] for entry in old.sort_candidates(chosen)]
    return ids,[raw[i] for i in ids],{'route':route,'positive_count':positives,
        'histogram_items_per_pass':sum(map(len,streams)),'prepass_reads':n if positives is not None else 0,
        'compaction_reads':positives if route=='compact' else 0}


def main():
    rng=random.Random(95020261021); checks=0; routes=set(); scratch=[(0x7fc00000,old.INVALID)]*CAPACITY
    for n in (1,6,31,32,33,127,128,129,1024,1025,2048,2049,4097,8191,8192,8193,65536):
        patterns=[[0]*n,[0x80000000 if i%2 else 0 for i in range(n)],
            [old.float_bits(1.)]*n,[old.float_bits(float(i)) for i in range(n)],
            [old.float_bits(float(n-i)) for i in range(n)],
            [old.float_bits(float(rng.randrange(9))) if i%11==0 else 0 for i in range(n)],
            [rng.randrange(1,0x7f800000) if i%11==0 else 0 for i in range(n)],
            [1 if i%11==0 else 0 for i in range(n)],
            [0x7f7fffff if i%11==0 else 0 for i in range(n)]]
        for count in sorted({0,1,min(n,9),min(n,10),min(n,11),min(n,31),min(n,32),min(n,33),min(n,1023),min(n,1024),min(n,1025)}):
            row=[0x80000000 if i%2 else 0 for i in range(n)]
            for i in rng.sample(range(n),count): row[i]=old.float_bits(float(1+i%5))
            patterns.append(row)
        for raw in patterns:
            for k in sorted({1,min(10,n),min(32,n)}):
                expected=sorted(range(n),key=lambda d:(-old.as_float(raw[d]),d))[:k]
                order=list(range(THREADS)); rng.shuffle(order)
                ids,bits,stats=select(raw,k,order,scratch)
                assert ids==expected and bits==[raw[i] for i in expected]
                routes.add(stats['route']); checks+=1
                if stats['route']=='compact': assert stats['histogram_items_per_pass']==stats['positive_count']<=CAPACITY
    # Counterexample: compaction order [128,1] is NOT original-ID tie order [1,128].
    raw=[0]*2049; raw[128]=raw[1]=old.float_bits(1.)
    ids,bits,stats=select(raw,1,scratch=scratch)
    assert ids==[1] and stats['route']=='compact'; checks+=1
    # All 64 membership bits, including high word bits 0 and 31.
    for lane in (0,127):
        raw=[0]*8192
        for d in range(lane,8192,128): raw[d]=old.float_bits(1.)
        ids,bits,stats=select(raw,32,scratch=scratch)
        assert ids==list(range(lane,lane+32*128,128)) and stats['positive_count']==64; checks+=1
    # Reuse one poisoned cache while alternating capacity overflow, sparse, zero and dense rows.
    for count in (1024,0,1025,730,10,9,8192,730):
        raw=[0x80000000 if i%2 else 0 for i in range(8192)]
        for d in rng.sample(range(8192),count): raw[d]=old.float_bits(float(1+d%17))
        expected=sorted(range(8192),key=lambda d:(-old.as_float(raw[d]),d))[:10]
        ids,bits,stats=select(raw,10,scratch=scratch)
        assert ids==expected and bits==[raw[i] for i in expected]; checks+=1
    for invalid in (0x7f800000,0xff800000,0x7fc00000,0x80000001,0xbf800000):
        for n in (1025,8193):
            raw=[0]*n; raw[-1]=invalid
            try: select(raw,10,scratch=scratch)
            except ValueError: checks+=1
            else: raise AssertionError('invalid score accepted')
    assert routes=={'full','compact','compact_zero_fill'}
    print(f'PASS: {checks} integer/thread MODEL cases: 1023/1024/1025 capacity, mask bits, original-ID ties, signed-zero bits, subnormals, dense/long fallback, cache reuse and invalid scores. '
          'Not C++ compilation, UB/barrier/device proof or performance evidence.')


if __name__=='__main__': main()
