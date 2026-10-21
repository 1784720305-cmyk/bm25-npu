"""Fused-shard FP32/selection/merge MODEL + fake-template checks; not C++/SIMT/device proof."""
from array import array
from bisect import bisect_left
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import resident_index as r, create_fused_score_topk as generator
from standalone import check_topk_positive_compact_model as selection, check_radix_model as bits
from standalone import query_score_edge_fixture as oracle


def dimensions(n, q, k, shards, debug, cores=56):
    r.require(all(type(x) is int for x in (n, q, k, shards, debug, cores)), 'integer dimensions required')
    r.require(1 <= n <= r.LIMIT and 1 <= q <= r.LIMIT//n and 1 <= k <= 32 and
        (1 <= shards <= 32 or shards == 64) and n%shards == 0 and k <= n//shards <= 8192 and debug in (0, 1) and cores > 0,
        'unsupported fused geometry')
    return min(cores, q*shards)


def simulate(index, values, offsets, words, k, shards, debug, cores, rng, memory=None, scratch=None):
    """Model grid-stride ownership and per-term barriers as explicit sequential phases.

    Cannot establish actual UB visibility, SIMT scheduling, compiler FP32 or speed.
    """
    n, q = index['num_docs'], len(offsets)-1
    blocks = dimensions(n, q, k, shards, debug, cores); width = n//shards
    if memory is None: memory = array('f', [float('nan')])*8192
    if scratch is None: scratch = [(0x7fc00000, bits.INVALID)]*1024
    partial_ids, partial_bits = [-123]*(q*shards*k), [0x7fc00000]*(q*shards*k)
    full = array('f', [float('nan')])*(q*n) if debug else array('f', [0])
    covered, routes, searches = set(), set(), 0
    t_offsets, docs, tf, norms, idfs = (values[name] for name in r.INDEX_TYPES)
    block_order = list(range(blocks)); rng.shuffle(block_order)
    for block in block_order:
        for task in range(block, q*shards, blocks):
            query, part = divmod(task, shards); first, last = part*width, (part+1)*width
            start, stop = offsets[query:query+2]
            valid = 0 <= start <= stop <= len(words) and stop-start <= 64
            active = []
            for doc in range(width): memory[doc] = 0
            if valid:
                for term in words[start:stop]:
                    lo = hi = 0; used = -1
                    if term < 0: valid = False
                    elif term < index['vocab_size']:
                        a, b = t_offsets[term:term+2]
                        if not 0 <= a <= b <= index['posting_count']: valid = False
                        else:
                            lo = bisect_left(docs, first, a, b); hi = bisect_left(docs, last, lo, b)
                            used = term; searches += 2
                    active.append((used, lo, hi))
            # Match the device's lack of early term-loop exit, including error cases.
            for term, lo, hi in active:
                if term < 0: continue
                writers = set(); lanes = list(range(128)); rng.shuffle(lanes)
                for lane in lanes:
                    for position in range(lo+lane, hi, 128):
                        doc = docs[position]
                        assert doc not in writers, 'resident index broke unique posting ownership'
                        writers.add(doc)
                        if not (first <= doc < last): valid = False; continue
                        frequency, norm, weight = tf[position], norms[doc], idfs[term]
                        if not (1 <= frequency <= 2**24 and 0 < norm <= bits.as_float(0x7f7fffff) and
                                0 <= weight <= bits.as_float(0x7f7fffff)):
                            valid = False; continue
                        try:
                            numerator = r.fp32(frequency*r.fp32(2.2))
                            contribution = r.fp32(r.fp32(weight*numerator)/r.fp32(frequency+norm))
                            total = r.fp32(memory[doc-first]+contribution)
                        except (OverflowError, ZeroDivisionError): valid = False; continue
                        if not (math.isfinite(contribution) and contribution >= 0 and math.isfinite(total) and total >= 0):
                            valid = False
                        else: memory[doc-first] = total
            if debug:
                for doc in range(width): full[query*n+first+doc] = memory[doc] if valid else -1
            if valid:
                lane_order = list(range(128)); rng.shuffle(lane_order)
                ids, raw, stats = selection.select([bits.float_bits(x) for x in memory[:width]], k, lane_order, scratch)
                routes.add(stats['route'])
            else: ids, raw = [-1]*k, [0xbf800000]*k
            begin = task*k; partial_ids[begin:begin+k] = ids; partial_bits[begin:begin+k] = raw
            assert task not in covered; covered.add(task)
    assert covered == set(range(q*shards)) and 0x7fc00000 not in partial_bits and -123 not in partial_ids
    return {'partial_ids': partial_ids, 'partial_bits': partial_bits, 'debug_scores': full,
        'routes': routes, 'searches': searches, 'full_score_gm_writes': q*n if debug else 0, 'blocks': blocks}


def merge_heads(result, q, n, k, shards):
    """Separate ordered-list merge model using GLOBAL IDs; invalid partial rows are rejected."""
    width = n//shards; ids, raw = [], []
    for query in range(q):
        heads = [0]*shards
        for part in range(shards):
            start = (query*shards+part)*k
            row = result['partial_ids'][start:start+k]
            r.require(all(0 <= x < width for x in row) and len(set(row)) == k, 'invalid partial IDs')
            r.require(row == sorted(row, key=lambda d: (-bits.as_float(result['partial_bits'][start+row.index(d)]), d)), 'invalid partial order')
        for rank in range(k):
            available = []
            for part, head in enumerate(heads):
                if head < k:
                    position = (query*shards+part)*k+head
                    doc = part*width+result['partial_ids'][position]
                    value = result['partial_bits'][position]
                    available.append((-bits.as_float(value), doc, part, value))
            unused_score, doc, winner, value = min(available)
            ids.append(doc); raw.append(value); heads[winner] += 1
    return ids, raw


def main():
    count = 0; rng = random.Random(95020261021); routes = set()
    memory = array('f', [float('nan')])*8192; scratch = [(0x7fc00000, bits.INVALID)]*1024
    def fails(call):
        nonlocal count
        try: call()
        except (ValueError, OSError): count += 1
        else: raise AssertionError('bad geometry/template accepted')
    for n, shards in ((1,1), (6,2), (32,1), (128,4), (1024,1), (1025,1), (2049,1), (8192,1), (16384,2)):
        documents = [([0]*(1+doc%3) if doc%11 == 0 else []) + ([1]*2 if doc%3 else []) + [2] for doc in range(n)]
        index, values = r.build_index(documents, 5)
        queries = [[0,0,1], [], [5,2147483646], [2], [1,2,1]]
        unused, words = r.make_queries(index, queries, 1)
        expected = r.diagnostic_scores(index, values, queries)
        r.compare(expected, oracle.independent_scores(documents, queries)); count += 1
        original = {key: r.encode(value) for key, value in values.items()}
        for k in sorted({1, min(10,n//shards), min(32,n//shards)}):
            actual = simulate(index, values, words['query_offsets.bin'], words['query_term_ids.bin'], k, shards, 1, 56, rng, memory, scratch)
            assert r.encode(actual['debug_scores']) == r.encode(expected)
            expected_ids = [doc for query in range(len(queries)) for doc in
                sorted(range(n), key=lambda d: (-expected[query*n+d], d))[:k]]
            ids, raw = merge_heads(actual, len(queries), n, k, shards)
            assert ids == expected_ids and raw == [bits.float_bits(expected[(i//k)*n+doc]) for i,doc in enumerate(ids)]
            for debug, cores in ((0,1), (0,56)):
                repeated = simulate(index, values, words['query_offsets.bin'], words['query_term_ids.bin'], k, shards, debug, cores, rng, memory, scratch)
                assert repeated['partial_ids'] == actual['partial_ids'] and repeated['partial_bits'] == actual['partial_bits']
                assert len(repeated['debug_scores']) == 1 and repeated['debug_scores'][0] == 0 and repeated['full_score_gm_writes'] == 0
                count += 2
            routes |= actual['routes']; count += 2
        assert original == {key: r.encode(value) for key,value in values.items()}; count += 1
    # Capacity boundary forces the exact dense path; repeated and changed queries reuse poisoned buffers.
    for positives in (0,9,10,11,1023,1024,1025,8192,730):
        documents = [[0] if doc < positives else [1] for doc in range(8192)]
        index, values = r.build_index(documents, 3)
        for queries in ([[0]], [[1]], [[0]], [[]], [[1]]):
            unused, words = r.make_queries(index, queries, 10)
            actual = simulate(index, values, words['query_offsets.bin'], words['query_term_ids.bin'], 10, 1, 1, 56, rng, memory, scratch)
            expected = r.diagnostic_scores(index, values, queries)
            assert r.encode(actual['debug_scores']) == r.encode(expected)
            expected_ids = sorted(range(8192), key=lambda doc: (-expected[doc],doc))[:10]
            assert actual['partial_ids'] == expected_ids
            routes |= actual['routes']; count += 2
    assert routes == {'compact', 'compact_zero_fill', 'full'}; count += 1
    # Q1 can use 64 real document partitions, exceeding the old merge's 32-list limit.
    # Reshape into two 32-list merges, then one two-list merge; no new merge algorithm.
    documents = [[0] if doc%11 else [1] for doc in range(8192)]
    index, values = r.build_index(documents, 3); queries = [[0,0,1]]
    unused, words = r.make_queries(index, queries, 10)
    actual = simulate(index, values, words['query_offsets.bin'], words['query_term_ids.bin'],10,64,1,56,rng)
    expected = r.diagnostic_scores(index, values, queries)
    assert actual['blocks'] == 56 and r.encode(actual['debug_scores']) == r.encode(expected)
    group_ids, group_raw = merge_heads(actual,2,8192//2,10,32)
    ids, raw = merge_heads({'partial_ids':group_ids,'partial_bits':group_raw},1,8192,10,2)
    expected_ids = sorted(range(8192),key=lambda doc:(-expected[doc],doc))[:10]
    assert ids == expected_ids and raw == [bits.float_bits(expected[doc]) for doc in ids]
    assert (ids,raw) == merge_heads(actual,1,8192,10,64)
    assert dimensions(65536,1,10,64,0,56) == dimensions(262144,8,10,32,0,56) == 56; count += 4
    # More tasks than cores, complete overwrite, exact 64 terms, malformed requests/data.
    index, values = r.build_index([[0,1], [], [1], [0,0]]*8, 3)
    queries = [[0,1]*32] + [([], [0,0], [3])[i%3] for i in range(64)]
    unused, words = r.make_queries(index, queries, 1)
    actual = simulate(index, values, words['query_offsets.bin'], words['query_term_ids.bin'], 1, 32, 1, 56, rng)
    assert r.encode(actual['debug_scores']) == r.encode(r.diagnostic_scores(index,values,queries)); count += 1
    for offsets, terms in (([0,65], [0]*65), ([-1,0],[0]), ([1,0],[0]), ([0,2],[0]), ([0,1],[-1])):
        bad = simulate(index,values,offsets,terms,1,1,1,56,rng)
        assert set(bad['debug_scores']) == {-1.} and set(bad['partial_ids']) == {-1}; count += 1
        fails(lambda: merge_heads(bad,1,32,1,1))
    for name, value in (('posting_tf.bin', -1), ('doc_norm.bin', float('nan')), ('idf.bin', float('inf'))):
        corrupted = copy.deepcopy(values); corrupted[name][0] = value
        bad = simulate(index,corrupted,[0,1],[0],1,1,1,56,rng)
        assert set(bad['partial_ids']) == {-1}; count += 1
    for options in ((8193,1,10,1,0), (7,1,1,2,0), (32,1,33,1,0), (32,1,1,33,0),
                    (32,1,2,32,0), (32,1,1,1,2), (32,0,1,1,0), (32,1,True,1,0)):
        fails(lambda: dimensions(*options))
    # Existing local-selection implementation is reused exactly, apart from UB pointer and row wrapper.
    fused = (ROOT/'npu_prototype/fused_score_topk/op_kernel/bm25_fused_score_top_k.cpp').read_text()
    old = (ROOT/'npu_prototype/topk_positive_compact/op_kernel/bm25_top_k.cpp').read_text()
    helper = old[old.index('namespace {'):old.index('__simt_vf__')].replace('kScratchBytes','kSelectScratchBytes').replace('__gm__ const uint32_t *scores','__ubuf__ const uint32_t *scores')
    assert helper in fused and 'atomicAdd(&scores' not in fused and 'kBufferBytes == 50816' in fused; count += 3
    selection_body = old[old.index('        uint32_t privatePositive ='):old.index('\n    }\n}\n}',old.index('        uint32_t privatePositive ='))]
    selection_body = selection_body.replace('asc_syncthreads(); continue;', 'asc_syncthreads(); return;')
    selection_body = selection_body.replace('badSmall == 0);\n            continue;', 'badSmall == 0);\n            return;')
    selection_body = selection_body.replace('continue; // explicit invalid result;', 'return; // explicit invalid result;').replace('continue; // uniform failure,','return; // uniform failure,')
    assert selection_body in fused; count += 1
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fused_FAKE_TEMPLATE_ONLY_',dir=ROOT/'results') as temp, redirect_stdout(io.StringIO()):
        base = Path(temp)
        for typed in (False,True):
            project = base/str(typed)
            def fake(command, **kwargs):
                assert command[command.index('-i')+1] == str(ROOT/'standalone/fused_score_topk_operator.json')
                project.mkdir(); (project/'build.sh').write_bytes(b'FAKE TEMPLATE')
                for name in generator.SOURCES:
                    path = project/name; path.parent.mkdir(exist_ok=True); path.write_bytes(b'KEEP TEMPLATE')
                r.write_json(project/'CMakePresets.json', {'configurePresets':[{'cacheVariables':{
                    'ASCEND_COMPUTE_UNIT':{'type':'STRING','value':'old'} if typed else 'old'}}]})
            with patch.object(generator,'generator_command',return_value=['FAKE_NEVER_RUN']), patch.object(generator.subprocess,'run',fake):
                generator.generate(project,'ai_core-ascend950')
            record = json.loads((project/'bm25_project.json').read_text())
            assert record['operator'] == 'fused_score_topk' and record['vendor_name'] == 'bm25_fused_score_topk'
            assert record['status'] == 'project_generated_not_compiled_not_device_verified'
            assert record['explicit_ub_bytes']+record['simt_local_bytes'] == 83584
            for name in generator.SOURCES:
                assert (project/name).read_bytes() == (ROOT/'npu_prototype/fused_score_topk'/name).read_bytes()
                assert record['source_sha256'][name] == hashlib.sha256((project/name).read_bytes()).hexdigest()
                assert (project/'bm25_template_backup'/name).read_bytes() == b'KEEP TEMPLATE'; count += 3
            count += 3
        with patch.object(generator.subprocess,'run') as child:
            fails(lambda: generator.generate(base/'False','ai_core-ascend950'))
            fails(lambda: generator.generate(base/'wrong','ai_core-ascend310'))
            assert not child.called; count += 1
        malformed = base/'malformed'; malformed.mkdir(); fails(lambda:generator.populate(malformed))
    print(f'PASS: {count} fused scoring/ordered FP32/full-score oracle/local selection/merge/overwrite/limit and fake-template checks. '
          'Python MODEL only; no C++ compilation, SIMT memory/synchronization proof, device execution or speedup claim.')


if __name__ == '__main__': main()
