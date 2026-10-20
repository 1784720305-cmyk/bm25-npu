"""Contiguous-range/cursor Python MODEL and fake-template generator tests, not compiled NPU tests."""
from array import array
from bisect import bisect_left
from contextlib import redirect_stdout
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
from standalone import resident_index as r, create_query_score_range as generator

THREADS, TILE, MAX_TERMS = 128, 2048, 64


def geometry(n, q, cores):
    partitions = min((n-1)//TILE+1, max(1, cores//min(cores, q)))
    return partitions, min(cores, q*partitions)


def simulate(index, values, offsets, terms, cores, rng, output=None):
    """Model phases/owners/cursor counts. Cannot prove SIMT barriers, register use or device speed."""
    n, q = index['num_docs'], len(offsets)-1
    partitions, blocks = geometry(n, q, cores)
    if output is None: output = array('f', [float('nan')])*(n*q)
    writes = array('i', [0])*(n*q)
    t_offsets, docs, tf, norms, idfs = (values[key] for key in r.INDEX_TYPES)
    block_order = list(range(blocks)); rng.shuffle(block_order)
    searches, contributions, lookahead = 0, 0, 0
    for block in block_order:
        for task in range(block, q*partitions, blocks):
            query, part = divmod(task, partitions)
            first, last = n*part//partitions, n*(part+1)//partitions
            begin, end = offsets[query:query+2]
            bad = not (0 <= begin <= end <= len(terms) and end-begin <= MAX_TERMS)
            cursors, ends, active = [], [], []
            if not bad:
                for term in terms[begin:end]:
                    lo = hi = 0; used = -1
                    if term < 0: bad = True
                    elif term < index['vocab_size']:
                        a, b = t_offsets[term:term+2]
                        if not 0 <= a <= b <= index['posting_count']: bad = True
                        else:
                            lo = bisect_left(docs, first, a, b); hi = bisect_left(docs, last, lo, b)
                            used = term; searches += 2
                    cursors.append(lo); ends.append(hi); active.append(used)
            if not bad:
                for tile in range(first, last, TILE):
                    stop = min(last, tile+TILE); local = array('f', [0])*TILE
                    for slot, term in enumerate(active):
                        start = cursors[slot]; consumed = [0]*THREADS; owners = set()
                        lanes = list(range(THREADS)); rng.shuffle(lanes)
                        if term >= 0:
                            for lane in lanes:
                                for pos in range(start+lane, ends[slot], THREADS):
                                    doc = docs[pos]
                                    if doc >= stop:
                                        lookahead += 1; break
                                    consumed[lane] += 1
                                    assert doc not in owners, 'multiple same-term writers'
                                    owners.add(doc)
                                    if not (tile <= doc < stop and 1 <= tf[pos] <= 2**24 and
                                            math.isfinite(norms[doc]) and norms[doc] > 0 and
                                            math.isfinite(idfs[term]) and idfs[term] >= 0):
                                        bad = True; continue
                                    numerator = r.fp32(tf[pos]*r.fp32(2.2))
                                    contribution = r.fp32(r.fp32(idfs[term]*numerator)/r.fp32(tf[pos]+norms[doc]))
                                    total = r.fp32(local[doc-tile]+contribution)
                                    if not math.isfinite(total) or total < 0: bad = True
                                    else: local[doc-tile] = total
                                    contributions += 1
                        cursors[slot] += sum(consumed)
                        # Validate the prefix count, independently of lane visitation order.
                        if term >= 0:
                            assert cursors[slot] == bisect_left(docs, stop, start, ends[slot])
                        if bad: break
                    if bad: break
                    for doc in range(tile, stop): output[query*n+doc] = local[doc-tile]
                if cursors != ends: bad = True
            for doc in range(first, last):
                if bad: output[query*n+doc] = -1.0
                writes[query*n+doc] += 1
    assert all(value == 1 for value in writes), 'range ownership gap or overlap'
    return output, {'lower_bound_calls':searches, 'posting_contributions':contributions,
                    'lookahead_id_reads':lookahead, 'partitions':partitions, 'blocks':blocks}


def main():
    checks = 0; rng = random.Random(20261021)
    for n in (1, 6, 127, 129, 2047, 2048, 2049, 4097, 16385):
        documents = [([0]*(1+doc%3)+([1, 2] if doc%3 else [])+[3]*(doc%5)) if doc%31 else [] for doc in range(n)]
        if n == 1: documents = [[0]]
        index, values = r.build_index(documents, 9)
        queries = [[], [9], [8], [0, 0, 1], [2, 3, 2], [2147483646, 1, 0]]
        if n <= 129: queries.append([0, 1]*32)
        unused, words = r.make_queries(index, queries, 1)
        expected = r.diagnostic_scores(index, values, queries)
        before = {name:r.encode(value) for name, value in values.items()}
        for cores in (1, 2, 7, 56):
            actual, counts = simulate(index, values, words['query_offsets.bin'], words['query_term_ids.bin'], cores, rng)
            assert r.encode(actual) == r.encode(expected); checks += 1
        a, b = [[0], [3], []], [[1], [], [0, 0]]
        output = array('f', [1234])*(3*n)
        for current in (a, b, a, [[], [], []], b):
            unused, words = r.make_queries(index, current, 1)
            simulate(index, values, words['query_offsets.bin'], words['query_term_ids.bin'], 7, rng, output)
            assert r.encode(output) == r.encode(r.diagnostic_scores(index, values, current)); checks += 1
        assert before == {name:r.encode(value) for name, value in values.items()}; checks += 1
    # More queries than cores; independent repeated IDs and physical padding for empty words.
    index, values = r.build_index([[0, 1], [1, 1], [], [2]]*10+[[0]], 9)
    queries = [([], [9], [0, 1], [1, 1], [2, 0])[i%5] for i in range(513)]
    unused, words = r.make_queries(index, queries, 3)
    for cores in (1, 56):
        actual, unused_counts = simulate(index, values, words['query_offsets.bin'], words['query_term_ids.bin'], cores, rng)
        assert r.encode(actual) == r.encode(r.diagnostic_scores(index, values, queries)); checks += 1
    for offsets, terms in (([0, 65], [0]*65), ([0, 2], [0]), ([-1, 0], [0]), ([1, 0], [0]), ([0, 1], [-1])):
        actual, unused = simulate(index, values, offsets, terms, 56, rng)
        assert set(actual) == {-1.0}; checks += 1
    invalid = {name:array(value.typecode, value) for name, value in values.items()}
    invalid['posting_tf.bin'][0] = -1
    actual, unused = simulate(index, invalid, [0, 1], [0], 1, rng)
    assert set(actual) == {-1.0}; checks += 1
    # Geometry-only workload derivation, not a timing prediction.
    assert geometry(262144, 8, 56) == (7, 56)
    assert 2*3*8*128 == 6144 and 2*3*8*7 == 336
    assert geometry(65536, 1, 56) == (32, 32); checks += 3
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='query_score_range_MODEL_TEMPLATE_ONLY_', dir=ROOT/'results') as temp:
        base = Path(temp)
        for typed in (False, True):
            project = base/('project_'+str(typed))
            def fake_generate(command, **kwargs):
                assert command[0] == 'TEST_ONLY_FAKE_GENERATOR'
                project.mkdir(); (project/'build.sh').write_bytes(b'TEST ONLY TEMPLATE\n')
                for path in ('op_host/bm25_query_score.cpp', 'op_kernel/bm25_query_score.cpp'):
                    file = project/path; file.parent.mkdir(exist_ok=True); file.write_bytes(b'TEST ONLY ORIGINAL TEMPLATE\n')
                r.write_json(project/'CMakePresets.json', {'configurePresets':[{'cacheVariables':{
                    'ASCEND_COMPUTE_UNIT':{'type':'STRING', 'value':'old'} if typed else 'old'}}]})
            with patch.object(generator, 'generator_command', return_value=['TEST_ONLY_FAKE_GENERATOR']), patch.object(generator.subprocess, 'run', fake_generate), redirect_stdout(io.StringIO()):
                generator.generate(project, 'ai_core-ascend950')
            metadata = json.loads((project/'bm25_project.json').read_text())
            assert metadata['vendor_name'] == 'bm25_query_score_range' and metadata['max_terms_per_query'] == 64
            assert metadata['status'] == 'project_generated_not_compiled_not_device_verified'
            assert all(metadata['source_sha256'][name] == hashlib.sha256((ROOT/'npu_prototype/query_score_range'/name).read_bytes()).hexdigest() for name in generator.SOURCES)
            assert (project/'bm25_template_backup/op_kernel/bm25_query_score.cpp').read_bytes() == b'TEST ONLY ORIGINAL TEMPLATE\n'; checks += 4
        code = (ROOT/'npu_prototype/query_score_range/op_kernel/bm25_query_score.cpp').read_text()
        assert 'for (uint32_t tile=first;' in code and code.count('LowerBound(postingDocs') == 2
        assert code.index('LowerBound(postingDocs') < code.index('for (uint32_t tile=first;')
        assert 'atomicAdd(&scores' not in code and 'bounds[1]-bounds[0]<=kTerms' in code
        assert '9568' in code; checks += 4
    print(f'PASS: {checks} contiguous-owner/forward-cursor/FP32/overwrite/tail/limit/geometry MODEL and fake-template checks. '
          'No Ascend C compilation, SIMT barrier proof, device correctness or speedup conclusion.')


if __name__ == '__main__': main()
