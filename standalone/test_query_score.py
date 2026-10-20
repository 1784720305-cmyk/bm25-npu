"""New score algorithm/ownership MODEL + real project-population tests. No CANN/NPU compilation."""
from array import array
from bisect import bisect_left
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import create_project as generator, resident_index as r, query_score_fixture as fixture

THREADS, TILE = 128, 2048


def simulate(index, values, offsets, terms, cores, randomizer, output=None):
    """Phase-ordered Python model, NOT the C++ kernel; barriers/ABI need device validation."""
    n, q, v = index['num_docs'], len(offsets)-1, index['vocab_size']
    r.require(cores > 0 and q > 0, 'bad model geometry')
    tiles = (n-1)//TILE+1
    if output is None: output = array('f',[float('nan')])*(n*q)
    r.require(len(output) == n*q, 'bad model output size')
    writers = array('i',[0])*(n*q)
    blocks = min(cores,q*tiles)
    t_offsets, docs, tf, norms, idfs = (values[key] for key in r.INDEX_TYPES)
    block_order = list(range(blocks))
    randomizer.shuffle(block_order)
    for block in block_order:
        for task in range(block,q*tiles,blocks):
            query, tile = divmod(task,tiles)
            first, last = tile*TILE,min((tile+1)*TILE,n)
            local = array('f',[0])*TILE
            a,b = offsets[query:query+2]
            invalid = not (0 <= a <= b <= len(terms))
            if not invalid:
                for term in terms[a:b]:
                    if term < 0:
                        invalid = True
                        break
                    if term >= v: continue
                    begin,end = t_offsets[term:term+2]
                    if not (0 <= begin <= end <= index['posting_count']):
                        invalid = True
                        break
                    begin = bisect_left(docs,first,begin,end)
                    end = bisect_left(docs,last,begin,end)
                    lane_order = list(range(THREADS))
                    randomizer.shuffle(lane_order)
                    owners = set()
                    for lane in lane_order:
                        for pos in range(begin+lane,end,THREADS):
                            doc = docs[pos]
                            r.require(doc not in owners, 'multiple writers within a term')
                            owners.add(doc)
                            if not (first <= doc < last and 1 <= tf[pos] <= 2**24 and
                                    math.isfinite(norms[doc]) and norms[doc] > 0 and
                                    math.isfinite(idfs[term]) and idfs[term] >= 0):
                                invalid = True
                                continue
                            numerator = r.fp32(tf[pos]*r.fp32(2.2))
                            contribution = r.fp32(r.fp32(idfs[term]*numerator)/r.fp32(tf[pos]+norms[doc]))
                            local[doc-first] = r.fp32(local[doc-first]+contribution)
                    if invalid: break
            for doc in range(first,last):
                output[query*n+doc] = -1.0 if invalid else local[doc-first]
                writers[query*n+doc] += 1
    assert all(value == 1 for value in writers), 'output gap or multiple block writers'
    return output


def main():
    checks = 0
    randomizer = random.Random(20261020)
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='query_score_checks_',dir=ROOT/'results') as temporary:
        root = Path(temporary)
        for n in (1,6,127,128,129,2047,2048,2049,4097):
            documents = [[0]*((doc%3)+1)+([1,2] if doc%2 else []) for doc in range(n)]
            if n > 1: documents[0] = []
            index, values = r.build_index(documents,8)
            queries = [[], [8], [0,0,1], [2,99,1], [7], [0], [1,2,1,0]]
            if n < 10: queries += [[0,1]*192]
            record, words = r.make_queries(index,queries,1)
            expected = r.diagnostic_scores(index,values,queries)
            for cores in (1,2,7,56):
                actual = simulate(index,values,words['query_offsets.bin'],words['query_term_ids.bin'],cores,randomizer)
                assert actual.tobytes() == expected.tobytes()
                checks += 1
            # Same buffers/index, genuinely different A/B/A inputs and changed match locations.
            queries_a, queries_b = [[0],[2],[]], [[1],[],[0,0]]
            output = array('f',[1234])*(n*3)
            before = {key:r.encode(value) for key,value in values.items()}
            for queries in (queries_a,queries_b,queries_a,[[],[],[]]):
                unused, words = r.make_queries(index,queries,1)
                physical = words['query_term_ids.bin'] or array('i',[0])
                simulate(index,values,words['query_offsets.bin'],physical,7,randomizer,output)
                assert output.tobytes() == r.diagnostic_scores(index,values,queries).tobytes()
                checks += 1
            assert before == {key:r.encode(value) for key,value in values.items()}
            checks += 1

        # 513 queries: multiple grid-stride tasks, tails, empty/unknown and duplicate terms.
        index, values = r.build_index([[0,1],[1,1],[0,2],[]]*10+[ [2] ],8)
        queries = [([], [100], [0,1], [1,1], [2,0])[i%5] for i in range(513)]
        unused, words = r.make_queries(index,queries,10)
        for cores in (1,56):
            actual = simulate(index,values,words['query_offsets.bin'],words['query_term_ids.bin'],cores,randomizer)
            assert actual.tobytes() == r.diagnostic_scores(index,values,queries).tobytes()
            checks += 1
        # Detectable negative output on malformed runtime boundaries/IDs or referenced TF.
        for offsets, terms in (([0,2], [0]), ([0,1], [-1]), ([-1,0], [0]), ([1,0], [0])):
            actual = simulate(index,values,offsets,terms,1,randomizer)
            assert list(actual) == [-1.0]*index['num_docs']
            checks += 1
        invalid = {key:array(value.typecode,value) for key,value in values.items()}
        invalid['posting_tf.bin'][0] = -1
        assert set(simulate(index,invalid,[0,1],[0],1,randomizer)) == {-1.0}
        checks += 1

        # Exercise actual populate(), exact source hashes, backups and both preset encodings.
        for operator, typed in (('query_score',False),('query_score',True),('contribution',True)):
            project = root/f'project_{operator}_{typed}'
            stem = generator.OPERATORS[operator][1]
            project.mkdir()
            (project/'build.sh').write_bytes(b'TEST TEMPLATE ONLY\n')
            for directory in ('op_host','op_kernel'):
                (project/directory).mkdir()
                (project/directory/(stem+'.cpp')).write_bytes(b'TEST GENERATED TEMPLATE\n')
            variables = {'ASCEND_COMPUTE_UNIT':{'type':'STRING','value':'old'} if typed else 'old'}
            r.write_json(project/'CMakePresets.json',{'configurePresets':[{'name':'test','cacheVariables':variables}]})
            generator.populate(project,'ascend950',generator.OPERATORS[operator][3],operator)
            metadata = json.loads((project/'bm25_project.json').read_text())
            assert metadata['status'] == 'project_generated_not_compiled_not_device_verified'
            for relative,digest in metadata['source_sha256'].items():
                assert r.digest((project/relative).read_bytes()) == digest
                origin = ROOT/'npu_prototype'
                if operator == 'query_score': origin /= operator
                assert (project/relative).read_bytes() == (origin/relative).read_bytes()
            assert (project/'bm25_template_backup/op_kernel'/(stem+'.cpp')).read_bytes() == b'TEST GENERATED TEMPLATE\n'
            preset = json.loads((project/'CMakePresets.json').read_text())['configurePresets'][0]['cacheVariables']
            assert preset['ASCEND_COMPUTE_UNIT']['value'] == 'ascend950'
            assert preset['vendor_name']['value'] == generator.OPERATORS[operator][3]
            checks += 6
        command = [sys.executable,'-B',str(ROOT/'standalone/create_project.py'),'--operator','query_score',
                   '--compute-resource','ai_core-ascend950','--vendor','bm25']
        attempt = subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        assert attempt.returncode != 0 and b'bm25_query_score' in attempt.stderr
        checks += 1

        with redirect_stdout(io.StringIO()): fixture.prepare(root/'fixtures')
        for case in ('A','B','empty'):
            qpath = root/'fixtures'/case
            idx, vals = r.validate_index(root/'fixtures/index')
            record, queries = r.validate_queries(qpath,idx)
            unused, words = r.make_queries(idx,queries,3)
            physical = words['query_term_ids.bin'] or array('i',[0])
            actual = simulate(idx,vals,words['query_offsets.bin'],physical,56,randomizer)
            actual_path = root/f'{case}_MODEL_NOT_DEVICE.bin'
            actual_path.write_bytes(r.encode(actual))
            with redirect_stdout(io.StringIO()): fixture.check(root/'fixtures/index',qpath,actual_path)
            actual[0] = float('nan')
            actual_path.write_bytes(r.encode(actual))
            try:
                with redirect_stdout(io.StringIO()): fixture.check(root/'fixtures/index',qpath,actual_path)
            except ValueError: pass
            else: raise AssertionError('invalid output accepted')
            checks += 2

    print(f'PASS: {checks} query-score MODEL/ownership/reuse/failure/fixture/project-copy checks. '
          'Model is not the C++ kernel; no CANN compilation, NPU execution or search timing.')


if __name__ == '__main__':
    main()
