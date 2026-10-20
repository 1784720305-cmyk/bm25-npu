"""Independent host roundtrip, raw TF, FP32, query reuse and rejection checks; no NPU."""
from array import array
from collections import Counter
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from standalone import resident_index as r
from standalone.scale_fixture import generate

(BASE/'results').mkdir(parents=True, exist_ok=True)
checks = 0


def expect_failure(call):
    global checks
    try:
        call()
    except (ValueError, KeyError, OSError, OverflowError):
        checks += 1
    else:
        raise AssertionError('Bad index/query was accepted')


def direct_scores(documents, queries, vocab):
    """Per-document loop, unlike the production diagnostic's per-term posting scan."""
    counts = [Counter(row) for row in documents]
    df = [sum(term in row for row in counts) for term in range(vocab)]
    avgdl = sum(map(len, documents))/len(documents)
    result = array('f')
    for query in queries:
        for doc, count in zip(documents, counts):
            total = 0.0
            norm = r.fp32(1.2*(1-.75+.75*len(doc)/avgdl))
            for term in query:
                frequency = count.get(term, 0)
                if frequency:
                    idf = r.fp32(math.log1p((len(documents)-df[term]+.5)/(df[term]+.5)))
                    product = r.fp32(frequency*r.fp32(2.2))
                    contribution = r.fp32(r.fp32(idf*product)/r.fp32(frequency+norm))
                    total = r.fp32(total+contribution)
            result.append(total)
    return result


with tempfile.TemporaryDirectory(prefix='resident_index_check_',dir=BASE/'results') as path:
    root = Path(path)
    randomizer = random.Random(20261020)
    for case in range(24):
        vocab = 8+case
        docs = [[], [0, 0, 1], [0, 0, 1]]
        docs += [[randomizer.randrange(vocab) for _ in range(randomizer.randrange(1, 40))]
                 for _ in range(4+case)]
        queries = [[], [vocab], [0, 0, 1], [1, vocab, 0], [vocab+1, vocab+2]]
        queries += [[randomizer.randrange(vocab) for _ in range(1+case%7)] for _ in range(case%5)]
        index, arrays = r.build_index(docs, vocab)
        record, query_arrays = r.make_queries(index, queries, min(len(docs), 1+case%32))
        ipath, qpath = root/f'i{case}', root/f'q{case}'
        r.save_arrays(ipath, index, arrays)
        r.save_arrays(qpath, record, query_arrays)
        actual_index, actual_arrays = r.validate_index(ipath)
        actual_record, actual_queries = r.validate_queries(qpath, actual_index)
        assert actual_index == index and actual_queries == queries
        assert r.diagnostic_scores(actual_index, actual_arrays, actual_queries).tobytes() == direct_scores(docs,queries,vocab).tobytes()
        # Every posting, not just terms selected by the test queries.
        decoded = [Counter() for _ in docs]
        for term in range(vocab):
            a,b = arrays['term_offsets.bin'][term:term+2]
            for pos in range(a,b):
                decoded[arrays['posting_doc_ids.bin'][pos]][term] = int(arrays['posting_tf.bin'][pos])
        assert decoded == [Counter(doc) for doc in docs]
        assert record['query_input_bytes'] == 4*sum(map(len,queries))+8*(len(queries)+1)
        assert record['topk_output_bytes'] == len(queries)*record['k']*8
        checks += 6

    # Index identity/raw bytes stay fixed for new queries and a different K.
    before = {p.name:p.read_bytes() for p in ipath.iterdir()}
    new_queries = [[vocab+100,0], [], [1,1,1]]
    new_record, new_arrays = r.make_queries(index,new_queries,1)
    new_path = root/'new_queries'
    r.save_arrays(new_path,new_record,new_arrays)
    assert r.validate_queries(new_path,index)[1] == new_queries
    assert before == {p.name:p.read_bytes() for p in ipath.iterdir()}
    different_index, unused = r.build_index(docs+[[0]],vocab)
    expect_failure(lambda:r.validate_queries(new_path,different_index))
    record_8, unused = r.make_queries(index,[[0,1,2]]*8,1)
    assert record_8['query_input_bytes'] == 168
    checks += 3

    for bad in ([[-1]], [[True]], [[2**31-1]], [], 'bad'):
        expect_failure(lambda:r.make_queries(index,bad,1))
    for bad in (0,33,True):
        expect_failure(lambda:r.make_queries(index,[[0]],bad))
    for doc, v in (([[]],8), ([[True]],8), ([[-1]],8), ([[8]],8), ([[0]],0)):
        expect_failure(lambda:r.build_index(doc,v))
    expect_failure(lambda:r.save_arrays(ipath,index,arrays))

    # Corrupt bytes, then self-consistent hashes with semantically invalid payloads.
    norm_path = ipath/'doc_norm.bin'
    old = norm_path.read_bytes()
    norm_path.write_bytes(old+b'bad')
    expect_failure(lambda:r.validate_index(ipath))
    norm_path.write_bytes(old)
    def forged_case(name, mutate):
        forged = {key:array(value.typecode,value) for key,value in arrays.items()}
        mutate(forged)
        meta = dict(index)
        meta['files'] = {key:r.digest(r.encode(value)) for key,value in forged.items()}
        meta['index_bytes'] = sum(len(r.encode(value)) for value in forged.values())
        meta['index_id'] = r.identity(meta)
        dest = root/name
        r.save_arrays(dest,meta,forged)
        expect_failure(lambda:r.validate_index(dest))
    forged_case('negative_tf',lambda a:a['posting_tf.bin'].__setitem__(0,-1))
    forged_case('nan_norm',lambda a:a['doc_norm.bin'].__setitem__(0,float('nan')))
    forged_case('wrong_norm',lambda a:a['doc_norm.bin'].__setitem__(0,100))
    forged_case('wrong_idf',lambda a:a['idf.bin'].__setitem__(0,100))
    forged_case('wrong_doc',lambda a:a['posting_doc_ids.bin'].__setitem__(0,len(docs)))
    forged_case('duplicate_doc',lambda a:a['posting_doc_ids.bin'].__setitem__(1,a['posting_doc_ids.bin'][0]))
    forged_case('bad_offset',lambda a:a['term_offsets.bin'].__setitem__(0,1))

    for case in ('correctness','performance'):
        source, packed = root/f'source_{case}', root/f'packed_{case}'
        with redirect_stdout(io.StringIO()):
            generate(source,128,8,10,32,20261020,32,case=case,query_terms=3)
            r.prepare(source,packed)
        report = json.loads((packed/'preparation_report.json').read_text())
        assert report['status'] == 'host_index_and_query_roundtrip_checked_device_not_run'
        assert report['device_verified'] is False
        assert not (packed/'scores.bin').exists() and not (packed/'timing.json').exists()
        if case == 'performance': assert report['query_input_bytes'] == 168
        expect_failure(lambda:r.prepare(source,packed))
        checks += 4
        source_meta = json.loads((source/'metadata.json').read_text())
        source_meta['corpus_sha256'] = '0'*64
        r.write_json(source/'metadata.json',source_meta)
        expect_failure(lambda:r.read_source(source))

print(f'PASS: {checks} resident-index host format/formula/immutable-query-reuse/invalid-data checks; temporary files removed. No NPU compilation, execution or benchmark.')
