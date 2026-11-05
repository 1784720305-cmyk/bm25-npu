"""Pinned BEIR SciFact preparation, natural-query quality, and separate Q8 performance."""

import argparse
import ast
import csv
import io
import json
import math
import random
import re
import sys
import urllib.request
import zipfile
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import final_support as support
from standalone import large_corpus as corpus
from standalone import resident_index as r
from standalone.run_bm25s_reference import check_source

URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
ARCHIVE_SHA256 = '536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165'
ARCHIVE_BYTES = 2816079


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def english_stopwords():
    check_source()
    tree = ast.parse((ROOT / 'reference_only/bm25s_original/bm25s/stopwords.py').read_text(encoding='utf-8'))
    return set(
        next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'STOPWORDS_EN' for target in node.targets)
        )
    )


def tokenize(text, stopwords):
    return [word for word in re.findall(r'(?u)\b\w\w+\b', text.lower()) if word not in stopwords]


def read_source(archive):
    r.require(
        archive.stat().st_size == ARCHIVE_BYTES and corpus.digest(archive) == ARCHIVE_SHA256,
        'SciFact archive differs from the pinned official BEIR package',
    )
    # Read only the three known members; no archive path is extracted to the filesystem.
    with zipfile.ZipFile(archive) as source:
        docs = [json.loads(line) for line in source.read('scifact/corpus.jsonl').decode('utf-8').splitlines()]
        queries = [json.loads(line) for line in source.read('scifact/queries.jsonl').decode('utf-8').splitlines()]
        judgments = csv.DictReader(io.StringIO(source.read('scifact/qrels/test.tsv').decode('utf-8')), delimiter='\t')
        qrels = {}
        count = 0
        for row in judgments:
            qid, did, score = row['query-id'], row['corpus-id'], int(row['score'])
            r.require(score > 0 and did not in qrels.get(qid, {}), 'Invalid/duplicate SciFact judgment')
            qrels.setdefault(qid, {})[did] = score
            count += 1
    docs = sorted(docs, key=lambda row: row['_id'])
    queries = {row['_id']: row['text'] for row in queries}
    doc_ids = [row['_id'] for row in docs]
    r.require(
        len(docs) == len(set(doc_ids)) == 5183 and len(queries) == 1109 and len(qrels) == 300 and count == 339,
        'SciFact split/count mismatch',
    )
    r.require(
        set(qrels).issubset(queries) and all(set(v).issubset(doc_ids) for v in qrels.values()),
        'Judgment references a missing query/document',
    )
    return docs, queries, qrels


def prepare(output, archive=None):
    r.require(not output.exists(), 'Fresh data directory required')
    r.require(output.is_relative_to(ROOT / 'results'), 'Data must stay under project results')
    output.mkdir(parents=True)
    if archive is None:
        archive = output / 'source.zip'
        print('Downloading official BEIR SciFact archive', flush=True)
        with urllib.request.urlopen(URL, timeout=30) as response, archive.open('xb') as target:
            received = 0
            while True:
                block = response.read(256 * 1024)
                if not block:
                    break
                received += len(block)
                r.require(received <= ARCHIVE_BYTES, 'Archive exceeds the pinned size')
                target.write(block)
    docs, texts, qrels = read_source(archive)
    stopwords = english_stopwords()
    doc_tokens = [tokenize(row.get('title', '') + ' ' + row['text'], stopwords) for row in docs]
    vocabulary = sorted({word for row in doc_tokens for word in row})
    word_ids = {word: i for i, word in enumerate(vocabulary)}
    qids = sorted(qrels)
    query_tokens = {qid: tokenize(texts[qid], stopwords) for qid in qids}
    query_ids = {qid: [word_ids[word] for word in row if word in word_ids] for qid, row in query_tokens.items()}
    r.require(
        all(len(row) <= 64 for row in query_ids.values()), 'Natural queries exceed the supported packet; no truncation'
    )
    documents = output / 'corpus_ids.jsonl'
    with documents.open('w', encoding='utf-8', newline='\n') as stream:
        for row in doc_tokens:
            stream.write(json.dumps([word_ids[word] for word in row]) + '\n')
    r.write_json(output / 'mapping.json', {'doc_ids': [row['_id'] for row in docs], 'vocabulary': vocabulary})
    r.write_json(output / 'queries.json', {qid: texts[qid] for qid in qids})
    r.write_json(output / 'qrels.json', qrels)
    performance = random.Random(20261105).sample(qids, 16)
    query_groups = {
        'quality': {'A': qids[:150], 'B': qids[150:]},
        'performance': {'A': performance[:8], 'B': performance[8:]},
    }
    for name, groups in query_groups.items():
        target = output / name
        description = corpus.prepare(
            target,
            len(docs),
            len(vocabulary),
            0,
            {case: [query_ids[qid] for qid in ids] for case, ids in groups.items()},
            corpus=documents,
            variable_queries=True,
        )
        r.write_json(target / 'qids.json', groups)
        print(f"Prepared {name}: Q{description['queries']}, natural query lengths preserved", flush=True)
    r.require(
        read(output / 'quality/index/metadata.json')['index_id']
        == read(output / 'performance/index/metadata.json')['index_id'],
        'Quality/performance corpus differs',
    )
    record = {
        'status': 'public_real_text_inputs_and_independent_oracles_prepared_device_not_run',
        'source': {
            'url': URL,
            'archive_sha256': ARCHIVE_SHA256,
            'archive_bytes': ARCHIVE_BYTES,
            'dataset_card': 'https://huggingface.co/datasets/BeIR/scifact',
            'qrels': 'https://huggingface.co/datasets/BeIR/scifact-qrels',
        },
        'docs': len(docs),
        'test_queries': len(qids),
        'judgments': 339,
        'vocab': len(vocabulary),
        'tokenizer': 'Matches frozen BM25S basic tokenizer: lowercase, (?u)\\b\\w\\w+\\b, original English stopwords, no stemming',
        'query_policy': 'No length truncation; OOV tokens omitted using the fixed document vocabulary; empty queries retain zero-score semantics',
        'query_lengths_after_OOV_filter': corpus.distribution([len(row) for row in query_ids.values()]),
        'OOV_tokens': sum(word not in word_ids for row in query_tokens.values() for word in row),
        'quality': 'All 300 judged queries, A/B 150 each, one checked round; not a Q8 performance result',
        'performance': 'Seeded 16-query sample, two Q8 packets, six rounds; labels not used for sampling',
    }
    record['file_sha256'] = {
        path.relative_to(output).as_posix(): corpus.digest(path) for path in output.rglob('*') if path.is_file()
    }
    r.write_json(output / 'preparation.json', record)
    print('PREPARED: ' + str(output / 'preparation.json'), flush=True)


def ranking_metrics(predictions, qrels):
    r.require(set(predictions) == set(qrels), 'All judged queries must be evaluated exactly once')
    totals = {'ndcg_at_10': [], 'recall_at_10': [], 'mrr_at_10': []}
    for qid, ranked in predictions.items():
        r.require(len(ranked) == 10 and len(set(ranked)) == 10, 'Invalid Top-10 ranking')
        truth = qrels[qid]
        gain = lambda rel: 2**rel - 1
        dcg = sum(gain(truth.get(did, 0)) / math.log2(pos + 2) for pos, did in enumerate(ranked))
        ideal = sum(gain(rel) / math.log2(pos + 2) for pos, rel in enumerate(sorted(truth.values(), reverse=True)[:10]))
        totals['ndcg_at_10'].append(dcg / ideal if ideal else 0)
        totals['recall_at_10'].append(sum(did in truth for did in ranked) / len(truth) if truth else 0)
        totals['mrr_at_10'].append(next((1 / (i + 1) for i, did in enumerate(ranked) if did in truth), 0))
    return {'queries': len(qrels), **{key: sum(values) / len(values) for key, values in totals.items()}}


def evaluate_quality(data, measurement):
    groups, mapping, qrels = read(data / 'quality/qids.json'), read(data / 'mapping.json'), read(data / 'qrels.json')
    report = read(measurement / 'report.json')
    r.require(report['status'] == 'actual_real_query_quality_checked', 'Quality round did not complete')
    results = {}
    for role in ('npu', 'original_native_t1', 'original_native_t8', 'original_native_t32'):
        predictions = {}
        for case, qids in groups.items():
            directory = measurement / 'round1' / role
            path = (
                directory / 'actual' / ('after_' + case) / 'top_ids.bin'
                if role == 'npu'
                else directory / case / 'top_ids.bin'
            )
            ids = array('i')
            ids.frombytes(path.read_bytes())
            r.require(
                ids.itemsize == 4 and sys.byteorder == 'little' and len(ids) == len(qids) * 10,
                'Unexpected ranking byte contract',
            )
            r.require(all(0 <= value < len(mapping['doc_ids']) for value in ids), 'Invalid document ID')
            for offset, qid in enumerate(qids):
                predictions[qid] = [mapping['doc_ids'][i] for i in ids[offset * 10 : (offset + 1) * 10]]
        r.write_json(measurement / (role + '.rankings.json'), predictions)
        results[role] = ranking_metrics(predictions, qrels)
    return results


def validate_data(data):
    record = read(data / 'preparation.json')
    r.require(record['source']['archive_sha256'] == ARCHIVE_SHA256, 'Different dataset source')
    for name, expected in record['file_sha256'].items():
        path = (data / name).resolve()
        r.require(path.is_relative_to(data) and corpus.digest(path) == expected, 'Prepared data changed: ' + name)
    for name in ('quality', 'performance'):
        corpus.validate(data / name)
    return record


def run(args):
    r.require(sys.platform == 'linux' and not args.output.exists(), 'A5 Linux and fresh output required')
    record = validate_data(args.data)
    args.output.mkdir(parents=True)
    report = {
        'status': 'running',
        'dataset': record,
        'performance_scope': 'Q8 sample, not the 150-query quality packets',
    }
    try:
        for name in ('quality', 'performance'):
            command = [
                sys.executable,
                '-u',
                ROOT / 'standalone/benchmark_final.py',
                '--real-input',
                '--cpu-baseline',
                'bm25s',
                '--input',
                args.data / name,
                '--output',
                args.output / name,
                '--vendors',
                args.vendors,
                '--host-build',
                args.host_build,
                '--dependency-dir',
                args.dependency_dir,
                '--device',
                args.device,
            ]
            if name == 'quality':
                command.append('--quality-only')
            support.execute(command, args.output / (name + '.log'), timeout=14400)
        report['quality'] = evaluate_quality(args.data, args.output / 'quality')
        performance = read(args.output / 'performance/report.json')
        r.require(performance['status'] == 'actual_final_search_six_rounds_checked', 'Performance rounds incomplete')
        report['performance'] = performance['bm25s_decision']['original_native']
        validate_data(args.data)
        report['status'] = 'actual_SciFact_all300_quality_and_Q8_six_round_search_checked'
        report['scope'] = (
            'Small public real corpus, declared tokenizer and BM25 parameters; native ties reported separately. No large-business-corpus claim.'
        )
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        r.write_json(args.output / 'report.json', report)
    print('REAL DATA QUALITY: ' + json.dumps(report['quality']), flush=True)
    print('REAL DATA PERFORMANCE: ' + json.dumps(report['performance']), flush=True)
    print('REPORT: ' + str(args.output / 'report.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--archive', type=Path)
    measure = sub.add_parser('run')
    for name in ('data', 'output', 'vendors', 'host-build', 'dependency-dir'):
        measure.add_argument('--' + name, type=Path, required=True)
    measure.add_argument('--device', type=int, default=5)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    r.require(
        args.output.is_relative_to(ROOT / 'results') and args.output != ROOT / 'results', 'Output must be under results'
    )
    if args.mode == 'prepare':
        prepare(args.output, args.archive)
    else:
        r.require(
            not args.output.is_relative_to(args.data) and not args.data.is_relative_to(args.output),
            'Independent data/output required',
        )
        run(args)


if __name__ == '__main__':
    main()
