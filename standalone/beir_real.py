"""Pinned public BEIR corpora, declared quality coverage, and separate fixed-Q8 performance."""

import argparse
import ast
import csv
import io
import json
import math
import random
import re
import shutil
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
DATASETS = {
    'scifact': {
        'bytes': 2816079,
        'sha256': '536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165',
        'docs': 5183,
        'queries': 1109,
        'test_queries': 300,
        'judgments': 339,
        'sort_docs': True,
        'grades': (1,),
        'status': 'actual_SciFact_all300_quality_and_Q8_six_round_search_checked',
    },
    'trec-covid': {
        'bytes': 73876720,
        'sha256': '120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0',
        'docs': 171332,
        'queries': 50,
        'test_queries': 50,
        'judgments': 66336,
        'sort_docs': False,
        'grades': (-1, 0, 1, 2),
        'status': 'actual_TREC_COVID_all50_quality_and_Q8_six_round_search_checked',
    },
    'msmarco': {
        'bytes': 1082258632,
        'sha256': 'f46aed8343564261c27228b7cf22dde5c9a70143fe34dc3c589bd5322308712a',
        'docs': 8841823,
        'queries': 509962,
        'test_queries': 43,
        'judgments': 9260,
        'sort_docs': False,
        'grades': (0, 1, 2, 3),
        'status': 'actual_MSMARCO_TREC_DL_test43_quality_and_Q8_six_round_search_checked',
    },
}
ARCHIVE_SHA256 = DATASETS['scifact']['sha256']
ARCHIVE_BYTES = DATASETS['scifact']['bytes']


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


class CorpusRows:
    """Reopen compressed corpus per pass; only the small legacy SciFact corpus is sorted in memory."""

    def __init__(self, archive, dataset):
        self.archive, self.dataset = archive, dataset

    def __iter__(self):
        with zipfile.ZipFile(self.archive) as source, source.open(self.dataset + '/corpus.jsonl') as stream:
            rows = (json.loads(line) for line in stream)
            if DATASETS[self.dataset]['sort_docs']:
                rows = sorted(rows, key=lambda row: row['_id'])
            yield from rows


def read_source(archive, dataset='scifact'):
    spec = DATASETS[dataset]
    r.require(
        archive.stat().st_size == spec['bytes'] and corpus.digest(archive) == spec['sha256'],
        'Archive differs from the pinned official BEIR package: ' + dataset,
    )
    # Read only the three known members; no archive path is extracted to the filesystem.
    with zipfile.ZipFile(archive) as source:
        rows = [json.loads(line) for line in source.read(dataset + '/queries.jsonl').decode('utf-8').splitlines()]
        queries = {row['_id']: row['text'] for row in rows}
        r.require(len(queries) == len(rows) == spec['queries'], 'Missing/duplicate source query ID')
        judgments = csv.DictReader(
            io.StringIO(source.read(dataset + '/qrels/test.tsv').decode('utf-8')), delimiter='\t'
        )
        qrels = {}
        count = 0
        for row in judgments:
            qid, did, score = row['query-id'], row['corpus-id'], int(row['score'])
            r.require(score in spec['grades'] and did not in qrels.get(qid, {}), 'Invalid/duplicate judgment')
            qrels.setdefault(qid, {})[did] = score
            count += 1
    r.require(
        len(qrels) == spec['test_queries'] and count == spec['judgments'] and set(qrels).issubset(queries),
        'Test split/judgment count mismatch',
    )
    return CorpusRows(archive, dataset), queries, qrels


def query_groups(qids, dataset):
    """Keep every test query; an odd test split gets one empty, unevaluated padding row."""
    qids = sorted(qids)
    r.require(len(qids) >= 16 and (len(qids) % 2 == 0 or dataset == 'msmarco'), 'Unsupported test split size')
    performance = random.Random(20261105).sample(qids, 16)
    half = (len(qids) + 1) // 2
    quality_b = qids[half:] + ([None] if len(qids) % 2 else [])
    return {
        'quality': {'A': qids[:half], 'B': quality_b},
        'performance': {'A': performance[:8], 'B': performance[8:]},
    }


def resource_plan(n, vocab, postings, groups, output):
    """Bound retained buffers using actual vocabulary/posting counts, outside Search timing."""
    index_bytes = 8 * postings + 4 * n + 12 * vocab + 8
    plans, prepared, measured = {}, 0, 0
    for name, cases in groups.items():
        q = len(cases['A'])
        plan = support.layout(n, vocab, postings, q, 10, q * 64)
        plans[name] = plan
        prepared += index_bytes + 8 * q * n
        rounds = 1 if name == 'quality' else 6
        measured += rounds * 4 * 8 * q * n + 8 * q * n + 8 * postings + 64 * vocab
    device = max(plan['known_device_bytes_upper_bound'] + plan['debug_score_bytes'] for plan in plans.values())
    disk = prepared + measured + 2 * 1024**3
    return {
        'docs': n,
        'vocab': vocab,
        'posting_records': postings,
        'stages': plans,
        'additional_disk_estimate_bytes': disk,
        'disk_free_bytes': shutil.disk_usage(output).free,
        'minimum_device_free_bytes_with_runner_margin': math.ceil((device + 1024**3) * 5 / 4),
        'scope': 'Known buffers and diagnostics plus margins, not a guarantee of upstream Python/JIT/runtime peak. Actual runner probes free memory and workspaces again before execution.',
    }


def prepare(output, archive=None, dataset='scifact'):
    spec = DATASETS[dataset]
    url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip'
    r.require(not output.exists(), 'Fresh data directory required')
    r.require(output.is_relative_to(ROOT / 'results'), 'Data must stay under project results')
    output.mkdir(parents=True)
    if archive is None:
        archive = output / 'source.zip'
        print('Downloading official BEIR archive: ' + dataset, flush=True)
        with urllib.request.urlopen(url, timeout=30) as response, archive.open('xb') as target:
            received = 0
            while True:
                block = response.read(256 * 1024)
                if not block:
                    break
                received += len(block)
                r.require(received <= spec['bytes'], 'Archive exceeds the pinned size')
                target.write(block)
                if received % (8 * 1024 * 1024) == 0 or received == spec['bytes']:
                    print(f'Downloaded {received}/{spec["bytes"]} bytes', flush=True)
    docs, texts, qrels = read_source(archive, dataset)
    stopwords = english_stopwords()
    vocabulary, doc_ids = set(), []
    for row in docs:
        doc_ids.append(row['_id'])
        vocabulary.update(tokenize(row.get('title', '') + ' ' + row['text'], stopwords))
        if len(doc_ids) % 50000 == 0:
            print('Vocabulary pass documents: ' + str(len(doc_ids)), flush=True)
    doc_id_set = set(doc_ids)
    r.require(len(doc_ids) == len(doc_id_set) == spec['docs'], 'Missing/duplicate corpus IDs')
    r.require(all(set(v).issubset(doc_id_set) for v in qrels.values()), 'Judgment refers to missing document')
    r.require(8 <= len(vocabulary) <= corpus.MAX_EXTERNAL_VOCAB, 'External vocabulary limit exceeded; no terms dropped')
    del doc_id_set
    vocabulary = sorted(vocabulary)
    word_ids = {word: i for i, word in enumerate(vocabulary)}
    qids = sorted(qrels)
    query_tokens = {qid: tokenize(texts[qid], stopwords) for qid in qids}
    query_ids = {qid: [word_ids[word] for word in row if word in word_ids] for qid, row in query_tokens.items()}
    r.require(
        all(len(row) <= 64 for row in query_ids.values()), 'Natural queries exceed the supported packet; no truncation'
    )
    documents = output / 'corpus_ids.jsonl'
    postings = 0
    with documents.open('w', encoding='utf-8', newline='\n') as stream:
        for ordinal, row in enumerate(docs):
            r.require(row['_id'] == doc_ids[ordinal], 'Corpus order changed between passes')
            tokens = tokenize(row.get('title', '') + ' ' + row['text'], stopwords)
            if dataset == 'msmarco':
                postings += len(set(tokens))
            stream.write(json.dumps([word_ids[word] for word in tokens]) + '\n')
            if (ordinal + 1) % 500000 == 0:
                print('Encoded corpus documents: ' + str(ordinal + 1), flush=True)
    r.write_json(output / 'mapping.json', {'doc_ids': doc_ids, 'vocabulary': vocabulary})
    r.write_json(output / 'queries.json', {qid: texts[qid] for qid in qids})
    r.write_json(output / 'qrels.json', qrels)
    groups_by_stage = query_groups(qids, dataset)
    half = len(groups_by_stage['quality']['A'])
    if dataset == 'msmarco':
        plan = resource_plan(len(doc_ids), len(vocabulary), postings, groups_by_stage, output)
        r.write_json(output / 'resource_plan.json', plan)
        print('RESOURCE PLAN: ' + json.dumps(plan), flush=True)
        r.require(
            plan['additional_disk_estimate_bytes'] < plan['disk_free_bytes'] * 4 // 5,
            'MS MARCO disk budget exceeded; see resource_plan.json; data is retained',
        )
    for name, groups in groups_by_stage.items():
        target = output / name
        print(f'Building {name} index/oracle: N{len(doc_ids)}, Q{len(groups["A"])}', flush=True)
        description = corpus.prepare(
            target,
            len(doc_ids),
            len(vocabulary),
            0,
            {case: [query_ids[qid] if qid is not None else [] for qid in ids] for case, ids in groups.items()},
            corpus=documents,
            variable_queries=True,
        )
        if dataset == 'msmarco':
            r.require(description['posting_records'] == postings, 'Source and builder posting counts disagree')
        r.write_json(target / 'qids.json', groups)
        print(f"Prepared {name}: Q{description['queries']}, natural query lengths preserved", flush=True)
    r.require(
        read(output / 'quality/index/metadata.json')['index_id']
        == read(output / 'performance/index/metadata.json')['index_id'],
        'Quality/performance corpus differs',
    )
    record = {
        'status': 'public_real_text_inputs_and_independent_oracles_prepared_device_not_run',
        'dataset': dataset,
        'source': {
            'url': url,
            'archive_sha256': spec['sha256'],
            'archive_bytes': spec['bytes'],
            'dataset_card': 'https://huggingface.co/datasets/BeIR/' + dataset,
            'qrels': 'https://huggingface.co/datasets/BeIR/' + dataset + '-qrels',
        },
        'docs': len(doc_ids),
        'test_queries': len(qids),
        'judgments': spec['judgments'],
        'judgment_grade_counts': {
            str(grade): sum(v == grade for rels in qrels.values() for v in rels.values()) for grade in spec['grades']
        },
        'metric_policy': 'trec_eval linear nDCG gains; relevance > 0 for recall/MRR; zero and unjudged (-1) have zero gain',
        'document_order': 'lexical source ID' if spec['sort_docs'] else 'official archive order',
        'vocab': len(vocabulary),
        'tokenizer': 'Matches frozen BM25S basic tokenizer: lowercase, (?u)\\b\\w\\w+\\b, original English stopwords, no stemming',
        'query_policy': 'No length truncation; OOV tokens omitted using the fixed document vocabulary; empty queries retain zero-score semantics',
        'query_lengths_after_OOV_filter': corpus.distribution([len(row) for row in query_ids.values()]),
        'OOV_tokens': sum(word not in word_ids for row in query_tokens.values() for word in row),
        'quality': f'All {len(qids)} judged queries, A/B {half} each, one checked round; not a Q8 performance result'
        + ('; one empty padding row excluded from metrics' if len(qids) % 2 else ''),
        'performance': 'Seeded 16-query sample, two Q8 packets, six rounds; labels not used for sampling',
    }
    if dataset == 'msmarco':
        record['evaluation_split'] = 'test: 43 graded queries (not the 6980-query dev split)'
    r.require(corpus.digest(archive) == spec['sha256'], 'Source archive changed during preparation')
    record['file_sha256'] = {
        path.relative_to(output).as_posix(): corpus.digest(path) for path in output.rglob('*') if path.is_file()
    }
    r.write_json(output / 'preparation.json', record)
    print('PREPARED: ' + str(output / 'preparation.json'), flush=True)


def ranking_metrics(predictions, qrels):
    r.require(bool(qrels) and set(predictions) == set(qrels), 'All judged queries must be evaluated exactly once')
    totals = {'ndcg_at_10': [], 'recall_at_10': [], 'mrr_at_10': []}
    for qid, ranked in predictions.items():
        r.require(len(ranked) == 10 and len(set(ranked)) == 10, 'Invalid Top-10 ranking')
        truth = {did: rel for did, rel in qrels[qid].items() if rel > 0}
        dcg = sum(truth.get(did, 0) / math.log2(pos + 2) for pos, did in enumerate(ranked))
        ideal = sum(rel / math.log2(pos + 2) for pos, rel in enumerate(sorted(truth.values(), reverse=True)[:10]))
        totals['ndcg_at_10'].append(dcg / ideal if ideal else 0)
        totals['recall_at_10'].append(sum(did in truth for did in ranked) / len(truth) if truth else 0)
        totals['mrr_at_10'].append(next((1 / (i + 1) for i, did in enumerate(ranked) if did in truth), 0))
    return {'queries': len(qrels), **{key: sum(values) / len(values) for key, values in totals.items()}}


def evaluate_quality(data, measurement):
    groups, mapping, qrels = read(data / 'quality/qids.json'), read(data / 'mapping.json'), read(data / 'qrels.json')
    qids = [qid for ids in groups.values() for qid in ids if qid is not None]
    r.require(len(qids) == len(set(qids)) and set(qids) == set(qrels), 'Quality must cover every test query once')
    padding = sum(qid is None for ids in groups.values() for qid in ids)
    r.require(padding == len(qids) % 2, 'Unexpected quality padding')
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
                if qid is not None:
                    predictions[qid] = [mapping['doc_ids'][i] for i in ids[offset * 10 : (offset + 1) * 10]]
        r.write_json(measurement / (role + '.rankings.json'), predictions)
        results[role] = ranking_metrics(predictions, qrels)
    return results


def validate_data(data, dataset='scifact'):
    record = read(data / 'preparation.json')
    r.require(
        record.get('dataset', 'scifact') == dataset
        and record['source']['archive_sha256'] == DATASETS[dataset]['sha256'],
        'Different dataset source',
    )
    for name, expected in record['file_sha256'].items():
        path = (data / name).resolve()
        r.require(path.is_relative_to(data) and corpus.digest(path) == expected, 'Prepared data changed: ' + name)
    for name in ('quality', 'performance'):
        corpus.validate(data / name)
    return record


def run(args):
    r.require(sys.platform == 'linux' and not args.output.exists(), 'A5 Linux and fresh output required')
    dataset = getattr(args, 'dataset', 'scifact')
    record = validate_data(args.data, dataset)
    args.output.mkdir(parents=True)
    report = {
        'status': 'running',
        'dataset': record,
        'performance_scope': 'Q8 sample, separate from the full-query quality packets',
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
        validate_data(args.data, dataset)
        report['status'] = DATASETS[dataset]['status']
        report['scope'] = (
            'Pinned public real corpus, declared tokenizer and BM25 parameters; native ties reported separately. No universal workload speed claim.'
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
    parser.add_argument('--dataset', choices=tuple(DATASETS), default='scifact')
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
        prepare(args.output, args.archive, args.dataset)
    else:
        r.require(
            not args.output.is_relative_to(args.data) and not args.data.is_relative_to(args.output),
            'Independent data/output required',
        )
        run(args)


if __name__ == '__main__':
    main()
