"""Test graded qrels, streaming corpus replay and external INT32 word IDs."""

import json
import math
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import beir_real as tool
from standalone import large_corpus


class BeirRealTests(unittest.TestCase):
    def test_graded_trec_gains_ignore_zero_and_unjudged_for_recall(self):
        ranked = ['zero', 'unjudged', 'b', 'a'] + [str(i) for i in range(6)]
        result = tool.ranking_metrics({'q': ranked}, {'q': {'a': 2, 'b': 1, 'zero': 0, 'unjudged': -1}})
        expected = (1 / math.log2(4) + 2 / math.log2(5)) / (2 + 1 / math.log2(3))
        self.assertAlmostEqual(expected, result['ndcg_at_10'])
        self.assertEqual(1, result['recall_at_10'])
        self.assertEqual(1 / 3, result['mrr_at_10'])

    def test_no_positive_judgments_do_not_create_false_relevance(self):
        result = tool.ranking_metrics({'q': [str(i) for i in range(10)]}, {'q': {'0': 0, '1': -1}})
        self.assertEqual({'queries': 1, 'ndcg_at_10': 0, 'recall_at_10': 0, 'mrr_at_10': 0}, result)

    def test_compressed_corpus_can_be_replayed_and_legacy_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / 'rows.zip'
            rows = [{'_id': 'b', 'text': 'second'}, {'_id': 'a', 'text': 'first'}]
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('fixture/corpus.jsonl', '\n'.join(map(json.dumps, rows)))
            with patch.dict(tool.DATASETS, {'fixture': {'sort_docs': False}}):
                corpus = tool.CorpusRows(archive, 'fixture')
                self.assertEqual(rows, list(corpus))
                self.assertEqual(rows, list(corpus))
            with patch.dict(tool.DATASETS, {'fixture': {'sort_docs': True}}):
                self.assertEqual(['a', 'b'], [r['_id'] for r in tool.CorpusRows(archive, 'fixture')])

    def test_external_word_ids_above_uint16_and_default_synthetic_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'documents.jsonl'
            source.write_text((json.dumps([65536, 1]) + '\n') * 16, encoding='utf-8')
            cases = {'A': [[65536]], 'B': [[1, 65536]]}
            record = large_corpus.prepare(root / 'external', 16, 65537, 0, cases, corpus=source, variable_queries=True)
            large_corpus.validate(root / 'external')
            self.assertEqual([[65536]], large_corpus.load_raw(root / 'external/A.txt'))
            self.assertEqual(65537, record['generation']['vocab'])
            with self.assertRaises(ValueError):
                large_corpus.prepare(root / 'synthetic', 16, 65537, 0, cases)
            with self.assertRaises(ValueError):
                large_corpus.prepare(root / 'too_large', 16, 1000001, 0, cases, corpus=source, variable_queries=True)

    def test_dataset_identity_cannot_be_silently_switched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'preparation.json').write_text(
                json.dumps({'source': {'archive_sha256': tool.ARCHIVE_SHA256}}), encoding='utf-8'
            )
            with self.assertRaises(ValueError):
                tool.validate_data(root, 'trec-covid')


if __name__ == '__main__':
    unittest.main()
