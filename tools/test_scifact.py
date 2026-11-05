"""Validate natural-query preparation and full-query quality arithmetic without device substitutes."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import large_corpus
from standalone.scifact import english_stopwords, ranking_metrics, read_source, tokenize


class SciFactTests(unittest.TestCase):
    def test_tokenization_retains_order_and_duplicates(self):
        self.assertEqual(['alpha', 'beta', 'alpha'], tokenize('The ALPHA, beta; alpha.', english_stopwords()))

    def test_all_queries_and_rank_discounts_are_required(self):
        predictions = {'one': ['a'] + [str(i) for i in range(9)], 'two': ['0', 'b'] + [str(i) for i in range(1, 9)]}
        result = ranking_metrics(predictions, {'one': {'a': 1}, 'two': {'b': 1}})
        self.assertEqual(2, result['queries'])
        self.assertAlmostEqual((1 + 1 / math.log2(3)) / 2, result['ndcg_at_10'])
        self.assertEqual(1, result['recall_at_10'])
        self.assertEqual(0.75, result['mrr_at_10'])
        with self.assertRaises(ValueError):
            ranking_metrics(predictions, {'one': {'a': 1}})
        with self.assertRaises(ValueError):
            ranking_metrics({'one': ['a'] * 10}, {'one': {'a': 1}})

    def test_variable_query_capacity_preserves_whole_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / 'docs.jsonl'
            documents.write_text(''.join(json.dumps([i % 8, (i + 1) % 8]) + '\n' for i in range(16)), encoding='utf-8')
            cases = {'A': [[0], [1, 2, 1, 3]], 'B': [[2, 3], []]}
            result = large_corpus.prepare(root / 'variable', 16, 8, 0, cases, corpus=documents, variable_queries=True)
            self.assertIsNone(result['query_length'])
            self.assertEqual(5, result['capacity_words'])
            large_corpus.validate(root / 'variable')
            self.assertEqual(cases['A'], large_corpus.load_raw(root / 'variable/A.txt'))
            with self.assertRaises(ValueError):
                large_corpus.prepare(root / 'default', 16, 8, 0, cases, corpus=documents)
            too_long = {'A': [[0] * 65], 'B': [[1]]}
            with self.assertRaises(ValueError):
                large_corpus.prepare(root / 'long', 16, 8, 0, too_long, corpus=documents, variable_queries=True)

    def test_archive_integrity_rejects_different_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / 'wrong.zip'
            archive.write_bytes(b'not the dataset')
            with self.assertRaises(ValueError):
                read_source(archive)


if __name__ == '__main__':
    unittest.main()
