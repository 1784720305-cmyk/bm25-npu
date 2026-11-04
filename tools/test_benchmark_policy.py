"""Check baseline selection and historical scope without substituting device results."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.benchmark_final import historical_npu_reference, measurement_roles
from standalone.bm25s_benchmark import decide
from standalone.large_corpus import load_raw


class BenchmarkPolicyTests(unittest.TestCase):
    def test_original_baseline_uses_native_workers_at_every_scale(self):
        expected = ['npu', 'original_native_t1', 'original_native_t8', 'original_native_t32']
        for docs in (1024, 10000000):
            self.assertEqual(expected, measurement_roles(docs, 'bm25s', True))
        self.assertEqual(['cpu', 'npu', 'cpu_directory'], measurement_roles(10000000, 'cpp', False))
        self.assertIn('cpu_single', measurement_roles(1024, 'cpp', False))
        self.assertIn('strict_adapter_t32', measurement_roles(10000000, 'cpp', True))

    def test_native_decision_does_not_need_or_claim_strict_adapter(self):
        def item(value):
            return {'summary': {c: {'median_batch_ms': value, 'p95_batch_ms': value * 1.1} for c in ('A', 'B')}}

        rows = [
            {
                'npu': item(1),
                'original_native_t1': item(4),
                'original_native_t8': item(2),
                'original_native_t32': item(3),
            }
            for _ in range(6)
        ]
        result = decide(rows, [1, 8, 32], {'A': True, 'B': True}, variants=('original_native',))
        self.assertEqual({'original_native'}, set(result))
        self.assertTrue(result['original_native']['A']['timing_gate_passed'])
        self.assertFalse(result['original_native']['A']['strict_equivalent_win'])
        rows[2]['npu'] = item(2.1)
        self.assertFalse(
            decide(rows, [1, 8, 32], {'A': True}, variants=('original_native',))['original_native']['A'][
                'timing_gate_passed'
            ]
        )

    def description(self):
        return {
            'dataset_kind': 'uniform_synthetic_not_MT_approved',
            'docs': 10000000,
            'queries': 8,
            'query_length': 3,
            'k': 10,
            'generation': {'seed': 20261014, 'vocab': 1024, 'source_jsonl': None, 'synthetic_length_range': [16, 48]},
            'cases': {case: {'terms': load_raw(ROOT / 'fixtures' / (case + '.txt'))} for case in ('A', 'B')},
        }

    def test_history_reports_difference_without_automatic_acceptance(self):
        old = json.loads(
            (ROOT / 'evidence/bm25s_current_npu_device_evidence_20261029.json').read_text(encoding='utf-8')
        )
        selected = {x['case']: x for x in old['summary'] if x['docs'] == 10000000 and x['variant'] == 'original_native'}
        rows = [
            {
                'npu': {
                    'summary': {
                        c: {
                            'median_batch_ms': selected[c]['npu_ms'] * 1.05,
                            'p95_batch_ms': selected[c]['npu_p95_ms'] * 1.04,
                        }
                        for c in ('A', 'B')
                    }
                }
            }
            for _ in range(6)
        ]
        result = historical_npu_reference(rows, self.description())
        self.assertEqual('historical_reference_for_review', result['status'])
        self.assertAlmostEqual(5, result['cases']['A']['median_change_percent'])
        self.assertAlmostEqual(4, result['cases']['B']['p95_change_percent'])
        self.assertNotIn('accepted', result)

    def test_real_or_changed_corpus_is_not_compared_with_synthetic_history(self):
        for change in ('source', 'seed', 'queries'):
            description = self.description()
            if change == 'source':
                description['generation']['source_jsonl'] = 'real.jsonl'
            elif change == 'seed':
                description['generation']['seed'] = 1
            else:
                description['cases']['A']['terms'][0][0] += 1
            self.assertEqual(
                'not_comparable_to_frozen_synthetic_workload', historical_npu_reference([], description)['status']
            )


if __name__ == '__main__':
    unittest.main()
