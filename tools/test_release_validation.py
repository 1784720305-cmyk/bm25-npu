"""Check bounded code specialization and the fixed acceptance workflow, not device speed."""

import contextlib
import copy
import io
import itertools
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from standalone import release_validation as v
from standalone import resident_index as r
from tools.capacity_sources import sources
from tools.dense_sources import dense_sources
from tools.split_sources import collector, split_sources


def section(text, start, end):
    return text[text.index(start) : text.index(end, text.index(start))]


def summary(wall):
    return {case: {'samples': 30, 'median_batch_ms': wall, 'p95_batch_ms': wall * 1.2} for case in ('A', 'B')}


class ReleaseTests(unittest.TestCase):
    def test_compact_body_and_dense_threshold_are_preserved(self):
        for instrumented in (False, True):
            host, kernel, plan = split_sources(instrumented)
            old_host, accepted, old_plan = sources(1024, instrumented)
            _, dense, _ = dense_sources(instrumented)
            self.assertEqual((host, plan), (old_host, old_plan))
            self.assertEqual(collector(kernel), collector(accepted))
            for begin, end in (
                ('inline void RegisterDenseThreshold(', '__aicore__ inline void VectorToScalar('),
                (
                    'inline void FusedScorePackSimt(',
                    '__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedCollectSimt(',
                ),
                ('void SelectLocal(', '// Three disjoint regions:'),
            ):
                self.assertEqual(section(kernel, begin, end), section(dense, begin, end))
            self.assertEqual(kernel.count('VF_CALL<FusedDenseCollectSimt>'), 1)
            self.assertEqual(kernel.count('VF_CALL<FusedCollectSimt>'), 1)
            self.assertIn('if (denseRoute)', kernel)
            self.assertIn('const bool compact = false;', kernel)
            self.assertNotIn('const bool compact = mode == 1;', kernel)
            self.assertIn('const bool dense = mode == 0 && count > kCapacity;', kernel)
            self.assertIn('if (mode != 1 && !dense)\n        return false;', kernel)
            self.assertIn('ScalarToVector();\n    return dense;', kernel)
            self.assertNotIn('ScalarToVector();\n    VectorToScalar();', kernel)

    def test_six_rounds_balance_each_npu_pair_and_position(self):
        orders = [v.process_order(i) for i in range(1, 7)]
        for order in orders:
            self.assertEqual(set(order), set(v.NPU_ROLES + v.CPU_ROLES))
        for role in v.NPU_ROLES:
            positions = [tuple(x for x in order if x in v.NPU_ROLES).index(role) for order in orders]
            self.assertEqual([positions.count(i) for i in range(3)], [2, 2, 2])
        for a, b in itertools.combinations(v.NPU_ROLES, 2):
            self.assertEqual(sum(order.index(a) < order.index(b) for order in orders), 3)
        self.assertEqual(sum(order[0] in v.CPU_ROLES for order in orders), 3)
        with self.assertRaises(ValueError):
            v.process_order(7)

    def test_overhead_is_retained_without_invented_stage_or_cpu_baseline(self):
        rows = [
            {
                role: {'summary': summary(wall)}
                for role, wall in (
                    ('baseline', 1.0),
                    ('dense_v1', 0.8),
                    ('split', 1.045),
                    ('original_native_t1', 2.0),
                    ('original_native_t8', 1.5),
                    ('original_native_t32', 1.8),
                )
            }
            for _ in range(6)
        ]
        workloads = {
            'x': {'per_round': rows, 'bm25s_construction': {'strict_ids_match_project': {'A': False, 'B': False}}}
        }
        result = v.summarize(workloads)['x']['A']
        self.assertAlmostEqual(result['split_vs_baseline_percent'], 4.5)
        self.assertEqual(result['paired_changes_percent'], [result['split_vs_baseline_percent']] * 6)
        self.assertIsNone(result['split_TopK_ticks_change_percent'])
        self.assertFalse(result['original_native_CPU']['strict_equivalent_win'])
        self.assertNotIn('strict_adapter', str(workloads))
        with self.assertRaises(ValueError):
            v.compare_search(rows[:5], 'baseline', 'split')

    def test_build_fingerprints_reject_wrong_candidate_and_merge(self):
        def setup(generator):
            modes = {}
            for mode in ('normal', 'profile'):
                host, kernel, plan = generator(mode == 'profile')
                modes[mode] = {
                    **plan,
                    'source_sha256': {
                        'op_host/bm25_fused_score_top_k.cpp': r.digest(host.encode()),
                        'op_kernel/bm25_fused_score_top_k.cpp': r.digest(kernel.encode()),
                    },
                }
            return {
                'variants': {'baseline': {'unchanged': True}, 'dense': modes},
                'host_build': 'same',
                'host_binary_sha256': {},
                'merge': 'same',
                'merge_files': {},
                'profile_trace_contract': v.TRACE_CONTRACT,
            }

        new, old = setup(split_sources), setup(dense_sources)
        with patch.object(v.dense_selector, 'check_setup'):
            self.assertEqual(set(v.check_builds(new, old)), set(v.NPU_ROLES))
            for key in ('merge', 'profile_trace_contract'):
                changed = copy.deepcopy(new)
                changed[key] = 'wrong'
                with self.assertRaises(ValueError):
                    v.check_builds(changed, old)
            with self.assertRaises(ValueError):
                v.check_builds(old, new)

    def test_quality_covers_all_real_queries_and_ignores_only_padding(self):
        with tempfile.TemporaryDirectory(dir=v.ROOT / 'results', prefix='release_quality_') as tmp:
            base = Path(tmp)
            (base / 'quality').mkdir()
            r.write_json(base / 'mapping.json', {'doc_ids': ['d' + str(i) for i in range(10)]})
            r.write_json(base / 'quality/qids.json', {'A': ['q1', 'q2'], 'B': ['q3', None]})
            r.write_json(base / 'qrels.json', {q: {'d0': 1} for q in ('q1', 'q2', 'q3')})
            for case in ('A', 'B'):
                (base / ('after_' + case)).mkdir()
                (base / ('after_' + case) / 'top_ids.bin').write_bytes(array('i', list(range(10)) * 2).tobytes())
            result = v.quality_metrics(base, base)
            self.assertEqual(result['queries'], 3)
            self.assertEqual(result['ndcg_at_10'], 1.0)
            r.write_json(base / 'quality/qids.json', {'A': ['q1', 'q2'], 'B': ['q1', None]})
            with self.assertRaises(ValueError):
                v.quality_metrics(base, base)

    def test_matrix_completes_once_or_retains_failure_without_retry(self):
        for fail in (False, True):
            with (
                self.subTest(fail=fail),
                tempfile.TemporaryDirectory(dir=v.ROOT / 'results', prefix='release_matrix_') as tmp,
            ):
                base = Path(tmp)
                host = base / 'host'
                host.mkdir()
                (host / 'bm25_large_search_benchmark').write_bytes(b'mocked runner')
                variants = {
                    role: {mode: {'aliases': str(base / role / mode)} for mode in ('normal', 'profile')}
                    for role in v.NPU_ROLES
                }
                for path in ('new', 'old'):
                    (base / path).mkdir()
                    r.write_json(base / path / 'build.json', {'host_build': str(host)})
                inputs = {}
                for i, name in enumerate(
                    ('scifact', 'trec_covid', 'msmarco', 'sparse8192', 'sparse65536', 'synthetic10m')
                ):
                    path = base / name
                    path.mkdir()
                    real = i < 3
                    if real:
                        r.write_json(path / 'preparation.json', {'dataset': ('scifact', 'trec-covid', 'msmarco')[i]})
                    desc = {'docs': (5183, 171332, 8841823, 8192, 65536, 10000000)[i], 'queries': 8, 'k': 10}
                    inputs[name] = {
                        'input': path,
                        'description': desc,
                        'index': {'vocab_size': 1, 'posting_count': 1, 'index_id': name},
                        'preparation_sha256': 'mock',
                        **({'real_data': path} if real else {}),
                    }
                args = SimpleNamespace(
                    build=base / 'new',
                    reference_build=base / 'old',
                    output=base / 'output',
                    host_build=host,
                    real_root=base,
                    legacy_root=base,
                    device=5,
                    dependency_dir=base,
                )
                counters = {'quality': 0, 'search': 0, 'cpu': 0, 'profile': 0, 'build': 0}
                bits = {'A_ids': 'same', 'A_scores': 'same', 'B_ids': 'same', 'B_scores': 'same'}

                def execute(command, log, env=None, timeout=None, _counters=counters):
                    if '--memory' in command:
                        r.write_json(Path(command[-1]), {'free': 2**50})
                    elif Path(command[0]).name == 'capacity_boundaries':
                        Path(command[-1]).mkdir()
                        r.write_json(
                            Path(command[-1]) / 'report.json',
                            {'status': 'actual_simd_selector_guards_and_strict_bit_rankings_passed', 'calls': 54},
                        )
                    else:
                        self.assertIn('build', command)
                        _counters['build'] += 1
                        built = Path(command[command.index('--built') + 1])
                        built.mkdir()
                        name = Path(command[command.index('--input') + 1]).name
                        r.write_json(
                            built / 'report.json',
                            {
                                'status': 'original_bm25s_large_index_and_formula_checked',
                                'index_id': name,
                                'strict_ids_match_project': {'A': True, 'B': True},
                            },
                        )

                def search(
                    runner,
                    record,
                    vendor,
                    output,
                    device,
                    warmup=5,
                    repeats=30,
                    _counters=counters,
                    _fail=fail,
                    _bits=bits,
                ):
                    key = 'quality' if repeats == 1 else 'search'
                    _counters[key] += 1
                    if _fail and key == 'search':
                        raise ValueError('injected failure')
                    output.mkdir(parents=True)
                    return {'summary': summary(1.0), 'TopK_sha256': _bits, 'report_sha256': 'mock'}

                def cpu(*args, _counters=counters):
                    _counters['cpu'] += 1
                    return {'summary': summary(2.0)}

                def profile(*args, _counters=counters):
                    _counters['profile'] += 1
                    return {case: {'topk_ticks': {'median': 1.0}} for case in ('A', 'B')}, 'mock'

                with (
                    patch.object(v, 'check_builds', return_value=variants),
                    patch.object(v, 'inputs_at', return_value=inputs),
                    patch.object(v, 'verify_unchanged_input'),
                    patch.object(v, 'quality_metrics', return_value={'queries': 1}),
                    patch.object(v, 'search', side_effect=search),
                    patch.object(v, 'cpu', side_effect=cpu),
                    patch.object(v, 'profile', side_effect=profile),
                    patch.object(
                        v.corpus,
                        'validate',
                        return_value=(
                            {'docs': 100, 'queries': 8, 'k': 10, 'capacity_words': 24},
                            {'vocab_size': 1, 'posting_count': 1},
                        ),
                    ),
                    patch.object(v.beir_real, 'validate_data'),
                    patch.object(v.s, 'execute', side_effect=execute),
                    patch.object(v.s, 'environment', return_value={}),
                    patch.object(v.s, 'host_available_bytes', return_value=2**50),
                    patch.object(
                        v.s,
                        'layout',
                        return_value={
                            'cpu_peak_known_bytes': 1,
                            'known_device_bytes_upper_bound': 1,
                            'debug_score_bytes': 1,
                        },
                    ),
                    patch.object(v.shutil, 'disk_usage', return_value=SimpleNamespace(free=2**50)),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'injected failure'):
                            v.run(args)
                    else:
                        v.run(args)
                report = v.read(args.output / 'report.json')
                self.assertFalse(report['candidate_promoted'])
                self.assertEqual(report['optimization_round_limit'], 1)
                if fail:
                    self.assertEqual(report['status'], 'failed')
                    self.assertEqual(counters['search'], 1)
                    self.assertIn('current_cell', report)
                else:
                    self.assertEqual(
                        report['status'], 'concentrated_preindex_checks_completed_selection_review_pending'
                    )
                    self.assertEqual(counters, {'quality': 9, 'search': 108, 'cpu': 108, 'profile': 54, 'build': 6})
                    self.assertEqual(len(report['summary']), 6)


if __name__ == '__main__':
    unittest.main()
