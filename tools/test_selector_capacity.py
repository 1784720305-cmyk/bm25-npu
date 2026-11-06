"""Capacity/layout, overflow, tie and experiment-report checks; no device simulation."""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from standalone.selector_capacity import ROOT, TRACE_CONTRACT, build, comparisons, profile_summary
from standalone.test_final import append_ranges, collect_streams
from tools.capacity_sources import CAPACITIES, FUSED, layout, once, sources


class CapacityTests(unittest.TestCase):
    def test_baseline_kernel_is_byte_identical(self):
        self.assertEqual(sources(1024)[1].encode(), (FUSED / 'op_kernel/bm25_fused_score_top_k.cpp').read_bytes())

    def test_ub_layout_and_declaration_cover_every_capacity(self):
        for capacity, scratch, total, dynamic in (
            (1024, 16736, 50848, 65536),
            (2048, 24928, 59040, 65536),
            (4096, 41312, 75424, 98304),
        ):
            host, kernel, plan = sources(capacity)
            self.assertEqual(
                plan,
                {'capacity': capacity, 'scratch_bytes': scratch, 'buffer_bytes': total, 'dynamic_ub_bytes': dynamic},
            )
            self.assertGreaterEqual(dynamic, total)
            self.assertIn(f'kSelectScratchBytes == {scratch}', kernel)
            self.assertIn(f'kBufferBytes == {total}', kernel)
            self.assertIn(f'kDynamicUbBytes = {dynamic}', host)
            self.assertNotIn('GetSystemCycle', kernel)
            self.assertNotIn('*debug != 3', host)

    def test_invalid_capacity_and_source_drift_fail_closed(self):
        for capacity in (0, 1025, 8192):
            with self.assertRaises(ValueError):
                layout(capacity)
        for text in ('missing', 'x x'):
            with self.assertRaises(ValueError):
                once(text, 'x', 'new')

    def test_profile_only_records_after_barriers_and_covers_pack(self):
        for capacity in CAPACITIES:
            host, kernel, _ = sources(capacity, True)
            self.assertIn('*debug != 3', host)
            self.assertIn('docs / (*shards) < 16', host)
            self.assertIn('if (debug != 3)\n                PackForSimd<true>', kernel)
            self.assertIn('CapacityProfilePack', kernel)
            self.assertLess(kernel.index('profileScored ='), kernel.index('VF_CALL<CapacityProfilePack>'))
            self.assertIn('VF_CALL<CapacityProfileTrace>', kernel)
            self.assertIn('static_cast<uint64_t>(task) * 8', kernel)
            self.assertNotIn('AscendC::GlobalTensor<int64_t> trace', kernel)
            self.assertIn('if (debug == 1)', kernel)

    def test_append_capacity_boundaries_never_truncate_selection_input(self):
        rng = random.Random(9504096)
        for capacity in CAPACITIES:
            for count in (capacity - 1, capacity, capacity + 1):
                positives = set(rng.sample(range(8192), count))
                compact, total, _ = append_ranges([(0, 8192)], positives, list(reversed(range(8))), capacity)
                self.assertEqual(total, count)
                self.assertEqual(len(compact), capacity)
                if count <= capacity:
                    self.assertEqual(set(compact[:count]), positives)
                    bits = [0x3F800000 if d % 3 else 0x40000000 for d in range(8192)]
                    streams = [compact[t:count:256] for t in range(256)]
                    for k in (1, 10, 32):
                        collect_streams(streams, bits, k)
                else:
                    self.assertGreater(total, capacity)  # Overflow must use dense fallback, not the bounded list.

    @staticmethod
    def profile():
        samples = []
        for cycle in range(30):
            for slot in range(2):
                sequence = 11 + cycle * 2 + slot
                values = [v for task in range(8) for v in (100, 120, 130, 2048, sequence, sequence, task, 2)]
                samples.append(
                    {
                        'case': 'B' if slot ^ (cycle & 1) else 'A',
                        'cycle': cycle,
                        'sequence': sequence,
                        'task_timestamps': values,
                    }
                )
        return {
            'status': 'device_stage_profile_traces_and_bit_checks_passed',
            'completed': True,
            'normal_profile_outputs_bit_identical': True,
            'guards_passed': True,
            'queries': 8,
            'tasks': 8,
            'trace_contract': TRACE_CONTRACT,
            'trace_words': 8,
            'samples': samples,
        }

    def test_profile_units_and_aggregation(self):
        result = profile_summary(self.profile(), 2048)
        self.assertEqual(result['A']['topk_ticks']['median'], 10)
        self.assertEqual(result['B']['score_and_append_ticks']['median'], 20)
        self.assertEqual(result['A']['topk_ticks']['samples'], 240)
        self.assertNotIn('median_batch_ms', result['A']['topk_ticks'])

    def test_incomplete_wrong_capacity_and_nonmonotonic_traces_rejected(self):
        for kind in ('guard', 'missing', 'wrong_capacity', 'nonmonotonic', 'cycle', 'stale', 'overlap', 'old_schema'):
            data = copy.deepcopy(self.profile())
            if kind == 'guard':
                data['guards_passed'] = False
            elif kind == 'missing':
                data['samples'][0]['task_timestamps'].pop()
            elif kind == 'wrong_capacity':
                data['samples'][0]['task_timestamps'][3] = 1024
            elif kind == 'cycle':
                data['samples'][0]['cycle'] = 9
            elif kind == 'stale':
                data['samples'][0]['task_timestamps'][5] -= 1
            elif kind == 'overlap':
                data['samples'][0]['task_timestamps'][6] = 1
            elif kind == 'old_schema':
                data['trace_words'] = 4
            else:
                data['samples'][0]['task_timestamps'][1] = 100
            with self.assertRaises(ValueError):
                profile_summary(data, 2048)

    def test_search_regression_is_retained_even_if_topk_improves(self):
        rows = []
        for _ in range(6):
            row = {'real': {}}
            for capacity, wall, ticks in ((1024, 1.0, 100), (2048, 1.05, 90), (4096, 0.9, 70)):
                row['real'][str(capacity)] = {
                    'search': {c: {'median_batch_ms': wall, 'p95_batch_ms': wall * 2} for c in ('A', 'B')},
                    'stages': {c: {'topk_ticks': {'median': ticks}} for c in ('A', 'B')},
                }
            rows.append(row)
        result = comparisons(rows, ['real'])
        self.assertAlmostEqual(result['real']['2048']['A']['search_change_percent'], 5)
        self.assertAlmostEqual(result['real']['2048']['A']['topk_ticks_change_percent'], -10)
        self.assertEqual(result['real']['2048']['A']['search_wins'], 0)

    def test_task_records_do_not_share_a_cache_line(self):
        old_lines = [(task * 32) // 64 for task in range(16)]
        self.assertLess(len(set(old_lines)), len(old_lines))
        starts = [task * 8 * 8 for task in range(16)]
        self.assertTrue(all(start % 64 == 0 for start in starts))
        self.assertEqual(len({start // 64 for start in starts}), 16)

    def test_mocked_package_build_keeps_baseline_and_template_untouched(self):
        # Exercise file/preset/install isolation, without claiming a CANN compilation.
        with tempfile.TemporaryDirectory(dir=ROOT / 'results', prefix='capacity_build_check_') as temporary:
            base = Path(temporary)
            template = base / 'template'
            template.mkdir()
            (template / 'build.sh').write_bytes(b'#!/bin/bash\n')
            preset = {'configurePresets': [{'cacheVariables': {'ASCEND_COMPUTE_UNIT': 'ascend950'}}]}
            (template / 'CMakePresets.json').write_text(json.dumps(preset), encoding='utf-8')
            vendors = base / 'accepted'
            for kind in ('bm25_fused_score_topk', 'bm25_topk_merge'):
                vendor = vendors / kind
                (vendor / 'op_api/lib').mkdir(parents=True)
                (vendor / 'op_api/lib/libcust_opapi.so').write_bytes(b'accepted')
                (vendor / 'binary_info_config.json').write_bytes(b'{}')
            original = {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob('*') if p.is_file()}
            installations = []

            def execute(command, log, env=None, cwd=None, timeout=900):
                log.write_bytes(b'mocked build, not CANN evidence\n')
                if command[:2] == ['bash', 'build.sh']:
                    packages = cwd / 'build_out'
                    packages.mkdir()
                    vendor = json.loads((cwd / 'final_source.json').read_text())['vendor_name']
                    (packages / 'operator.run').write_text(vendor, encoding='utf-8')
                elif command[0] == 'bash':
                    prefix = Path(str(command[2]).split('=', 1)[1])
                    name = Path(command[1]).read_text()
                    vendor = prefix / 'vendors' / name
                    (vendor / 'op_api/lib').mkdir(parents=True)
                    (vendor / 'op_api/lib/libcust_opapi.so').write_bytes(name.encode())
                    (vendor / 'binary_info_config.json').write_bytes(b'{}')
                    installations.append(vendor)
                elif command[:2] == ['cmake', '--build']:
                    target = Path(command[2])
                    target.mkdir()
                    for name in ('capacity_profile', 'capacity_boundaries'):
                        (target / name).write_bytes(b'mocked binary')
                elif str(command[1]).endswith('run_clang_tidy.py'):
                    target = Path(command[3]) / 'bm25_clang_tidy'
                    target.mkdir()
                    report = {
                        'production_compile_db_coverage_complete': True,
                        'checks': [
                            {
                                'file': str(
                                    ROOT / 'feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/capacity_profile.cpp'
                                ),
                                'exit_code': 0,
                            }
                        ],
                    }
                    (target / 'report.json').write_text(json.dumps(report), encoding='utf-8')

            args = SimpleNamespace(
                output=base / 'experiment',
                template=template,
                vendors=vendors,
                toolkit_home=base,
                jobs=1,
                clang_tidy='mocked-tidy',
            )
            with (
                patch('standalone.selector_capacity.s.execute', side_effect=execute),
                patch('standalone.selector_capacity.s.aliases') as aliases,
            ):
                build(args)
                self.assertEqual(aliases.call_count, 6)
            self.assertEqual(len(set(installations)), 6)
            for p, content in original.items():
                self.assertEqual((base / p).read_bytes(), content)
            result = json.loads((args.output / 'build.json').read_text())
            self.assertEqual(result['status'], 'experimental_packages_and_host_built_device_pending')
            with self.assertRaises(ValueError):
                build(args)
            old_build = args.output
            old_report = (old_build / 'build.json').read_bytes()
            args.output = base / 'repair'
            args.reuse_normal_build = old_build
            with (
                patch('standalone.selector_capacity.s.execute', side_effect=execute),
                patch('standalone.selector_capacity.s.aliases') as aliases,
            ):
                build(args)
                self.assertEqual(aliases.call_count, 3)
            self.assertEqual(len(installations), 9)  # Only three replacement diagnostic packages.
            repaired = json.loads((args.output / 'build.json').read_text())
            self.assertEqual(repaired['profile_trace_contract'], TRACE_CONTRACT)
            for capacity in CAPACITIES:
                self.assertEqual(repaired['variants'][str(capacity)], result['variants'][str(capacity)])
            self.assertEqual((old_build / 'build.json').read_bytes(), old_report)


if __name__ == '__main__':
    unittest.main()
