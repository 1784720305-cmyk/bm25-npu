"""Exact dense-threshold models and build isolation checks; not device evidence."""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from standalone.dense_selector import ROOT, STATUS, build, check_setup, compare
from standalone.selector_capacity import TRACE_CONTRACT
from standalone.test_final import collect_streams
from tools.capacity_sources import sources
from tools.dense_sources import dense_sources


def threshold(bits, k):
    if not all((raw & 0x7FFFFFFF) < 0x7F800000 and (raw < 0x80000000 or raw == 0x80000000) for raw in bits):
        raise ValueError('Invalid raw score')
    keys = [(~(raw & 0x7FFFFFFF)) & 0xFFFFFFFF for raw in bits]
    prefix = mask = 0
    rank = k
    for shift in (24, 16, 8, 0):
        counts = [0] * 256
        for key in keys:
            if key & mask == prefix:
                counts[(key >> shift) & 255] += 1
        total = 0
        for bucket, count in enumerate(counts):
            if total < rank <= total + count:
                rank -= total
                prefix |= bucket << shift
                break
            total += count
        else:
            raise AssertionError('Rank missing')
        mask |= 255 << shift
    return prefix, rank


def body(text, begin, end):
    return text[text.index(begin) : text.index(end, text.index(begin))]


class DenseTests(unittest.TestCase):
    def test_unchanged_capacity_memory_scoring_and_legacy_fallback(self):
        for profile in (False, True):
            old_host, old_kernel, old_plan = sources(1024, profile)
            host, kernel, plan = dense_sources(profile)
            self.assertEqual(host, old_host)
            self.assertEqual(plan, old_plan)
            self.assertEqual(plan['buffer_bytes'], 50848)
            self.assertEqual(plan['dynamic_ub_bytes'], 65536)
            for first, last in (
                ('void SelectLocal(', '// Three disjoint regions:'),
                ('inline void FusedScorePackSimt(', 'inline void FusedCollectSimt('),
            ):
                self.assertEqual(body(kernel, first, last), body(old_kernel, first, last))
            self.assertIn('const bool dense = mode == 0 && count > kCapacity;', kernel)
            self.assertIn('if (mode != 1 && !dense)', kernel)
            self.assertIn('const bool compact = mode == 1;', kernel)
            self.assertIn('RegisterDenseThreshold(memory, memory + base, width, k)', kernel)
            self.assertIn('V::And(canonical, bits, magnitudeMask, valid)', kernel)

    def test_exact_dense_cdf_and_original_id_ties(self):
        rng = random.Random(95020261106)
        for width, count in ((1025, 1025), (2049, 2049), (4097, 4097), (8192, 1025), (8192, 4399), (8192, 8192)):
            for pattern in range(4):
                bits = [0x80000000 if d % 2 else 0 for d in range(width)]
                positions = rng.sample(range(width), count)
                for d in positions:
                    values = (
                        0x3F800000,
                        (1, 2, 0x007FFFFF, 0x00800000, 0x7F7FFFFF)[d % 5],
                        0x3F800000 + d % 3,
                        rng.randrange(1, 0x7F800000),
                    )
                    bits[d] = values[pattern]
                normalized = {d: raw & 0x7FFFFFFF for d, raw in enumerate(bits)}
                keys = [(~normalized[d]) & 0xFFFFFFFF for d in range(width)]
                streams = [list(range(t, width, 256)) for t in range(256)]
                for k in (1, 10, 32):
                    prefix, rank = threshold(bits, k)
                    self.assertEqual(prefix, sorted(keys)[k - 1])
                    self.assertEqual(rank, k - sum(key < prefix for key in keys))
                    collect_streams(streams, normalized, k)

    def test_invalid_fp32_bits_are_rejected_before_dense_threshold(self):
        for raw in (0xBF800000, 0x7F800000, 0x7FC00000, 0xFF800000, 0xFFFFFFFF):
            with self.assertRaises(ValueError):
                threshold([0x3F800000] * 1025 + [raw], 10)

    def test_study_keeps_search_regression_when_stage_improves(self):
        rows = []
        for _ in range(6):
            row = {'real': {}}
            for role, wall, ticks in (('baseline', 1.0, 100), ('dense', 1.05, 80)):
                row['real'][role] = {
                    'search': {c: {'median_batch_ms': wall, 'p95_batch_ms': wall * 2} for c in ('A', 'B')},
                    'stages': {c: {'topk_ticks': {'median': ticks}} for c in ('A', 'B')},
                }
            rows.append(row)
        result = compare(rows, ['real'])['real']['A']
        self.assertAlmostEqual(result['median_change_percent'], 5.0)
        self.assertAlmostEqual(result['topk_ticks_change_percent'], -20.0)
        self.assertEqual(result['median_wins'], 0)
        with self.assertRaises(ValueError):
            compare(rows[:5], ['real'])

    def test_mocked_build_installs_only_two_new_packages(self):
        # Create a minimal actual file/preset layout; all compilation calls are mocked.
        from standalone import resident_index as r
        from standalone.selector_capacity import package_fingerprint

        with tempfile.TemporaryDirectory(dir=ROOT / 'results', prefix='dense_build_check_') as tmp:
            base = Path(tmp)
            baseline_dir = base / 'old_build'
            baseline_dir.mkdir()
            host = baseline_dir / 'host_build'
            host.mkdir()
            for name in ('capacity_profile', 'capacity_boundaries'):
                (host / name).write_bytes(b'validated host')

            def vendor_at(path, label):
                (path / 'op_api/lib').mkdir(parents=True)
                (path / 'op_api/lib/libcust_opapi.so').write_bytes(label.encode())
                (path / 'binary_info_config.json').write_bytes(b'{}')
                return package_fingerprint(path)

            merge = base / 'accepted_merge'
            merge_files = vendor_at(merge, 'merge')
            old = {
                'status': 'experimental_packages_and_host_built_device_pending',
                'profile_trace_contract': TRACE_CONTRACT,
                'host_build': str(host),
                'host_binary_sha256': {p.name: r.digest(p.read_bytes()) for p in host.iterdir()},
                'baseline_merge': str(merge),
                'merge_installed_files': merge_files,
                'variants': {},
            }
            for profile in (False, True):
                h, k, plan = sources(1024, profile)
                key = '1024_profile' if profile else '1024'
                vendor = base / key
                old['variants'][key] = {
                    **plan,
                    'vendor': str(vendor),
                    'installed_files': vendor_at(vendor, key),
                    'aliases': str(base / ('old_alias_' + key)),
                    'source_sha256': {
                        'op_host/bm25_fused_score_top_k.cpp': r.digest(h.encode()),
                        'op_kernel/bm25_fused_score_top_k.cpp': r.digest(k.encode()),
                    },
                }
            r.write_json(baseline_dir / 'build.json', old)
            original = {p: p.read_bytes() for p in base.rglob('*') if p.is_file()}
            template = base / 'template'
            template.mkdir()
            (template / 'build.sh').write_bytes(b'#!/bin/bash\n')
            r.write_json(
                template / 'CMakePresets.json',
                {'configurePresets': [{'cacheVariables': {'ASCEND_COMPUTE_UNIT': 'ascend950'}}]},
            )
            installs = []

            def execute(cmd, log, env=None, cwd=None, timeout=900):
                log.write_bytes(b'mocked, not CANN evidence')
                if cmd[:2] == ['bash', 'build.sh']:
                    target = cwd / 'build_out'
                    target.mkdir()
                    (target / 'operator.run').write_text(
                        json.loads((cwd / 'final_source.json').read_text())['vendor_name']
                    )
                else:
                    prefix = Path(cmd[2].split('=', 1)[1])
                    vendor = prefix / 'vendors' / Path(cmd[1]).read_text()
                    vendor_at(vendor, 'candidate')
                    installs.append(vendor)

            args = SimpleNamespace(baseline_build=baseline_dir, output=base / 'candidate', template=template)
            with (
                patch('standalone.dense_selector.s.execute', side_effect=execute),
                patch('standalone.dense_selector.s.aliases'),
            ):
                build(args)
            self.assertEqual(len(installs), 2)
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)
            setup = json.loads((args.output / 'build.json').read_text())
            self.assertEqual(setup['status'], STATUS)
            check_setup(setup)
            broken = copy.deepcopy(setup)
            del broken['variants']['dense']
            with self.assertRaises(ValueError):
                check_setup(broken)


if __name__ == '__main__':
    unittest.main()
