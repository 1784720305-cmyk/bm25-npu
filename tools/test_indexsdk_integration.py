"""Independent SDK source merge and public-output contracts; no CANN/device claim."""

import contextlib
import io
import re
import shutil
import sys
import tarfile
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from standalone import resident_index as r
from tools import indexsdk_integration as sdk
from tools import run_clang_tidy as tidy


def archive(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as output:
        for name, payload, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(payload)
            output.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


class IntegrationTests(unittest.TestCase):
    def test_header_scope_covers_owned_headers_through_sdk_parent_alias_only(self):
        with tempfile.TemporaryDirectory(dir=sdk.ROOT / 'results', prefix='sdk_header_scope_') as tmp:
            source = Path(tmp)
            owned = (
                'common/bm25/index_data.h',
                'ascendhost/src/bm25/runtime.h',
                'ascendhost/include/index/AscendIndexBm25.h',
                'ascendhost/include/impl/Bm25RuntimeBinding.h',
                'test/ascendhost/bm25/reference/cpu_index_search.h',
            )
            upstream = (
                'ascenddaemon/AscendResourcesProxy.h',
                'common/utils/SocUtils.h',
                'ascendhost/include/impl/OriginalSdk.h',
            )
            for name in owned + upstream:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'// fixture\n')
            expression, names = tidy.owned_header_filter(source)
            self.assertEqual(set(names), set(owned))
            for name in owned + upstream:
                for prefix in (source.as_posix() + '/', source.as_posix() + '/ascendhost/../'):
                    for path in (prefix + name, (prefix + name).replace('/', '\\')):
                        self.assertEqual(bool(re.search(expression, path)), name in owned, path)
            self.assertIsNone(re.search(expression, source.as_posix() + '/common/bm25/../utils/SocUtils.h'))

    def test_static_recovery_keeps_binaries_packages_sources_and_original_failure(self):
        with tempfile.TemporaryDirectory(dir=sdk.ROOT / 'results', prefix='sdk_resume_') as tmp:
            output = Path(tmp)
            host = output / 'host'
            host.mkdir()
            tree = output / 'sdk'
            toolkit = output / 'toolkit'
            toolkit.mkdir()
            (host / 'CMakeCache.txt').write_bytes(
                (
                    'CMAKE_HOME_DIRECTORY:INTERNAL='
                    + str(tree / sdk.PREFIX / 'ascendhost')
                    + '\nASCEND_TOOLKIT_PATH:UNINITIALIZED='
                    + str(toolkit)
                    + '\nBUILD_ASCEND_BM25:BOOL=ON\n'
                ).encode()
            )
            for name in ('libascendfaiss.so', 'bm25_sdk_benchmark', 'bm25_sdk_shared_check'):
                (host / name).write_bytes(('retained fixture ' + name).encode())
            vendor = output / 'vendor'
            (vendor / 'op_api/lib').mkdir(parents=True)
            (vendor / 'op_api/lib/libcust_opapi.so').write_bytes(b'retained package fixture')
            (vendor / 'binary_info_config.json').write_bytes(b'{}')
            result = {
                'status': 'failed',
                'current_stage': 'clang_tidy',
                'integration': str(output / 'integration'),
                'stages': ['operator_build', 'operator_install', 'configure', 'host_build', 'unit_tests'],
                'fused_vendor': str(vendor),
                'merge': str(vendor),
                'fused_files': sdk.package_fingerprint(vendor),
                'merge_files': sdk.package_fingerprint(vendor),
                'error': 'prior static failure fixture',
            }
            r.write_json(output / 'build.json', result)
            original = (output / 'build.json').read_bytes()
            (output / 'clang_tidy.log').write_bytes(b'prior failure log\n')
            (host / 'bm25_clang_tidy').mkdir()
            (host / 'bm25_clang_tidy/report.json').write_bytes(b'{"status":"failed"}\n')
            commands = []

            def execute(command, log, env=None, cwd=None, timeout=None):
                commands.append([str(part) for part in command])
                Path(log).write_bytes(b'mocked recovery stage, no CANN build claim\n')
                if command[:2] == ['cmake', '--install']:
                    target = output / 'installed/host/lib'
                    target.mkdir(parents=True)
                    shutil.copyfile(host / 'libascendfaiss.so', target / 'libascendfaiss.so')

            with (
                patch.object(sdk, 'verify_tree', return_value=({}, tree)),
                patch.object(sdk.s, 'execute', side_effect=execute),
                patch.object(
                    sdk.subprocess,
                    'check_output',
                    return_value='\n'.join(
                        'AscendIndexBm25::' + name for name in ('Init(', 'Search(', 'Close(', 'GetLastError(')
                    ),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                (vendor / 'op_api/lib/libcust_opapi.so').write_bytes(b'changed package fixture')
                with self.assertRaisesRegex(ValueError, 'package changed'):
                    sdk.resume_build(SimpleNamespace(build=output, clang_tidy=Path('clang-tidy')))
                self.assertFalse((output / 'resume_01').exists())
                self.assertEqual((output / 'build.json').read_bytes(), original)
                (vendor / 'op_api/lib/libcust_opapi.so').write_bytes(b'retained package fixture')
                sdk.resume_build(SimpleNamespace(build=output, clang_tidy=Path('clang-tidy')))
            record = sdk.read(output / 'build.json')
            self.assertEqual(record['status'], 'integrated_IndexSDK_build_static_UT_install_passed_device_pending')
            self.assertTrue(record['resume_history'][0]['retained_binaries_unchanged'])
            self.assertEqual((output / 'resume_01/build_before.json').read_bytes(), original)
            self.assertEqual((output / 'resume_01/clang_tidy_before.log').read_bytes(), b'prior failure log\n')
            self.assertTrue((output / 'resume_01/clang_tidy_before/report.json').is_file())
            self.assertEqual(len(commands), 5)
            self.assertTrue(commands[0][1].endswith('run_clang_tidy.py'))
            self.assertFalse(any('build.sh' in cmd or str(host) in cmd and '--build' in cmd for cmd in commands))
            self.assertEqual(record['fused_files'], sdk.package_fingerprint(vendor))

    def test_archive_cannot_escape_or_create_links(self):
        with tempfile.TemporaryDirectory(dir=sdk.ROOT / 'results', prefix='sdk_archive_') as tmp:
            base = Path(tmp)
            target = base / 'tree'
            target.mkdir()
            for name, kind in (('../outside', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)):
                with self.assertRaises(ValueError):
                    sdk.extract_archive(archive([(name, b'bad', kind)]), target)
            self.assertFalse((base / 'outside').exists())
            sdk.extract_archive(archive([('ok/source.txt', b'accepted', tarfile.REGTYPE)]), target)
            self.assertEqual((target / 'ok/source.txt').read_bytes(), b'accepted')

    def test_source_merge_changes_only_the_sdk_build_entry(self):
        with tempfile.TemporaryDirectory(dir=sdk.ROOT / 'results', prefix='sdk_merge_') as tmp:
            target = Path(tmp)
            host = target / sdk.PREFIX / 'ascendhost/CMakeLists.txt'
            host.parent.mkdir(parents=True)
            host.write_bytes(b'PROJECT(ascendhost)\nADD_LIBRARY(ascendfaiss_host SHARED original.cpp)\n')
            original = host.read_bytes()
            added, patch_text = sdk.materialize(target)
            self.assertTrue(host.read_bytes().startswith(original))
            self.assertIn('INCLUDE(${CMAKE_CURRENT_LIST_DIR}/cmake/Bm25.cmake)', host.read_text())
            self.assertIn('/dev/null', patch_text)
            self.assertIn('ascendhost/src/bm25/AscendIndexBm25.cpp', patch_text)
            self.assertEqual(len(added), len(set(added)))
            for folder, sha_name in (('op_host', 'host_sha256'), ('op_kernel', 'kernel_sha256')):
                selected = sdk.split_sources(False)[0 if folder == 'op_host' else 1]
                actual = target / sdk.PREFIX / 'ops/ascendc/bm25/fused' / folder / 'bm25_fused_score_top_k.cpp'
                self.assertEqual(actual.read_text(encoding='utf-8'), selected)
            with self.assertRaises(ValueError):
                sdk.materialize(target)

    def test_public_results_check_int64_fp32_order_and_positive_times(self):
        with tempfile.TemporaryDirectory(dir=sdk.ROOT / 'results', prefix='sdk_output_') as tmp:
            target = Path(tmp)
            desc = {'queries': 1, 'k': 2, 'docs': 10}
            samples = [
                {'cycle': i // 2, 'case': 'B' if (i % 2) ^ ((i // 2) & 1) else 'A', 'call_wall_ms': 0.02}
                for i in range(20)
            ]
            report = {
                'status': 'actual_integrated_SDK_public_Search_and_shared_context_checked',
                'SDK_context_device_stream_survived_Close': True,
                'device': 5,
                'queries': 1,
                'k': 2,
                'warmup': 1,
                'repeats': 10,
                'samples': samples,
            }
            r.write_json(target / 'report.json', report)
            for case in ('A', 'B'):
                (target / (case + '_ids.bin')).write_bytes(array('q', [0, 1]).tobytes())
                (target / (case + '_scores.bin')).write_bytes(array('f', [2.0, 1.0]).tobytes())
            self.assertEqual(sdk.verify_sdk_output(target, desc, 5, 1, 10)['summary']['A']['samples'], 10)
            (target / 'A_ids.bin').write_bytes(array('i', [0, 1]).tobytes())
            with self.assertRaisesRegex(ValueError, 'INT64'):
                sdk.verify_sdk_output(target, desc, 5, 1, 10)
            report['samples'][0]['call_wall_ms'] = 0
            r.write_json(target / 'report.json', report)
            with self.assertRaises(ValueError):
                sdk.verify_sdk_output(target, desc, 5, 1, 10)

    def test_sdk_quality_is_bound_to_frozen_rankings_and_qrels(self):
        with tempfile.TemporaryDirectory(dir=sdk.ROOT / 'results', prefix='sdk_quality_') as tmp:
            base = Path(tmp)
            (base / 'quality').mkdir()
            r.write_json(base / 'mapping.json', {'doc_ids': ['d' + str(i) for i in range(10)]})
            r.write_json(base / 'quality/qids.json', {'A': ['q'], 'B': [None]})
            r.write_json(base / 'qrels.json', {'q': {'d0': 1}})
            bits = {}
            for case in ('A', 'B'):
                (base / (case + '_ids.bin')).write_bytes(array('q', range(10)).tobytes())
                (base / (case + '_scores.bin')).write_bytes(array('f', [1.0] * 10).tobytes())
                bits[case + '_ids'] = r.digest(array('i', range(10)).tobytes())
                bits[case + '_scores'] = sdk.s.digest(base / (case + '_scores.bin'))
            metrics = sdk.beir_real.ranking_metrics({'q': ['d' + str(i) for i in range(10)]}, {'q': {'d0': 1}})
            frozen = {'TopK_sha256': bits, 'metrics': metrics}
            self.assertEqual(sdk.sdk_quality(base, base, frozen), metrics)
            (base / 'A_scores.bin').write_bytes(array('f', [2.0] * 10).tobytes())
            with self.assertRaisesRegex(ValueError, 'bits differ'):
                sdk.sdk_quality(base, base, frozen)

    def test_prepare_captures_measured_build_and_report_without_touching_reference(self):
        with tempfile.TemporaryDirectory(dir=sdk.ROOT / 'results', prefix='sdk_prepare_') as tmp:
            base = Path(tmp)
            reference = base / 'reference'
            reference.mkdir()
            selected = base / 'selected'
            selected.mkdir()
            vendor = base / 'vendor'
            (vendor / 'op_api/lib').mkdir(parents=True)
            (vendor / 'op_api/lib/libcust_opapi.so').write_bytes(b'mocked package')
            (vendor / 'binary_info_config.json').write_bytes(b'{}')
            (vendor / 'dynamic').mkdir()
            captured_header = vendor / 'dynamic/bm25_fused_score_top_k_tiling.h'
            captured_header.write_bytes(b'// captured header fixture, not CANN evidence\n')
            h, k, _ = sdk.split_sources(False)
            generated = {'host_sha256': r.digest(h.encode()), 'kernel_sha256': r.digest(k.encode())}
            build = {
                'variants': {
                    'dense': {
                        'normal': {
                            'vendor': str(vendor),
                            'package_sha256': 'package',
                            'installed_files': sdk.package_fingerprint(vendor),
                            'source_sha256': {
                                'op_host/bm25_fused_score_top_k.cpp': generated['host_sha256'],
                                'op_kernel/bm25_fused_score_top_k.cpp': generated['kernel_sha256'],
                                'op_kernel/bm25_fused_score_top_k_tiling.h': sdk.s.digest(captured_header),
                            },
                        }
                    }
                }
            }
            r.write_json(selected / 'build.json', build)
            freeze = base / 'freeze.json'
            r.write_json(
                freeze,
                {
                    'selection': {
                        'variant': 'split',
                        'expected_standalone_runner_sha256': 'runner',
                        'generated_normal': generated,
                        'package': {'package_sha256': 'package'},
                    }
                },
            )
            names = ('scifact', 'trec_covid', 'msmarco', 'sparse8192', 'sparse65536', 'synthetic10m')
            quality = {
                name: {
                    role: {'metrics': {'queries': count}, 'TopK_sha256': {'same': 'bits'}}
                    for role in sdk.release.NPU_ROLES
                }
                for name, count in (('scifact', 300), ('trec_covid', 50), ('msmarco', 43))
            }
            pre = {
                'status': 'concentrated_preindex_checks_completed_selection_review_pending',
                'runner_sha256': 'runner',
                'workloads': {
                    name: {
                        'per_round': [
                            {role: {} for role in sdk.release.NPU_ROLES + sdk.release.CPU_ROLES} for _ in range(6)
                        ]
                    }
                    for name in names
                },
                'quality': quality,
                'build_report_sha256': {str(selected / 'build.json'): sdk.s.digest(selected / 'build.json')},
            }
            report = base / 'pre.json'
            r.write_json(report, pre)
            tar = archive(
                [((sdk.PREFIX / 'ascendhost/CMakeLists.txt').as_posix(), b'PROJECT(ascendhost)\n', tarfile.REGTYPE)]
            )
            args = SimpleNamespace(
                sdk_source=reference, selected_build=selected, preindex_report=report, output=base / 'integration'
            )
            with (
                patch.object(sdk, 'FREEZE', freeze),
                patch.object(sdk.subprocess, 'check_output', side_effect=[sdk.BASE, tar]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                sdk.prepare(args)
                record, _ = sdk.verify_tree(args.output)
                self.assertEqual(record['selected_package_sha256'], 'package')
                self.assertEqual((args.output / 'preindex_report.json').read_bytes(), report.read_bytes())
                self.assertEqual(list(reference.iterdir()), [])
                (args.output / 'selected_build.json').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'report changed'):
                    sdk.verify_tree(args.output)

    def test_integrated_tidy_checks_bm25_units_and_ignores_unrelated_sdk_sources(self):
        with tempfile.TemporaryDirectory(dir=sdk.ROOT / 'results', prefix='sdk_tidy_') as tmp:
            base = Path(tmp)
            source = base / 'source'
            build = base / 'build'
            build.mkdir()
            files = [
                'ascendhost/src/bm25/' + name
                for name in (
                    'runtime.cpp',
                    'fused_runtime.cpp',
                    'planned_search.cpp',
                    'large_search.cpp',
                    'AscendIndexBm25.cpp',
                )
            ]
            files += ['ascendhost/src/old_sdk.cpp', 'test/ascendhost/bm25/sdk_benchmark.cpp']
            entries = []
            for name in files:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'// fixture\n')
                entries.append({'directory': str(build), 'file': str(path), 'arguments': ['g++', '-c', str(path)]})
            r.write_json(build / 'compile_commands.json', entries)
            result = SimpleNamespace(returncode=0, stdout='declaration-only fake clang tool', stderr='')
            argv = ['run_clang_tidy.py', '--build-dir', str(build), '--source-root', str(source), '--include-tests']
            with (
                patch.object(sys, 'argv', argv),
                patch.object(tidy.subprocess, 'run', return_value=result),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(tidy.main(), 0)
            checked = sdk.read(build / 'bm25_clang_tidy/report.json')
            self.assertEqual(len(checked['checks']), 6)
            self.assertFalse(any(item['file'].endswith('old_sdk.cpp') for item in checked['checks']))
            self.assertTrue(checked['production_compile_db_coverage_complete'])


if __name__ == '__main__':
    unittest.main()
