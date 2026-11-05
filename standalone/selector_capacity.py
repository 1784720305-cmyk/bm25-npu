"""Build isolated 1024/2048/4096 packages and compare real/sparse input on A5."""

import argparse
import itertools
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import build_final
from standalone import final_support as s
from standalone import large_corpus as corpus
from standalone import resident_index as r
from tools.capacity_sources import CAPACITIES, once, write_variant


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def locked_sources():
    lock = read(ROOT / 'source_lock.json')
    for group in ('frozen_operators', 'retained_core_source_sha256', 'build_and_check_source_sha256'):
        for name, sha in lock[group].items():
            r.require(s.digest(ROOT / name) == sha, 'Source lock mismatch: ' + name)
    for name, sha in lock.get('experimental_capacity_study', {}).get('source_sha256', {}).items():
        r.require(s.digest(ROOT / name) == sha, 'Experiment source changed: ' + name)


def preset_vendor(path, vendor):
    data = read(path)
    count = 0
    for preset in data['configurePresets']:
        variables = preset.get('cacheVariables', {})
        if 'ASCEND_COMPUTE_UNIT' in variables:
            variables['vendor_name'] = {'type': 'STRING', 'value': vendor}
            count += 1
    r.require(count > 0, 'Missing CANN configure preset')
    r.write_json(path, data)


def package_fingerprint(vendor):
    # Include registration, tiling and device binaries, not just the API .so.
    files = {}
    for p in sorted(vendor.rglob('*')):
        if p.is_file():
            r.require(not p.is_symlink() or p.resolve().is_relative_to(vendor.resolve()), 'External package symlink')
            files[p.relative_to(vendor).as_posix()] = s.digest(p)
    r.require(any(p.endswith('binary_info_config.json') for p in files), 'Missing device registration')
    r.require('op_api/lib/libcust_opapi.so' in files, 'Missing API library')
    return files


def build(args):
    output = args.output.resolve()
    r.require(not output.exists() and output.is_relative_to(ROOT / 'results'), 'New results output required')
    # Check the complete template before creating any output or invoking its build scripts.
    r.require((args.template / 'build.sh').is_file(), 'Actual fused CANN template required')
    merge = (args.vendors / 'bm25_topk_merge').resolve(strict=True)
    output.mkdir(parents=True)
    record = {
        'status': 'building_experimental_packages_device_pending',
        'variants': {},
        'baseline_vendors': str(args.vendors.resolve()),
        'baseline_merge': str(merge),
        'execution_source_lock_sha256': s.digest(ROOT / 'source_lock.json'),
        'merge_installed_files': package_fingerprint(merge),
    }
    try:
        for capacity in CAPACITIES:
            for profile in (False, True):
                key = f'{capacity}' + ('_profile' if profile else '')
                vendor_name = 'bm25_capacity_' + key
                project = output / ('project_' + key)
                build_final.prepare(args.template.resolve(), project, 'fused')
                variant = write_variant(project, capacity, profile)
                preset_vendor(project / 'CMakePresets.json', vendor_name)
                variant.update(vendor_name=vendor_name, project=str(project))
                r.write_json(project / 'final_source.json', variant)
                s.execute(['bash', 'build.sh'], output / (key + '_build.log'), cwd=project, timeout=3600)
                packages = list((project / 'build_out').glob('*.run'))
                r.require(len(packages) == 1, 'One actual CANN package required')
                prefix = output / ('installation_' + key)
                r.require(not prefix.exists(), 'Refuse existing installation')
                prefix.mkdir()
                s.execute(
                    ['bash', packages[0], '--install-path=' + str(prefix)], output / (key + '_install.log'), timeout=300
                )
                vendor = prefix / 'vendors' / vendor_name
                variant.update(
                    package_sha256=s.digest(packages[0]),
                    installed_files=package_fingerprint(vendor),
                    vendor=str(vendor),
                    aliases=str(output / ('runtime_' + key) / 'vendors'),
                )
                s.aliases(Path(variant['aliases']), vendor, merge)
                record['variants'][key] = variant
                r.write_json(output / 'build.json', record)
        host = output / 'host_project'
        host.mkdir()
        check = (ROOT / 'feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/selector_check.cpp').read_text(
            encoding='utf-8'
        )
        check = once(check, '1023, 1024,  1025, width', '1023, 1024,  1025, 2047, 2048, 2049, 4095, 4096, 4097, width')
        (host / 'selector_boundaries.cpp').write_bytes(check.encode('utf-8'))
        cmake = '''cmake_minimum_required(VERSION 3.17)
project(bm25_capacity_diagnostics LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BM25_BUILD_NPU ON CACHE BOOL "" FORCE)
set(BM25_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory("@ROOT@" core)
add_executable(capacity_profile "@ROOT@/feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/capacity_profile.cpp")
add_executable(capacity_boundaries selector_boundaries.cpp)
target_compile_definitions(capacity_boundaries PRIVATE BM25_SIMD_CHECK_SHA="@CHECK@")
foreach(target capacity_profile capacity_boundaries)
    target_link_libraries(${target} PRIVATE bm25_host)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach()
'''.replace('@ROOT@', ROOT.as_posix()).replace('@CHECK@', s.digest(host / 'selector_boundaries.cpp'))
        (host / 'CMakeLists.txt').write_bytes(cmake.encode('utf-8'))
        host_build = output / 'host_build'
        s.execute(
            [
                'cmake',
                '-S',
                host,
                '-B',
                host_build,
                '-DCMAKE_BUILD_TYPE=Release',
                '-DTOOLKIT_HOME=' + str(args.toolkit_home.resolve()),
                '-DBM25_VENDOR_DIR=' + record['variants']['1024']['aliases'],
            ],
            output / 'host_configure.log',
        )
        s.execute(['cmake', '--build', host_build, '--parallel', args.jobs], output / 'host_build.log', timeout=900)
        s.execute(
            [
                sys.executable,
                ROOT / 'tools/run_clang_tidy.py',
                '--build-dir',
                host_build,
                '--clang-tidy',
                args.clang_tidy,
                '--include-tests',
            ],
            output / 'host_clang_tidy.log',
            timeout=900,
        )
        tidy = read(host_build / 'bm25_clang_tidy/report.json')
        profile_file = ROOT / 'feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/capacity_profile.cpp'
        r.require(
            tidy['production_compile_db_coverage_complete']
            and any(Path(x['file']).resolve() == profile_file and x['exit_code'] == 0 for x in tidy['checks']),
            'New diagnostic must pass real-header clang-tidy',
        )
        record.update(
            status='experimental_packages_and_host_built_device_pending',
            host_build=str(host_build),
            host_binary_sha256={n: s.digest(host_build / n) for n in ('capacity_profile', 'capacity_boundaries')},
            production_clang_tidy_report_sha256=s.digest(host_build / 'bm25_clang_tidy/report.json'),
        )
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        r.write_json(output / 'build.json', record)
    print('BUILD REPORT:', output / 'build.json', flush=True)


def profile_summary(data, capacity):
    r.require(
        data.get('status') == 'device_stage_profile_traces_and_bit_checks_passed'
        and data.get('completed') is True
        and data.get('normal_profile_outputs_bit_identical') is True
        and data.get('guards_passed') is True
        and len(data['samples']) == 60,
        'Incomplete stage checks',
    )
    r.require(data['queries'] == 8 and type(data['tasks']) is int and data['tasks'] >= 8, 'Profile dimensions differ')
    result = {}
    for case in ('A', 'B'):
        samples = [x for x in data['samples'] if x['case'] == case]
        r.require([x['cycle'] for x in samples] == list(range(30)), 'Incomplete profile sample cycles')
        score, topk = [], []
        for sample in samples:
            values = sample['task_timestamps']
            r.require(len(values) == 4 * data['tasks'], 'Missing task timestamps')
            r.require(all(x == capacity for x in values[3::4]), 'Loaded device capacity differs from the variant')
            for start, middle, end in zip(values[::4], values[1::4], values[2::4]):
                r.require(
                    all(type(x) is int for x in (start, middle, end)) and 0 <= start < middle < end,
                    'Invalid device timestamps',
                )
                score.append(middle - start)
                topk.append(end - middle)
        result[case] = {'score_and_append_ticks': s.summarize(score, 1), 'topk_ticks': s.summarize(topk, 1)}
        # summarize() provides generic arithmetic, but its ms/QPS labels do not apply to ticks.
        for stage in result[case].values():
            stage['median'] = stage.pop('median_batch_ms')
            stage['p95'] = stage.pop('p95_batch_ms')
            stage['mean'] = stage.pop('mean_batch_ms')
            stage.pop('search_time_throughput_qps')
    return result


def comparisons(rows, workloads):
    result = {}
    for name in workloads:
        result[name] = {}
        for capacity in CAPACITIES[1:]:
            result[name][str(capacity)] = {}
            for case in ('A', 'B'):
                current = [x[name][str(capacity)]['search'][case]['median_batch_ms'] for x in rows]
                base = [x[name]['1024']['search'][case]['median_batch_ms'] for x in rows]
                current_p95 = [x[name][str(capacity)]['search'][case]['p95_batch_ms'] for x in rows]
                base_p95 = [x[name]['1024']['search'][case]['p95_batch_ms'] for x in rows]
                topk = [x[name][str(capacity)]['stages'][case]['topk_ticks']['median'] for x in rows]
                topk_base = [x[name]['1024']['stages'][case]['topk_ticks']['median'] for x in rows]
                result[name][str(capacity)][case] = {
                    'search_ms': statistics.median(current),
                    'baseline_search_ms': statistics.median(base),
                    'search_change_percent': 100 * (statistics.median(current) / statistics.median(base) - 1),
                    'search_P95_change_percent': 100
                    * (statistics.median(current_p95) / statistics.median(base_p95) - 1),
                    'search_wins': sum(a < b for a, b in zip(current, base)),
                    'search_P95_wins': sum(a < b for a, b in zip(current_p95, base_p95)),
                    'paired_search_change_percent': [100 * (a / b - 1) for a, b in zip(current, base)],
                    'topk_ticks_change_percent': 100 * (statistics.median(topk) / statistics.median(topk_base) - 1),
                }
    return result


def run(args):
    setup = read(args.build / 'build.json')
    r.require(setup['status'] == 'experimental_packages_and_host_built_device_pending', 'Build incomplete')
    output = args.output.resolve()
    r.require(not output.exists() and output.is_relative_to(ROOT / 'results'), 'New results output required')
    workloads = {}
    for value in args.workload:
        name, separator, path = value.partition('=')
        r.require(
            separator
            and name.isidentifier()
            and name not in workloads
            and name not in ('round', 'capacity_order', 'measurement_order'),
            'Use distinct NAME=INPUT workloads',
        )
        inputs = Path(path).resolve(strict=True)
        desc, index = corpus.validate(inputs)
        r.require(desc['queries'] == 8 and desc['k'] == 10, 'Fixed Q8/K10 input required')
        workloads[name] = (inputs, desc, index)
    r.require(
        len(workloads) >= 2 and any(d['generation'].get('source_jsonl') is None for _, d, _ in workloads.values()),
        'Include a real workload and an original sparse synthetic workload',
    )
    r.require(
        any(
            d.get('dataset_kind') == 'real' or d['generation'].get('source_jsonl') is not None
            for _, d, _ in workloads.values()
        ),
        'Missing real-input workload',
    )
    binaries = {n: Path(setup['host_build']) / n for n in ('capacity_profile', 'capacity_boundaries')}
    for name, binary in binaries.items():
        r.require(s.digest(binary) == setup['host_binary_sha256'][name], 'Diagnostic binary changed')
    runner = args.host_build.resolve() / 'bm25_large_search_benchmark'
    runner_sha = s.digest(runner)
    r.require(
        package_fingerprint(Path(setup['baseline_merge'])) == setup['merge_installed_files'], 'Merge package changed'
    )
    for variant in setup['variants'].values():
        r.require(
            package_fingerprint(Path(variant['vendor'])) == variant['installed_files'], 'Experiment package changed'
        )
    output.mkdir(parents=True)
    record = {
        'status': 'running_capacity_experiment',
        'per_round': [],
        'build_report_sha256': s.digest(args.build / 'build.json'),
        'runner_sha256': runner_sha,
        'workloads': {
            k: {
                'input': str(v[0]),
                'index_id': v[2]['index_id'],
                'preparation_sha256': s.digest(v[0] / 'preparation_report.json'),
            }
            for k, v in workloads.items()
        },
        'stage_units': 'raw GetSystemCycle ticks; no unverified A5 frequency conversion',
        'stage_scope': 'Instrumented package per-leaf scoring+append and Pack+threshold+collect; barriers and timestamp overhead remain. Not additive to uninstrumented full Search.',
        'performance_acceptance': 'pending_review_no_candidate_promoted',
    }
    try:
        for key, variant in setup['variants'].items():
            dest = output / ('boundaries_' + key)
            s.execute(
                [binaries['capacity_boundaries'], variant['aliases'], args.device, dest],
                output / (dest.name + '.log'),
                s.environment(Path(variant['aliases'])),
            )
            check = read(dest / 'report.json')
            r.require(
                check['calls'] == 54
                and check['status'] == 'actual_simd_selector_guards_and_strict_bit_rankings_passed',
                'Boundary checks failed',
            )
        # Every capacity appears twice in each ordinal position; retain all six permutations.
        for round_number, order in enumerate(itertools.permutations(CAPACITIES), 1):
            row = {
                'round': round_number,
                'capacity_order': list(order),
                'measurement_order': ['search', 'profile'] if round_number % 2 else ['profile', 'search'],
            }
            record['per_round'].append(row)
            for name, (inputs, desc, index) in workloads.items():
                r.require(
                    s.digest(inputs / 'preparation_report.json') == record['workloads'][name]['preparation_sha256'],
                    'Workload preparation changed',
                )
                corpus.validate(inputs)
                row[name] = {}
                for capacity in order:
                    target = output / f'round{round_number}' / name / str(capacity)
                    target.mkdir(parents=True)
                    normal = setup['variants'][str(capacity)]
                    profile = setup['variants'][str(capacity) + '_profile']
                    for mode in ('search', 'profile') if round_number % 2 else ('profile', 'search'):
                        if mode == 'search':
                            words = max(sum(map(len, desc['cases'][c]['terms'])) for c in ('A', 'B'))
                            cmd = [
                                runner,
                                inputs / 'index',
                                inputs / 'A.txt',
                                inputs / 'B.txt',
                                target / 'actual',
                                args.device,
                                normal['aliases'],
                                10,
                                8,
                                words,
                                5,
                                30,
                                inputs / 'A/expected_scores.bin',
                                inputs / 'B/expected_scores.bin',
                            ]
                            s.execute(cmd, target / 'search.log', s.environment(Path(normal['aliases'])))
                            measured = s.verify_run(target / 'actual', inputs, desc, index, 'npu', args.device)
                            r.write_json(target / 'search.json', measured)
                        else:
                            s.execute(
                                [
                                    binaries['capacity_profile'],
                                    inputs,
                                    profile['aliases'],
                                    args.device,
                                    target / 'profile.json',
                                ],
                                target / 'profile.log',
                                s.environment(Path(profile['aliases'])),
                            )
                            stages = profile_summary(read(target / 'profile.json'), capacity)
                    cross_package = {}
                    for case in ('A', 'B'):
                        for kind in ('scores', 'ids'):
                            normal_topk = target / 'actual' / ('before_' + case) / ('top_' + kind + '.bin')
                            profile_topk = target / f'profile.json.gold_{case}_{kind}.bin'
                            r.require(
                                s.same_stream(normal_topk, profile_topk), 'Normal/profile package TopK bits differ'
                            )
                            cross_package[case + '_' + kind] = s.digest(profile_topk)
                    row[name][str(capacity)] = {
                        'search': measured['summary'],
                        'stages': stages,
                        'normal_profile_package_topk_sha256': cross_package,
                        'search_report_sha256': s.digest(target / 'search.json'),
                        'profile_report_sha256': s.digest(target / 'profile.json'),
                    }
                    r.write_json(output / 'report.json', record)
                    print(name, capacity, json.dumps(row[name][str(capacity)], ensure_ascii=False), flush=True)
        for name, (inputs, _, _) in workloads.items():
            corpus.validate(inputs)
            r.require(
                s.digest(inputs / 'preparation_report.json') == record['workloads'][name]['preparation_sha256'],
                'Workload preparation changed during run',
            )
        r.require(
            package_fingerprint(Path(setup['baseline_merge'])) == setup['merge_installed_files'],
            'Merge package changed during run',
        )
        r.require(s.digest(runner) == runner_sha, 'Search runner changed')
        for name, binary in binaries.items():
            r.require(s.digest(binary) == setup['host_binary_sha256'][name], 'Diagnostic binary changed during run')
        for variant in setup['variants'].values():
            r.require(
                package_fingerprint(Path(variant['vendor'])) == variant['installed_files'],
                'Experiment package changed during run',
            )
        record.update(
            status='capacity_correctness_and_measurements_completed_acceptance_pending',
            comparison=comparisons(record['per_round'], workloads),
        )
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        r.write_json(output / 'report.json', record)
    print('COMPARISON:', json.dumps(record['comparison'], ensure_ascii=False), flush=True)
    print('REPORT:', output / 'report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('build')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--vendors', type=Path, required=True)
    p.add_argument('--toolkit-home', type=Path, default=Path('/usr/local/Ascend/cann-9.2.0/x86_64-linux'))
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--clang-tidy', default='clang-tidy')
    p = sub.add_parser('run')
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--host-build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workload', action='append', required=True)
    p.add_argument('--device', type=int, default=5)
    args = parser.parse_args()
    r.require(
        sys.platform == 'linux' and os.environ.get('ASCEND_OPP_PATH'), 'Use A5 Linux after sourcing CANN set_env.sh'
    )
    locked_sources()
    (build if args.mode == 'build' else run)(args)


if __name__ == '__main__':
    main()
