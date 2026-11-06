"""A5 paired study of the 1024 baseline and a dense-row SIMD threshold candidate."""

import argparse
import copy
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
from standalone.selector_capacity import (
    TRACE_CONTRACT,
    locked_sources,
    package_fingerprint,
    preset_vendor,
    profile_summary,
    read,
)
from tools.capacity_sources import sources
from tools.dense_sources import dense_sources

STATUS = 'dense_study_packages_built_device_pending'


def check_setup(setup, baseline_only=False):
    r.require(
        setup['status'] == STATUS and setup['profile_trace_contract'] == TRACE_CONTRACT, 'Dense study build incomplete'
    )
    r.require(
        set(setup['variants']) == ({'baseline'} if baseline_only else {'baseline', 'dense'}),
        'Study variants incomplete',
    )
    for path, sha in setup['host_binary_sha256'].items():
        r.require(s.digest(Path(setup['host_build']) / path) == sha, 'Diagnostic host binary changed')
    r.require(package_fingerprint(Path(setup['merge'])) == setup['merge_files'], 'Merge changed')
    for variants in setup['variants'].values():
        r.require(set(variants) == {'normal', 'profile'}, 'Normal/profile pair incomplete')
        for variant in variants.values():
            r.require(
                package_fingerprint(Path(variant['vendor'])) == variant['installed_files'], 'Operator package changed'
            )


def build(args, source_generator=dense_sources, vendor_base='bm25_dense_vector_1024'):
    baseline_path = args.baseline_build.resolve(strict=True)
    old = read(baseline_path / 'build.json')
    r.require(
        old['status'] == 'experimental_packages_and_host_built_device_pending'
        and old.get('profile_trace_contract') == TRACE_CONTRACT,
        'A successful v2 capacity build is required',
    )
    output = args.output.resolve()
    r.require(not output.exists() and output.is_relative_to(ROOT / 'results'), 'New results directory required')
    r.require((args.template / 'build.sh').is_file(), 'Actual CANN template required')
    baseline = {
        key: copy.deepcopy(old['variants']['1024' + ('_profile' if key == 'profile' else '')])
        for key in ('normal', 'profile')
    }
    for mode, variant in baseline.items():
        host, kernel, _ = sources(1024, mode == 'profile')
        for folder, text in (('op_host', host), ('op_kernel', kernel)):
            r.require(
                variant['source_sha256'][folder + '/bm25_fused_score_top_k.cpp'] == r.digest(text.encode()),
                'Baseline source differs from the validated implementation',
            )
    setup = {
        'status': STATUS,
        'profile_trace_contract': TRACE_CONTRACT,
        'host_build': old['host_build'],
        'host_binary_sha256': old['host_binary_sha256'],
        'merge': old['baseline_merge'],
        'merge_files': old['merge_installed_files'],
        'variants': {'baseline': baseline},
        'baseline_build_sha256': s.digest(baseline_path / 'build.json'),
        'current_source_lock_sha256': s.digest(ROOT / 'source_lock.json'),
        'capacity': 1024,
        'candidate_promoted': False,
        'candidate_generator': source_generator.__module__,
    }
    check_setup(setup, baseline_only=True)
    output.mkdir(parents=True)
    setup['status'] = 'building_dense_candidate'
    try:
        candidate = setup['variants']['dense'] = {}
        for mode in ('normal', 'profile'):
            project = output / ('project_' + mode)
            build_final.prepare(args.template.resolve(), project, 'fused')
            host, kernel, plan = source_generator(mode == 'profile')
            for folder, text in (('op_host', host), ('op_kernel', kernel)):
                (project / folder / 'bm25_fused_score_top_k.cpp').write_bytes(text.encode('utf-8'))
            vendor_name = vendor_base + ('_profile' if mode == 'profile' else '')
            preset_vendor(project / 'CMakePresets.json', vendor_name)
            source_hashes = {
                p.relative_to(project).as_posix(): s.digest(p)
                for p in project.rglob('*')
                if p.is_file() and p.suffix in ('.cpp', '.h')
            }
            variant = {**plan, 'project': str(project), 'vendor_name': vendor_name, 'source_sha256': source_hashes}
            r.write_json(project / 'final_source.json', variant)
            s.execute(['bash', 'build.sh'], output / (mode + '_build.log'), cwd=project, timeout=3600)
            packages = list((project / 'build_out').glob('*.run'))
            r.require(len(packages) == 1, 'One actual CANN package required')
            prefix = output / ('installation_' + mode)
            prefix.mkdir()
            s.execute(
                ['bash', packages[0], '--install-path=' + str(prefix)], output / (mode + '_install.log'), timeout=300
            )
            vendor = prefix / 'vendors' / vendor_name
            alias = output / ('runtime_' + mode) / 'vendors'
            s.aliases(alias, vendor, Path(setup['merge']))
            variant.update(
                vendor=str(vendor),
                aliases=str(alias),
                installed_files=package_fingerprint(vendor),
                package_sha256=s.digest(packages[0]),
            )
            candidate[mode] = variant
        setup['status'] = STATUS
        check_setup(setup)
    except (Exception, KeyboardInterrupt) as error:
        setup.update(status='failed', error=str(error))
        raise
    finally:
        r.write_json(output / 'build.json', setup)
    print('BUILD REPORT:', output / 'build.json', flush=True)


def compare(rows, names, include_stages=True):
    r.require(len(rows) == 6, 'Six complete paired rounds required')
    result = {}
    for name in names:
        result[name] = {}
        for case in ('A', 'B'):
            base = [x[name]['baseline']['search'][case] for x in rows]
            candidate = [x[name]['dense']['search'][case] for x in rows]
            b, c = ([x['median_batch_ms'] for x in group] for group in (base, candidate))
            bp, cp = ([x['p95_batch_ms'] for x in group] for group in (base, candidate))
            result[name][case] = {
                'baseline_ms': statistics.median(b),
                'dense_ms': statistics.median(c),
                'median_change_percent': 100 * (statistics.median(c) / statistics.median(b) - 1),
                'P95_change_percent': 100 * (statistics.median(cp) / statistics.median(bp) - 1),
                'paired_change_percent': [100 * (v / ref - 1) for v, ref in zip(c, b)],
                'median_wins': sum(v < ref for v, ref in zip(c, b)),
                'P95_wins': sum(v < ref for v, ref in zip(cp, bp)),
            }
            if include_stages:
                bt, ct = (
                    [x[name][role]['stages'][case]['topk_ticks']['median'] for x in rows]
                    for role in ('baseline', 'dense')
                )
                result[name][case]['topk_ticks_change_percent'] = 100 * (
                    statistics.median(ct) / statistics.median(bt) - 1
                )
    return result


def run(args):
    setup = read(args.build / 'build.json')
    check_setup(setup)
    output = args.output.resolve()
    r.require(not output.exists() and output.is_relative_to(ROOT / 'results'), 'New results directory required')
    inputs = {}
    for value in args.workload:
        name, separator, path = value.partition('=')
        r.require(
            separator and name.isidentifier() and name not in inputs and name not in ('round', 'order'),
            'Use distinct NAME=INPUT workloads',
        )
        path = Path(path).resolve(strict=True)
        desc, index = corpus.validate(path)
        r.require(desc['queries'] == 8 and desc['k'] == 10, 'Fixed Q8/K10 required')
        inputs[name] = (path, desc, index)
    r.require(
        len(inputs) >= 2
        and any(d['generation'].get('source_jsonl') for _, d, _ in inputs.values())
        and any(d['generation'].get('source_jsonl') is None for _, d, _ in inputs.values()),
        'Include real and original sparse inputs',
    )
    runner = args.host_build.resolve(strict=True) / 'bm25_large_search_benchmark'
    runner_sha = s.digest(runner)
    host = Path(setup['host_build'])
    output.mkdir(parents=True)
    report = {
        'status': 'running_dense_selector_study',
        'per_round': [],
        'runner_sha256': runner_sha,
        'build_report_sha256': s.digest(args.build / 'build.json'),
        'profile_trace_contract': TRACE_CONTRACT,
        'workloads': {
            name: {
                'path': str(p),
                'index_id': index['index_id'],
                'preparation_sha256': s.digest(p / 'preparation_report.json'),
            }
            for name, (p, _, index) in inputs.items()
        },
        'capacity': 1024,
        'candidate_promoted': False,
        'stage_units': 'raw system ticks; separate instrumented execution, not additive batch time',
        'performance_acceptance': 'pending_review',
    }
    try:
        for role, variants in setup['variants'].items():
            for mode, variant in variants.items():
                target = output / ('boundaries_' + role + '_' + mode)
                s.execute(
                    [host / 'capacity_boundaries', variant['aliases'], args.device, target],
                    output / (target.name + '.log'),
                    s.environment(Path(variant['aliases'])),
                )
                check = read(target / 'report.json')
                r.require(
                    check['status'] == 'actual_simd_selector_guards_and_strict_bit_rankings_passed'
                    and check['calls'] == 54,
                    'Boundary checks incomplete',
                )
        for ordinal in range(1, 7):
            order = ('baseline', 'dense') if ordinal % 2 else ('dense', 'baseline')
            row = {'round': ordinal, 'order': list(order)}
            report['per_round'].append(row)
            for name, (path, desc, index) in inputs.items():
                corpus.validate(path)
                r.require(
                    s.digest(path / 'preparation_report.json') == report['workloads'][name]['preparation_sha256'],
                    'Input preparation changed',
                )
                row[name] = {}
                for role in order:
                    target = output / f'round{ordinal}' / name / role
                    target.mkdir(parents=True)
                    variants = setup['variants'][role]
                    for mode in ('search', 'profile') if ordinal % 2 else ('profile', 'search'):
                        if mode == 'search':
                            words = max(sum(map(len, desc['cases'][c]['terms'])) for c in ('A', 'B'))
                            vendor = Path(variants['normal']['aliases'])
                            s.execute(
                                [
                                    runner,
                                    path / 'index',
                                    path / 'A.txt',
                                    path / 'B.txt',
                                    target / 'actual',
                                    args.device,
                                    vendor,
                                    10,
                                    8,
                                    words,
                                    5,
                                    30,
                                    path / 'A/expected_scores.bin',
                                    path / 'B/expected_scores.bin',
                                ],
                                target / 'search.log',
                                s.environment(vendor),
                            )
                            measured = s.verify_run(target / 'actual', path, desc, index, 'npu', args.device)
                            r.write_json(target / 'search.json', measured)
                        else:
                            vendor = Path(variants['profile']['aliases'])
                            s.execute(
                                [host / 'capacity_profile', path, vendor, args.device, target / 'profile.json'],
                                target / 'profile.log',
                                s.environment(vendor),
                            )
                            stages = profile_summary(read(target / 'profile.json'), 1024)
                    bits = {}
                    for case in ('A', 'B'):
                        for kind in ('scores', 'ids'):
                            normal = target / 'actual' / ('before_' + case) / ('top_' + kind + '.bin')
                            profile = target / f'profile.json.gold_{case}_{kind}.bin'
                            r.require(s.same_stream(normal, profile), 'Normal/profile TopK bits differ')
                            bits[case + '_' + kind] = s.digest(profile)
                    row[name][role] = {
                        'search': measured['summary'],
                        'stages': stages,
                        'TopK_sha256': bits,
                        'search_report_sha256': s.digest(target / 'search.json'),
                        'profile_report_sha256': s.digest(target / 'profile.json'),
                    }
                    if role == order[-1]:
                        r.require(
                            row[name]['baseline']['TopK_sha256'] == row[name]['dense']['TopK_sha256'],
                            'Candidate/baseline TopK bits differ',
                        )
                    r.write_json(output / 'report.json', report)
                    print(name, role, json.dumps(row[name][role], ensure_ascii=False), flush=True)
        for name, (path, _, _) in inputs.items():
            corpus.validate(path)
            r.require(
                s.digest(path / 'preparation_report.json') == report['workloads'][name]['preparation_sha256'],
                'Input preparation changed during run',
            )
        check_setup(setup)
        r.require(s.digest(runner) == runner_sha, 'Search binary changed')
        report.update(
            status='dense_selector_correctness_and_six_round_measurements_completed',
            comparison=compare(report['per_round'], inputs),
        )
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        r.write_json(output / 'report.json', report)
    print('COMPARISON:', json.dumps(report['comparison'], ensure_ascii=False), flush=True)
    print('REPORT:', output / 'report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('build')
    p.add_argument('--baseline-build', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--host-build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workload', action='append', required=True)
    p.add_argument('--device', type=int, default=5)
    args = parser.parse_args()
    r.require(sys.platform == 'linux' and os.environ.get('ASCEND_OPP_PATH'), 'Use A5 Linux after sourcing CANN')
    locked_sources()
    study = read(ROOT / 'source_lock.json')['experimental_dense_study']
    for path, sha in study['source_sha256'].items():
        r.require(s.digest(ROOT / path) == sha, 'Dense experiment source changed: ' + path)
    (build if args.mode == 'build' else run)(args)


if __name__ == '__main__':
    main()
