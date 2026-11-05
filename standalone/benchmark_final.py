"""One final implementation, frozen CPU opponents, optional original BM25S; six rounds."""

import argparse
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import final_support as s
from standalone import large_corpus as corpus
from standalone import resident_index as r


def measurement_roles(docs, cpu_baseline, include_bm25s):
    r.require(cpu_baseline in ('cpp', 'bm25s'), 'Unknown primary CPU baseline')
    if cpu_baseline == 'bm25s':
        return ['npu'] + [f'original_native_t{workers}' for workers in (1, 8, 32)]
    roles = ['cpu', 'npu', 'cpu_directory']
    if docs <= 65536:
        roles += ['cpu_single', 'cpu_directory_single']
    if include_bm25s:
        roles += [
            f'{variant}_t{workers}' for variant in ('original_native', 'strict_adapter') for workers in (1, 8, 32)
        ]
    return roles


def historical_npu_reference(rows, description):
    generation = description.get('generation', {})
    matching = (
        description.get('dataset_kind') == 'uniform_synthetic_not_MT_approved'
        and generation.get('seed') == 20261014
        and generation.get('vocab') == 1024
        and generation.get('synthetic_length_range') == [16, 48]
        and (description.get('queries'), description.get('query_length'), description.get('k')) == (8, 3, 10)
        and not generation.get('source_jsonl')
        and all(
            description.get('cases', {}).get(case, {}).get('terms')
            == corpus.load_raw(ROOT / 'fixtures' / (case + '.txt'))
            for case in ('A', 'B')
        )
    )
    if not matching:
        return {'status': 'not_comparable_to_frozen_synthetic_workload'}
    evidence = ROOT / 'evidence/bm25s_current_npu_device_evidence_20261029.json'
    old = json.loads(evidence.read_text(encoding='utf-8'))
    selected = {
        item['case']: item
        for item in old['summary']
        if item['docs'] == description['docs'] and item['variant'] == 'original_native'
    }
    if set(selected) != {'A', 'B'}:
        return {'status': 'no_historical_reference_for_this_scale'}
    cases = {}
    for case in ('A', 'B'):
        median = statistics.median(row['npu']['summary'][case]['median_batch_ms'] for row in rows)
        p95 = statistics.median(row['npu']['summary'][case]['p95_batch_ms'] for row in rows)
        cases[case] = {
            'historical_median_batch_ms': selected[case]['npu_ms'],
            'current_median_batch_ms': median,
            'median_change_percent': 100 * (median / selected[case]['npu_ms'] - 1),
            'historical_p95_batch_ms': selected[case]['npu_p95_ms'],
            'current_p95_batch_ms': p95,
            'p95_change_percent': 100 * (p95 / selected[case]['npu_p95_ms'] - 1),
        }
    return {
        'status': 'historical_reference_for_review',
        'cases': cases,
        'evidence': str(evidence),
        'evidence_sha256': s.digest(evidence),
        'scope': 'Separate historical A5 session, not a paired control or automatic regression acceptance threshold.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output', 'vendors'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--host-build', type=Path, default=ROOT / 'results/final_host_build')
    p.add_argument('--npu-runner', type=Path)
    p.add_argument('--cpu-runner', type=Path)
    p.add_argument('--directory-runner', type=Path)
    p.add_argument('--device', type=int, default=5)
    p.add_argument('--bm25s', action='store_true')
    p.add_argument('--cpu-baseline', choices=('cpp', 'bm25s'), default='cpp')
    p.add_argument('--real-input', action='store_true', help='External corpus with natural variable-length queries')
    p.add_argument(
        '--quality-only', action='store_true', help='One checked round for full-query quality, not a performance gate'
    )
    p.add_argument('--dependency-dir', type=Path, default=ROOT / 'results/bm25s_numba_py39')
    a = p.parse_args()
    a.bm25s = a.bm25s or a.cpu_baseline == 'bm25s'
    for k, v in vars(a).items():
        if isinstance(v, Path):
            setattr(a, k, v.resolve())
    r.require(sys.platform == 'linux' and not a.output.exists(), 'A5 Linux and fresh results required')
    r.require(
        a.output.is_relative_to(ROOT / 'results')
        and not a.output.is_relative_to(a.input)
        and not a.input.is_relative_to(a.output),
        'Independent results directory required',
    )
    expected = {
        'bm25_fused_score_topk': 'bm25_fused_score_topk_warp_append_large',
        'bm25_topk_merge': 'bm25_topk_merge_tree',
    }
    for alias, target in expected.items():
        r.require((a.vendors / alias).resolve(strict=True).name == target, 'Final vendor mismatch: ' + alias)
    description, index = corpus.validate(a.input)
    n, q, k = description['docs'], description['queries'], description['k']
    if a.real_input:
        r.require(
            a.cpu_baseline == 'bm25s'
            and description['query_length'] is None
            and description['dataset_kind'] == 'external_token_ID_jsonl_not_MT_approved',
            'Real input requires an external variable-query corpus and original BM25S baseline',
        )
    else:
        r.require((q, k, description['query_length']) == (8, 10, 3), 'Declared final benchmark is Q8/3words/K10')
    r.require(not a.quality_only or a.real_input, 'Quality-only requires real input')
    rounds, warmup, repeats = (1, 1, 10) if a.quality_only else (6, 5, 30)
    words = description.get('capacity_words', 3 * q)
    roles = measurement_roles(n, a.cpu_baseline, a.bm25s)
    runners = {
        'npu': a.npu_runner or a.host_build / 'bm25_large_search_benchmark',
    }
    if a.cpu_baseline == 'cpp':
        runners.update(
            cpu=a.cpu_runner or a.host_build / 'bm25_cpu_search_benchmark',
            cpu_directory=a.directory_runner or a.host_build / 'bm25_directory_cpu_benchmark',
        )
    r.require(all(x.is_file() for x in runners.values()), 'Required actual host binary missing')
    plan = s.layout(n, index['vocab_size'], index['posting_count'], q, k, words)
    r.require(plan['cpu_peak_known_bytes'] + 1024**3 < s.host_available_bytes() * 4 // 5, 'Host budget exceeded')
    disk = (
        rounds * sum(not role.startswith(('original_native', 'strict_adapter')) for role in roles) * 4 * q * n * 4
        + (8 * index['posting_count'] + 8 * q * n if a.bm25s else 0)
        + 1024**3
    )
    r.require(disk < shutil.disk_usage(a.output.parent).free * 4 // 5, 'Diagnostic-output disk budget exceeded')
    a.output.mkdir(parents=True)
    report = {
        'status': 'running',
        'docs': n,
        'workload': description,
        'per_round': [],
        'runner_sha256': {role: s.digest(path) for role, path in runners.items()},
        'scope': 'Final Warp Append+tree Merge only; complete steady-state Search, no experimental candidate.',
        'primary_cpu_baseline': 'original_native' if a.cpu_baseline == 'bm25s' else 'cpp',
        'performance_acceptance': 'pending_review',
        'measurement_mode': 'quality_only' if a.quality_only else 'six_round_performance',
    }

    def save():
        r.write_json(a.output / 'report.json', report)

    save()
    try:
        s.execute(
            [runners['npu'], '--memory', a.device, a.output / 'memory.json'],
            a.output / 'memory.log',
            s.environment(a.vendors),
        )
        memory = json.loads((a.output / 'memory.json').read_text(encoding='utf-8'))
        r.require(
            plan['known_device_bytes_upper_bound'] + plan['debug_score_bytes'] + 1024**3 < memory['free'] * 4 // 5,
            'Device budget exceeded',
        )
        report['memory_probe'] = memory
        bm_script = ROOT / 'standalone/bm25s_benchmark.py'
        built = a.output / 'bm25s_index'
        if a.bm25s:
            s.execute(
                [
                    sys.executable,
                    '-u',
                    bm_script,
                    'build',
                    '--input',
                    a.input,
                    '--built',
                    built,
                    '--output',
                    built,
                    '--dependency-dir',
                    a.dependency_dir,
                ],
                a.output / 'bm25s_build.log',
                timeout=14400,
            )
            report['bm25s_construction'] = json.loads((built / 'report.json').read_text(encoding='utf-8'))
        for ordinal in range(rounds):
            corpus.validate(a.input)
            order = roles if ordinal % 2 == 0 else list(reversed(roles))
            row = {'round': ordinal + 1, 'process_order': list(order)}
            for role in order:
                target = a.output / f'round{ordinal + 1}' / role
                target.parent.mkdir(parents=True, exist_ok=True)
                if role.startswith(('original_native', 'strict_adapter')):
                    variant, workers = role.rsplit('_t', 1)
                    cmd = [
                        sys.executable,
                        '-u',
                        bm_script,
                        'measure',
                        '--input',
                        a.input,
                        '--built',
                        built,
                        '--output',
                        target,
                        '--dependency-dir',
                        a.dependency_dir,
                        '--workers',
                        workers,
                        '--variant',
                        variant,
                        '--warmup',
                        str(warmup),
                        '--repeats',
                        str(repeats),
                    ]
                    s.execute(cmd, target.parent / (role + '.log'), timeout=3600)
                    result = json.loads((target / 'report.json').read_text(encoding='utf-8'))
                else:
                    target.mkdir()
                    actual = target / 'actual'
                    single = role.endswith('_single')
                    base = role.removesuffix('_single')
                    threads = 1 if single else 32
                    args = [runners[base], a.input / 'index', a.input / 'A.txt', a.input / 'B.txt', actual]
                    if base == 'npu':
                        args += [
                            a.device,
                            a.vendors,
                            k,
                            q,
                            words,
                            warmup,
                            repeats,
                            a.input / 'A/expected_scores.bin',
                            a.input / 'B/expected_scores.bin',
                        ]
                    else:
                        args += [threads, k, threads, q, words, warmup, repeats]
                    s.execute(
                        args,
                        target / 'run.log',
                        s.environment(a.vendors) if base == 'npu' else os.environ.copy(),
                        timeout=3600,
                    )
                    result = s.verify_run(
                        actual,
                        a.input,
                        description,
                        index,
                        base,
                        a.device,
                        warmup=warmup,
                        repeats=repeats,
                        threads=threads,
                    )
                    result['command'] = list(map(str, args))
                    r.write_json(target / 'report.json', result)
                row[role] = {
                    'summary': result['summary'],
                    'report_sha256': s.digest(target / 'report.json'),
                    'directory': str(target),
                }
                print(f'N={n} {role} complete Search: ' + json.dumps(result['summary'], ensure_ascii=False), flush=True)
            report['per_round'].append(row)
            save()
        cpu_roles = [role for role in roles if role.startswith('cpu')]
        if cpu_roles:
            report['cpu_decision'] = s.cpu_gate(
                [{**row, 'candidate': row['npu']} for row in report['per_round']], cpu_roles
            )
        if a.bm25s and not a.quality_only:
            from standalone.bm25s_benchmark import decide

            report['bm25s_decision'] = decide(
                report['per_round'],
                [1, 8, 32],
                report['bm25s_construction']['strict_ids_match_project'],
                variants=('original_native',) if a.cpu_baseline == 'bm25s' else ('original_native', 'strict_adapter'),
            )
        if a.cpu_baseline == 'bm25s':
            report['historical_npu_comparison'] = historical_npu_reference(report['per_round'], description)
            print(
                'Historical NPU comparison: ' + json.dumps(report['historical_npu_comparison'], ensure_ascii=False),
                flush=True,
            )
        corpus.validate(a.input)
        r.require(
            all(s.digest(path) == report['runner_sha256'][role] for role, path in runners.items()), 'Runner changed'
        )
        report['status'] = (
            'actual_real_query_quality_checked' if a.quality_only else 'actual_final_search_six_rounds_checked'
        )
        save()
        print('REPORT: ' + str(a.output / 'report.json'), flush=True)
        print(
            'Final Search decision: '
            + json.dumps({k: v for k, v in report.items() if k.endswith('decision')}, ensure_ascii=False),
            flush=True,
        )
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
