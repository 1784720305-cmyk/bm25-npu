"""One final collector adjustment, one fixed matrix, then IndexSDK integration review."""

import argparse
import itertools
import json
import os
import shutil
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import beir_real, dense_selector
from standalone import final_support as s
from standalone import large_corpus as corpus
from standalone import resident_index as r
from standalone.bm25s_benchmark import decide
from standalone.selector_capacity import TRACE_CONTRACT, locked_sources, profile_summary, read
from tools.dense_sources import dense_sources
from tools.split_sources import split_sources

NPU_ROLES = ('baseline', 'dense_v1', 'split')
CPU_ROLES = tuple('original_native_t' + str(w) for w in (1, 8, 32))
DATASETS = ('scifact', 'trec-covid', 'msmarco')


def process_order(ordinal):
    r.require(1 <= ordinal <= 6, 'Exactly six fixed rounds')
    npu = list(itertools.permutations(NPU_ROLES))[ordinal - 1]
    cpu = CPU_ROLES if ordinal % 2 else tuple(reversed(CPU_ROLES))
    return list(npu + cpu if ordinal % 2 else cpu + npu)


def check_builds(new, old):
    for setup, generator in ((new, split_sources), (old, dense_sources)):
        dense_selector.check_setup(setup)
        for mode, variant in setup['variants']['dense'].items():
            host, kernel, plan = generator(mode == 'profile')
            for directory, source in (('op_host', host), ('op_kernel', kernel)):
                r.require(
                    variant['source_sha256'][directory + '/bm25_fused_score_top_k.cpp'] == r.digest(source.encode()),
                    'Candidate generated source changed',
                )
            r.require(all(variant[key] == value for key, value in plan.items()), 'Candidate UB layout changed')
    r.require(new['variants']['baseline'] == old['variants']['baseline'], 'Different accepted baseline packages')
    for key in ('host_build', 'host_binary_sha256', 'merge', 'merge_files', 'profile_trace_contract'):
        r.require(new[key] == old[key], 'Different baseline dependency: ' + key)
    return {
        'baseline': new['variants']['baseline'],
        'dense_v1': old['variants']['dense'],
        'split': new['variants']['dense'],
    }


def inputs_at(real_root, legacy_root):
    result = {}
    for dataset in DATASETS:
        data = real_root / (dataset.replace('-', '_') + '_data_01')
        beir_real.validate_data(data, dataset)
        result[dataset.replace('-', '_')] = {'input': data / 'performance', 'real_data': data}
    for count in (8192, 65536):
        result['sparse' + str(count)] = {'input': legacy_root / 'results/score_bitmap_scales1/inputs' / str(count)}
    result['synthetic10m'] = {'input': legacy_root / 'results/large_scale_close1/inputs/10000000'}
    for record in result.values():
        path = record['input'].resolve(strict=True)
        description, index = corpus.validate(path)
        r.require((description['queries'], description['k']) == (8, 10), 'Fixed Q8/K10 performance packets required')
        record.update(
            input=path,
            description=description,
            index=index,
            preparation_sha256=s.digest(path / 'preparation_report.json'),
        )
    return result


def verify_unchanged_input(record, rehash=False):
    corpus.validate(record['input'], rehash=rehash)
    r.require(
        s.digest(record['input'] / 'preparation_report.json') == record['preparation_sha256'], 'Input metadata changed'
    )


def search(runner, record, vendor, output, device, warmup=5, repeats=30):
    path, description, index = (record[key] for key in ('input', 'description', 'index'))
    output.mkdir(parents=True)
    command = [
        runner,
        path / 'index',
        path / 'A.txt',
        path / 'B.txt',
        output / 'actual',
        device,
        vendor,
        description['k'],
        description['queries'],
        description.get('capacity_words', description['queries'] * 3),
        warmup,
        repeats,
        path / 'A/expected_scores.bin',
        path / 'B/expected_scores.bin',
    ]
    s.execute(command, output / 'search.log', s.environment(vendor), timeout=7200)
    measured = s.verify_run(output / 'actual', path, description, index, 'npu', device, warmup=warmup, repeats=repeats)
    r.write_json(output / 'search.json', measured)
    bits = {
        case + '_' + kind: s.digest(output / 'actual' / ('before_' + case) / ('top_' + kind + '.bin'))
        for case in ('A', 'B')
        for kind in ('scores', 'ids')
    }
    return {
        'summary': measured['summary'],
        'TopK_sha256': bits,
        'report_sha256': s.digest(output / 'search.json'),
        'directory': str(output),
    }


def profile(record, host, vendor, output, device, expected_bits):
    s.execute(
        [host / 'capacity_profile', record['input'], vendor, device, output / 'profile.json'],
        output / 'profile.log',
        s.environment(vendor),
        timeout=7200,
    )
    stages = profile_summary(read(output / 'profile.json'), 1024)
    bits = {
        case + '_' + kind: s.digest(output / ('profile.json.gold_' + case + '_' + kind + '.bin'))
        for case in ('A', 'B')
        for kind in ('scores', 'ids')
    }
    r.require(bits == expected_bits, 'Normal/profile output bits differ')
    return stages, s.digest(output / 'profile.json')


def cpu(record, built, output, dependency, workers):
    s.execute(
        [
            sys.executable,
            '-u',
            ROOT / 'standalone/bm25s_benchmark.py',
            'measure',
            '--input',
            record['input'],
            '--built',
            built,
            '--output',
            output,
            '--dependency-dir',
            dependency,
            '--workers',
            workers,
            '--variant',
            'original_native',
            '--warmup',
            5,
            '--repeats',
            30,
        ],
        output.parent / (output.name + '.log'),
        timeout=7200,
    )
    measured = read(output / 'report.json')
    r.require(
        measured['status'] == 'actual_original_bm25s_large_search_checked'
        and measured['variant'] == 'original_native'
        and measured['index_id'] == record['index']['index_id']
        and measured['requested_query_workers'] == workers
        and measured['measured_cycles'] == 30
        and measured['warmup_cycles'] == 5,
        'Incomplete original-native CPU check',
    )
    return {
        'summary': measured['summary'],
        'report_sha256': s.digest(output / 'report.json'),
        'native_id_differences_from_strict': measured['native_id_differences_from_strict'],
        'directory': str(output),
    }


def compare_search(rows, reference, candidate):
    # Keep all paired changes, P95 and win counts; reuse the audited two-arm arithmetic.
    include_stages = all('stages' in row[reference] and 'stages' in row[candidate] for row in rows)
    mapped = [
        {
            'one': {
                label: {
                    'search': row[role]['summary'],
                    **({'stages': row[role]['stages']} if include_stages else {}),
                }
                for label, role in (('baseline', reference), ('dense', candidate))
            }
        }
        for row in rows
    ]
    return dense_selector.compare(mapped, ['one'], include_stages=include_stages)['one']


def quality_metrics(data, actual):
    mapping, groups, qrels = (read(data / file) for file in ('mapping.json', 'quality/qids.json', 'qrels.json'))
    predictions = {}
    for case, qids in groups.items():
        ids = array('i')
        ids.frombytes((actual / ('after_' + case) / 'top_ids.bin').read_bytes())
        r.require(
            sys.byteorder == 'little' and ids.itemsize == 4 and len(ids) == len(qids) * 10, 'Ranking layout differs'
        )
        r.require(all(0 <= value < len(mapping['doc_ids']) for value in ids), 'Ranking ID out of range')
        for offset, qid in enumerate(qids):
            if qid is not None:
                r.require(qid not in predictions, 'Duplicate query')
                predictions[qid] = [mapping['doc_ids'][value] for value in ids[offset * 10 : (offset + 1) * 10]]
    r.require(set(predictions) == set(qrels), 'Incomplete full-query quality coverage')
    return beir_real.ranking_metrics(predictions, qrels)


def summarize(workloads):
    result = {}
    for name, record in workloads.items():
        rows = record['per_round']
        r.require(len(rows) == 6, 'Incomplete fixed matrix')
        comparisons = {role: compare_search(rows, 'baseline', role) for role in ('dense_v1', 'split')}
        comparisons['split_vs_dense_v1'] = compare_search(rows, 'dense_v1', 'split')
        cpu_decision = decide(
            [{**row, 'npu': row['split']} for row in rows],
            [1, 8, 32],
            record['bm25s_construction']['strict_ids_match_project'],
            variants=('original_native',),
        )['original_native']
        record.update(comparison=comparisons, original_native_CPU_decision=cpu_decision)
        result[name] = {
            case: {
                'split_vs_baseline_percent': comparisons['split'][case]['median_change_percent'],
                'split_vs_baseline_P95_percent': comparisons['split'][case]['P95_change_percent'],
                'paired_changes_percent': comparisons['split'][case]['paired_change_percent'],
                'split_vs_dense_v1_percent': comparisons['split_vs_dense_v1'][case]['median_change_percent'],
                'split_TopK_ticks_change_percent': comparisons['split'][case].get('topk_ticks_change_percent'),
                'original_native_CPU': cpu_decision[case],
            }
            for case in ('A', 'B')
        }
    return result


def run(args):
    new, old = read(args.build / 'build.json'), read(args.reference_build / 'build.json')
    variants = check_builds(new, old)
    inputs = inputs_at(args.real_root, args.legacy_root)
    output = args.output.resolve()
    r.require(
        not output.exists() and output.is_relative_to(ROOT / 'results'), 'Fresh independent results directory required'
    )
    r.require(
        all(
            not output.is_relative_to(item['input']) and not item['input'].is_relative_to(output)
            for item in inputs.values()
        ),
        'Output overlaps input',
    )
    runner = args.host_build.resolve(strict=True) / 'bm25_large_search_benchmark'
    runner_sha = s.digest(runner)
    host = Path(new['host_build'])
    build_hashes = {
        str(path): s.digest(path) for path in (args.build / 'build.json', args.reference_build / 'build.json')
    }
    # Retain all diagnostic files. Include full-query quality, all three NPU arms,
    # CPU construction, and a 50% margin; every subprocess is sequential.
    disk = 1024**3
    plans = []
    for item in inputs.values():
        desc, index = item['description'], item['index']
        disk += (
            3 * 6 * 16 * desc['queries'] * desc['docs']
            + 8 * index['posting_count']
            + 8 * desc['queries'] * desc['docs']
        )
        plans.append(
            s.layout(
                desc['docs'],
                index['vocab_size'],
                index['posting_count'],
                desc['queries'],
                desc['k'],
                desc.get('capacity_words', desc['queries'] * 3),
            )
        )
        if 'real_data' in item:
            qdesc, qindex = corpus.validate(item['real_data'] / 'quality')
            disk += 3 * 16 * qdesc['queries'] * qdesc['docs']
            plans.append(
                s.layout(
                    qdesc['docs'],
                    qindex['vocab_size'],
                    qindex['posting_count'],
                    qdesc['queries'],
                    qdesc['k'],
                    qdesc['capacity_words'],
                )
            )
    r.require(disk * 1.5 < shutil.disk_usage(output.parent).free, 'Insufficient disk for the complete retained matrix')
    r.require(
        max(plan['cpu_peak_known_bytes'] for plan in plans) + 1024**3 < s.host_available_bytes() * 0.8,
        'Host budget exceeded',
    )
    output.mkdir(parents=True)
    report = {
        'status': 'running_concentrated_preindex_validation',
        'candidate_promoted': False,
        'optimization_round_limit': 1,
        'next_phase': 'freeze_selected_variant_then_IndexSDK_integration_and_postintegration_tests',
        'primary_CPU_baseline': 'frozen original-native BM25S; workers1/8/32; strict adapter excluded',
        'runner_sha256': runner_sha,
        'build_report_sha256': build_hashes,
        'source_lock_sha256': s.digest(ROOT / 'source_lock.json'),
        'manifest_sha256': s.digest(ROOT / 'package_manifest.json'),
        'profile_trace_contract': TRACE_CONTRACT,
        'exclusive_core_reservation_verified': False,
        'real_preparation_sha256': {
            name: s.digest(item['real_data'] / 'preparation.json')
            for name, item in inputs.items()
            if 'real_data' in item
        },
        'quality': {},
        'workloads': {},
        'scope': 'One fixed matrix. Profile only SciFact and the two sparse inputs; no tuning loop or automatic promotion. Full-query quality and Q8 performance remain separate.',
    }
    save = lambda: r.write_json(output / 'report.json', report)
    save()
    try:
        s.execute(
            [runner, '--memory', args.device, output / 'memory.json'],
            output / 'memory.log',
            s.environment(Path(variants['split']['normal']['aliases'])),
        )
        memory = read(output / 'memory.json')
        r.require(
            max(plan['known_device_bytes_upper_bound'] + plan['debug_score_bytes'] for plan in plans) + 1024**3
            < memory['free'] * 0.8,
            'Device budget exceeded',
        )
        report['memory_probe'] = memory
        for mode, variant in variants['split'].items():
            report['current_cell'] = {'stage': 'boundary', 'role': 'split', 'mode': mode}
            save()
            target = output / ('boundaries_split_' + mode)
            s.execute(
                [host / 'capacity_boundaries', variant['aliases'], args.device, target],
                output / (target.name + '.log'),
                s.environment(Path(variant['aliases'])),
            )
            check = read(target / 'report.json')
            r.require(
                check['status'] == 'actual_simd_selector_guards_and_strict_bit_rankings_passed'
                and check['calls'] == 54,
                'New package boundary checks incomplete',
            )
        for name, item in inputs.items():
            if 'real_data' not in item:
                continue
            path = item['real_data'] / 'quality'
            desc, index = corpus.validate(path)
            quality_input = {'input': path, 'description': desc, 'index': index}
            bits = []
            report['quality'][name] = {}
            for role in NPU_ROLES:
                report['current_cell'] = {'stage': 'full_query_quality', 'workload': name, 'role': role}
                save()
                target = output / 'quality' / name / role
                measured = search(
                    runner,
                    quality_input,
                    Path(variants[role]['normal']['aliases']),
                    target,
                    args.device,
                    warmup=1,
                    repeats=1,
                )
                bits.append(measured['TopK_sha256'])
                report['quality'][name][role] = {
                    'metrics': quality_metrics(item['real_data'], target / 'actual'),
                    'report_sha256': measured['report_sha256'],
                    'TopK_sha256': measured['TopK_sha256'],
                }
                save()
            r.require(all(value == bits[0] for value in bits), 'Full-query quality bits differ between NPU versions')
        for name, item in inputs.items():
            target = output / 'performance' / name
            target.mkdir(parents=True)
            built = target / 'bm25s_index'
            report['current_cell'] = {'stage': 'original_native_index_build', 'workload': name}
            save()
            s.execute(
                [
                    sys.executable,
                    '-u',
                    ROOT / 'standalone/bm25s_benchmark.py',
                    'build',
                    '--input',
                    item['input'],
                    '--built',
                    built,
                    '--output',
                    built,
                    '--dependency-dir',
                    args.dependency_dir,
                ],
                target / 'bm25s_build.log',
                timeout=14400,
            )
            construction = read(built / 'report.json')
            r.require(
                construction['status'] == 'original_bm25s_large_index_and_formula_checked'
                and construction['index_id'] == item['index']['index_id'],
                'CPU construction mismatch',
            )
            record = report['workloads'][name] = {
                'input': str(item['input']),
                'index_id': item['index']['index_id'],
                'preparation_sha256': item['preparation_sha256'],
                'bm25s_construction': construction,
                'per_round': [],
            }
            stable_bits = None
            for ordinal in range(1, 7):
                verify_unchanged_input(item)
                row = {'round': ordinal, 'process_order': process_order(ordinal)}
                record['per_round'].append(row)
                for role in row['process_order']:
                    directory = target / ('round' + str(ordinal)) / role
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    report['current_cell'] = {'workload': name, 'round': ordinal, 'role': role}
                    save()
                    if role in CPU_ROLES:
                        measured = cpu(item, built, directory, args.dependency_dir, int(role.rsplit('_t', 1)[1]))
                    else:
                        measured = search(
                            runner, item, Path(variants[role]['normal']['aliases']), directory, args.device
                        )
                        r.require(
                            stable_bits is None or measured['TopK_sha256'] == stable_bits,
                            'TopK bits differ across arms or rounds',
                        )
                        stable_bits = measured['TopK_sha256']
                        if item['description']['docs'] <= 65536:
                            stages, sha = profile(
                                item,
                                host,
                                Path(variants[role]['profile']['aliases']),
                                directory,
                                args.device,
                                stable_bits,
                            )
                            measured.update(stages=stages, profile_report_sha256=sha)
                    row[role] = measured
                    save()
                    print(
                        'PASS',
                        name,
                        ordinal,
                        role,
                        json.dumps({case: measured['summary'][case]['median_batch_ms'] for case in ('A', 'B')}),
                        flush=True,
                    )
            verify_unchanged_input(item, rehash=True)
        check_builds(new, old)
        r.require(
            s.digest(runner) == runner_sha and all(s.digest(Path(path)) == sha for path, sha in build_hashes.items()),
            'Runner/build report changed',
        )
        for item in inputs.values():
            if 'real_data' in item:
                beir_real.validate_data(
                    item['real_data'], read(item['real_data'] / 'preparation.json').get('dataset', 'scifact')
                )
        report.update(
            summary=summarize(report['workloads']),
            status='concentrated_preindex_checks_completed_selection_review_pending',
        )
        report.pop('current_cell', None)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        save()
    print('SUMMARY:', json.dumps(report['summary'], ensure_ascii=False), flush=True)
    print('REPORT:', output / 'report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('build')
    p.add_argument('--baseline-build', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    for name in ('build', 'reference-build', 'host-build', 'dependency-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--real-root', type=Path, default=ROOT / 'results')
    p.add_argument('--legacy-root', type=Path, default=ROOT.parent / 'bm25-npu')
    p.add_argument('--device', type=int, default=5)
    args = parser.parse_args()
    r.require(sys.platform == 'linux' and os.environ.get('ASCEND_OPP_PATH'), 'Use A5 Linux after sourcing CANN')
    locked_sources()
    lock = read(ROOT / 'source_lock.json')
    for group in ('experimental_dense_study', 'release_validation'):
        for name, sha in lock[group]['source_sha256'].items():
            r.require(s.digest(ROOT / name) == sha, 'Validation tool source changed: ' + name)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.mode == 'build':
        dense_selector.build(args, source_generator=split_sources, vendor_base='bm25_dense_split_1024')
    else:
        run(args)


if __name__ == '__main__':
    main()
