"""Materialize the frozen BM25 selection in an independent IndexSDK tree and test its public API."""

import argparse
import difflib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import beir_real, build_final
from standalone import final_support as s
from standalone import large_corpus as corpus
from standalone import release_validation as release
from standalone import resident_index as r
from standalone.selector_capacity import locked_sources, package_fingerprint, preset_vendor, read
from tools.split_sources import split_sources

BASE = '59213c4ca5ebb643c031c8b819e90e3a5cc21a8c'
PREFIX = Path('feature_retrieval/src/ascendfaiss')
FREEZE = ROOT / 'evidence/concentrated_preindex_a5_20261106.json'


def files_at(root):
    return {p.relative_to(root).as_posix(): s.digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def verify_tree(output):
    record = read(output / 'integration.json')
    sdk = output / 'sdk'
    r.require(files_at(sdk) == record['file_sha256'], 'Integrated source tree changed')
    r.require(s.digest(FREEZE) == record['freeze_evidence_sha256'], 'Selected release evidence changed')
    r.require(
        s.digest(output / 'preindex_report.json') == record['preindex_report_sha256'],
        'Frozen preintegration report changed',
    )
    r.require(
        s.digest(output / 'selected_build.json') == record['selected_build_sha256'],
        'Frozen selected-build report changed',
    )
    return record, sdk


def extract_archive(blob, destination):
    # Extract only tracked regular files/directories, never a path supplied by an archive link.
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:') as archive:
        for entry in archive:
            target = (destination / entry.name).resolve()
            r.require(target.is_relative_to(destination.resolve()), 'Archive path escapes the independent tree')
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                r.require(entry.isfile(), 'Unsupported archive link or special file')
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(entry) as stream:
                    target.write_bytes(stream.read())
                if os.name != 'nt':
                    target.chmod(entry.mode & 0o777)


def materialize(sdk, kernel_header=None):
    source = ROOT / PREFIX
    original_cmake = sdk / PREFIX / 'ascendhost/CMakeLists.txt'
    r.require(original_cmake.is_file(), 'Original IndexSDK ascendhost build entry missing')
    before = original_cmake.read_bytes()
    paths = []
    for folder in ('common/bm25', 'ascendhost/src/bm25', 'ops/ascendc/bm25', 'ut/cases/bm25'):
        paths += [p for p in (source / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [source / 'ascendhost/include/index/AscendIndexBm25.h']
    paths += list((source / 'ascendhost/include/impl').glob('Bm25*.h'))
    paths += [p for p in (source / 'test/ascendhost/bm25').rglob('*.h')]
    paths += [source / 'test/ascendhost/bm25/sdk_shared_check.cpp', source / 'test/ascendhost/bm25/sdk_benchmark.cpp']
    added = []
    for path in paths:
        relative = PREFIX / path.relative_to(source)
        target = sdk / relative
        r.require(not target.exists(), 'BM25 file collides with the original SDK: ' + str(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        added.append(relative.as_posix())
    host, kernel, _ = split_sources(False)
    for folder, text in (('op_host', host), ('op_kernel', kernel)):
        (sdk / PREFIX / 'ops/ascendc/bm25/fused' / folder / 'bm25_fused_score_top_k.cpp').write_bytes(text.encode())
    if kernel_header is not None:
        relative = PREFIX / 'ops/ascendc/bm25/fused/op_kernel/bm25_fused_score_top_k_tiling.h'
        r.require(not (sdk / relative).exists(), 'Captured kernel header collides with SDK source')
        (sdk / relative).write_bytes(kernel_header)
        added.append(relative.as_posix())
    cmake = sdk / PREFIX / 'ascendhost/cmake/Bm25.cmake'
    r.require(not cmake.exists(), 'BM25 module already exists')
    cmake.parent.mkdir(exist_ok=True)
    cmake.write_bytes((ROOT / 'integration/indexsdk/Bm25.cmake').read_bytes())
    added.append(cmake.relative_to(sdk).as_posix())
    original_cmake.write_bytes(
        before.rstrip(b'\r\n')
        + b'\n\n# Integrated sparse BM25; enable with -DBUILD_ASCEND_BM25=ON.\nINCLUDE(${CMAKE_CURRENT_LIST_DIR}/cmake/Bm25.cmake)\n'
    )
    patch = ''.join(
        difflib.unified_diff(
            before.decode().splitlines(True),
            original_cmake.read_text(encoding='utf-8').splitlines(True),
            fromfile='a/' + original_cmake.relative_to(sdk).as_posix(),
            tofile='b/' + original_cmake.relative_to(sdk).as_posix(),
        )
    )
    for name in sorted(added):
        text = (sdk / name).read_text(encoding='utf-8').splitlines(True)
        patch += ''.join(difflib.unified_diff([], text, fromfile='/dev/null', tofile='b/' + name))
    return added, patch


def prepare(args):
    frozen = read(FREEZE)
    r.require(frozen['selection']['variant'] == 'split', 'A frozen split selection is required')
    preindex = read(args.preindex_report)
    selected = read(args.selected_build / 'build.json')
    r.require(
        preindex['status'] == 'concentrated_preindex_checks_completed_selection_review_pending',
        'Completed concentrated matrix required',
    )
    r.require(
        preindex['runner_sha256'] == frozen['selection']['expected_standalone_runner_sha256'],
        'Different preintegration runner',
    )
    if 'run_snapshot' in frozen:
        for key in (
            'build_report_sha256',
            'source_lock_sha256',
            'manifest_sha256',
            'quality',
            'real_preparation_sha256',
        ):
            r.require(preindex[key] == frozen['run_snapshot'][key], 'Frozen report snapshot differs: ' + key)
    r.require(
        set(preindex['workloads']) == {'scifact', 'trec_covid', 'msmarco', 'sparse8192', 'sparse65536', 'synthetic10m'},
        'Incomplete concentrated workload matrix',
    )
    for item in preindex['workloads'].values():
        r.require(
            len(item['per_round']) == 6
            and all(set(release.NPU_ROLES + release.CPU_ROLES).issubset(row) for row in item['per_round']),
            'Incomplete concentrated rounds',
        )
    r.require(set(preindex['quality']) == {'scifact', 'trec_covid', 'msmarco'}, 'Missing full-query quality checks')
    for name, count in (('scifact', 300), ('trec_covid', 50), ('msmarco', 43)):
        quality = preindex['quality'][name]
        r.require(set(quality) == set(release.NPU_ROLES), 'Missing quality variant')
        r.require(
            all(
                value['metrics']['queries'] == count and value['TopK_sha256'] == quality['split']['TopK_sha256']
                for value in quality.values()
            ),
            'Full-query quality coverage/bits differ',
        )
    r.require(
        preindex['build_report_sha256'][str(args.selected_build / 'build.json')]
        == s.digest(args.selected_build / 'build.json'),
        'Selected package build differs from the measured build',
    )
    normal = selected['variants']['dense']['normal']
    r.require(
        normal['package_sha256'] == frozen['selection']['package']['package_sha256'],
        'Different measured package artifact',
    )
    for folder, sha_name in (('op_host', 'host_sha256'), ('op_kernel', 'kernel_sha256')):
        r.require(
            normal['source_sha256'][folder + '/bm25_fused_score_top_k.cpp']
            == frozen['selection']['generated_normal'][sha_name],
            'Selected normal source differs',
        )
    r.require(package_fingerprint(Path(normal['vendor'])) == normal['installed_files'], 'Measured package changed')
    header_sha = normal['source_sha256']['op_kernel/bm25_fused_score_top_k_tiling.h']
    headers = [
        path
        for path in Path(normal['vendor']).rglob('bm25_fused_score_top_k_tiling.h')
        if path.is_file() and s.digest(path) == header_sha
    ]
    r.require(len(headers) == 1, 'Measured CANN kernel tiling header missing or ambiguous')
    output = args.output.resolve()
    r.require(not output.exists() and output.is_relative_to(ROOT / 'results'), 'Fresh independent results required')
    sdk_source = args.sdk_source.resolve(strict=True)
    base = subprocess.check_output(['git', '-C', str(sdk_source), 'rev-parse', BASE], text=True).strip()
    r.require(base == BASE, 'Pinned SDK reference is unavailable')
    blob = subprocess.check_output(['git', '-C', str(sdk_source), 'archive', '--format=tar', BASE])
    output.mkdir(parents=True)
    sdk = output / 'sdk'
    sdk.mkdir()
    extract_archive(blob, sdk)
    original = files_at(sdk)
    added, patch = materialize(sdk, headers[0].read_bytes())
    actual = files_at(sdk)
    modified = [name for name, sha in original.items() if actual.get(name) != sha]
    r.require(modified == [(PREFIX / 'ascendhost/CMakeLists.txt').as_posix()], 'Unexpected upstream SDK edit')
    for mode, source_name in (('op_host', 'host_sha256'), ('op_kernel', 'kernel_sha256')):
        path = sdk / PREFIX / 'ops/ascendc/bm25/fused' / mode / 'bm25_fused_score_top_k.cpp'
        r.require(
            s.digest(path) == frozen['selection']['generated_normal'][source_name], 'Selected operator source mismatch'
        )
    (output / 'indexsdk-bm25.patch').write_bytes(patch.encode())
    record = {
        'status': 'frozen_BM25_integrated_into_independent_IndexSDK_source_build_pending',
        'base_commit': BASE,
        'freeze_evidence_sha256': s.digest(FREEZE),
        'selected_variant': 'split',
        'original_reference_modified': False,
        'modified_upstream_files': modified,
        'added_files': sorted(added),
        'file_sha256': actual,
        'patch_sha256': s.digest(output / 'indexsdk-bm25.patch'),
        'preindex_report_sha256': s.digest(args.preindex_report),
        'selected_build_sha256': s.digest(args.selected_build / 'build.json'),
        'selected_package_sha256': normal['package_sha256'],
    }
    (output / 'preindex_report.json').write_bytes(args.preindex_report.read_bytes())
    (output / 'selected_build.json').write_bytes((args.selected_build / 'build.json').read_bytes())
    r.write_json(output / 'integration.json', record)
    print('INTEGRATED SOURCE:', sdk, flush=True)
    print('REVIEW PATCH:', output / 'indexsdk-bm25.patch', flush=True)


def build(args):
    _, sdk = verify_tree(args.integration)
    output = args.output.resolve()
    r.require(
        not output.exists() and output.is_relative_to(ROOT / 'results'), 'Fresh integrated build directory required'
    )
    selected_build = read(args.selected_build / 'build.json')
    selected = selected_build['variants']['dense']['normal']
    r.require(
        selected['package_sha256'] == read(args.integration / 'integration.json')['selected_package_sha256'],
        'Different frozen package reference',
    )
    r.require(
        s.digest(args.selected_build / 'build.json')
        == read(args.integration / 'integration.json')['selected_build_sha256'],
        'Frozen build metadata changed',
    )
    r.require(package_fingerprint(Path(selected['vendor'])) == selected['installed_files'], 'Frozen package changed')
    merge = Path(selected_build['merge'])
    r.require(package_fingerprint(merge) == selected_build['merge_files'], 'Merge changed')
    output.mkdir(parents=True)
    result = {
        'status': 'building_integrated_IndexSDK',
        'integration': str(args.integration),
        'stages': [],
        'candidate_promoted': False,
    }

    def stage(name, command, env=None, cwd=None):
        result['current_stage'] = name
        r.write_json(output / 'build.json', result)
        s.execute(command, output / (name + '.log'), env, cwd=cwd, timeout=14400)
        result['stages'].append(name)

    try:
        project = output / 'operator_project'
        build_final.prepare(args.template, project, 'fused')
        for folder in ('op_host', 'op_kernel'):
            for source in (sdk / PREFIX / 'ops/ascendc/bm25/fused' / folder).iterdir():
                if source.suffix in ('.cpp', '.h'):
                    (project / folder / source.name).write_bytes(source.read_bytes())
        preset_vendor(project / 'CMakePresets.json', 'bm25_indexsdk_split_1024')
        for folder in ('op_host', 'op_kernel'):
            for name in ('bm25_fused_score_top_k.cpp', 'bm25_fused_score_top_k_tiling.h'):
                expected = selected['source_sha256'][folder + '/' + name]
                r.require(s.digest(project / folder / name) == expected, 'Merged CANN source/header differs')
        operator_sources = {
            path.relative_to(project).as_posix(): s.digest(path)
            for path in project.rglob('*')
            if path.is_file() and path.suffix in ('.cpp', '.h')
        }
        r.write_json(
            project / 'final_source.json',
            {
                'kind': 'fused',
                'vendor': 'bm25_indexsdk_split_1024',
                'frozen_package_reference': selected['package_sha256'],
                'source_sha256': operator_sources,
            },
        )
        result['operator_source_sha256'] = operator_sources
        stage('operator_build', ['bash', 'build.sh'], cwd=project)
        packages = list((project / 'build_out').glob('*.run'))
        r.require(len(packages) == 1, 'Expected one actual merged-source operator package')
        install = output / 'operator_install'
        install.mkdir()
        stage('operator_install', ['bash', packages[0], '--install-path=' + str(install)])
        vendor = install / 'vendors/bm25_indexsdk_split_1024'
        aliases = output / 'runtime/vendors'
        s.aliases(aliases, vendor, merge)
        result.update(
            vendors=str(aliases),
            fused_vendor=str(vendor),
            merge=str(merge),
            fused_files=package_fingerprint(vendor),
            merge_files=package_fingerprint(merge),
        )
        host = output / 'host'
        command = [
            'cmake',
            '-S',
            sdk / PREFIX / 'ascendhost',
            '-B',
            host,
            '-DBUILD_ASCEND_SEARCH=OFF',
            '-DBUILD_OCK=OFF',
            '-DBUILD_ASCEND_BM25=ON',
            '-DBM25_BUILD_INTEGRATION_TESTS=ON',
            '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DFAISS_HOME=' + str(args.faiss_home),
            '-DASCEND_HOME=/usr/local/Ascend',
            '-DASCEND_TOOLKIT_PATH=' + str(args.toolkit_home),
            '-DBM25_VENDOR_DIR=' + str(aliases),
            '-DCMAKE_SHARED_LINKER_FLAGS=-fopenmp -Wl,--no-undefined',
            '-DCMAKE_CXX_STANDARD_LIBRARIES:STRING=' + str(args.openblas),
        ]
        stage('configure', command)
        stage('host_build', ['cmake', '--build', host, '--parallel', args.jobs])
        stage('unit_tests', ['ctest', '--test-dir', host, '--output-on-failure'])
        stage(
            'clang_tidy',
            [
                sys.executable,
                ROOT / 'tools/run_clang_tidy.py',
                '--build-dir',
                host,
                '--source-root',
                sdk / PREFIX,
                '--include-tests',
                '--clang-tidy',
                args.clang_tidy,
            ],
        )
        prefix = output / 'installed'
        stage('install', ['cmake', '--install', host, '--prefix', prefix])
        relocated = output / 'relocated'
        shutil.copytree(prefix, relocated)
        consumer = output / 'consumer'
        consumer.mkdir()
        (consumer / 'main.cpp').write_bytes(
            b'#include "faiss/ascend/AscendIndexBm25.h"\n#include "common/ErrorCode.h"\nint main() { faiss::ascend::AscendIndexBm25 model; return model.Close() == ascend::APP_ERR_OK ? 0 : 1; }\n'
        )
        (consumer / 'CMakeLists.txt').write_bytes(
            b'cmake_minimum_required(VERSION 3.17)\nproject(installed_bm25 LANGUAGES CXX)\nset(CMAKE_CXX_STANDARD 17)\nadd_executable(consumer main.cpp)\ntarget_include_directories(consumer PRIVATE "${SDK_PREFIX}/include" "${SDK_PREFIX}/include/faiss/ascend" "${TOOLKIT_HOME}/include")\ntarget_link_libraries(consumer PRIVATE "${SDK_PREFIX}/host/lib/libascendfaiss.so")\n'
        )
        stage(
            'consumer_configure',
            [
                'cmake',
                '-S',
                consumer,
                '-B',
                consumer / 'build',
                '-DSDK_PREFIX=' + str(relocated),
                '-DTOOLKIT_HOME=' + str(args.toolkit_home),
            ],
        )
        stage('consumer_build', ['cmake', '--build', consumer / 'build'])
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(relocated / 'host/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
        stage('consumer_run', [consumer / 'build/consumer'], env)
        symbols = subprocess.check_output(['nm', '-D', '-C', str(host / 'libascendfaiss.so')], text=True)
        r.require(
            all('AscendIndexBm25::' + name in symbols for name in ('Init(', 'Search(', 'Close(', 'GetLastError(')),
            'Public API is not exported by the real SDK library',
        )
        (output / 'exported_symbols.log').write_bytes(symbols.encode())
        verify_tree(args.integration)
        result.update(
            status='integrated_IndexSDK_build_static_UT_install_passed_device_pending',
            host=str(host),
            toolkit_home=str(args.toolkit_home),
            sdk_source=str(sdk),
            library_sha256=s.digest(host / 'libascendfaiss.so'),
            benchmark_sha256=s.digest(host / 'bm25_sdk_benchmark'),
            shared_check_sha256=s.digest(host / 'bm25_sdk_shared_check'),
        )
        result.pop('current_stage', None)
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        r.write_json(output / 'build.json', result)
    print('SDK BUILD REPORT:', output / 'build.json', flush=True)


def verify_sdk_output(path, description, device, warmup, repeats):
    record = read(path / 'report.json')
    r.require(
        record['status'] == 'actual_integrated_SDK_public_Search_and_shared_context_checked'
        and record['SDK_context_device_stream_survived_Close'] is True,
        'SDK public-call/lifetime check incomplete',
    )
    for key, value in (
        ('device', device),
        ('queries', description['queries']),
        ('k', description['k']),
        ('warmup', warmup),
        ('repeats', repeats),
    ):
        r.require(record.get(key) == value, 'SDK report contract differs: ' + key)
    samples = record['samples']
    r.require(len(samples) == 2 * repeats, 'Incomplete SDK samples')
    for i, item in enumerate(samples):
        cycle, slot = divmod(i, 2)
        r.require(
            item['cycle'] == cycle and item['case'] == ('B' if slot ^ (cycle & 1) else 'A'), 'SDK sample order differs'
        )
    result = {
        'summary': {
            case: s.summarize(
                [item['call_wall_ms'] for item in samples if item['case'] == case], description['queries']
            )
            for case in ('A', 'B')
        },
        'report_sha256': s.digest(path / 'report.json'),
    }
    for case in ('A', 'B'):
        ids = array('q')
        ids.frombytes((path / (case + '_ids.bin')).read_bytes())
        scores = array('f')
        scores.frombytes((path / (case + '_scores.bin')).read_bytes())
        r.require(
            sys.byteorder == 'little'
            and ids.itemsize == 8
            and scores.itemsize == 4
            and len(ids) == len(scores) == description['queries'] * description['k'],
            'Public INT64/FP32 result shape differs',
        )
        r.require(
            all(0 <= value < description['docs'] for value in ids)
            and all(math.isfinite(value) and value >= 0 for value in scores),
            'Invalid public SDK IDs/scores',
        )
    return result


def sdk_quality(data, output, frozen):
    groups, mapping, qrels = (read(data / name) for name in ('quality/qids.json', 'mapping.json', 'qrels.json'))
    predictions = {}
    for case, qids in groups.items():
        ids = array('q')
        ids.frombytes((output / (case + '_ids.bin')).read_bytes())
        r.require(
            len(ids) == len(qids) * 10
            and all(0 <= value < len(mapping['doc_ids']) and value <= 2**31 - 1 for value in ids),
            'Invalid public quality IDs',
        )
        id_bits = r.digest(array('i', ids).tobytes())
        r.require(
            id_bits == frozen['TopK_sha256'][case + '_ids']
            and s.digest(output / (case + '_scores.bin')) == frozen['TopK_sha256'][case + '_scores'],
            'Postintegration quality bits differ from frozen selected version',
        )
        for i, qid in enumerate(qids):
            if qid is not None:
                r.require(qid not in predictions, 'Duplicate SDK quality query')
                predictions[qid] = [mapping['doc_ids'][value] for value in ids[i * 10 : (i + 1) * 10]]
    r.require(set(predictions) == set(qrels), 'Incomplete public SDK quality coverage')
    metrics = beir_real.ranking_metrics(predictions, qrels)
    r.require(metrics == frozen['metrics'], 'Postintegration quality metrics changed')
    return metrics


def device(args):
    build_record = read(args.build / 'build.json')
    r.require(
        build_record['status'] == 'integrated_IndexSDK_build_static_UT_install_passed_device_pending',
        'Complete SDK build/install/static gates required',
    )
    verify_tree(Path(build_record['integration']))
    host = Path(build_record['host'])
    aliases = Path(build_record['vendors'])
    r.require(
        s.digest(host / 'libascendfaiss.so') == build_record['library_sha256']
        and s.digest(host / 'bm25_sdk_benchmark') == build_record['benchmark_sha256']
        and s.digest(host / 'bm25_sdk_shared_check') == build_record['shared_check_sha256'],
        'Integrated SDK binary changed',
    )
    r.require(
        package_fingerprint(Path(build_record['fused_vendor'])) == build_record['fused_files']
        and package_fingerprint(Path(build_record['merge'])) == build_record['merge_files'],
        'Integrated operator package changed',
    )
    inputs = release.inputs_at(args.real_root, args.legacy_root)
    frozen = read(Path(build_record['integration']) / 'preindex_report.json')
    for name, item in inputs.items():
        prior = frozen['workloads'][name]
        r.require(
            item['index']['index_id'] == prior['index_id']
            and item['preparation_sha256'] == prior['preparation_sha256'],
            'Postintegration input differs from the frozen matrix: ' + name,
        )
    r.require(
        s.digest(args.host_build / 'bm25_large_search_benchmark') == frozen['runner_sha256'],
        'Different preintegration standalone runner',
    )
    output = args.output.resolve()
    r.require(not output.exists() and output.is_relative_to(ROOT / 'results'), 'Fresh SDK device output required')
    disk = 1024**3 + sum(
        6 * 16 * item['description']['queries'] * item['description']['docs'] for item in inputs.values()
    )
    r.require(disk * 1.5 < shutil.disk_usage(output.parent).free, 'Insufficient retained diagnostic-output disk space')
    output.mkdir(parents=True)
    env = s.environment(aliases)
    env['MX_INDEX_USE_ONLINEOP'] = '1'
    env['LD_LIBRARY_PATH'] = (
        str(host) + ':' + str(Path(build_record['toolkit_home']) / 'lib64') + ':' + env.get('LD_LIBRARY_PATH', '')
    )
    report = {
        'status': 'running_postintegration_public_SDK_checks',
        'quality': {},
        'workloads': {},
        'candidate_promoted': False,
        'SDK_library_sha256': build_record['library_sha256'],
        'SDK_benchmark_sha256': build_record['benchmark_sha256'],
        'freeze_evidence_sha256': s.digest(FREEZE),
        'CPU_scope': 'Original-native CPU measurements are frozen in the concentrated preintegration record, not re-timed in this API-overhead comparison.',
    }
    save = lambda: r.write_json(output / 'report.json', report)
    save()
    try:
        linked = subprocess.check_output(['ldd', str(host / 'bm25_sdk_benchmark')], env=env, text=True)
        r.require(str(host / 'libascendfaiss.so') in linked, 'SDK benchmark resolved a different libascendfaiss')
        (output / 'linked_library.log').write_bytes(linked.encode())
        s.execute(
            [host / 'bm25_sdk_shared_check', inputs['scifact']['input'], aliases, args.device, output / 'shared.json'],
            output / 'shared.log',
            env,
            timeout=7200,
        )
        shared = read(output / 'shared.json')
        r.require(
            shared['status'] == 'shared_SDK_context_oracle_retry_lifetime_and_stream_passed'
            and shared['init_close_cycles'] == 2,
            'Shared-resource lifecycle incomplete',
        )
        for name, item in inputs.items():
            if 'real_data' in item:
                path = item['real_data'] / 'quality'
                desc, _ = corpus.validate(path)
                target = output / 'quality' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                s.execute(
                    [host / 'bm25_sdk_benchmark', path, aliases, args.device, target, 1, 10],
                    target.parent / (name + '.log'),
                    env,
                    timeout=7200,
                )
                result = verify_sdk_output(target, desc, args.device, 1, 10)
                result['metrics'] = sdk_quality(item['real_data'], target, frozen['quality'][name]['split'])
                report['quality'][name] = result
                save()
            rows = report['workloads'][name] = []
            for ordinal in range(1, 7):
                release.verify_unchanged_input(item)
                order = ('sdk', 'standalone') if ordinal % 2 else ('standalone', 'sdk')
                row = {'round': ordinal, 'process_order': list(order)}
                rows.append(row)
                for role in order:
                    target = output / 'performance' / name / ('round' + str(ordinal)) / role
                    target.parent.mkdir(parents=True, exist_ok=True)
                    report['current_cell'] = {'workload': name, 'round': ordinal, 'role': role}
                    save()
                    if role == 'sdk':
                        s.execute(
                            [host / 'bm25_sdk_benchmark', item['input'], aliases, args.device, target, 5, 30],
                            target.parent / 'sdk.log',
                            env,
                            timeout=7200,
                        )
                        measured = verify_sdk_output(target, item['description'], args.device, 5, 30)
                    else:
                        measured = release.search(
                            args.host_build / 'bm25_large_search_benchmark', item, aliases, target, args.device
                        )
                    row[role] = measured
                    save()
                for case in ('A', 'B'):
                    ids = array('q')
                    ids.frombytes((target.parent / 'sdk' / (case + '_ids.bin')).read_bytes())
                    expected = array('i')
                    expected.frombytes((item['input'] / case / 'expected_top_ids.bin').read_bytes())
                    r.require(ids.tolist() == expected.tolist(), 'Integrated public IDs differ from strict oracle')
                    r.require(
                        (target.parent / 'sdk' / (case + '_scores.bin')).read_bytes()
                        == (target.parent / 'standalone/actual' / ('before_' + case) / 'top_scores.bin').read_bytes(),
                        'SDK/standalone selected FP32 bits differ',
                    )
            release.verify_unchanged_input(item, rehash=True)
        verify_tree(Path(build_record['integration']))
        for item in inputs.values():
            if 'real_data' in item:
                beir_real.validate_data(
                    item['real_data'], read(item['real_data'] / 'preparation.json').get('dataset', 'scifact')
                )
        r.require(
            s.digest(host / 'libascendfaiss.so') == build_record['library_sha256']
            and s.digest(host / 'bm25_sdk_benchmark') == build_record['benchmark_sha256'],
            'SDK library/benchmark mutated',
        )
        r.require(
            package_fingerprint(Path(build_record['fused_vendor'])) == build_record['fused_files']
            and package_fingerprint(Path(build_record['merge'])) == build_record['merge_files'],
            'Integrated package changed during device checks',
        )
        report['comparison'] = {
            name: release.compare_search(rows, 'standalone', 'sdk') for name, rows in report['workloads'].items()
        }
        report.update(status='postintegration_actual_public_SDK_quality_and_six_round_Search_checked')
        report.pop('current_cell', None)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        save()
    print('SDK COMPARISON:', json.dumps(report['comparison'], ensure_ascii=False), flush=True)
    print('SDK REPORT:', output / 'report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--sdk-source', type=Path, required=True)
    p.add_argument('--preindex-report', type=Path, required=True)
    p.add_argument('--selected-build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('build')
    for name in (
        'integration',
        'selected-build',
        'template',
        'output',
        'toolkit-home',
        'faiss-home',
        'openblas',
        'clang-tidy',
    ):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--jobs', type=int, default=8)
    p = sub.add_parser('device')
    for name in ('build', 'host-build', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--real-root', type=Path, default=ROOT / 'results')
    p.add_argument('--legacy-root', type=Path, default=ROOT.parent / 'bm25-npu')
    p.add_argument('--device', type=int, default=5)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    lock = read(ROOT / 'source_lock.json')
    locked_sources()
    for name, sha in lock['indexsdk_integration']['source_sha256'].items():
        r.require(s.digest(ROOT / name) == sha, 'Integration source changed: ' + name)
    if args.mode != 'prepare':
        r.require(
            sys.platform == 'linux' and os.environ.get('ASCEND_OPP_PATH'), 'Actual A5 Linux/CANN environment required'
        )
    {'prepare': prepare, 'build': build, 'device': device}[args.mode](args)


if __name__ == '__main__':
    main()
