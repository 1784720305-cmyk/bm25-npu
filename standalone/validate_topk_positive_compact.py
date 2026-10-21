"""独立正分压缩Top-K的集中设备验收；不修改旧流程，不测性能，失败即停。"""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import validate_topk_radix_collect as old
from standalone.create_topk_positive_compact import SOURCES, VENDOR
from standalone.prepare_topk_parallel_runtime import prototypes
from standalone.topk_fixture import write_input, check
from standalone.validate_topk_merge import digest, write_json

HOST_SOURCES = ('npu_prototype/topk_positive_compact_host/CMakeLists.txt',
                'npu_prototype/topk_positive_compact_host/positive_compact_runner.cpp',
                'npu_prototype/topk_radix_host/radix_host_common.h')
TOOLS = ('standalone/validate_topk_positive_compact.py', 'standalone/validate_topk_radix_collect.py',
         'standalone/create_topk_positive_compact.py', 'standalone/create_project.py',
         'standalone/prepare_topk_parallel_runtime.py', 'standalone/topk_fixture.py',
         'standalone/validate_topk_merge.py', 'standalone/accumulate_fixture.py', 'check_result.py')
STATUS = 'positive_compact_all_37_cases_device_and_full_sort_passed'
BOUNDARIES = tuple((65, docs, k) for docs in (1024, 1025, 8192, 8193) for k in (1, 10, 32))


def host_hashes():
    return {name: digest(ROOT/name) for name in HOST_SOURCES}


def validate_install(vendor, project):
    vendor, project = vendor.resolve(strict=True), project.resolve(strict=True)
    info = json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    required = {'operator': 'topk_positive_compact', 'vendor_name': VENDOR, 'compute_unit': 'ascend950',
                'positive_capacity': 1024, 'compaction_min_docs': 1025, 'membership_mask_max_docs': 8192,
                'selection_rule': 'FP32_score_desc_original_ID_asc_signed_zero_equal'}
    if vendor.name != VENDOR or any(info.get(key) != value for key, value in required.items()):
        raise ValueError('需要独立950正分压缩工程、容量/规则及实际安装vendor')
    for name in SOURCES:
        if digest(project/name) != info['source_sha256'][name] or digest(project/name) != digest(ROOT/'npu_prototype/topk_positive_compact'/name):
            raise ValueError('当前源码、生成工程和工程指纹不同：'+name)
    header = vendor/'op_api/include/aclnn_bm25_top_k.h'
    # Both declarations were read from this new project's actual A5 build log, not inferred from IR.
    if prototypes(header) != old.PROTOTYPES:
        raise ValueError('实际安装头文件不匹配已读到的两阶段声明')
    library = (vendor/'op_api/lib/libcust_opapi.so').resolve(strict=True)
    for path in (header, library):
        copies = list((project/'build_out').rglob(path.name))
        if not path.stat().st_size or not any(digest(p) == digest(path) for p in copies):
            raise ValueError('安装头文件/API库没有相同字节的实际打包产物：'+path.name)
    folder = vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_top_k'
    objects = sorted(folder.glob('*.o'))
    if not objects:
        raise ValueError('缺少实际ascend950设备目标文件')
    hashes = {}
    for obj in objects:
        copies = list((project/'build_out').rglob(obj.name))
        if not copies or any(digest(p) != digest(obj) for p in copies):
            raise ValueError('安装设备目标文件与实际构建产物不同：'+obj.name)
        hashes[str(obj.relative_to(vendor))] = digest(obj)
    return {'api_library': str(library), 'api_sha256': digest(library), 'header_sha256': digest(header),
            'project_source_sha256': info['source_sha256'], 'installed_device_object_sha256': hashes,
            'project_metadata_sha256': digest(project/'bm25_project.json')}


def boundary_rows(docs, k, alternate):
    """65 rows also reuse cores; alternating rows change compact/overflow/zero branches."""
    counts = (0, 1, k-1, k, k+1, 31, 32, 33, 730, 1023, 1024, 1025, docs)
    rows, labels = [], []
    for query in range(65):
        case = (query + (9 if alternate else 0)) % 16
        row = [-0.0 if (doc+query) % 2 else 0.0 for doc in range(docs)]
        if case < len(counts):
            positive = min(docs, counts[case])
            ids = list(range(docs))
            random.Random(95021+docs+query+(9999 if alternate else 0)).shuffle(ids)
            for rank, doc in enumerate(ids[:positive]):
                # Adjacent floats, subnormals, maximum finite, and many equal scores.
                bits = (1, 2, 0x3f800000, 0x3f800001, 0x7f7fffff)[rank % 5]
                row[doc] = old.from_bits(bits)
            label = 'positive_count_'+str(positive)
        elif case == 13:
            row[1] = row[128] = 1.0
            label = 'original_ID_1_beats_compaction_position_for_ID_128'
        elif case == 14:
            # High member-mask bits and lane tails, including 31/32/63 when present.
            for doc in (0, 127, 128, 3968, 4096, 4095, 4223, 8064, 8191):
                if doc < docs:
                    row[doc] = old.from_bits(1)
            label = 'membership_bits_and_thread_tail'
        else:
            # All equal; rows in the eligible interval must take overflow/full-scan path.
            row = [1.0]*docs
            label = 'dense_equal_overflow'
        rows.extend(row)
        labels.append(label)
    return rows, labels


def prepare_boundary(directory, q, docs, k):
    if q != 65:
        raise ValueError('集中边界套件要求65行')
    labels = {}
    for name, alternate in (('input_a', False), ('input_b', True)):
        rows, labels[name] = boundary_rows(docs, k, alternate)
        write_input(rows, q, docs, k, directory/name, 'POSITIVE_COMPACT_UNIT_ONLY',
                    '1024 capacity, row limits, zero fill, original ID and raw FP32 bits')
    write_json(directory/'row_cases.json', labels)


def validate_record(record, q, docs, k, device, calls):
    expected = {'status': 'positive_compact_device_calls_and_cpp_full_sort_passed',
                'queries': q, 'docs': docs, 'k': k, 'device': device, 'validated_calls': calls,
                'declared_user_gm_scratch_bytes': 0, 'declared_ub_algorithm_bytes': 16704,
                'input_calls': [(calls+1)//2, calls//2], 'every_call_poisoned': True,
                'every_call_full_sort_checked': True, 'host_source_sha256': host_hashes()}
    if (set(record) != set(expected) | {'workspace_bytes'} or
            type(record.get('workspace_bytes')) is not int or not 0 <= record['workspace_bytes'] <= 2**63-1 or
            any(type(record.get(key)) is not type(value) or record.get(key) != value for key, value in expected.items()) or
            any(type(n) is not int for n in record['input_calls'])):
        raise ValueError('设备记录缺失、尺寸/覆写/重复/工作区不符，或调用器未按当前源码编译')


def run(output, runner, vendor, project, device, calls):
    output, runner, vendor, project = (p.resolve() for p in (output, runner, vendor, project))
    if output.exists() or not runner.is_file():
        raise ValueError('结果目录须为新目录；请先编译独立调用器')
    if type(device) is not int or not 0 <= device <= 2147483647 or type(calls) is not int or not 5 <= calls <= 100:
        raise ValueError('设备号须为非负int32，交替调用次数须为5..100')
    installation = validate_install(vendor, project)
    runner_sha = digest(runner)
    tools = {name: digest(ROOT/name) for name in TOOLS+HOST_SOURCES}
    environment = os.environ.copy()
    environment['ASCEND_CUSTOM_OPP_PATH'] = str(vendor)
    # libcust_opapi is opened by absolute path; competing vendor registrations are excluded.
    output.mkdir(parents=True)
    report = {'status': 'running', 'device': device, 'calls_per_case': calls,
              'runner_sha256': runner_sha, 'tool_source_sha256': tools, 'installation': installation,
              'cases': [], 'scope': 'isolated FP32 Top-K correctness; no Search or performance claim'}
    path = output/'report.json'
    write_json(path, report)
    try:
        old.execute([str(runner), '--check-library', installation['api_library']], output/'library.log', environment)
        specs = [('regression', spec) for spec in old.SPECS]+[('capacity', spec) for spec in BOUNDARIES]
        for index, (kind, (q, docs, k)) in enumerate(specs):
            directory = output/f'case{index}_{kind}_q{q}_d{docs}_k{k}'
            directory.mkdir()
            actual = directory/'actual'
            item = {'kind': kind, 'queries': q, 'docs': docs, 'k': k, 'status': 'running'}
            report['cases'].append(item)
            write_json(path, report)
            with redirect_stdout(io.StringIO()):
                (old.prepare_pair if kind == 'regression' else prepare_boundary)(directory, q, docs, k)
            input_names = [str(p.relative_to(directory)).replace('\\', '/') for p in directory.rglob('*') if p.is_file()]
            before = {name: digest(directory/name) for name in input_names}
            command = [str(runner), str(directory), str(actual), str(q), str(docs), str(k), str(device),
                       installation['api_library'], str(calls)]
            item.update(command=command, input_sha256=before)
            write_json(path, report)
            old.execute(command, directory/'device.log', environment)
            if (validate_install(vendor, project) != installation or digest(runner) != runner_sha or
                    any(digest(ROOT/name) != value for name, value in tools.items()) or
                    any(digest(directory/name) != value for name, value in before.items())):
                raise ValueError('验收过程中源码、安装文件、调用器或输入发生改变')
            record = json.loads((actual/'device_validation.json').read_text(encoding='utf-8'))
            validate_record(record, q, docs, k, device, calls)
            with (directory/'check.log').open('w', encoding='utf-8', newline='\n') as log:
                with redirect_stdout(log):
                    for variant in ('a', 'b'):
                        check(directory/f'input_{variant}', actual/f'{variant}_top_scores.bin', actual/f'{variant}_top_ids.bin')
            item.update(status='passed', validation=record,
                        output_sha256={name: digest(actual/name) for name in
                                       ('a_top_scores.bin', 'a_top_ids.bin', 'b_top_scores.bin', 'b_top_ids.bin')},
                        device_log_sha256=digest(directory/'device.log'), check_log_sha256=digest(directory/'check.log'))
            write_json(path, report)
            print(f'通过{index+1}/{len(specs)}：{kind} Q={q} D={docs} K={k}；{calls}次实际交替、poison及全部编号/原位核对。', flush=True)
        report['status'] = STATUS
        write_json(path, report)
        print(f'37组独立设备验收全部通过：{path}。尚未接入完整Search，不作性能结论。', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error) or type(error).__name__
        if report['cases'] and report['cases'][-1]['status'] == 'running':
            report['cases'][-1]['status'] = report['status']
        write_json(path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runner', default=str(ROOT/'results/topk_positive_compact_host_build/bm25_topk_positive_compact_runner'))
    parser.add_argument('--vendor', default=str(ROOT/'results/custom_opp/vendors'/VENDOR))
    parser.add_argument('--project', default=str(ROOT/'results/Bm25TopKPositiveCompact'))
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--calls', type=int, default=5)
    args = parser.parse_args()
    try:
        if sys.platform != 'linux':
            raise ValueError('实际设备验收请在A5 Linux执行')
        run(Path(args.output), Path(args.runner), Path(args.vendor), Path(args.project), args.device, args.calls)
    except KeyboardInterrupt:
        print('验收中断，保留日志；不生成CPU替代输出。', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f'FAILED：{error}；停止，不生成CPU替代输出或通过报告。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
