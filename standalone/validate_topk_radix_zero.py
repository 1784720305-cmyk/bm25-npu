"""独立Radix Top-K实机验收：交替输入、输出poison、逐次完整排序核对；失败即停。"""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.prepare_topk_parallel_runtime import prototypes
from standalone.topk_fixture import write_input, check
from standalone.validate_topk_merge import digest, write_json

VARIANT = 'exact_radix_zero_aggregate_11_11_10_deterministic_ties_v1'
# Retain all 13 Warp dimension cases, then exercise the 128-thread tile boundaries.
SPECS = ((6, 6, 1), (6, 6, 3), (6, 6, 6), (65, 32, 32), (513, 40, 32),
         (65, 513, 1), (65, 513, 32), (65, 1025, 10), (65, 1025, 32),
         (65, 2048, 10), (65, 2049, 32), (65, 4097, 32), (1, 65536, 10),
         (65, 127, 10), (65, 128, 32), (65, 129, 10), (65, 257, 10),
         (65, 1025, 1), (1, 65536, 32), (65, 8192, 10),
         (65, 8192, 32), (3, 65536, 32), (257, 129, 10))
SMOKE_SPECS = ((6, 6, 3), (65, 129, 10))
SOURCE_FILES = ('op_host/bm25_top_k.cpp', 'op_host/bm25_top_k_tiling.h', 'op_kernel/bm25_top_k.cpp')
HOST_SOURCE_PATHS = ('npu_prototype/topk_radix_zero_host/CMakeLists.txt',
                     'npu_prototype/topk_radix_host/radix_host_common.h',
                     'npu_prototype/topk_radix_host/radix_runner.cpp')
# Verbatim signatures supplied from the ACTUAL installed Radix header, 2026-10-16.
PROTOTYPES = {
    'aclnnBm25TopKGetWorkspaceSize': 'constaclTensor*scores,int64_tk,constaclTensor*topScoresOut,constaclTensor*topIdsOut,uint64_t*workspaceSize,aclOpExecutor**executor',
    'aclnnBm25TopK': 'void*workspace,uint64_tworkspaceSize,aclOpExecutor*executor,aclrtStreamstream'}


def dimensions(q, docs, k):
    if any(type(n) is not int for n in (q, docs, k)) or not (
            q >= 1 and docs >= 1 and q*docs <= 2147483647 and 1 <= k <= min(docs, 32)):
        raise ValueError('诊断尺寸非法：Q*D须在INT32范围，K须在1..min(D,32)')


def validate_install(vendor, project):
    vendor, project = vendor.resolve(strict=True), project.resolve(strict=True)
    info = json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    if info['operator'] != 'topk_radix_zero' or info['vendor_name'] != 'bm25_topk_radix_zero' or vendor.name != 'bm25_topk_radix_zero':
        raise ValueError('需要独立topk_radix_zero工程和实际安装vendor')
    for name in SOURCE_FILES:
        actual = digest(project/name)
        if actual != info['source_sha256'][name] or actual != digest(ROOT/'npu_prototype/topk_radix_zero'/name):
            raise ValueError(f'工程与当前包的Radix源码不一致：{name}')
    header = vendor/'op_api/include/aclnn_bm25_top_k.h'
    if prototypes(header) != PROTOTYPES:
        raise ValueError('实际Radix头文件声明与已核对的接口不同')
    library = (vendor/'op_api/lib/libcust_opapi.so').resolve(strict=True)
    if library.stat().st_size == 0:
        raise ValueError('安装API库为空')
    # Match installed device objects to actual build outputs, not just the API header/library.
    installed = vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_top_k'
    objects = sorted(installed.glob('*.o'))
    if not objects:
        raise ValueError('安装目录没有实际ascend950 Top-K设备目标文件')
    hashes = {}
    for obj in objects:
        matches = list((project/'build_out').rglob(obj.name))
        actual = digest(obj)
        if not matches or any(digest(p) != actual for p in matches):
            raise ValueError('安装设备目标文件与工程构建产物不一致：'+obj.name)
        hashes[str(obj.relative_to(vendor))] = actual
    return {'api_library': str(library), 'api_sha256': digest(library), 'header_sha256': digest(header),
            'project_source_sha256': info['source_sha256'], 'installed_device_object_sha256': hashes,
            'scope': 'source/header/API/device-object file checks; not device execution evidence'}


def prepare_pair(directory, q, docs, k):
    dimensions(q, docs, k)
    rng = random.Random(950+docs*100+k)
    rows = []
    for query in range(q):
        case = query % 16
        if case == 0:
            row = [0.]*docs
        elif case == 1:
            row = [1.]*docs
        elif case == 2:
            row = [float(i) for i in range(docs)]
        elif case == 3:
            row = [float(docs-i) for i in range(docs)]
        elif case == 4:
            row = [float(rng.randrange(9)) for _ in range(docs)]
        elif case == 5:
            row = [0.]*docs
            for i in {0, min(31,docs-1), docs//2, docs-1}:
                row[i] = 99. # Fewer than K positives for K=10/32; many boundary-zero members.
        elif case == 6:
            row = [from_bits(0x3f800001 if i % 2 else 0x3f800000) for i in range(docs)]
        elif case == 7:
            row = [-0. if i % 2 else 0. for i in range(docs)]
        elif case == 8:
            row = [from_bits(1) if i % 3 == 0 else 0. for i in range(docs)]
        elif case == 9:
            row = [from_bits(0x7f7fffff) if i % 7 == 0 else 0. for i in range(docs)]
        elif case == 10:
            row = [from_bits(rng.randrange(0x7f800000)) for _ in range(docs)]
        elif case == 11:
            row = [1.]*docs
            for i in {0, min(127,docs-1), min(128,docs-1), docs-1}:
                row[i] = 2. # A score tie crossing a thread-tile boundary.
        elif case in (12, 13, 15):
            # K, K+1 or K-1 positives exercise the zero threshold and its ties.
            row = [-0. if i % 2 else 0. for i in range(docs)]
            for i in range(min(docs, k + {12: 0, 13: 1, 15: -1}[case])):
                row[docs-1 if i == 0 else i-1] = from_bits(1+i)
        else:
            row = [from_bits(0x3f800000 + i % 2) if i % 11 == 0 else
                   (-0. if i % 2 else 0.) for i in range(docs)]
        if q == 1:
            row = [float(rng.randrange(1000))/8 for _ in range(docs)]
        rows.append(row)
    for name, alternate in (('input_a', False), ('input_b', True)):
        values = []
        for query in range(q):
            row = rows[(query+1) % q] if alternate else rows[query]
            if alternate:
                # Change both row type and ID membership while keeping exact FP32 input bits.
                shift = 1+docs//3
                row = row[shift:]+row[:shift]
            values.extend(row)
        write_input(values, q, docs, k, directory/name, 'SYNTHETIC_RADIX_TOPK_UNIT_ONLY',
                    'alternating-input correctness fixture, not BM25/performance data')


def from_bits(raw):
    return struct.unpack('<f', struct.pack('<I', raw))[0]


def execute(command, log, environment):
    print('实际NPU命令：'+json.dumps(command, ensure_ascii=False), flush=True)
    with log.open('wb') as stream:
        try:
            result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        finally:
            stream.flush()
            print(log.read_text(encoding='utf-8', errors='replace'), flush=True)
    result.check_returncode()


def validate_record(record, q, docs, k, device, calls):
    expected = {'status': 'radix_device_calls_and_cpp_full_sort_passed', 'queries': q, 'docs': docs, 'k': k,
                'device': device, 'validated_calls': calls, 'declared_user_gm_scratch_bytes': 0,
                'declared_ub_algorithm_bytes': 8512, 'input_calls': [(calls+1)//2, calls//2],
                'every_call_poisoned': True, 'every_call_full_sort_checked': True}
    if (type(record.get('workspace_bytes')) is not int or not 0 <= record['workspace_bytes'] <= 2**63-1 or
            set(record) != set(expected) | {'workspace_bytes'} or
            any(type(record.get(name)) is not type(value) or record.get(name) != value for name, value in expected.items()) or
            any(type(value) is not int for value in record['input_calls'])):
        raise ValueError('调用器未报告本次尺寸、实际工作区、逐次覆写/交替输入/全排序验收')


def run(output, runner, vendor, project, device, calls, suite):
    output, runner, vendor, project = (p.resolve() for p in (output, runner, vendor, project))
    if output.exists() or not runner.is_file():
        raise ValueError('结果目录须为新目录，独立调用器须先编译')
    if type(device) is not int or not 0 <= device <= 2147483647 or type(calls) is not int or not 5 <= calls <= 100:
        raise ValueError('设备号须为非负int32；交替调用次数须在5..100，确保A和B均重复')
    if suite not in ('smoke', 'all'):
        raise ValueError('未知验收范围')
    installation = validate_install(vendor, project)
    runner_sha = digest(runner)
    tools = {str(p.relative_to(ROOT)): digest(p) for p in
             [ROOT/'standalone/validate_topk_radix_zero.py', ROOT/'standalone/topk_fixture.py']+
             [ROOT/n for n in HOST_SOURCE_PATHS]}
    environment = os.environ.copy()
    environment['ASCEND_CUSTOM_OPP_PATH'] = str(vendor) # Only this vendor; no competing Bm25TopK registrations.
    output.mkdir(parents=True)
    report = {'status': 'running', 'variant': VARIANT, 'suite': suite, 'device': device, 'calls_per_case': calls,
              'runner_sha256': runner_sha, 'tool_source_sha256': tools, 'installation': installation, 'cases': [],
              'scope': 'isolated actual Radix correctness only; not four-stage BM25 or performance evidence'}
    path = output/'report.json'
    write_json(path, report)
    try:
        execute([str(runner), '--check-library', installation['api_library']], output/'library.log', environment)
        for index, (q, docs, k) in enumerate(SMOKE_SPECS if suite == 'smoke' else SPECS):
            case_dir = output/f'case{index}_q{q}_d{docs}_k{k}'
            case_dir.mkdir()
            actual = case_dir/'actual'
            item = {'queries': q, 'docs': docs, 'k': k, 'status': 'running'}
            report['cases'].append(item)
            write_json(path, report)
            prepare_pair(case_dir, q, docs, k)
            before = {name: digest(case_dir/name/'scores.bin') for name in ('input_a', 'input_b')}
            command = [str(runner), str(case_dir), str(actual), str(q), str(docs), str(k), str(device),
                       installation['api_library'], str(calls)]
            item['command'] = command
            write_json(path, report)
            execute(command, case_dir/'device.log', environment)
            if (validate_install(vendor, project) != installation or digest(runner) != runner_sha or
                    any(digest(ROOT/n) != h for n, h in tools.items()) or
                    any(digest(case_dir/n/'scores.bin') != h for n, h in before.items())):
                raise ValueError('验收过程中源码、安装文件、调用器或输入改变')
            record = json.loads((actual/'device_validation.json').read_text(encoding='utf-8'))
            validate_record(record, q, docs, k, device, calls)
            with (case_dir/'check.log').open('w', encoding='utf-8', newline='\n') as stream:
                with redirect_stdout(stream):
                    for variant in ('a', 'b'):
                        check(case_dir/f'input_{variant}', actual/f'{variant}_top_scores.bin', actual/f'{variant}_top_ids.bin')
            item.update(status='passed', input_sha256=before, validation=record,
                        output_sha256={n: digest(actual/n) for n in
                                       ('a_top_scores.bin','a_top_ids.bin','b_top_scores.bin','b_top_ids.bin')},
                        device_log_sha256=digest(case_dir/'device.log'))
            write_json(path, report)
            print(f'通过：Q={q} D={docs} K={k}，{calls}次交替实际调用逐次完整核对，A/B独立Python全排序一致。', flush=True)
        report['status'] = ('radix_smoke_device_and_full_sort_passed' if suite == 'smoke' else
                            'radix_all_unit_device_and_full_sort_passed')
        write_json(path, report)
        print(f'{suite}验收完成，报告：{path}。冒烟通过后仍需all；全套通过后才接四段。不测性能、不作加速结论。', flush=True)
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
    parser.add_argument('--runner', default=str(ROOT/'results/topk_radix_zero_host_build/bm25_topk_radix_zero_runner'))
    parser.add_argument('--vendor', default=str(ROOT/'results/custom_opp/vendors/bm25_topk_radix_zero'))
    parser.add_argument('--project', default=str(ROOT/'results/Bm25TopKRadixZero'))
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--calls', type=int, default=5)
    parser.add_argument('--suite', choices=('smoke','all'), default='all')
    args = parser.parse_args()
    try:
        if sys.platform != 'linux':
            raise ValueError('请在A5 Linux主机执行实际设备验收')
        run(Path(args.output), Path(args.runner), Path(args.vendor), Path(args.project), args.device, args.calls, args.suite)
    except KeyboardInterrupt:
        print('验收中断，保留日志，不把未完成项目记为通过。', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f'FAILED：{error}；停止，不生成CPU替代输出或通过报告。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
