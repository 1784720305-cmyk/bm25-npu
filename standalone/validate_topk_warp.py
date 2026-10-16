"""新warp Top-K独立设备验收：全排序核对与同进程重复，失败即停止，无CPU替代。"""
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
from standalone.prepare_topk_parallel_runtime import partition_count, prototypes
from standalone.topk_fixture import write_input, check
from standalone.validate_topk_merge import digest, write_json

VARIANT = 'warp_sort32_cached_head_topk_v1'
# Includes grid-stride query loops, each P transition, multi-tile tails, K=1/32/all docs.
SPECS = ((6, 6, 1), (6, 6, 3), (6, 6, 6), (65, 32, 32), (513, 40, 32),
         (65, 513, 1), (65, 513, 32), (65, 1025, 10), (65, 1025, 32),
         (65, 2048, 10), (65, 2049, 32), (65, 4097, 32), (1, 65536, 10))
SOURCE_FILES = ('op_host/bm25_top_k.cpp', 'op_host/bm25_top_k_tiling.h', 'op_kernel/bm25_top_k.cpp')
# Read verbatim from the actual newly installed generated header supplied by the user.
PROTOTYPES = {
    'aclnnBm25TopKGetWorkspaceSize': 'constaclTensor*scores,int64_tk,constaclTensor*topScoresOut,constaclTensor*topIdsOut,uint64_t*workspaceSize,aclOpExecutor**executor',
    'aclnnBm25TopK': 'void*workspace,uint64_tworkspaceSize,aclOpExecutor*executor,aclrtStreamstream'}


def scratch_bytes(q, docs, k):
    if any(type(n) is not int for n in (q, docs, k)) or not (
            1 <= q and 1 <= docs and q * docs <= 2147483647 and 1 <= k <= min(docs, 32)):
        raise ValueError('诊断尺寸非法：Q*D须在INT32范围，K须在1..min(D,32)')
    return q * partition_count(docs) * 12


def validate_install(vendor, project):
    vendor, project = vendor.resolve(strict=True), project.resolve(strict=True)
    info = json.loads((project / 'bm25_project.json').read_text(encoding='utf-8'))
    if info['operator'] != 'topk_warp' or info['vendor_name'] != 'bm25_topk_warp' or vendor.name != 'bm25_topk_warp':
        raise ValueError('需要独立topk_warp工程和实际安装vendor')
    for name in SOURCE_FILES:
        if digest(project / name) != info['source_sha256'][name] or digest(project / name) != digest(ROOT / 'npu_prototype/topk_warp' / name):
            raise ValueError(f'工程源码或当前包指纹不一致：{name}')
    header = vendor / 'op_api/include/aclnn_bm25_top_k.h'
    if prototypes(header) != PROTOTYPES:
        raise ValueError('实际生成接口与用户提供的新warp头文件声明不同')
    library = (vendor / 'op_api/lib/libcust_opapi.so').resolve(strict=True)
    if library.stat().st_size == 0:
        raise ValueError('实际API库为空')
    return {'api_library': str(library), 'api_sha256': digest(library), 'header_sha256': digest(header),
            'project_source_sha256': info['source_sha256'],
            'scope': 'file/prototype/source checks; actual kernel build and installation evidenced by device-side logs'}


def prepare(directory, q, docs, k):
    scratch_bytes(q, docs, k)
    rng = random.Random(950 + docs * 100 + k)
    scores = []
    for query in range(q):
        case = query % 8
        if case == 0:
            row = [0.] * docs
        elif case == 1:
            row = [1.] * docs
        elif case == 2:
            row = [float(i) for i in range(docs)]
        elif case == 3:
            row = [float(docs - i) for i in range(docs)]
        elif case == 4:
            row = [float(rng.randrange(9)) for _ in range(docs)]
        elif case == 5:
            row = [1.] * docs
            for i in (0, min(31, docs - 1), docs // 2, docs - 1):
                row[i] = 99.
        elif case == 6:
            row = [1. if i % 2 == 0 else struct.unpack('<f', struct.pack('<I', 0x3f800001))[0] for i in range(docs)]
        else:
            row = [-0. if i % 2 else 0. for i in range(docs)]
        scores.extend(row)
    # Large single-query case must exercise arbitrary values, not just all-zero input.
    if q == 1:
        scores = [float(rng.randrange(1000)) / 8 for _ in range(docs)]
    write_input(scores, q, docs, k, directory, 'SYNTHETIC_WARP_TOPK_UNIT_ONLY', 'deterministic unit fixture; not a retrieval workload')


def execute(command, log_path, environment):
    print('实际NPU命令：' + json.dumps(command, ensure_ascii=False), flush=True)
    with log_path.open('wb') as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    print(log_path.read_text(encoding='utf-8', errors='replace'), flush=True)
    result.check_returncode()


def run(output, runner, vendor, project, device, system_bytes, calls):
    output, runner, vendor, project = (p.resolve() for p in (output, runner, vendor, project))
    if output.exists() or not runner.is_file():
        raise ValueError('结果目录须为新目录，调用器须先编译')
    if type(device) is not int or not 0 <= device <= 2147483647 or type(calls) is not int or not 2 <= calls <= 100:
        raise ValueError('设备号须为非负int32；同进程调用次数须在2..100')
    if type(system_bytes) is not int or not 1 <= system_bytes <= 2**63 - 1:
        raise ValueError('系统工作区须为已核实的正整数')
    installation = validate_install(vendor, project)
    runner_sha = digest(runner)
    environment = os.environ.copy()
    environment['ASCEND_CUSTOM_OPP_PATH'] = str(vendor)
    output.mkdir(parents=True)
    report = {'status': 'running', 'variant': VARIANT, 'device': device, 'calls_per_case': calls,
              'system_bytes': system_bytes, 'runner_sha256': runner_sha, 'installation': installation, 'cases': [],
              'scope': 'isolated actual warp Top-K correctness with synthetic scores; not four-stage BM25 or a performance result'}
    report_path = output / 'report.json'
    write_json(report_path, report)
    try:
        execute([str(runner), '--check-library', installation['api_library']], output / 'library.log', environment)
        for index, (q, docs, k) in enumerate(SPECS):
            case_dir = output / f'case{index}_q{q}_d{docs}_k{k}'
            case_dir.mkdir()
            input_dir, actual = case_dir / 'input', case_dir / 'actual'
            item = {'queries': q, 'docs': docs, 'k': k, 'status': 'running'}
            report['cases'].append(item)
            write_json(report_path, report)
            prepare(input_dir, q, docs, k)
            scratch = scratch_bytes(q, docs, k)
            command = [str(runner), str(input_dir), str(actual), str(q), str(docs), str(k), str(device),
                       installation['api_library'], str(scratch), str(system_bytes), str(calls)]
            item['command'] = command
            write_json(report_path, report)
            execute(command, case_dir / 'device.log', environment)
            if validate_install(vendor, project) != installation or digest(runner) != runner_sha:
                raise ValueError('验收过程中源码、头文件、API库或调用器已改变')
            record = json.loads((actual / 'device_validation.json').read_text(encoding='utf-8'))
            expected = {'status': 'device_calls_completed_full_sort_not_yet_checked', 'queries': q, 'docs': docs,
                        'k': k, 'device': device, 'validated_calls': calls, 'workspace_bytes': system_bytes + scratch,
                        'user_scratch_bytes': scratch, 'system_bytes': system_bytes, 'every_call_poisoned': True}
            if record != expected:
                raise ValueError('调用器未报告本次尺寸、严格工作区和每次覆写重复验证')
            with (case_dir / 'check.log').open('w', encoding='utf-8', newline='\n') as stream:
                with redirect_stdout(stream):
                    check(input_dir, actual / 'top_scores.bin', actual / 'top_ids.bin')
            item.update(status='passed', user_scratch_bytes=scratch, input_sha256=digest(input_dir / 'scores.bin'),
                        output_sha256={name: digest(actual / name) for name in ('top_scores.bin', 'top_ids.bin')},
                        device_log_sha256=digest(case_dir / 'device.log'), validation=record)
            write_json(report_path, report)
            print(f'通过：Q={q}、D={docs}、K={k}；{calls}次同进程实际调用结果字节一致，且符合独立完整排序。', flush=True)
        report['status'] = 'warp_topk_unit_device_calls_and_full_sort_checks_passed'
        write_json(report_path, report)
        print(f'{len(SPECS)}组独立warp Top-K设备验收通过。报告：{report_path}', flush=True)
        print('下一步才接四段流程；本验收不测性能，不证明加速。', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error) or type(error).__name__
        if report['cases'] and report['cases'][-1]['status'] == 'running':
            report['cases'][-1]['status'] = report['status']
        write_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runner', default=str(ROOT / 'results/topk_warp_host_build/bm25_topk_warp_runner'))
    parser.add_argument('--vendor', default=str(ROOT / 'results/custom_opp/vendors/bm25_topk_warp'))
    parser.add_argument('--project', default=str(ROOT / 'results/Bm25TopKWarp'))
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--calls', type=int, default=3)
    parser.add_argument('--system-workspace-bytes', type=int, default=16777216,
                        help='当前CANN既有四段日志核实为16777216；尺寸不符即停止，不自动放宽')
    args = parser.parse_args()
    try:
        if sys.platform != 'linux':
            raise ValueError('请在A5 Linux主机执行实际设备验收')
        run(Path(args.output), Path(args.runner), Path(args.vendor), Path(args.project), args.device,
            args.system_workspace_bytes, args.calls)
    except KeyboardInterrupt:
        print('验收已中断，保留报告，不将未完成用例记为通过。', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f'FAILED：{error}；停止，不产生CPU替代输出或通过报告。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
