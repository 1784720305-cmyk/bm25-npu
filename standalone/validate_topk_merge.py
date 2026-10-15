"""新合并算子的独立设备验收；CPU仅准备单元测试候选/核对，不作性能结论。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPECS = ((65, 64, 1, 1), (65, 64, 2, 3), (65, 96, 3, 10),
         (65, 256, 8, 10), (65, 1024, 16, 32), (65, 2048, 32, 32), (65, 32, 1, 32))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, record):
    path.write_bytes((json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def pack(kind, values):
    return struct.pack('<' + kind * len(values), *values)


def unpack(path, kind, count):
    data = path.read_bytes()
    if len(data) != count * 4:
        raise ValueError(f'文件尺寸错误：{path}')
    return list(struct.unpack('<' + kind * count, data))


def dimensions(q, docs, shards, k):
    if any(type(v) is not int for v in (q, docs, shards, k)) or not (
            1 <= q and 1 <= docs and q * docs <= 2147483647 and
            1 <= shards <= 32 and 1 <= k <= 32 and docs % shards == 0 and docs // shards >= k):
        raise ValueError('不支持的Q/文章数/分块数/K：须整除，每块至少K篇')


def prepare(directory, q, docs, shards, k):
    dimensions(q, docs, shards, k)
    if directory.exists():
        raise ValueError('请选择新的测试目录')
    rng = random.Random(950 + shards * 100 + k)
    rows = []
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
            row = [float(rng.randrange(5)) for _ in range(docs)]
        elif case == 5:
            row = [0.] * docs
            row[-1] = row[docs // 2] = 99.
        elif case == 6:
            row = [1. if i % 2 == 0 else 1.0000001192092896 for i in range(docs)]
        else:
            row = [-0. if i % 2 else 0. for i in range(docs)]
        rows.extend(row)
    directory.mkdir(parents=True)
    (directory / 'scores.bin').write_bytes(pack('f', rows))
    partial_scores, partial_ids = [], []
    width = docs // shards
    for query in range(q):
        row = rows[query * docs:(query + 1) * docs]
        for shard in range(shards):
            base = shard * width
            selected = sorted(range(width), key=lambda i: (-row[base + i], i))[:k]
            partial_ids.extend(selected)
            partial_scores.extend(row[base + i] for i in selected)
    (directory / 'partial_scores.bin').write_bytes(pack('f', partial_scores))
    (directory / 'partial_ids.bin').write_bytes(pack('i', partial_ids))
    names = ('scores.bin', 'partial_scores.bin', 'partial_ids.bin')
    record = {'status': 'cpu_prepared_merge_unit_input_not_device_result',
              'queries': q, 'docs': docs, 'shards': shards, 'k': k,
              'candidate_producer': 'CPU unit-fixture only; not the NPU first stage and not a performance experiment',
              'sha256': {name: digest(directory / name) for name in names}}
    write_json(directory / 'fixture.json', record)
    return record


def validate_input(directory):
    info = json.loads((directory / 'fixture.json').read_text(encoding='utf-8'))
    q, docs, shards, k = (info[key] for key in ('queries', 'docs', 'shards', 'k'))
    dimensions(q, docs, shards, k)
    for name in ('scores.bin', 'partial_scores.bin', 'partial_ids.bin'):
        if digest(directory / name) != info['sha256'][name]:
            raise ValueError(f'输入指纹改变：{name}')
    scores = unpack(directory / 'scores.bin', 'f', q * docs)
    partial = unpack(directory / 'partial_scores.bin', 'f', q * shards * k)
    ids = unpack(directory / 'partial_ids.bin', 'i', q * shards * k)
    if any(not math.isfinite(v) or v < 0 for v in scores):
        raise ValueError('完整分数非法')
    width = docs // shards
    for query in range(q):
        for shard in range(shards):
            base = query * docs + shard * width
            selected = sorted(range(width), key=lambda i: (-scores[base + i], i))[:k]
            offset = (query * shards + shard) * k
            if ids[offset:offset + k] != selected or pack('f', partial[offset:offset + k]) != pack('f', [scores[base + i] for i in selected]):
                raise ValueError('候选不是该文章块按统一规则选出的前K')
    return info, scores


def check(directory, actual):
    info, scores = validate_input(directory)
    q, docs, k = (info[key] for key in ('queries', 'docs', 'k'))
    actual_ids = unpack(actual / 'top_ids.bin', 'i', q * k)
    actual_scores = (actual / 'top_scores.bin').read_bytes()
    if len(actual_scores) != q * k * 4:
        raise ValueError('输出分数尺寸错误')
    expected_ids, expected_scores = [], []
    for query in range(q):
        base = query * docs
        selected = sorted(range(docs), key=lambda i: (-scores[base + i], i))[:k]
        expected_ids.extend(selected)
        expected_scores.extend(scores[base + i] for i in selected)
    if actual_ids != expected_ids or actual_scores != pack('f', expected_scores):
        raise ValueError('合并结果与完整文章独立排序不一致（编号/顺序/分数字节）')
    return {'queries': q, 'docs': docs, 'shards': info['shards'], 'k': k,
            'output_sha256': {name: digest(actual / name) for name in ('top_scores.bin', 'top_ids.bin')}}


def validate_install(vendor, project):
    project_info = json.loads((project / 'bm25_project.json').read_text(encoding='utf-8'))
    if project_info['operator'] != 'topk_merge' or project_info['vendor_name'] != 'bm25_topk_merge':
        raise ValueError('需要实际独立topk_merge工程')
    for name in ('op_host/bm25_top_k_merge.cpp', 'op_host/bm25_top_k_merge_tiling.h', 'op_kernel/bm25_top_k_merge.cpp'):
        if digest(project / name) != project_info['source_sha256'][name] or digest(project / name) != digest(ROOT / 'npu_prototype/topk_merge' / name):
            raise ValueError(f'工程源码和当前包不一致：{name}')
    header = vendor / 'op_api/include/aclnn_bm25_top_k_merge.h'
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', header.read_text(encoding='utf-8'), flags=re.S)
    expected = {
        'aclnnBm25TopKMergeGetWorkspaceSize': 'constaclTensor*partialScores,constaclTensor*partialIds,int64_tk,int64_tdocCount,constaclTensor*topScoresOut,constaclTensor*topIdsOut,uint64_t*workspaceSize,aclOpExecutor**executor',
        'aclnnBm25TopKMerge': 'void*workspace,uint64_tworkspaceSize,aclOpExecutor*executor,aclrtStreamstream'}
    for name, parameters in expected.items():
        matches = re.findall(r'\baclnnStatus\s+' + name + r'\s*\((.*?)\)\s*;', text, flags=re.S)
        if len(matches) != 1 or re.sub(r'\s+', '', matches[0]) != parameters:
            raise ValueError(f'实际声明与用户提供的头文件不一致：{name}')
    library = (vendor / 'op_api/lib/libcust_opapi.so').resolve(strict=True)
    return {'api_library': str(library), 'api_sha256': digest(library), 'header_sha256': digest(header),
            'project_source_sha256': project_info['source_sha256'],
            'installation_scope': 'source/header/API file checks; installed kernel provenance also requires the build/install logs'}


def run(output, runner, vendor, project, device):
    if output.exists() or not runner.is_file():
        raise ValueError('结果目录须为新目录，调用器须已编译')
    if type(device) is not int or not 0 <= device <= 2147483647:
        raise ValueError('设备号非法')
    install = validate_install(vendor, project)
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment['ASCEND_CUSTOM_OPP_PATH'] = str(vendor.resolve())
    cases = []
    for index, (q, docs, shards, k) in enumerate(SPECS):
        case = output / f'case{index}_s{shards}_k{k}'
        input_dir, actual = case / 'input', case / 'actual'
        prepare(input_dir, q, docs, shards, k)
        validate_input(input_dir)
        command = [str(runner.resolve()), str(input_dir.resolve()), str(actual.resolve()),
                   str(q), str(docs), str(shards), str(k), str(device), install['api_library']]
        print('实际设备命令：' + json.dumps(command, ensure_ascii=False), flush=True)
        with (case / 'device.log').open('wb') as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        print((case / 'device.log').read_text(encoding='utf-8', errors='replace'), flush=True)
        result.check_returncode()
        # Ensure validation really covers the same installed API files throughout this suite.
        if validate_install(vendor, project) != install:
            raise ValueError('运行过程中实际安装库/头文件/源码发生改变')
        record = check(input_dir, actual)
        record.update(command=command, device_log_sha256=digest(case / 'device.log'))
        write_json(case / 'checked.json', record)
        cases.append(record)
        print(f'通过：S={shards}、K={k}，完整排名一致；程序内部执行3次带输出覆写检查的调用。', flush=True)
    report = {'status': 'merge_unit_device_calls_and_independent_checks_passed', 'device': device,
              'runner_sha256': digest(runner), 'installation': install, 'cases': cases,
              'scope': 'CPU-produced candidates for isolated merge unit tests; not four-stage BM25, NPU first-stage validation or a performance benchmark'}
    write_json(output / 'report.json', report)
    print('七组设备验收通过。仅验新合并算子，下一步接四段流程；不证明性能收益。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runner', default=str(ROOT / 'results/topk_merge_host_build/bm25_topk_merge_runner'))
    parser.add_argument('--vendor', default=str(ROOT / 'results/custom_opp/vendors/bm25_topk_merge'))
    parser.add_argument('--project', default=str(ROOT / 'results/Bm25TopKMerge'))
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    try:
        run(Path(args.output), Path(args.runner), Path(args.vendor), Path(args.project), args.device)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'FAILED：{error}；停止，不生成CPU替代结果或成功报告。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
