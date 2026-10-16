"""实际调用官方FP32 Top-K并严格核对；独立参考，不执行BM25算分、不替代自研交付。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_result import read_floats
from standalone.compare_compute_outputs import NPU_STATUSES, validate_producer
from standalone.pipeline_fixture import check as check_pipeline, validate_input
from standalone.topk_fixture import validate as validate_scores

PROTOTYPES = {
    'aclnnTopkGetWorkspaceSize': 'constaclTensor*self,int64_tk,int64_tdim,boollargest,boolsorted,aclTensor*valuesOut,aclTensor*indicesOut,uint64_t*workspaceSize,aclOpExecutor**executor',
    'aclnnTopk': 'void*workspace,uint64_tworkspaceSize,aclOpExecutor*executor,constaclrtStreamstream',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, record):
    path.write_bytes((json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def official_install(toolkit):
    header = toolkit / 'include/aclnnop/aclnn_topk.h'
    library = toolkit / 'lib64/libopapi.so'
    text = header.read_text(encoding='utf-8')
    for name, expected in PROTOTYPES.items():
        found = re.findall(r'aclnnStatus\s+' + name + r'\s*\(([^;]+)\)\s*;', text)
        require(len(found) == 1 and re.sub(r'\s+', '', found[0]) == expected,
                '实际官方声明与已提供A5头文件不同，停止：' + name)
    return {'header': str(header.resolve()), 'header_sha256': digest(header),
            'library': str(library.resolve(strict=True)), 'library_sha256': digest(library)}


def source_input(fixture, directory, producer, shards):
    require(type(shards) is int and 1 <= shards <= 32, '分块数应在1..32')
    if fixture is not None:
        require(directory is None and producer is None and shards == 1,
                '边界fixture入口不同时接受BM25来源或额外分块')
        info = json.loads((fixture / 'metadata.json').read_text(encoding='utf-8'))
        q, d, k = (info[name] for name in ('num_queries', 'num_docs', 'k'))
        path = fixture / 'scores.bin'
        validate_scores(read_floats(path), q, d, k)
        require(q*d <= 2147483647 and info['input_shape'] == [q, d] and info['output_shape'] == [q, k],
                'fixture形状不匹配或超过调用器上限')
        require(info['ties'] == 'exact_FP32_score_desc_doc_id_asc' and digest(path) == info['scores_sha256'],
                'fixture同分规则或输入指纹不匹配')
        origin = {'kind': 'selection_fixture_only', 'metadata_sha256': digest(fixture / 'metadata.json'),
                  'fixture': str(fixture.resolve()), 'input_origin': info['input_origin']}
    else:
        require(directory is not None and producer is not None, '须同时提供--input和--scores-from')
        info, _ = validate_input(directory)
        record = validate_producer(producer, info, NPU_STATUSES)
        check_pipeline(directory, producer)
        q, d, k = (info[name] for name in ('num_queries', 'num_docs', 'k'))
        require(record['queries'] == q and record['docs'] == d and record['k'] == k,
                '实际NPU生产者形状与原始输入不符')
        path = producer / 'scores.bin'
        origin = {'kind': 'actual_npu_score_file', 'producer': str(producer.resolve()),
                  'metadata_sha256': digest(producer / 'run_metadata.json'),
                  'input_sha256': info['input_sha256'], 'producer_output_sha256': record['output_sha256']}
    require(d % shards == 0 and d//shards >= k, '文章数须整除分块数，每块至少K篇')
    rows, cols = q*shards, d//shards
    path = path.resolve(strict=True)
    validate_scores(read_floats(path), rows, cols, k)
    return {'rows': rows, 'cols': cols, 'k': k, 'original_queries': q, 'original_docs': d,
            'shards': shards, 'scores_path': str(path), 'scores_sha256': digest(path), 'origin': origin}


def check_output(scores_path, directory, rows, cols, k):
    raw = scores_path.read_bytes()
    require(len(raw) == rows*cols*4, '总分输入尺寸不匹配')
    scores = struct.unpack('<' + 'f' * (rows*cols), raw)
    validate_scores(scores, rows, cols, k)
    expected_ids, expected_score_bits = [], []
    for r in range(rows):
        base = r*cols
        chosen = sorted(range(cols), key=lambda i: (-scores[base+i], i))[:k]
        expected_ids.extend(chosen)
        expected_score_bits.extend(raw[(base+i)*4:(base+i+1)*4] for i in chosen)
    expected_values = b''.join(expected_score_bits)
    expected_indices = struct.pack('<' + 'q' * len(expected_ids), *expected_ids)
    hashes = {}
    for value_name, id_name in (('bootstrap_scores.bin', 'bootstrap_ids.bin'), ('top_scores.bin', 'top_ids.bin')):
        value_path, id_path = directory/value_name, directory/id_name
        require(len(id_path.read_bytes()) == rows*k*8, '官方索引必须为INT64[行数,K]')
        require(id_path.read_bytes() == expected_indices,
                '官方Top-K入选集合或同分次序不满足严格规则：' + id_name)
        require(value_path.read_bytes() == expected_values,
                '官方分数与对应输入FP32字节不一致：' + value_name)
        hashes[value_name], hashes[id_name] = digest(value_path), digest(id_path)
    return hashes


def validate_timing(path, info, device, warmup, repeats):
    t = json.loads(path.read_text(encoding='utf-8'))
    require(t['status'] == 'official_topk_calls_completed_checks_pending' and t['api'] == 'aclnnTopk'
            and t['mode'] == 'resident' and t['indices_dtype'] == 'int64', '官方调用器计时类型不匹配')
    for name, expected in dict(rows=info['rows'], cols=info['cols'], k=info['k'], device=device,
                               warmup=warmup, repeats=repeats, additional_plain_warmup=1,
                               validated_poisoned_calls=warmup+2, verified_api_symbols=2, input_upload_count=1).items():
        require(type(t[name]) is int and t[name] == expected, '调用器参数/重复记录不匹配：' + name)
    require(type(t['workspace_capacity_bytes']) is int and t['workspace_capacity_bytes'] >= 0, '官方临时空间非法')
    require(len(t['samples']) == repeats, '正式计时样本数不匹配')
    for s in t['samples']:
        require(all(type(s[n]) in (int, float) and math.isfinite(s[n]) and s[n] >= 0
                    for n in ('prepare_ms', 'launch_sync_ms', 'batch_path_ms')), '非法计时值')
        require(s['batch_path_ms'] + 1e-9 >= s['prepare_ms'] + s['launch_sync_ms'], '计时边界不一致')
    median = sorted(s['batch_path_ms'] for s in t['samples'])[max(1, math.ceil(repeats*.5))-1]
    require(type(t['median_batch_path_ms']) in (int, float) and math.isfinite(t['median_batch_path_ms'])
            and math.isclose(t['median_batch_path_ms'], median, rel_tol=1e-12, abs_tol=1e-12),
            '中位数与原始样本不符')
    return t


def run(fixture, directory, producer, output, runner, toolkit, shards, device, warmup, repeats):
    require(type(device) is int and 0 <= device <= 2147483647, '设备编号非法')
    require(all(type(v) is int and 1 <= v <= 10000 for v in (warmup, repeats)), '预热/正式轮次非法')
    require(sys.byteorder == 'little', '当前二进制约定为小端')
    output, runner, toolkit = output.resolve(), runner.resolve(strict=True), toolkit.resolve(strict=True)
    require(not output.exists(), '输出目录已存在，请选择新目录；不覆盖原始结果')
    install = official_install(toolkit)
    info = source_input(fixture, directory, producer, shards)
    runner_sha = digest(runner)
    command = [str(runner), info['scores_path'], str(output/'device'), str(info['rows']), str(info['cols']),
               str(info['k']), str(device), str(toolkit), str(warmup), str(repeats), 'resident']
    environment = os.environ.copy()
    environment.pop('ASCEND_CUSTOM_OPP_PATH', None)  # Child only; official reference has no custom vendor dependency.
    print('官方独立对照：读取同一FP32总分，不重算贡献；输出INT64编号。', flush=True)
    print(f'形状=[{info["rows"]},{info["cols"]}] K={info["k"]}；原查询数={info["original_queries"]}，分块数={shards}', flush=True)
    print('实际命令：' + json.dumps(command, ensure_ascii=False), flush=True)
    (output/'device').mkdir(parents=True)
    try:
        with (output/'runner.log').open('w', encoding='utf-8', newline='\n') as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace', env=environment)
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                code = child.wait()
            finally:
                child.stdout.close()
                child.wait()
        require(code == 0, '官方调用器失败，退出码=' + str(code))
        log_text = (output/'runner.log').read_text(encoding='utf-8')
        for name in PROTOTYPES:
            require(log_text.count('verified_api_symbol=' + name + ' owner=') == 1,
                    '实际符号来源核对日志缺失：' + name)
        require(digest(runner) == runner_sha and official_install(toolkit) == install, '运行期间调用器或官方库/头文件改变')
        require(source_input(fixture, directory, producer, shards) == info, '运行期间总分或来源记录改变')
        hashes = check_output(Path(info['scores_path']), output/'device', info['rows'], info['cols'], info['k'])
        timing = validate_timing(output/'device/official_timing.json', info, device, warmup, repeats)
        record = {'status': 'official_reference_device_calls_and_strict_ranking_checks_passed',
                  'scope': 'Official FP32 selection only; prepared scores resident. Not custom BM25 delivery, complete retrieval, pure-kernel timing or a speedup claim.',
                  'api': 'aclnnTopk', 'indices_dtype': 'int64', 'input': info, 'installation': install,
                  'device_kernel_type_verified': False,
                  'command': command, 'runner_sha256': runner_sha, 'output_sha256': hashes,
                  'timing_sha256': digest(output/'device/official_timing.json'),
                  'runner_log_sha256': digest(output/'runner.log'), 'timing': timing}
        write_json(output/'run_metadata.json', record)
        print('官方实际调用及首末完整排序核对通过：入选集合、同分编号升序和分数字节一致。')
        print(f'整批resident中位数={timing["median_batch_path_ms"]:.6g}ms；含接口准备、提交和同步，不含上传下载。')
        print('这是官方单算子参考，尚不能与四段BM25整链直接计算算法加速比。')
        return record
    except BaseException as error:
        write_json(output/'run_metadata.json', {'status': 'failed', 'error': str(error), 'command': command,
                                              'scope': 'No accepted correctness/performance result; no CPU fallback.'})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', help='已有topk_fixture.py生成的选择输入；与--input/--scores-from互斥')
    p.add_argument('--input', help='原BM25串联输入，用于核对生产者来源')
    p.add_argument('--scores-from', help='实际NPU串联结果目录，含scores.bin与run_metadata.json')
    p.add_argument('--output', required=True)
    p.add_argument('--runner', default=str(ROOT/'results/official_topk_host_build/bm25_official_topk_runner'))
    p.add_argument('--toolkit', default='/usr/local/Ascend/cann-9.2.0')
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--device', type=int, default=2)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=30)
    a = p.parse_args()
    try:
        run(Path(a.fixture) if a.fixture else None, Path(a.input) if a.input else None,
            Path(a.scores_from) if a.scores_from else None, Path(a.output), Path(a.runner), Path(a.toolkit),
            a.shards, a.device, a.warmup, a.repeats)
    except (OSError, ValueError, KeyError, TypeError, struct.error, subprocess.SubprocessError) as error:
        print('FAILED：' + str(error) + '；停止，不生成CPU替代结果。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
