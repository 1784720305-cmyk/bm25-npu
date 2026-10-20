"""Read-only provenance/result/timing checks for complete raw-ID Search measurements."""
from array import array
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import resident_index as r, run_search_scale as scale, run_search as npu, run_cpu_index_search as cpu

COMMON_SOURCES = ('npu_prototype/search_benchmark_host/CMakeLists.txt',
    'npu_prototype/search_benchmark_host/benchmark_common.h', 'npu_prototype/search_host/search_inputs.h')
CPU_SOURCES = COMMON_SOURCES+('npu_prototype/search_benchmark_host/cpu_benchmark.cpp',
    'cpu_reference/index_search/cpu_index_search.h', 'cpu_reference/topk_document_shards.h')
NPU_SOURCES = COMMON_SOURCES+('npu_prototype/search_benchmark_host/npu_benchmark.cpp',
    'npu_prototype/search_host/npu_search.h', 'npu_prototype/resident_chain_host/resident_chain_common.h',
    'npu_prototype/query_score_host/query_score_runtime.h')
TIMER = 'caller steady_clock before Search through returned CPU Top-K'
CHECKPOINTS = ('before_A', 'before_B', 'after_A', 'after_B')
digest = scale.digest


def sources(names):
    return {name:digest(ROOT/name) for name in names}


def cache_inputs(inputs, index, records):
    """Prepare untimed independent/F32 oracles once, not once per measured process."""
    unused, values = r.validate_index(inputs/'index')
    cache = {}
    for name in ('A', 'B', 'empty'):
        record, queries = r.validate_queries(inputs/name, index)
        scores = r.diagnostic_scores(index, values, queries)
        q, n, k = record['num_queries'], index['num_docs'], record['k']
        raw, original = scale.chain.read(inputs/name/'expected_scores.bin', 'f', q*n)
        r.require(all(math.isfinite(x) and x >= 0 for x in original), '独立公式含非法值')
        r.compare(scores, original)
        ids = [doc for row in range(q) for doc in sorted(range(n), key=lambda i:(-scores[row*n+i], i))[:k]]
        cache[name] = {'scores':scores, 'raw':r.encode(scores), 'original':original,
                       'queries':queries, 'q':q, 'n':n, 'k':k, 'ids':ids}
    return cache


def check_output(path, reference, backend, shards):
    q, n, k = (reference[key] for key in ('q', 'n', 'k'))
    raw, full = scale.chain.read(path/'scores.bin', 'f', q*n)
    top_raw, top = scale.chain.read(path/'top_scores.bin', 'f', q*k)
    unused, ids = scale.chain.read(path/'top_ids.bin', 'i', q*k)
    r.require(all(math.isfinite(x) and x >= 0 for values in (full, top) for x in values), '实际分数含NaN/Inf/负值')
    errors = {'independent_formula':r.compare(full, reference['original']),
              'same_fp32_formula':r.compare(full, reference['scores'])}
    r.require(list(ids) == reference['ids'], '最终编号/顺序与同输入FP32参考不一致')
    if backend == 'cpu':
        r.require(raw == reference['raw'], 'CPU完整分数与有序FP32公式不逐字节一致')
    else:
        scale.chain.dimensions(n, k, shards)
        part_raw, parts = scale.chain.read(path/'partial_scores.bin', 'f', q*shards*k)
        unused, local = scale.chain.read(path/'partial_ids.bin', 'i', q*shards*k)
        r.require(all(math.isfinite(x) and x >= 0 for x in parts), '局部候选分数非法')
        width = n//shards
        for row in range(q):
            for shard in range(shards):
                base = row*n+shard*width; pos = (row*shards+shard)*k
                chosen = sorted(range(width), key=lambda i:(-full[base+i], i))[:k]
                r.require(list(local[pos:pos+k]) == chosen, '局部候选编号或同分顺序错误')
                bits = b''.join(raw[(base+i)*4:(base+i+1)*4] for i in chosen)
                r.require(part_raw[pos*4:(pos+k)*4] == bits, '局部候选未保留原始FP32分数字节')
    for row in range(q):
        base = row*n
        chosen = (reference['ids'][row*k:(row+1)*k] if backend == 'cpu' else
                  sorted(range(n), key=lambda i:(-full[base+i], i))[:k])
        r.require(list(ids[row*k:(row+1)*k]) == chosen, '最终编号与实际完整分数严格排序不一致')
        bits = b''.join(raw[(base+i)*4:(base+i+1)*4] for i in chosen)
        r.require(top_raw[row*k*4:(row+1)*k*4] == bits, 'Top-K未保留实际FP32分数字节')
    return {'all_queries_strict_ids_passed':True, 'selected_fp32_bits_preserved':True, 'max_abs_errors':errors}


def validate_acceptance(path, inputs, index, records, snapshot, cache):
    path = path.resolve()
    report = json.loads((path/'report.json').read_text(encoding='utf-8'))
    k = records['A']['k']
    r.require(report.get('status') == scale.STATUS and report.get('index_id') == index['index_id'] and
              report.get('input_sha256') == snapshot and type(report.get('k')) is int and report['k'] == k and
              report.get('source_is_real_corpus') is False, '需要本输入已经通过的正常规模CPU/NPU Search报告')
    r.require(report.get('host_source_sha256') == sources(scale.HOST_SOURCES) and
              report.get('tool_source_sha256') == sources(scale.TOOL_SOURCES), '已有验收源码已改变')
    device, shards, threads, cpu_shards, capq, capwords = (report.get(key) for key in
        ('device', 'shards', 'requested_threads', 'cpu_document_shards', 'capacity_queries', 'capacity_words'))
    r.require(r.integer(device, 0, 2147483647) and r.integer(shards, 1, 32) and r.integer(threads, 1, 65536) and
              r.integer(cpu_shards, 1, min(threads, index['num_docs'])) and r.integer(capq, 1, 10000) and
              r.integer(capwords, 0, 16777216), '已有验收设备/线程/容量记录非法')
    workspace = report['installation']['topk_acceptance']['workspace_bytes']
    cc, nc = report.get('cpu_command', []), report.get('npu_command', [])
    r.require(len(cc) == 10 and cc[1:] == [str(inputs/'index'), str(path/'requests.tsv'), str(path/'cpu'), str(threads),
              str(k), str(cpu_shards), str(capq), str(capwords), 'raw_query_ids_v1'] and
              len(nc) == 11 and nc[1:5] == [str(inputs/'index'), str(path/'requests.tsv'), str(path/'device'), str(device)] and
              nc[6:] == [str(k), str(shards), str(capq), str(capwords), str(workspace)], '已有验收实际命令不属于本输入/规则')
    r.require(digest(Path(cc[0])) == report.get('cpu_runner_sha256') and digest(Path(nc[0])) == report.get('npu_runner_sha256') and
              digest(path/'cpu.log') == report.get('cpu_log_sha256') and digest(path/'device.log') == report.get('device_log_sha256'),
              '已有验收主机程序或日志改变')
    scale.unchanged(report.get('raw_query_sha256', {}), '已有验收原始查询改变')
    r.require(len(report.get('raw_query_sha256', {})) == 4, '已有验收原始查询/计划记录缺失')
    output_hashes = {backend+'/'+f'call{i}_{name}/'+file:digest(path/backend/f'call{i}_{name}'/file)
        for backend, files in (('cpu', cpu.FILES), ('device', scale.chain.FILES))
        for i, name in enumerate(scale.chain.SEQUENCE) for file in files}
    r.require(output_hashes == report.get('output_sha256'), '已有完整输出指纹改变')
    cpu.check_session(path/'cpu', index, records, threads, k, cpu_shards, capq, capwords)
    npu.check_session(path/'device', index, records, device, k, shards, capq, capwords, workspace)
    # Recheck each unique query; all repeated outputs already have exact byte checks above.
    for i, name in ((0, 'A'), (1, 'B'), (3, 'empty')):
        check_output(path/'cpu'/f'call{i}_{name}', cache[name], 'cpu', cpu_shards)
        check_output(path/'device'/f'call{i}_{name}', cache[name], 'npu', shards)
    return {'directory':str(path), 'report_sha256':digest(path/'report.json'), 'device':device,
            'index_id':index['index_id'], 'scope':'prior actual normal-scale correctness; not a timing sample'}


def summarize(values, q):
    r.require(values and all(npu.finite_positive(value) for value in values), '计时样本为空或非法')
    ordered = sorted(values)
    return {'samples':len(values), 'median_batch_ms':statistics.median(values),
            'p95_batch_ms':ordered[math.ceil(.95*len(values))-1], 'mean_batch_ms':statistics.mean(values),
            'search_time_throughput_qps':q*1000/statistics.mean(values)}


def check_session(path, backend, index, records, partitions, threads, device, capq, capwords, warmup, repeats, workspace):
    data = json.loads((path/'benchmark_session.json').read_text(encoding='utf-8'))
    q, k = records['A']['num_queries'], records['A']['k']
    wanted = {'status':'search_benchmark_calls_completed_checks_pending', 'backend':backend,
        'docs':index['num_docs'], 'queries':q, 'k':k, 'partitions':partitions, 'capacity_queries':capq,
        'capacity_words':capwords, 'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory', 'timer_policy':TIMER,
        'index_load_batches':1, 'index_bytes':index['index_bytes'], 'cold_synthetic_empty_queries':capq,
        'checkpoint_search_calls':4, 'warmup_cycles':warmup, 'warmup_search_calls':2*warmup, 'warmup_topk_bit_checks':2*warmup,
        'measured_cycles':repeats, 'measured_search_calls':2*repeats, 'measured_topk_bit_checks':2*repeats,
        'measured_debug_download_calls':0, 'measured_poison_calls':0, 'measured_file_io_calls':0,
        'diagnostic_checkpoints_outside_measurement':True, 'index_readback_byte_identical':True}
    if backend == 'cpu':
        wanted.update(requested_threads=threads, debug_source_reread_batches=1,
                      fp32_flags='-O3 -march=native -ffp-contract=off -fno-fast-math')
        r.require(r.integer(data.get('actual_threads'), 1, threads), 'CPU实际线程数非法')
    else:
        wanted.update(device=device, index_tensor_h2d_calls=5, pipeline_stages=3, debug_index_d2h_calls=5, intermediate_host_roundtrips=0)
        ws = data.get('workspace_capacity_bytes', [])
        r.require(len(ws) == 3 and all(type(x) is int and x >= 0 for x in ws) and ws[1] == workspace, 'NPU实际工作区不符')
    r.require(all(type(data.get(key)) is type(value) and data[key] == value for key, value in wanted.items()) and
              npu.finite_positive(data.get('cold_init_ms')), '正式计时边界/计数/尺寸/索引复用记录不符')
    r.require(data.get('host_source_sha256') == sources(CPU_SOURCES if backend == 'cpu' else NPU_SOURCES),
              '计时程序不是当前源码编译，请重新配置编译独立计时工程')
    samples = data.get('samples', [])
    r.require(len(samples) == 2*repeats, '正式计时样本数量不符')
    for pos, item in enumerate(samples):
        cycle, slot = divmod(pos, 2)
        case = 'B' if slot^(cycle&1) else 'A'
        r.require(type(item.get('cycle')) is int and item['cycle'] == cycle and item.get('case') == case and
                  npu.finite_positive(item.get('call_wall_ms')) and npu.finite_positive(item.get('internal_search_ms')) and
                  item['call_wall_ms'] >= item['internal_search_ms'], '正式计时次序/完整边界/内部时间非法')
        expected_bytes = (0, 0) if backend == 'cpu' else (records[case]['query_input_bytes'], 8*q*k)
        r.require(all(type(item.get(key)) is int and item[key] == wanted for key, wanted in
                      zip(('query_h2d_bytes', 'topk_d2h_bytes'), expected_bytes)), '正式查询/Top-K应用传输字节不符')
    summary = {name:summarize([row['call_wall_ms'] for row in samples if row['case'] == name], q) for name in ('A', 'B')}
    return data, summary


def check_checkpoints(path, backend, cache, partitions):
    checks = {}
    files = cpu.FILES if backend == 'cpu' else scale.chain.FILES
    for name in ('A', 'B'):
        before, after = path/('before_'+name), path/('after_'+name)
        checks[name] = check_output(before, cache[name], backend, partitions)
        r.require(all((before/file).read_bytes() == (after/file).read_bytes() for file in files),
                  '正式计时前后完整输出不逐字节一致：'+name)
    hashes = {name+'/'+file:digest(path/name/file) for name in CHECKPOINTS for file in files}
    return checks, hashes
