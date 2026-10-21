"""Fused public Search-specific metadata validation; reuse unchanged independent array checks."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import search_benchmark_checks as common, resident_index as r, run_search as npu
sources=common.sources
summarize=common.summarize
TIMER=common.TIMER
HOST_SOURCES=('npu_prototype/fused_search_host/CMakeLists.txt','npu_prototype/fused_search_host/fused_search.h',
 'npu_prototype/fused_search_host/fused_api.h','npu_prototype/fused_search_host/fused_benchmark.cpp',
 'npu_prototype/fused_chain_host/fused_runtime.h','npu_prototype/search_host/search_inputs.h',
 'npu_prototype/search_benchmark_host/benchmark_common.h')
def check_session(path, backend, index, records, partitions, threads, device, capq, capwords, warmup, repeats, workspace):
    r.require(backend == 'npu', '融合记录只接受实际NPU后端')
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
        wanted.update(device=device, index_tensor_h2d_calls=5, pipeline_stages=2, normal_full_score_elements=0, debug_placeholder_elements=1, diagnostic_extra_chains=4, debug_index_d2h_calls=5, intermediate_host_roundtrips=0)
        ws = data.get('workspace_capacity_bytes', [])
        r.require(len(ws) == 2 and all(type(x) is int and x >= 0 for x in ws), 'NPU实际工作区不符')
    r.require(all(type(data.get(key)) is type(value) and data[key] == value for key, value in wanted.items()) and
              npu.finite_positive(data.get('cold_init_ms')), '正式计时边界/计数/尺寸/索引复用记录不符')
    r.require(data.get('host_source_sha256') == sources(HOST_SOURCES),
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


