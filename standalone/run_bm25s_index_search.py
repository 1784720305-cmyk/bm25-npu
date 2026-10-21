"""原版BM25S Numba：原始查询词编号到CPU Top-K的完整Search参考，原生/严格适配分开。"""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import run_bm25s_reference as original, resident_index as r
from standalone import search_scale_fixture as fixture, search_benchmark_checks as checks
from standalone import benchmark_search as resources

STATUS = 'original_bm25s_numba_raw_ID_search_and_separate_strict_adapter_passed'
TIMER = 'caller perf_counter_ns before Search through returned CPU Top-K; raw grouped integer word IDs'
TOOLS = tuple(sorted(set(resources.TOOL_SOURCES) | {
    'standalone/run_bm25s_reference.py', 'standalone/run_bm25s_index_search.py',
    'standalone/prepare_bm25s_numba.py'}))


class SearchSession:
    """Only index/configuration is retained here; no expected answers or per-query cache."""
    def __init__(self, np, model, vocab_size, k, max_queries, max_words, workers, strict):
        self.np, self.model, self.vocab = np, model, vocab_size
        self.k, self.maxq, self.maxwords, self.workers, self.strict = k, max_queries, max_words, workers, strict
        r.require(r.integer(k, 1, min(32, model.scores['num_docs'])) and r.integer(vocab_size, 1, r.LIMIT-1) and
                  r.integer(max_queries, 1, 10000) and r.integer(max_words, 0, r.LIMIT) and
                  r.integer(workers, 1, 256) and type(strict) is bool, 'Search配置非法')
        r.require(model.vocab_dict.get('') == vocab_size and vocab_size in model.unique_token_ids_set and
                  len(model.scores['indptr']) == vocab_size+2 and
                  model.scores['indptr'][vocab_size] == model.scores['indptr'][vocab_size+1],
                  '原版索引必须包含真实的空词零列，不能用第一个词代替空查询')

    def search(self, queries):
        # Caller starts the clock before this validation; no request normalization is cached.
        r.require(type(queries) is list and 1 <= len(queries) <= self.maxq, '查询批次非法/超过声明容量')
        total = 0
        for row in queries:
            r.require(type(row) is list and len(row) <= self.maxwords-total, '查询词总数超过声明容量')
            for word in row: r.require(r.integer(word, 0, r.LIMIT-1), '词编号必须为非负INT32整数，拒绝bool')
            total += len(row)
        if self.strict:
            # Necessary compatibility work stays INSIDE Search, including the all-empty case.
            packed = [[word for word in row if word < self.vocab] or [self.vocab] for row in queries]
            values, ids = original.adapted_call(self.np, self.model, packed, self.k, self.workers)
        else:
            # Upstream's type probe rejects a first empty row before its empty-token handling.
            # Only this required empty-row compatibility step is ours; it stays INSIDE Search.
            native_queries = [row if row else [self.vocab] for row in queries]
            values, ids = original.native_call(self.model, native_queries, self.k, self.workers)
        np = self.np
        r.require(values.shape == ids.shape == (len(queries), self.k) and values.dtype == np.dtype('float32') and
                  ids.dtype == np.dtype('int32') and np.isfinite(values).all() and (values >= 0).all() and
                  (ids >= 0).all() and (ids < self.model.scores['num_docs']).all(), 'Search输出尺寸/类型/分数/编号非法')
        for scores, row_ids in zip(values, ids):
            r.require(len(np.unique(row_ids)) == self.k and (scores[:-1] >= scores[1:]).all(), 'Search重复编号或分数未降序')
            if self.strict:
                r.require(not ((scores[:-1] == scores[1:]) & (row_ids[:-1] >= row_ids[1:])).any(), '严格Search同分编号规则不符')
        return values, ids


def timed_search(session, raw):
    start = time.perf_counter_ns()
    result = session.search(raw)
    elapsed = (time.perf_counter_ns()-start)/1e6
    return result, elapsed


def index_hashes(model):
    return {name: hashlib.sha256(model.scores[name].tobytes()).hexdigest() for name in ('data', 'indices', 'indptr')}


def compiled_signatures(model, workers):
    from bm25s.numba import retrieve_utils
    native = retrieve_utils._retrieve_internal_jitted_serial if workers == 1 else retrieve_utils._retrieve_internal_jitted_parallel
    result = {'original_scorer': [str(x) for x in model._compute_relevance_from_scores.nopython_signatures],
              'original_native_serial_or_parallel': [str(x) for x in native.nopython_signatures]}
    r.require(all(result.values()), '原版Numba未实际生成nopython编译签名，不回退NumPy')
    return result


def run(inputs, output, workers, warmup=5, repeats=30, dependency_dir=None):
    inputs, output = inputs.resolve(), output.resolve()
    r.require(not output.exists(), '结果目录已存在，请换新名字，不覆盖')
    r.require(workers and len(set(workers)) == len(workers) and all(r.integer(n, 1, 256) for n in workers) and
              r.integer(warmup, 1, 10000) and r.integer(repeats, 10, 100000), '线程/预热/测量配置非法')
    output.mkdir(parents=True)
    report = {'status': 'running', 'timer_policy': TIMER, 'measurements': [],
              'scope': 'BM25S precomputed-contribution CPU Search reference; no NPU execution or hardware-only speedup claim'}
    path = output/'report.json'; r.write_json(path, report)
    try:
        r.require(not any(name in sys.modules for name in ('numba', 'llvmlite', 'bm25s')), '请用独立进程加载原版及所选私有Numba')
        dependency_record = None
        if dependency_dir is not None:
            from standalone.prepare_bm25s_numba import installed_record
            dependency_dir = dependency_dir.resolve(strict=True)
            print('stage=check_existing_private_dependencies_no_install', flush=True)
            dependency_record = installed_record(dependency_dir)
            sys.path.insert(0, str(dependency_dir))
        r.require('NUMBA_DISABLE_JIT' not in os.environ, '存在NUMBA_DISABLE_JIT设置，停止，不回退解释执行')
        numba = importlib.import_module('numba'); llvmlite = importlib.import_module('llvmlite')
        if dependency_dir is not None:
            r.require(all(dependency_dir in Path(module.__file__).resolve().parents for module in (numba, llvmlite)), '实际加载的Numba/llvmlite不是所选私有目录')
        r.require(max(workers) <= numba.config.NUMBA_NUM_THREADS, '请求Numba线程数超过该进程上限')
        library, np, source_record = original.load_reference()
        print('stage=validate_same_resident_index_corpus_and_raw_queries', flush=True)
        index, records, snapshot = fixture.validate_inputs(inputs)
        cache = checks.cache_inputs(inputs, index, records)
        corpus = json.loads((inputs/'source/corpus.json').read_text(encoding='utf-8'))
        vocab_size, k = corpus['vocab_size'], records['A']['k']
        raw = {name: value['queries'] for name, value in cache.items()}
        raw_before = json.dumps(raw, ensure_ascii=False, separators=(',', ':'))
        capq, capwords = max(map(len, raw.values())), max(sum(map(len, rows)) for rows in raw.values())
        vocab = {str(i): i for i in range(vocab_size)}; vocab[''] = vocab_size
        print('stage=build_original_precomputed_contribution_index', flush=True)
        start = time.perf_counter_ns()
        model = library.BM25(method='atire', idf_method='lucene', k1=1.2, b=.75, dtype='float32', int_dtype='int32',
                             backend='numba', csc_backend='numpy', auto_compile=False)
        model.index((corpus['documents'], vocab), create_empty_token=False, show_progress=False)
        build_ms = (time.perf_counter_ns()-start)/1e6
        model.activate_numba_scorer()
        hashes = index_hashes(model)
        start = time.perf_counter_ns(); full, oracles, sorted_scores = {}, {}, {}
        for name, queries in raw.items():
            # Debug-only preparation, not an alternative Search input or cached answer path.
            packed = [[word for word in row if word < vocab_size] or [vocab_size] for row in queries]
            rows = np.asarray([model.get_scores_from_ids(row) for row in packed], dtype='float32')
            original.compare_scores(np, rows, np.asarray(cache[name]['original'], dtype='float32').reshape(rows.shape), 'BM25S与独立原始语料公式：'+name)
            original.compare_scores(np, rows, np.asarray(cache[name]['scores'], dtype='float32').reshape(rows.shape), 'BM25S与有序FP32计算：'+name)
            full[name] = rows; oracles[name] = original.full_oracle(np, rows, k)
            r.require(np.array_equal(oracles[name].ravel(), np.asarray(cache[name]['ids'], dtype='int32')), 'BM25S自身严格排名与同输入CPU参考不同，停止定位舍入，不放宽同分')
            sorted_scores[name] = np.asarray([np.sort(row)[-k:][::-1] for row in rows])
            (output/f'{name}_full_scores.bin').write_bytes(rows.astype('<f4').tobytes())
        debug_ms = (time.perf_counter_ns()-start)/1e6
        report.update(index_id=index['index_id'], docs=index['num_docs'], queries=records['A']['num_queries'], k=k,
            source_is_real_corpus=False, input_sha256=snapshot, bm25s_source=source_record,
            tool_source_sha256=checks.sources(TOOLS), packages=original.package_versions(), python=platform.python_version(),
            resources=resources.host_resources(), capacity_queries=capq, capacity_words=capwords,
            private_dependency_record=dependency_record, dependency_scope='verified_private_py39' if dependency_dir else 'local_environment_interface_test_only',
            index_build_precomputation_ms=build_ms, debug_formula_and_first_scorer_jit_ms=debug_ms,
            index_array_bytes=sum(model.scores[name].nbytes for name in ('data', 'indices', 'indptr')),
            index_sha256=hashes, full_score_sha256={name: original.digest(output/f'{name}_full_scores.bin') for name in full},
            parameters={'backend': 'numba', 'csc_backend': 'numpy', 'method': 'atire', 'idf_method': 'lucene', 'k1': 1.2, 'b': .75},
            notes=['Native tie differences are recorded, not silently accepted as the strict contract.',
                   'Strict adapter uses public original compiled scorer, NumPy threshold/tie selection and per-call query thread pool; not claimed globally optimal.',
                   'Index build/precomputation, JIT, warmup, debug formula/output checks and IO excluded; all request validation/normalization/packing and required result checks INCLUDED.',
                   'Numba native parallelism is across queries; requested workers are not physical CPU cores or measured active worker counts.',
                   'One model/process, repeated calls per configuration; not independent process rounds or a paired NPU speedup experiment.'])
        r.write_json(path, report)
        for worker in workers:
            # Compile/check both modes before measuring either. No measured request is pre-normalized.
            sessions = {strict: SearchSession(np, model, vocab_size, k, capq, capwords, worker, strict) for strict in (False, True)}
            first_ms, mismatches, saved = {}, {}, {}
            print(f'stage=compile_and_check_raw_ID_search workers={worker}', flush=True)
            for strict, session in sessions.items():
                label = 'strict_adapter' if strict else 'original_native'
                first_ms[label] = {}; mismatches[label] = {}
                for name in ('A', 'B', 'A', 'empty', 'B'):
                    result, elapsed = timed_search(session, raw[name])
                    difference = original.check_selection(np, result, full[name], oracles[name], sorted_scores[name], k, strict)
                    if name not in first_ms[label]: first_ms[label][name] = elapsed
                    mismatches[label][name] = difference
                    previous = saved.get((label, name))
                    bits = (result[0].tobytes(), result[1].tobytes())
                    r.require(previous is None or bits == previous, '换查询后的BM25S原始输出字节不一致')
                    saved[label, name] = bits
            signatures = compiled_signatures(model, worker)
            for strict, session in sessions.items():
                label = 'strict_adapter' if strict else 'original_native'
                print(f'stage=measure_full_raw_ID_search variant={label} workers={worker}', flush=True)
                def check_result(name, result):
                    original.check_selection(np, result, full[name], oracles[name], sorted_scores[name], k, strict)
                    r.require((result[0].tobytes(), result[1].tobytes()) == saved[label, name], 'Search输出变化，停止')
                for cycle in range(warmup):
                    for name in (('A', 'B') if cycle % 2 == 0 else ('B', 'A')):
                        check_result(name, session.search(raw[name]))
                samples = []; final = {}
                for cycle in range(repeats):
                    for name in (('A', 'B') if cycle % 2 == 0 else ('B', 'A')):
                        result, elapsed = timed_search(session, raw[name])
                        # Oracle/repeat checking happens only AFTER the caller stopped its clock.
                        check_result(name, result)
                        samples.append({'cycle': cycle, 'case': name, 'call_wall_ms': elapsed})
                        final[name] = result
                outputs = {}
                for name, result in final.items():
                    for suffix, array in (('scores', result[0].astype('<f4')), ('ids', result[1].astype('<i4'))):
                        file = output/f'{label}_t{worker}_{name}_{suffix}.bin'; file.write_bytes(array.tobytes())
                        outputs[file.name] = original.digest(file)
                summary = {name: checks.summarize([item['call_wall_ms'] for item in samples if item['case'] == name], records[name]['num_queries']) for name in ('A', 'B')}
                item = {'variant': label, 'requested_query_workers': worker, 'maximum_active_queries': min(worker, records['A']['num_queries']),
                    'strict_tie_contract': strict, 'native_tie_differences': mismatches[label], 'compiled_signatures': signatures,
                    'initial_calls_including_possible_jit_ms': first_ms[label], 'warmup_cycles': warmup, 'measured_cycles': repeats,
                    'samples': samples, 'summary': summary, 'output_sha256': outputs, 'every_result_checked_outside_timer': True}
                report['measurements'].append(item); r.write_json(path, report)
                print(f'{label} workers={worker}：A整批中位数={summary["A"]["median_batch_ms"]:.6g}ms、p95={summary["A"]["p95_batch_ms"]:.6g}ms；B={summary["B"]["median_batch_ms"]:.6g}ms；原生A/B同分差异行数={len(mismatches[label]["A"])}/{len(mismatches[label]["B"])}。', flush=True)
        original.source_still_matches(source_record)
        r.require(index_hashes(model) == hashes and json.dumps(raw, ensure_ascii=False, separators=(',', ':')) == raw_before, '索引或原始查询在运行中改变')
        r.require(checks.sources(TOOLS) == report['tool_source_sha256'] and all(original.digest(Path(name)) == value for name, value in snapshot.items()), '输入或工具源码改变')
        if dependency_dir is not None: r.require(installed_record(dependency_dir) == dependency_record, '所选私有依赖发生改变')
        report.update(status=STATUS, numba_threading_layer=numba.threading_layer() if any(w > 1 for w in workers) else 'serial_only')
        r.write_json(path, report)
        print('BM25S Numba原始词编号Search及独立严格适配通过；原版源码未改，报告保存所有A/B样本。尚非与NPU同期配对或最优CPU结论。', flush=True)
        return report
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        r.write_json(path, report); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dependency-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, nargs='+', default=[1, 8]); p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=30)
    a = p.parse_args()
    try: run(a.input, a.output, a.workers, a.warmup, a.repeats, a.dependency_dir)
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不回退NumPy、不改原版或生成替代答案。', file=sys.stderr); return 1
    return 0


if __name__ == '__main__': sys.exit(main())
