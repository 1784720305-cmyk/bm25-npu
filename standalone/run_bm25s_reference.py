"""原版BM25S离线CPU对照；保留源码、原生同分行为及独立严格规则适配计时。"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reference_only/bm25s_original'
COMMIT = '03eac5a8fb3aa4958ce37415a4cf2220d35157f7'
sys.path.insert(0, str(ROOT))
from standalone.pipeline_fixture import validate_input
from standalone.validate_topk_merge import digest, write_json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_source():
    record = json.loads((SOURCE/'source.json').read_text(encoding='utf-8'))
    require(record.get('commit') == COMMIT and len(record.get('files', {})) == 25,
            'BM25S原版提交或源码清单不匹配')
    for name, expected in record['files'].items():
        path = (SOURCE/name).resolve(strict=True)
        require(SOURCE.resolve() in path.parents and digest(path) == expected, '原版源码缺失或改变：'+name)
    return record


def load_reference():
    record = check_source()
    require(not any(n == 'bm25s' or n.startswith('bm25s.') for n in sys.modules),
            '进程已加载其他BM25S，须使用新的独立Python进程')
    # The unchanged upstream selection module optionally probes JAX at import time.
    # Keep even that probe on CPU; measured selection is explicitly numpy below.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    sys.path.insert(0, str(SOURCE))
    library = importlib.import_module('bm25s')
    import numpy as np
    require(Path(library.__file__).resolve() == (SOURCE/'bm25s/__init__.py').resolve(),
            '未加载本包固定版本的原BM25S')
    return library, np, record


def source_still_matches(record):
    require(check_source() == record, '运行过程中BM25S原版来源改变')
    for name, module in list(sys.modules.items()):
        if name == 'bm25s' or name.startswith('bm25s.'):
            path = getattr(module, '__file__', None)
            if path:
                relative = Path(path).resolve().relative_to(SOURCE.resolve()).as_posix()
                require(relative in record['files'] and digest(Path(path)) == record['files'][relative],
                        '实际加载的BM25S子模块不在原版清单内：'+name)


def package_versions():
    result = {}
    for name in ('numpy', 'scipy', 'numba', 'llvmlite', 'jax'):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def compare_scores(np, actual, expected, label):
    require(actual.shape == expected.shape and np.isfinite(actual).all() and (actual >= 0).all(),
            label+'尺寸、有限性或非负性不符')
    delta = np.abs(actual.astype('float64')-expected.astype('float64'))
    # Same original contract, no tolerance widening for a different CPU opponent.
    failed = delta > 1e-6 + 2e-6*np.abs(expected.astype('float64'))
    if failed.any():
        q, d = np.argwhere(failed)[0]
        raise ValueError(f'{label}不符：query={q} doc={d} actual={actual[q,d]} expected={expected[q,d]}')
    return float(delta.max(initial=0))


def read_corpus(directory, info):
    path = directory/'corpus.json'
    require(digest(path) == info.get('corpus_sha256'), '原始语料文件指纹不匹配')
    corpus = json.loads(path.read_text(encoding='utf-8'))
    require(corpus.get('kind') == 'synthetic_numeric_corpus_v1' and corpus.get('k1') == 1.2 and
            corpus.get('b') == .75 and corpus.get('idf_formula') == 'log1p((N-df+0.5)/(df+0.5))',
            '语料公式或类型不匹配')
    docs, queries, vocab = corpus['documents'], corpus['queries'], corpus['vocab_size']
    require(type(vocab) is int and 1 <= vocab < 2**31-1 and len(docs) == info['num_docs'] and
            len(queries) == info['num_queries'], '语料尺寸不匹配')
    require(all(isinstance(row, list) and all(type(t) is int and 0 <= t < vocab for t in row) for row in docs),
            '文章必须是词表范围内的整数词编号')
    require(all(isinstance(row, list) and all(type(t) is int and 0 <= t < 2**31-1 for t in row) for row in queries),
            '查询必须是非负整数词编号')
    require(sum(map(len, docs)) > 0, '不能为全空语料建索引')
    return corpus


def independent_formula(np, corpus):
    """Use original documents/statistics, not BM25S index or prepared TF/IDF arrays."""
    docs, queries = corpus['documents'], corpus['queries']
    n = len(docs)
    terms = sorted({t for q in queries for t in q if t < corpus['vocab_size']})
    tf = {t: np.zeros(n, dtype='float64') for t in terms}
    for i, doc in enumerate(docs):
        count = Counter(doc)
        for t in terms:
            tf[t][i] = count.get(t, 0)
    lengths = np.asarray(list(map(len, docs)), dtype='float64')
    norm = 1.2*(1-.75+.75*lengths/lengths.mean())
    idf = {t: math.log1p((n-int(np.count_nonzero(tf[t]))+.5)/(int(np.count_nonzero(tf[t]))+.5))
           for t in terms}
    scores = np.zeros((len(queries), n), dtype='float64')
    for q, query in enumerate(queries):
        for t in query:  # Duplicate query terms count repeatedly; unknown terms contribute zero.
            if t in tf:
                scores[q] += idf[t]*(tf[t]*2.2)/(tf[t]+norm)
    return scores.astype('float32')


def strict_select(np, row, k):
    """Wrapper only: exact Kth threshold, fill ties by ascending ID, sort selected K."""
    threshold = np.partition(row, len(row)-k)[len(row)-k]
    better = np.flatnonzero(row > threshold)
    equal = np.flatnonzero(row == threshold)  # Numeric equality deliberately unifies +/-0.
    selected = np.concatenate((better, equal[:k-len(better)]))
    order = np.lexsort((selected, -row[selected]))
    ids = selected[order].astype('int32')
    return row[ids], ids


def full_oracle(np, rows, k):
    doc_ids = np.arange(rows.shape[1], dtype='int32')
    return np.asarray([np.lexsort((doc_ids, -row))[:k] for row in rows], dtype='int32')


def native_call(model, queries, k, workers):
    result = model.retrieve(queries, k=k, sorted=True, return_as='tuple', n_threads=0 if workers == 1 else workers,
                            show_progress=False, backend_selection=model.backend)
    return result.scores, result.documents


def adapted_call(np, model, queries, k, workers):
    def one(query):
        return strict_select(np, model.get_scores_from_ids(query), k)
    if workers == 1:
        result = list(map(one, queries))
    else:
        # Same per-batch thread-pool lifetime as original retrieve's numpy path.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            result = list(pool.map(one, queries))
    return np.asarray([v[0] for v in result]), np.asarray([v[1] for v in result])


def check_selection(np, result, rows, oracle, sorted_scores, k, strict):
    values, ids = result
    require(values.shape == ids.shape == (len(rows), k) and values.dtype == np.dtype('float32') and
            np.issubdtype(ids.dtype, np.integer), '检索输出尺寸或类型不符')
    mismatches = []
    for q in range(len(rows)):
        require((ids[q] >= 0).all() and (ids[q] < rows.shape[1]).all() and len(np.unique(ids[q])) == k,
                'Top-K编号越界或重复')
        require(np.array_equal(values[q].view('uint32'), rows[q, ids[q]].view('uint32')),
                'Top-K分数不是对应BM25S分数的原始FP32字节')
        require(np.array_equal(values[q], sorted_scores[q]), '原版输出不是自身分数的前K大降序序列')
        if not np.array_equal(ids[q], oracle[q]):
            mismatches.append({'query': q, 'native_ids': ids[q].tolist(), 'strict_ids': oracle[q].tolist(),
                               'selected_member_set_differs': set(ids[q].tolist()) != set(oracle[q].tolist())})
    if strict:
        require(not mismatches, '严格适配未满足同分编号升序，停止')
    return mismatches


def measure(np, call, check, warmup, repeats):
    first_start = time.perf_counter_ns()
    first_result = call()
    first_call_ms = (time.perf_counter_ns()-first_start)/1e6
    check(first_result)
    for _ in range(warmup):
        check(call())
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns()-start)/1e6)
        check(result)  # Entire check excluded from measured interval.
    return result, {'samples_batch_ms': samples, 'median_batch_ms': statistics.median(samples),
                    'p95_batch_ms': float(np.percentile(samples, 95)), 'warmup': warmup, 'repeats': repeats,
                    'initial_call_before_warmup_ms': first_call_ms,
                    'every_result_checked_outside_timer': True}


def run(directory, output, threads, warmup, repeats, csc_backend='numpy', backend='numpy', dependency_dir=None):
    directory, output = directory.resolve(), output.resolve()
    require(not output.exists(), '结果目录已存在，请指定新目录，不覆盖历史结果')
    require(threads and len(set(threads)) == len(threads) and all(type(t) is int and 1 <= t <= 256 for t in threads),
            '查询线程配置须为不重复的1..256整数')
    require(type(warmup) is int and 1 <= warmup <= 10000 and type(repeats) is int and 1 <= repeats <= 10000,
            '预热和测量次数须在1..10000')
    require(csc_backend in ('numpy', 'scipy'), 'CSC后端非法')
    require(backend in ('numpy', 'numba'), '查询后端非法')
    require(dependency_dir is None or backend == 'numba', '私有Numba依赖只能用于明确的numba路径')
    output.mkdir(parents=True)
    report = {'status': 'running', 'scope': 'CPU BM25S reference only, not NPU execution or a speedup claim.', 'measurements': []}
    path = output/'report.json'
    write_json(path, report)
    try:
        dependency_record = None
        if dependency_dir is not None:
            from standalone.prepare_bm25s_numba import installed_record
            dependency_record = installed_record(dependency_dir)
            require(not any(n in sys.modules for n in ('numba','llvmlite')), '进程已加载其他Numba/llvmlite，停止')
            sys.path.insert(0, str(dependency_dir.resolve()))
        if backend == 'numba':
            require('NUMBA_DISABLE_JIT' not in os.environ, '设置了NUMBA_DISABLE_JIT，停止，不退回NumPy或解释执行')
            numba = importlib.import_module('numba')
            require(max(threads) <= numba.config.NUMBA_NUM_THREADS, '请求的Numba线程数超过该进程上限')
            if dependency_dir is not None:
                require(dependency_dir.resolve() in Path(numba.__file__).resolve().parents,
                        '没有加载所选私有Numba，停止')
        library, np, source = load_reference()
        info, _ = validate_input(directory)
        require('corpus_sha256' in info, '需要带corpus.json的scale_fixture语料，不能用其他随机语料代替')
        before = {name: digest(directory/name) for name in
                  ('metadata.json', 'corpus.json', 'tf.bin', 'doc_norm.bin', 'idf.bin', 'offsets.bin',
                   'expected_contributions.bin', 'expected_scores.bin')}
        for name in ('expected_contributions.bin', 'expected_scores.bin'):
            require(before[name] == info['reference_sha256'][name], '原期望文件指纹改变：'+name)
        corpus = read_corpus(directory, info)
        expected = np.fromfile(directory/'expected_scores.bin', dtype='<f4').reshape(info['num_queries'], info['num_docs'])
        formula = independent_formula(np, corpus)
        reference_error = compare_scores(np, expected, formula, '打包参考与原始语料公式')
        # Explicit identity mapping, including a real empty column for all-unknown/empty queries.
        vocab = {str(t): t for t in range(corpus['vocab_size'])}
        vocab[''] = corpus['vocab_size']
        queries = [[t for t in query if t < corpus['vocab_size']] or [corpus['vocab_size']]
                   for query in corpus['queries']]
        model = library.BM25(method='atire', idf_method='lucene', k1=1.2, b=.75, dtype='float32', int_dtype='int32',
                             backend=backend, csc_backend=csc_backend, auto_compile=False)
        print('stage=build_original_bm25s_index', flush=True)
        start = time.perf_counter_ns()
        model.index((corpus['documents'], vocab), create_empty_token=False, show_progress=False)
        build_ms = (time.perf_counter_ns()-start)/1e6
        if backend == 'numba':
            # Public original API: only compile the scorer; keep identical index construction.
            model.activate_numba_scorer()
        score_prepare_start = time.perf_counter_ns()
        rows = np.asarray([model.get_scores_from_ids(q) for q in queries])
        score_prepare_ms = (time.perf_counter_ns()-score_prepare_start)/1e6
        if backend == 'numba':
            require(bool(model._compute_relevance_from_scores.nopython_signatures), '原版Numba scorer没有生成编译签名')
        score_error = compare_scores(np, rows, formula, 'BM25S与独立原始语料公式')
        compare_scores(np, rows, expected, 'BM25S与现有参考总分')
        k = info['k']
        oracle = full_oracle(np, rows, k)
        formula_ids = full_oracle(np, formula, k)
        # Arithmetic tolerance never permits silently changing ranking membership/order.
        require(np.array_equal(oracle, formula_ids), 'BM25S自身严格排名与独立公式不同，停止，不按容差合并同分')
        sorted_scores = np.asarray([np.sort(row)[-k:][::-1] for row in rows])
        native_start = time.perf_counter_ns()
        first_native = native_call(model, queries, k, 1)
        first_native_ms = (time.perf_counter_ns()-native_start)/1e6
        native_mismatches = check_selection(np, first_native, rows, oracle, sorted_scores, k, False)
        check_selection(np, adapted_call(np, model, queries, k, 1), rows, oracle, sorted_scores, k, True)
        full_scores_path = output/'bm25s_full_scores.bin'
        full_scores_path.write_bytes(rows.astype('<f4').tobytes())
        report.update(input_directory=str(directory), input_sha256=before, bm25s_source=source,
                      wrapper_sha256=digest(Path(__file__)), python=platform.python_version(), platform=platform.platform(),
                      packages=package_versions(), queries=info['num_queries'], docs=info['num_docs'], k=k,
                      parameters={'method': 'atire', 'idf_method': 'lucene', 'k1': 1.2, 'b': .75, 'dtype': 'float32',
                                  'backend': backend, 'selection_backend': backend, 'csc_backend': csc_backend},
                      index_build_ms=build_ms, index_array_bytes=sum(model.scores[n].nbytes for n in ('data', 'indices', 'indptr')),
                      score_validation_prepare_ms_including_possible_jit=score_prepare_ms,
                      first_native_call_ms_including_possible_jit=first_native_ms,
                      private_dependency_record=dependency_record,
                      index_postings=len(model.scores['data']),
                      full_score_output={'file': full_scores_path.name, 'sha256': digest(full_scores_path),
                                         'dtype': 'little_endian_float32', 'shape': list(rows.shape)},
                      correctness={'all_query_full_scores_original_tolerance_passed': True,
                                   'original_reference_formula_max_abs_error': reference_error,
                                   'bm25s_formula_max_abs_error': score_error,
                                   'strict_adapter_ids_equal_independent_full_sort': True,
                                   'native_tie_mismatch_rows': native_mismatches,
                                   'native_matches_strict_id_contract': not native_mismatches},
                      query_encoding='Original integer word IDs, duplicates retained, unknown words omitted; empty gets explicitly indexed zero column.',
                      environment={n: os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','JAX_PLATFORMS')},
                      scope='Pre-encoded query IDs to Top-K from an original precomputed BM25S index. Index build/precomputation, '
                            'tokenization, corpus loading, validation, file IO excluded. Native retrieve and own strict-tie '
                            'adapter timed SEPARATELY; original numpy uses a per-call thread pool, original numba uses '
                            'its compiled serial/parallel retrieval. Adapter uses public original scorer plus numpy selection '
                            'and a per-call query thread pool. First calls/JIT/warmup/checks excluded from samples. '
                            'No query-time BM25 formula in BM25S. Not identical computation to current NPU, not full retrieval '
                            'or global-best CPU acceptance; shared load and affinity not controlled.')
        write_json(path, report)
        print(f'全部{len(queries)}条查询总分及严格适配排名通过；原版同分规则不匹配行数={len(native_mismatches)}。', flush=True)
        for workers in threads:
            for label, strict in ((f'original_{backend}_retrieve', False), (f'{backend}_scores_plus_strict_tie_adapter', True)):
                print(f'stage=measure_cpu_reference variant={label} query_workers={workers}', flush=True)
                call = (lambda: adapted_call(np, model, queries, k, workers)) if strict else (lambda: native_call(model, queries, k, workers))
                check = lambda result: check_selection(np, result, rows, oracle, sorted_scores, k, strict)
                result, timing = measure(np, call, check, warmup, repeats)
                if backend == 'numba' and not strict:
                    from bm25s.numba import retrieve_utils
                    compiled = (retrieve_utils._retrieve_internal_jitted_serial if workers == 1 else
                                retrieve_utils._retrieve_internal_jitted_parallel)
                    require(bool(compiled.nopython_signatures), '原版Numba检索未生成编译签名')
                prefix = f'{label}_t{workers}'
                values_path, ids_path = output/(prefix+'_scores.bin'), output/(prefix+'_ids.bin')
                values_path.write_bytes(result[0].astype('<f4').tobytes())
                ids_path.write_bytes(result[1].astype('<i8').tobytes())
                item = {'variant': label, 'requested_query_workers': workers,
                        'maximum_active_queries': min(workers, len(queries)), 'scorer_backend': backend,
                        'native_compiled_serial_or_parallel_checked': backend == 'numba' and not strict,
                        'strict_id_contract': strict or not native_mismatches, **timing,
                        'batch_queries': len(queries), 'ids_storage': 'little_endian_int64',
                        'output_sha256': {p.name: digest(p) for p in (values_path, ids_path)}}
                report['measurements'].append(item)
                write_json(path, report)
                print(f'{label} workers={workers}: batch_median_ms={timing["median_batch_ms"]:.6g} p95_ms={timing["p95_batch_ms"]:.6g}', flush=True)
        require(all(digest(directory/n) == h for n, h in before.items()), '运行过程中输入/参考文件改变')
        require(digest(Path(__file__)) == report['wrapper_sha256'], '运行过程中调用脚本改变')
        source_still_matches(source)
        if dependency_dir is not None:
            require(installed_record(dependency_dir) == dependency_record, '运行过程中私有依赖改变')
        report['status'] = f'original_bm25s_{backend}_and_separate_strict_adapter_checks_completed'
        write_json(path, report)
        print(f'原版CPU对照运行及核对完成：{path}', flush=True)
        print('原版计时与严格适配计时分别报告；未执行NPU，不计算硬件加速比，不代表最优CPU/完整检索达标。', flush=True)
        return report
    except (Exception, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error) or type(error).__name__
        write_json(path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check-source')
    r = sub.add_parser('run')
    r.add_argument('--input', required=True)
    r.add_argument('--output', required=True)
    r.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    r.add_argument('--warmup', type=int, default=5)
    r.add_argument('--repeats', type=int, default=30)
    r.add_argument('--csc-backend', choices=('numpy', 'scipy'), default='numpy')
    r.add_argument('--backend', choices=('numpy', 'numba'), default='numpy')
    r.add_argument('--dependency-dir', type=Path, help='已离线安装并核对的项目私有Numba目录；不修改系统环境')
    args = parser.parse_args()
    try:
        if args.command == 'check-source':
            source = check_source()
            print('原版源码指纹通过：', source['commit'], len(source['files']), '个文件；未建索引或检索。')
            print('现有依赖：', json.dumps(package_versions()))
        else:
            run(Path(args.input), Path(args.output), args.threads, args.warmup, args.repeats,
                args.csc_backend, args.backend, args.dependency_dir)
    except KeyboardInterrupt:
        print('对照中断，保留部分报告；未完成不记为通过。', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f'FAILED：{error}；停止，不使用其他算法或参考文件替代实际BM25S输出。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
