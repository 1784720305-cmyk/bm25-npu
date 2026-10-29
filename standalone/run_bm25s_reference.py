"""Unmodified pinned upstream BM25S loader, native call and separate strict adapter."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,importlib,json,os,sys
from importlib import metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reference_only/bm25s_original'
COMMIT='03eac5a8fb3aa4958ce37415a4cf2220d35157f7'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
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
