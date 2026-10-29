"""Same measured raw-ID Search wrapper; upstream code remains unchanged."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r,run_bm25s_reference as original,final_support as checks
TIMER='caller perf_counter_ns before Search through returned CPU Top-K; raw grouped integer word IDs'
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
