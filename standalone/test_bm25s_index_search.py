"""Raw-ID boundary tests, plus actual LOCAL original Numba integration; not yellow-zone performance proof."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from standalone import run_bm25s_index_search as tool, scale_fixture


class FakeModel:
    """Only used for deterministic API/boundary tests; never written as a device or benchmark result."""
    backend = 'numba'
    def __init__(self):
        self.vocab_dict = {str(i): i for i in range(4)}; self.vocab_dict[''] = 4
        self.unique_token_ids_set = set(range(5))
        self.scores = {'num_docs': 6, 'indptr': np.zeros(6, dtype='int32')}
        self.calls = []
    def get_scores_from_ids(self, row):
        scores = np.zeros(6, dtype='float32')
        for word in row:
            if word < 4:
                scores += np.asarray([((doc+word) % 4) for doc in range(6)], dtype='float32')
        return scores
    def retrieve(self, queries, **kwargs):
        self.calls.append(queries)
        result = [tool.original.strict_select(np, self.get_scores_from_ids([t for t in row if t < 4]), kwargs['k']) for row in queries]
        return SimpleNamespace(scores=np.asarray([v[0] for v in result]), documents=np.asarray([v[1] for v in result]))


def main():
    count = 0
    def fails(call):
        nonlocal count
        try: call()
        except (ValueError, OSError): count += 1
        else: raise AssertionError('invalid request accepted')
    for strict in (False, True):
        model = FakeModel(); session = tool.SearchSession(np, model, 4, 3, 4, 12, 1, strict)
        for raw in ([[], [0]], [[9999], [0, 0]], [[], [], []], [[4, 999, 0], [1, 1, 1], []]):
            before = json.dumps(raw)
            values, ids = session.search(raw)
            expected = [tool.original.strict_select(np, model.get_scores_from_ids([t for t in row if t < 4]), 3) for row in raw]
            assert np.array_equal(ids, np.asarray([v[1] for v in expected]))
            assert np.array_equal(values.view('uint32'), np.asarray([v[0] for v in expected]).view('uint32'))
            assert json.dumps(raw) == before; count += 3
        for raw in ([], (), [None], [[True]], [[-1]], [[tool.r.LIMIT]], [[1.0]], [[0]*13], [[]]*5):
            before = len(model.calls); fails(lambda: session.search(raw)); assert len(model.calls) == before; count += 1
        with patch.object(tool.time, 'perf_counter_ns', side_effect=[1000000, 3000000]) as clock:
            method = tool.original.adapted_call if strict else tool.original.native_call
            def during_clock(*args):
                assert clock.call_count == 1
                # Empty handling and unknown filtering must be observable inside the timed callee.
                packed = args[2] if strict else args[1]
                assert packed[0] == [4]
                assert packed[1] == ([0] if strict else [9999, 0])
                return method(*args)
            with patch.object(tool.original, 'adapted_call' if strict else 'native_call', during_clock):
                unused, elapsed = tool.timed_search(session, [[], [9999, 0]])
            assert elapsed == 2.0 and clock.call_count == 2; count += 3
        for bad in (None, 0):
            broken = FakeModel()
            if bad is None: broken.vocab_dict.pop('')
            else: broken.scores['indptr'][5] = 1
            fails(lambda: tool.SearchSession(np, broken, 4, 3, 4, 12, 1, strict))
    # Output protocol mistakes must fail, even if a lower-level implementation returns normally.
    model = FakeModel(); session = tool.SearchSession(np, model, 4, 3, 4, 12, 1, False)
    for values, ids in ((np.zeros((1, 3), dtype='float64'), np.arange(3, dtype='int32').reshape(1, 3)),
                        (np.zeros((1, 3), dtype='float32'), np.zeros((1, 3), dtype='int32')),
                        (np.asarray([[0, 1, 2]], dtype='float32'), np.arange(3, dtype='int32').reshape(1, 3)),
                        (np.full((1, 3), float('nan'), dtype='float32'), np.arange(3, dtype='int32').reshape(1, 3))):
        with patch.object(tool.original, 'native_call', return_value=(values, ids)):
            fails(lambda: session.search([[0]]))
    print(f'PASS: {count} FAKE-model raw-ID/empty/unknown/duplicate/capacity/output/timer-boundary checks.', flush=True)
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bm25s_raw_ID_LOCAL_', dir=ROOT/'results') as temp:
        root = Path(temp); source, inputs, output = root/'source', root/'inputs', root/'output'
        with redirect_stdout(io.StringIO()):
            scale_fixture.generate(source, 128, 8, 10, 128, 20261014, 0, 'performance', 3)
            tool.fixture.prepare(source, inputs)
        # The actual unchanged upstream code and local Numba run here, including serial and parallel JIT.
        report = tool.run(inputs, output, [1, 2], 1, 10, None)
        assert report['status'] == tool.STATUS and report['dependency_scope'] == 'local_environment_interface_test_only'
        assert len(report['measurements']) == 4
        for item in report['measurements']:
            assert len(item['samples']) == 20 and set(item['summary']) == {'A', 'B'}
            assert all(item['compiled_signatures'].values())
            assert [row['case'] for row in item['samples'][:4]] == ['A', 'B', 'B', 'A']
            assert all(row['samples'] == 10 for row in item['summary'].values())
            if item['variant'] == 'strict_adapter': assert all(not rows for rows in item['native_tie_differences'].values())
        assert any(item['native_tie_differences']['empty'] for item in report['measurements'] if item['variant'] == 'original_native')
        fails(lambda: tool.run(inputs, output, [1], 1, 10, None))
        fails(lambda: tool.run(inputs, root/'invalid_workers', [1, 1], 1, 10, None))
        print('PASS: actual LOCAL unchanged BM25S + Numba serial/parallel raw-ID Search, all scores/strict ranks/empty/repeated outputs; '
              'first JIT separated and all four measurements checked. Not target Python3.9/Numba0.60 or NPU performance proof.')


if __name__ == '__main__': main()
