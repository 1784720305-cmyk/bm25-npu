"""Boundary fixtures, independent oracle and fail-stop orchestration tests. No ACLNN/NPU."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, query_score_edge_fixture as edges
from standalone import run_query_score as tool, validate_query_score_edges as suite
from standalone.accumulate_fixture import independent_scores as original_oracle


def main():
    checks = 0
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='score_edges_HOST_ONLY_',dir=ROOT/'results') as temp:
        base = Path(temp)
        inputs = base/'input'
        with redirect_stdout(io.StringIO()): edges.prepare(inputs)
        for n in edges.SIZES:
            directory = inputs/f'n{n}'
            index,records,snapshot = tool.validate_inputs(directory)
            documents,cases = edges.definition(edges.profile(n))
            assert index['num_docs'] == n and records['A']['num_queries'] == 65
            assert records['B']['num_queries'] == 7 and records['empty']['query_term_count'] == 0
            assert index['index_id'] == r.build_index(documents,edges.VOCAB_SIZE)[0]['index_id']
            checks += 3
            for marker in (0,127,128,2046,2047,2048,2049,4095,4096,n-1):
                if marker < n: assert 6 in documents[marker]; checks += 1
            assert any(not doc for doc in documents)
            assert all(7 not in doc and 8 not in doc for doc in documents)
            assert any(9 in q and 2147483646 in q for q in cases['A'])
            checks += 3
            # Check optimized oracle against the previous independent implementation,
            # which computes DF by rescanning the original documents for each match.
            if n == 129:
                assert list(edges.independent_scores(documents,cases['B'])) == original_oracle(documents,cases['B'])
                checks += 1
            ref = directory/'A/expected_scores.bin'
            raw = ref.read_bytes()
            ref.write_bytes(bytes(len(raw)))
            try: tool.validate_inputs(directory)
            except ValueError: checks += 1
            else: raise AssertionError('forged expected scores accepted')
            ref.write_bytes(raw)
            assert all(tool.digest(Path(p)) == value for p,value in snapshot.items())
            checks += 1

        for spec in ({'kind':edges.VERSION,'num_docs':True},
                     {'kind':edges.VERSION,'num_docs':2046},
                     {'kind':'unknown','num_docs':2048},
                     {'kind':edges.VERSION,'num_docs':2048,'extra':1}):
            try: edges.definition(spec)
            except ValueError: checks += 1
            else: raise AssertionError('unsupported profile accepted')

        # MOCK only the orchestration: no executable, ACL library or device output used.
        for failure in (None,2):
            output = base/('mock_pass' if failure is None else 'mock_fail')
            calls = []
            def mock_session(source,target,runner,vendor,project,device):
                calls.append(source.name)
                if len(calls) == failure: raise ValueError('INTENTIONAL MOCK DEVICE FAILURE')
                target.mkdir()
                index,unused = r.validate_index(source/'index')
                r.write_json(target/'report.json',{
                    'status':'device_score_session_and_independent_formula_checks_passed',
                    'device':device,'index_id':index['index_id'],'test_only_mock':True})
            with patch.object(tool,'run',mock_session),redirect_stdout(io.StringIO()):
                if failure is None: suite.run(inputs,output,Path('MOCK'),Path('MOCK'),Path('MOCK'),2)
                else:
                    try: suite.run(inputs,output,Path('MOCK'),Path('MOCK'),Path('MOCK'),2)
                    except ValueError: pass
                    else: raise AssertionError('device failure ignored')
            report = json.loads((output/'report.json').read_text())
            assert len(calls) == (5 if failure is None else 2)
            assert len(report['cases']) == (5 if failure is None else 1)
            assert report['status'] == ('all_five_device_score_edge_sessions_passed' if failure is None else 'failed')
            checks += 3
            with patch.object(tool,'run') as child:
                try: suite.run(inputs,output,Path('MOCK'),Path('MOCK'),Path('MOCK'),2)
                except ValueError: pass
                else: raise AssertionError('existing report overwritten')
                assert not child.called
                checks += 1
    print(f'PASS: {checks} edge-fixture/oracle/tamper/fail-stop checks. '
          'MOCK suite orchestration removed with temporary data. No C++ build, ACLNN or NPU execution.')


if __name__ == '__main__': main()
