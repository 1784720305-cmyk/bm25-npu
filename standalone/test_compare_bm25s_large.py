"""Actual local unchanged BM25S integration; not A5 performance evidence."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import compare_bm25s_large as tool


def main():
    import numpy as np
    cases={'A':[[i%8,(i+1)%8,i%8] for i in range(8)],
           'B':[[(i+2)%8,(i+3)%8,(i+4)%8] for i in range(8)]}
    with tempfile.TemporaryDirectory(prefix='bm25s_large_LOCAL_',dir=ROOT/'results') as temp:
        base=Path(temp);inputs=base/'input';built=base/'built'
        with redirect_stdout(io.StringIO()):
            description=tool.corpus.prepare(inputs,128,8,2026,cases)
            assert list(tool.ReplayCorpus(description))==list(tool.ReplayCorpus(description))
            def worker(code):
                process=subprocess.run([sys.executable,'-c',code,str(inputs),str(built),str(base)],cwd=ROOT,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                assert process.returncode==0,process.stdout
            worker("from pathlib import Path; import sys; from standalone import compare_bm25s_large as t; t.build(Path(sys.argv[1]),Path(sys.argv[2]),None)")
            construction=tool.read(built/'report.json')
            assert construction['status']=='original_bm25s_large_index_and_formula_checked'
            assert all(construction['strict_ids_match_project'].values())
            for workers in (1,2):
                for strict in (False,True):
                    out=base/f'w{workers}_s{strict}'
                    worker("from pathlib import Path; import sys; from standalone import compare_bm25s_large as t; "
                           f"t.measure(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3])/'{out.name}',None,{workers},{strict},1,10)")
                    report=tool.read(out/'report.json')
                    assert report['status']=='actual_original_bm25s_large_search_checked'
                    assert len(report['samples'])==20 and all(report['compiled_signatures'].values())
                    assert report['index_hashes']==construction['index_hashes']
                    if strict:assert not any(report['native_id_differences_from_strict'].values())
        # Separate fresh interpreter: verify the replayable corpus produces the exact
        # upstream eager index obtained from ordinary in-memory document lists.
        code='''from pathlib import Path
from types import SimpleNamespace
from standalone import compare_bm25s_large as t
import sys
base=Path(sys.argv[1]);d,_=t.corpus.validate(base/'input')
lib,np,source,numba=t.load_library(None)
m=t.new_model(lib);m.activate_numba_csc()
v={str(i):i for i in range(8)};v['']=8
m.index((list(t.ReplayCorpus(d)),v),create_empty_token=False,show_progress=False)
assert t.hashes(m)==t.read(base/'built/report.json')['index_hashes']
'''
        subprocess.run([sys.executable,'-c',code,str(base)],cwd=ROOT,check=True)
    item=lambda x:{'summary':{c:{'median_batch_ms':x,'p95_batch_ms':x*1.1} for c in ('A','B')}}
    rows=[{'npu':item(1),**{f'{v}_t{w}':item(2 if w==1 else 3)
        for v in ('original_native','strict_adapter') for w in (1,8)}} for i in range(6)]
    decision=tool.decide(rows,[1,8],{'A':True,'B':False})
    assert decision['strict_adapter']['A']['strict_equivalent_win']
    assert not decision['strict_adapter']['B']['strict_equivalent_win']
    assert not decision['original_native']['A']['strict_equivalent_win']
    rows[2]['npu']=item(2.1)
    assert not tool.decide(rows,[1,8],{'A':True,'B':True})['strict_adapter']['A']['timing_gate_passed']
    print('PASS: actual LOCAL upstream index parity, repeated corpus passes, FP32 formula/strict ranks, native/strict serial+parallel Search, nopython signatures, immutable index and six-round gates. Not A5 performance proof.')


if __name__=='__main__':main()
