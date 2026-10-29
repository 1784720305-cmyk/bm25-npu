"""Paired original BM25S/NPU Search; unchanged upstream indexer and retrieval code."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import large_corpus as corpus, run_bm25s_index_search as api

r = api.r
SIZES = (1024,8192,65536,262144,1048576,4194304,10000000)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class ReplayCorpus:
    """Original indexer's repeated passes without retaining millions of Python lists."""
    def __init__(self, description):
        self.description = description

    def __len__(self):
        return self.description['docs']

    def __iter__(self):
        d = self.description; g = d['generation']
        source = Path(g['source_jsonl']) if g.get('source_jsonl') else None
        if source:
            r.require(corpus.digest(source) == g['source_sha256'], 'Corpus JSONL changed')
        return corpus.documents(d['docs'],g['vocab'],g['seed'],source)


def load_library(dependency):
    if dependency:
        from standalone.prepare_bm25s_numba import installed_record
        installed_record(dependency)
        sys.path.insert(0,str(dependency))
    r.require('NUMBA_DISABLE_JIT' not in os.environ,'JIT disabled')
    numba=importlib.import_module('numba')
    if dependency:
        r.require(dependency.resolve() in Path(numba.__file__).resolve().parents,'Wrong Numba')
    library,np,source=api.original.load_reference()
    return library,np,source,numba


def hashes(model):
    # No array-sized bytes copy for a multi-GB eager index.
    import hashlib
    return {key:hashlib.sha256(memoryview(model.scores[key]).cast('B')).hexdigest()
            for key in ('data','indices','indptr')}


def new_model(library):
    return library.BM25(method='atire',idf_method='lucene',k1=1.2,b=.75,
        dtype='float32',int_dtype='int32',backend='numba',csc_backend='numpy',auto_compile=False)


def build(inputs,output,dependency):
    d,index=corpus.validate(inputs)
    library,np,source,numba=load_library(dependency)
    r.require(not output.exists(),'Build output exists')
    output.mkdir(parents=True)
    model=new_model(library)
    # Public original accelerator; only construction, outside Search timing.
    model.activate_numba_csc()
    vocab={str(i):i for i in range(index['vocab_size'])};vocab['']=index['vocab_size']
    start=time.perf_counter_ns()
    model.index(SimpleNamespace(ids=ReplayCorpus(d),vocab=vocab),create_empty_token=False,show_progress=False)
    build_ms=(time.perf_counter_ns()-start)/1e6
    model.activate_numba_scorer()
    r.require(len(model.scores['data'])==d['posting_records'],'Original posting count differs')
    model.save(output/'model',show_progress=False)
    raw={c:corpus.load_raw(inputs/(c+'.txt')) for c in ('A','B')}
    agreement={};maximum_error={}
    for case,queries in raw.items():
        full=np.lib.format.open_memmap(output/(case+'_scores.npy'),mode='w+',dtype='float32',shape=(len(queries),d['docs']))
        gold=np.memmap(inputs/case/'expected_scores.bin',mode='r',dtype='<f4',shape=full.shape)
        ids=[];values=[];errors=[]
        for q,query in enumerate(queries):
            full[q]=model.get_scores_from_ids(query)
            errors.append(api.original.compare_scores(np,full[q:q+1],gold[q:q+1],'Original BM25S vs ordered FP32'))
            v,i=api.original.strict_select(np,full[q],d['k']);ids.append(i);values.append(v)
        ids=np.asarray(ids,dtype='int32');values=np.asarray(values,dtype='float32')
        expected=np.fromfile(inputs/case/'expected_top_ids.bin',dtype='<i4').reshape(ids.shape)
        agreement[case]=bool(np.array_equal(ids,expected));maximum_error[case]=max(errors)
        np.save(output/(case+'_strict_ids.npy'),ids);np.save(output/(case+'_strict_values.npy'),values)
        full.flush();del full,gold
    api.original.source_still_matches(source)
    result=dict(status='original_bm25s_large_index_and_formula_checked',index_id=index['index_id'],
        build_precompute_ms=build_ms,bm25s_source=source,index_hashes=hashes(model),
        index_array_bytes=sum(model.scores[k].nbytes for k in ('data','indices','indptr')),
        strict_ids_match_project=agreement,maximum_score_error=maximum_error,
        source_files={str(p):corpus.digest(p) for p in (output/'model').iterdir() if p.is_file()},
        oracle_files={str(p):corpus.digest(p) for p in output.glob('*.npy')},
        packages=api.original.package_versions(),construction='Unchanged public BM25.index + original activate_numba_csc; replayable original documents, no TF-index injection')
    r.write_json(output/'report.json',result)
    print('BM25S build/formula: '+json.dumps(result,ensure_ascii=False),flush=True)


def check_result(np,result,full,ids,values,strict):
    scores,selected=result
    r.require(np.array_equal(scores,np.take_along_axis(full,selected,axis=1)), 'Selected score differs from original full scores')
    r.require(np.array_equal(scores,values),'Original native did not select exact top score multiset')
    if strict:r.require(np.array_equal(selected,ids),'Strict adapter IDs differ from original full-sort contract')
    return int(np.count_nonzero(selected!=ids))


def measure(inputs,built,output,dependency,workers,strict,warmup=5,repeats=30):
    d,index=corpus.validate(inputs,rehash=False)
    prior=read(built/'report.json');r.require(prior['index_id']==index['index_id'],'Different built corpus')
    library,np,source,numba=load_library(dependency)
    r.require(workers<=numba.config.NUMBA_NUM_THREADS,'Numba worker limit too small')
    r.require(not output.exists(),'Measurement output exists');output.mkdir(parents=True)
    model=new_model(library);model.load_scores(built/'model',num_docs=d['docs'],mmap=False)
    model.vocab_dict=read(built/'model/vocab.index.json');model.unique_token_ids_set=set(model.vocab_dict.values())
    model.nonoccurrence_array=None;model.activate_numba_scorer()
    before=hashes(model);r.require(before==prior['index_hashes'],'Loaded eager index changed')
    raw={c:corpus.load_raw(inputs/(c+'.txt')) for c in ('A','B')}
    session=api.SearchSession(np,model,index['vocab_size'],d['k'],d['queries'],3*d['queries'],workers,strict)
    full={c:np.load(built/(c+'_scores.npy'),mmap_mode='r') for c in raw}
    ids={c:np.load(built/(c+'_strict_ids.npy')) for c in raw}
    values={c:np.load(built/(c+'_strict_values.npy')) for c in raw}
    saved={};native_differences={};samples=[]
    def verify(case,result):
        diff=check_result(np,result,full[case],ids[case],values[case],strict)
        bits=tuple(x.tobytes() for x in result)
        r.require(case not in saved or bits==saved[case],'Repeated BM25S output changed')
        saved[case]=bits;native_differences[case]=diff
    for case in ('A','B','A','B'):
        verify(case,session.search(raw[case]))
    # Native and strict paths compile different upstream functions.
    if strict:
        signatures={'original_scorer':[str(s) for s in model._compute_relevance_from_scores.nopython_signatures]}
    else:
        from bm25s.numba import retrieve_utils
        function=retrieve_utils._retrieve_internal_jitted_serial if workers==1 else retrieve_utils._retrieve_internal_jitted_parallel
        signatures={'original_retrieval':[str(s) for s in function.nopython_signatures]}
    r.require(all(signatures.values()),'Actual upstream nopython signatures missing')
    for cycle in range(warmup):
        for case in (('A','B') if cycle%2==0 else ('B','A')):
            verify(case,session.search(raw[case]))
    for cycle in range(repeats):
        for case in (('A','B') if cycle%2==0 else ('B','A')):
            result,elapsed=api.timed_search(session,raw[case]);verify(case,result)
            samples.append(dict(case=case,cycle=cycle,call_wall_ms=elapsed))
    r.require(hashes(model)==before,'BM25S index mutated');api.original.source_still_matches(source)
    report=dict(status='actual_original_bm25s_large_search_checked',variant='strict_adapter' if strict else 'original_native',
        index_id=index['index_id'],requested_query_workers=workers,maximum_active_queries=min(workers,d['queries']),
        timer_policy=api.TIMER,warmup_cycles=warmup,measured_cycles=repeats,
        native_id_differences_from_strict=native_differences,strict_ids_match_project=prior['strict_ids_match_project'],
        compiled_signatures=signatures,samples=samples,index_hashes=before,packages=api.original.package_versions(),
        source_sha256=source,summary={c:api.checks.summarize([x['call_wall_ms'] for x in samples if x['case']==c],d['queries']) for c in raw})
    r.write_json(output/'report.json',report)
    print('BM25S complete Search: '+json.dumps(report,ensure_ascii=False),flush=True)


def decide(rows,workers,strict_equivalence):
    result={}
    for variant in ('original_native','strict_adapter'):
        cases={}
        for case in ('A','B'):
            cpu=[min(row[f'{variant}_t{w}']['summary'][case]['median_batch_ms'] for w in workers) for row in rows]
            cp95=[min(row[f'{variant}_t{w}']['summary'][case]['p95_batch_ms'] for w in workers) for row in rows]
            npu=[row['npu']['summary'][case]['median_batch_ms'] for row in rows]
            np95=[row['npu']['summary'][case]['p95_batch_ms'] for row in rows]
            wins=sum(n<c for n,c in zip(npu,cpu))
            passed=len(rows)==6 and wins==6 and statistics.median(np95)<statistics.median(cp95)
            cases[case]=dict(cpu_medians_ms=cpu,npu_medians_ms=npu,cpu_p95_ms=cp95,npu_p95_ms=np95,
                cpu_over_npu=statistics.median(cpu)/statistics.median(npu),npu_wins=wins,
                timing_gate_passed=passed,strict_equivalent_win=bool(passed and variant=='strict_adapter' and strict_equivalence[case]))
        result[variant]=cases
    return result


def run(a):
    from standalone import warp_append as accepted
    r.require(sys.platform=='linux' and not a.output.exists(),'A5 Linux and fresh output required')
    r.require(a.output.is_relative_to(ROOT/'results'),'Output must be inside project results')
    r.require(all(not a.output.is_relative_to(p) and not p.is_relative_to(a.output)
                  for p in (a.accepted,a.dependency_dir)),'Output overlaps accepted evidence or dependencies')
    sweep=read(a.accepted/'report.json')
    r.require(sweep['status']==accepted.CHECKED,'Completed warp append sweep required')
    r.require(a.sizes==sorted(set(a.sizes)) and all(n in SIZES for n in a.sizes),'Use unique increasing accepted scale points')
    r.require(a.workers and len(set(a.workers))==len(a.workers) and all(1<=w<=256 for w in a.workers),'Invalid workers')
    own={str(p):corpus.digest(p) for p in (Path(__file__),ROOT/'standalone/run_bm25s_index_search.py',ROOT/'standalone/run_bm25s_reference.py')}
    a.output.mkdir(parents=True);report=dict(status='running',measurement='new same-session paired original BM25S and accepted NPU',tiers=[],tool_sha256=own)
    def save():
        r.write_json(a.output/'report.json',report)
        with (a.output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['docs','variant','case','cpu_ms','npu_ms','cpu_over_npu','npu_wins_of_6','timing_gate_passed','strict_equivalent_win'])
            for tier in report['tiers']:
                for variant,cases in tier.get('decision',{}).items():
                    for case,result in cases.items():
                        writer.writerow([tier['docs'],variant,case,statistics.median(result['cpu_medians_ms']),
                            statistics.median(result['npu_medians_ms']),result['cpu_over_npu'],result['npu_wins'],
                            result['timing_gate_passed'],result['strict_equivalent_win']])
    save()
    try:
        for n in a.sizes:
            link=next(x for x in sweep['tiers'] if x['docs']==n)
            point=Path(link['report']);r.require(corpus.digest(point)==link['report_sha256'],'Accepted point report changed')
            old=read(point);accepted.check_affected(old['affected_checks'])
            r.require(old['source_sha256']==accepted.g.checks.sources(accepted.OWN+accepted.retained.OWN+accepted.retained.HOST),'Accepted code changed')
            proofs={**old['previous_sha256'],**old['candidate_sha256']}
            accepted.simd.close.scale.unchanged(proofs,'Accepted artifacts changed')
            oldrun=Path(old['per_round'][0]['candidate']['directory'])/'report.json'
            r.require(corpus.digest(oldrun)==old['per_round'][0]['candidate']['report_sha256'],'NPU process proof changed')
            frozen=read(oldrun);command=frozen['command'];runner=Path(command[0]);inputs=Path(command[1]).parent
            r.require(corpus.digest(runner)==frozen['runner_sha256'],'NPU binary changed')
            d,index=corpus.validate(inputs);tier=dict(docs=n,input=str(inputs),workload=d,status='pending',per_round=[])
            report['tiers'].append(tier);save()
            # Conservative host construction allowance, not a measured peak.
            need=64*d['posting_records']+128*n+1024**3
            disk_need=8*d['posting_records']+8*d['queries']*n+6*4*d['queries']*n*4+1024**3
            tier.update(estimated_host_construction_bytes=need,estimated_disk_bytes=disk_need)
            if need>=accepted.retained.host_available_bytes()*4//5 or disk_need>=shutil.disk_usage(a.output).free*4//5:
                tier['status']='not_measured_memory_or_disk_budget';save();continue
            base=a.output/f'N{n}';base.mkdir();built=base/'bm25s_index'
            def worker_cmd(mode,out):
                return [sys.executable,'-u',str(Path(__file__)),mode,'--input',str(inputs),'--built',str(built),
                        '--output',str(out),'--dependency-dir',str(a.dependency_dir)]
            accepted.g.execute(worker_cmd('build',built),base/'build.log',timeout=14400)
            construction=read(built/'report.json');tier['construction']=construction;save()
            roles=[f'{v}_t{w}' for v in ('original_native','strict_adapter') for w in a.workers]+['npu']
            for ordinal in range(6):
                row=dict(round=ordinal+1,process_order=roles if ordinal%2==0 else list(reversed(roles)))
                for role in row['process_order']:
                    target=base/f'round{ordinal+1}'/role;target.parent.mkdir(parents=True,exist_ok=True)
                    if role=='npu':
                        actual=target/'actual';target.mkdir();cmd=list(command);cmd[4]=str(actual)
                        accepted.g.execute(cmd,target/'run.log',accepted.g.environment(Path(cmd[6])),timeout=3600)
                        item=accepted.retained.verify_run(actual,inputs,d,index,'npu',old['device'])
                        item.update(command=cmd,runner_sha256=corpus.digest(runner));r.write_json(target/'report.json',item)
                    else:
                        variant,w=role.rsplit('_t',1);cmd=worker_cmd('measure',target)+['--workers',w,'--variant',variant]
                        accepted.g.execute(cmd,target.parent/(role+'.log'),timeout=3600)
                        item=read(target/'report.json')
                        r.require(item['status']=='actual_original_bm25s_large_search_checked','BM25S measurement failed')
                    row[role]=dict(directory=str(target),summary=item['summary'],report_sha256=corpus.digest(target/'report.json'))
                    print(f'N={n} {role} complete Search: '+json.dumps(item['summary'],ensure_ascii=False),flush=True)
                tier['per_round'].append(row);save()
            accepted.simd.close.scale.unchanged(proofs,'Accepted artifacts changed during run')
            for files in ('source_files','oracle_files'):
                accepted.simd.close.scale.unchanged(construction[files],'BM25S built index/oracles changed')
            corpus.validate(inputs)
            tier.update(status='six_paired_rounds_checked',decision=decide(tier['per_round'],a.workers,construction['strict_ids_match_project']))
            save();print(f'N={n} BM25S comparison decision: '+json.dumps(tier['decision'],ensure_ascii=False),flush=True)
        r.require(all(corpus.digest(Path(p))==h for p,h in own.items()),'Comparison tool changed')
        report['status']='actual_bm25s_npu_requested_coverage_checked' if all(t['status']=='six_paired_rounds_checked' for t in report['tiers']) else 'incomplete_unmeasured_budget_points'
        report['strict_stable_npu_winning_docs']=[t['docs'] for t in report['tiers'] if t.get('decision') and all(x['strict_equivalent_win'] for x in t['decision']['strict_adapter'].values())]
        save();print('BM25S/NPU coverage: '+json.dumps(report,ensure_ascii=False),flush=True)
        return 0 if report['status']=='actual_bm25s_npu_requested_coverage_checked' else 2
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',error=str(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','build','measure'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--built',type=Path)
    p.add_argument('--dependency-dir',type=Path,required=True)
    p.add_argument('--accepted',type=Path,default=ROOT/'results/warp_append_scales1')
    p.add_argument('--sizes',type=int,nargs='+',default=list(SIZES));p.add_argument('--workers',type=int,nargs='+',default=[1,8,32])
    p.add_argument('--variant',choices=['original_native','strict_adapter'],default='original_native');a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    try:
        if a.mode=='run':return run(a)
        r.require(a.input is not None and a.built is not None,'Worker requires --input/--built')
        if a.mode=='build':build(a.input,a.output,a.dependency_dir)
        else:
            r.require(len(a.workers)==1,'One worker configuration per process')
            measure(a.input,a.built,a.output,a.dependency_dir,a.workers[0],a.variant=='strict_adapter')
        return 0
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED: '+str(error)+'; no replacement results or CPU/NPU speedup claim.',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
