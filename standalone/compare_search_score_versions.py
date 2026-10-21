"""Complete Search comparison: unchanged binaries/Top-K, isolated old/range score libraries; no profiling."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, benchmark_search as bench, search_benchmark_checks as checks
from standalone import search_scale_fixture as fixture, run_search_scale as scale, run_search as search
from standalone import run_query_score_range as score, validate_query_score_range_boundaries as boundary

SLOTS=('bm25_query_score','bm25_topk_radix_collect','bm25_topk_merge')
STATUS='complete_search_score_versions_paired_and_all_checks_passed'
TOOL_SOURCES=tuple(sorted(set(bench.TOOL_SOURCES)|set(boundary.TOOL_SOURCES)|{
    'standalone/compare_search_score_versions.py'}))
digest=score.digest


def validate_boundaries(directory,installation):
    directory=directory.resolve(strict=True)
    record=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    cases=record.get('cases',[])
    r.require(record.get('status')=='all_three_range_boundary_device_sessions_passed' and
        record.get('kind')==boundary.boundaries.VERSION and len(cases)==len(boundary.boundaries.CASES),
        '需要已通过的三组新评分设备边界报告')
    evidence={str(directory/'report.json'):digest(directory/'report.json')}
    for case,(name,(n,q)) in zip(cases,boundary.boundaries.CASES.items()):
        child=directory/name; report_path=child/'report.json'
        r.require(case.get('name')==name and case.get('num_docs')==n and case.get('queries_A')==q and
            case.get('report_sha256')==digest(report_path),'边界子报告缺失/变更')
        saved=json.loads(report_path.read_text(encoding='utf-8'))
        command=saved.get('command',[])
        r.require(saved.get('status')=='range_boundary_device_score_session_and_independent_formula_checks_passed' and
            saved.get('installation')==installation and len(command)==6,'边界设备实现或安装产物不符')
        inputs=Path(command[1]).parent
        index,records,snapshot=boundary.validate_inputs(inputs)
        r.require(index['num_docs']==n and records['A']['num_queries']==q and
            saved.get('index_id')==index['index_id'] and saved.get('input_sha256')==snapshot and
            saved.get('actual_max_query_terms')==boundary.enforce_query_limits(inputs,index,records) and
            saved.get('host_rejection')==boundary.boundaries.check_rejection(inputs,index),'边界输入/64词/拒绝记录不符')
        runner=Path(command[0]); device=saved.get('device')
        r.require(r.integer(device,0,2147483647) and device==record.get('device') and
            command==[str(runner),str(inputs/'index'),str(child/'requests.tsv'),str(child/'device'),str(device),installation['vendor']] and
            saved.get('runner_sha256')==digest(runner) and
            saved.get('host_source_sha256')==checks.sources(score.HOST_SOURCES) and
            saved.get('tool_source_sha256')==checks.sources(boundary.TOOL_SOURCES),'边界命令/程序/验收源码改变')
        session=score.check_session(child/'device',index,records)
        r.require(session==saved.get('session') and session.get('device')==device and
            session.get('host_source_sha256')==saved['host_source_sha256'],'边界实际会话记录不符')
        hashes={}
        with redirect_stdout(io.StringIO()):
            for i,label in enumerate(score.SEQUENCE):
                path=child/'device'/f'call{i}_{label}_scores.bin'
                score.fixture.check(inputs/'index',inputs/label,path)
                hashes[path.name]=digest(path); evidence[str(path)]=digest(path)
        r.require(hashes==saved.get('output_sha256') and digest(child/'device.log')==saved.get('device_log_sha256'),
            '边界设备输出或日志改变')
        for path in (report_path,child/'requests.tsv',child/'device.log',child/'device/device_session.json',runner):
            evidence[str(path)]=digest(path)
        evidence.update(snapshot)
        suite_path=inputs.parent/'suite.json'
        r.require(digest(suite_path)==record.get('suite_sha256'),'边界输入套件改变')
        evidence[str(suite_path)]=digest(suite_path)
    return {'directory':str(directory),'report_sha256':digest(directory/'report.json'),'evidence_sha256':evidence,
            'scope':'prior score-only actual device proof; no old timing reused'}


def create_runtime(path,targets):
    r.require(not path.exists(),'独立运行目录已存在，不覆盖')
    path.mkdir(parents=True)
    for slot in SLOTS:
        (path/slot).symlink_to(targets[slot].resolve(strict=True),target_is_directory=True)
    return validate_runtime(path,targets)


def validate_runtime(path,targets):
    mapping={}
    for slot in SLOTS:
        alias=path/slot; target=targets[slot].resolve(strict=True)
        r.require(alias.is_symlink() and alias.resolve(strict=True)==target,'运行库别名被替换：'+slot)
        header={'bm25_query_score':'aclnn_bm25_query_score.h',
                'bm25_topk_radix_collect':'aclnn_bm25_top_k.h','bm25_topk_merge':'aclnn_bm25_top_k_merge.h'}[slot]
        mapping[slot]={'actual_directory':str(target),'api_sha256':digest(alias/'op_api/lib/libcust_opapi.so'),
                       'header_sha256':digest(alias/'op_api/include'/header)}
    return mapping


def preflight(directory,inputs,index,records,cache,raw,runner,vendors,installation,device,shards,capq,capwords):
    directory.mkdir(parents=True)
    k=records['A']['k']; workspace=installation['topk_acceptance']['workspace_bytes']
    plan=directory/'requests.tsv'
    plan.write_bytes(('\n'.join(f'call{i}_{name}\t{raw/(name+".txt")}' for i,name in enumerate(score.SEQUENCE))+'\n').encode('utf-8'))
    command=[str(runner),str(inputs/'index'),str(plan),str(directory/'actual'),str(device),str(vendors),
             str(k),str(shards),str(capq),str(capwords),str(workspace)]
    import os
    env=os.environ.copy(); env['ASCEND_CUSTOM_OPP_PATH']=':'.join(str((vendors/slot).resolve(strict=True)) for slot in SLOTS)
    print('先核对五次完整Search：'+json.dumps(command,ensure_ascii=False),flush=True)
    score.execute(command,directory/'run.log',env)
    session=search.check_session(directory/'actual',index,records,device,k,shards,capq,capwords,workspace)
    validation={}; hashes={}
    for i,name in enumerate(score.SEQUENCE):
        label=f'call{i}_{name}'; actual=directory/'actual'/label
        validation[label]=checks.check_output(actual,cache[name],'npu',shards)
        hashes.update({label+'/'+file:digest(actual/file) for file in scale.chain.FILES})
    result={'status':'five_full_search_calls_and_independent_rankings_passed','command':command,
            'runner_sha256':digest(runner),'session':session,'checks':validation,'output_sha256':hashes,
            'log_sha256':digest(directory/'run.log'),'scope':'untimed correctness gate before paired measurements'}
    r.write_json(directory/'report.json',result)
    return result


def pair_summary(rows):
    result={}
    for case in ('A','B'):
        old=[row['baseline']['summary'][case]['median_batch_ms'] for row in rows]
        new=[row['candidate']['summary'][case]['median_batch_ms'] for row in rows]
        result[case]={'baseline_run_medians_ms':old,'candidate_run_medians_ms':new,
            'baseline_median_of_run_medians_ms':statistics.median(old),
            'candidate_median_of_run_medians_ms':statistics.median(new),
            'paired_baseline_over_candidate_ratios':[a/b for a,b in zip(old,new)],
            'candidate_faster_rounds':sum(a>b for a,b in zip(old,new)),'rounds':len(rows)}
    return result


def run(inputs,output,runner,diagnostic_runner,vendors,score_project,range_project,top_project,merge_project,
        top_acceptance,boundary_acceptance,device=2,shards=32,rounds=6,warmup=5,repeats=30):
    inputs,output,runner,diagnostic_runner,vendors=(p.resolve() for p in (inputs,output,runner,diagnostic_runner,vendors))
    r.require(not output.exists(),'结果目录已存在，请换新名字，不覆盖')
    r.require(runner.is_file() and diagnostic_runner.is_file() and r.integer(device,0,2147483647) and
        r.integer(shards,1,32) and r.integer(rounds,2,20) and rounds%2==0 and
        r.integer(warmup,1,10000) and r.integer(repeats,10,100000),'程序缺失或卡号/轮数/预热/测量参数非法')
    index,records,snapshot=fixture.validate_inputs(inputs)
    k,q=records['A']['k'],records['A']['num_queries']
    scale.chain.dimensions(index['num_docs'],k,shards)
    capq=max(item['num_queries'] for item in records.values())
    capwords=max(item['query_term_count'] for item in records.values())
    # Existing public PackQueries validates total capacity INSIDE Search. This
    # experiment deliberately only admits whole batches <=64 words, so no row can
    # reach the range kernel with >64 words. Same exact capacity on both versions.
    r.require(1<=capwords<=64,'本轮复用入口只支持整批最多64词；更大批次需另行加入计时内逐行保护，不能绕过')
    cache=checks.cache_inputs(inputs,index,records)
    baseline=scale.chain.validate_install(vendors,score_project,top_project,merge_project,top_acceptance)
    range_install=score.validate_install(vendors/'bm25_query_score_range',range_project)
    prior=validate_boundaries(boundary_acceptance,range_install)
    candidate=dict(baseline); candidate['score']=range_install
    sources=checks.sources(TOOL_SOURCES)
    hosts=checks.sources(tuple(sorted(set(checks.NPU_SOURCES)|set(search.HOST_SOURCES))))
    targets={name:(vendors/name).resolve(strict=True) for name in SLOTS}
    new_targets=dict(targets); new_targets['bm25_query_score']=(vendors/'bm25_query_score_range').resolve(strict=True)
    output.mkdir(parents=True)
    report={'status':'running','index_id':index['index_id'],'docs':index['num_docs'],'queries':q,'k':k,
        'device':device,'shards':shards,'capacity_queries':capq,'capacity_words':capwords,
        'supported_request_scope':'whole batch <=64 words; existing timed PackQueries capacity guard unchanged',
        'metric':'single_query_latency' if q==1 else 'batch_latency_and_search_time_throughput',
        'rounds':rounds,'warmup_cycles':warmup,'measured_cycles':repeats,'input_sha256':snapshot,
        'boundary_acceptance':prior,'baseline_installation':baseline,'candidate_installation':candidate,
        'runner_sha256':digest(runner),'diagnostic_runner_sha256':digest(diagnostic_runner),
        'tool_source_sha256':sources,'host_source_sha256':hosts,'resources':bench.host_resources(),
        'per_round':[],'source_is_real_corpus':False,
        'scope':'Same NPU/binary/Top-K/input; only score library differs. Complete query-ID to CPU Top-K Search wall time. No profiling, CPU or BM25S speedup claim.'}
    r.write_json(output/'report.json',report)
    try:
        runtime={}; mappings={}
        for role,mapped in (('baseline',targets),('candidate',new_targets)):
            runtime[role]=output/('runtime_'+role)/'vendors'
            mappings[role]=create_runtime(runtime[role],mapped)
        report['runtime_mapping']=mappings
        raw=output/'raw_queries'; raw.mkdir()
        for name in records: search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
        raw_hashes={str(p):digest(p) for p in raw.iterdir()}; report['raw_query_sha256']=raw_hashes
        report['preflight']={}
        for role,installation in (('baseline',baseline),('candidate',candidate)):
            report['preflight'][role]=preflight(output/('preflight_'+role),inputs,index,records,cache,raw,
                diagnostic_runner,runtime[role],installation,device,shards,capq,capwords)
            r.write_json(output/'report.json',report)
        # Both whole-Search formula/local/global rankings must pass BEFORE timed runs.
        for round_id in range(rounds):
            order=['baseline','candidate'] if round_id%2==0 else ['candidate','baseline']
            row={'round':round_id+1,'process_order':order,'status':'running'}
            report['per_round'].append(row); r.write_json(output/'report.json',report)
            print(f'完整Search新旧配对{round_id+1}/{rounds}：{order}',flush=True)
            for role in order:
                mapped=targets if role=='baseline' else new_targets
                r.require(validate_runtime(runtime[role],mapped)==mappings[role],'独立运行目录改变')
                row[role]=bench.run_one(output/f'round{round_id+1}'/role,'npu',inputs,index,records,cache,raw,
                    runner,runtime[role],baseline if role=='baseline' else candidate,
                    1,1,shards,device,capq,capwords,warmup,repeats)
                r.write_json(output/'report.json',report)
            row['cross_checks']={label:scale.cpu.compare_npu(Path(row['baseline']['directory'])/'actual'/label,
                Path(row['candidate']['directory'])/'actual'/label,q,index['num_docs'],k) for label in checks.CHECKPOINTS}
            row['status']='both_search_versions_and_all_cross_checks_passed'
            r.write_json(output/'report.json',report)
        scale.unchanged(snapshot,'运行期间输入改变'); scale.unchanged(raw_hashes,'原始查询改变')
        scale.unchanged(prior['evidence_sha256'],'既有评分设备验收证据改变')
        r.require(baseline==scale.chain.validate_install(vendors,score_project,top_project,merge_project,top_acceptance) and
            range_install==score.validate_install(vendors/'bm25_query_score_range',range_project),'安装产物改变')
        r.require(checks.sources(TOOL_SOURCES)==sources and checks.sources(tuple(hosts))==hosts and
            digest(runner)==report['runner_sha256'] and digest(diagnostic_runner)==report['diagnostic_runner_sha256'],
            '源码或调用器在运行期间改变')
        r.require(validate_runtime(runtime['baseline'],targets)==mappings['baseline'] and
            validate_runtime(runtime['candidate'],new_targets)==mappings['candidate'],'运行目录映射改变')
        report.update(status=STATUS,paired_summary=pair_summary(report['per_round']))
        r.write_json(output/'report.json',report)
        for name,summary in report['paired_summary'].items():
            print(f'{name}完整Search：旧版各轮ms={summary["baseline_run_medians_ms"]}；新版={summary["candidate_run_medians_ms"]}；新版更快{summary["candidate_faster_rounds"]}/{rounds}轮。',flush=True)
        print('全部正确性/严格排名/配对检查通过。报告保留所有样本和p95；只比较评分替换对Search的影响，不据此宣称超过CPU或BM25S。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        r.write_json(output/'report.json',report); raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'results/search_benchmark_host_build/bm25_npu_search_benchmark')
    p.add_argument('--diagnostic-runner',type=Path,default=ROOT/'results/search_host_build/bm25_search_runner')
    p.add_argument('--vendors',type=Path,default=ROOT/'results/custom_opp/vendors')
    p.add_argument('--score-project',type=Path,default=ROOT/'results/Bm25QueryScore')
    p.add_argument('--range-project',type=Path,default=ROOT/'results/Bm25QueryScoreRange')
    p.add_argument('--topk-project',type=Path,default=ROOT/'results/Bm25TopKRadixCollect')
    p.add_argument('--merge-project',type=Path,default=ROOT/'results/Bm25TopKMerge')
    p.add_argument('--topk-acceptance',type=Path,default=ROOT/'results/radix_collect_topk_validation1')
    p.add_argument('--boundary-acceptance',type=Path,default=ROOT/'results/query_score_range_boundaries_check1')
    p.add_argument('--device',type=int,default=2); p.add_argument('--shards',type=int,default=32)
    p.add_argument('--rounds',type=int,default=6); p.add_argument('--warmup',type=int,default=5); p.add_argument('--repeats',type=int,default=30)
    args=p.parse_args()
    try: run(args.input,args.output,args.runner,args.diagnostic_runner,args.vendors,args.score_project,args.range_project,
        args.topk_project,args.merge_project,args.topk_acceptance,args.boundary_acceptance,args.device,args.shards,
        args.rounds,args.warmup,args.repeats)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不回退CPU或其他评分实现，不生成成功汇总。',file=sys.stderr); return 1
    return 0


if __name__=='__main__': sys.exit(main())
