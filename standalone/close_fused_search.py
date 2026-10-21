"""Frozen fused kernel closure: reuse device acceptance, gate normal workload, pair full Search, conditionally pair CPU."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import run_fused_chain as fused, fused_search_checks as fc
from standalone import benchmark_compact_search_cpu as prior, benchmark_search as bench
from standalone import search_benchmark_checks as checks, search_scale_fixture as fixture
from standalone import run_search_scale as scale, resident_index as r, run_search as search

digest=fused.digest
STATUS='fused_complete_search_decision_finished'


def acceptance(path,inputs,vendors,project,merge_project,k,shards,create=False,device=5):
    """Never restart an existing running/failed acceptance; inspect its immutable evidence."""
    if not path.exists():
        r.require(create,'当前实机验收未找到；先等待其完成，不另起验收：'+str(path))
        fused.run(inputs,path,ROOT/'results/fused_chain_host_build/bm25_fused_chain_runner',vendors,project,merge_project,device,k,shards)
    record=json.loads((path/'report.json').read_text(encoding='utf-8'))
    r.require(record.get('status')==fused.STATUS,'现有融合验收未通过/尚在运行；停止，不重启、不测速：'+str(path))
    index,records,snapshot,references=fused.validate_inputs(inputs)
    r.require(record.get('input_sha256')==snapshot and record.get('installation')==fused.installed(vendors,project,merge_project)
        and record.get('k')==k and record.get('shards')==shards,'融合验收不是当前输入/安装/参数')
    r.require(record.get('host_source_sha256')==checks.sources(fused.HOST_SOURCES)
        and record.get('tool_source_sha256')==checks.sources(fused.TOOL_SOURCES),'验收期间源码改变')
    cmd=record.get('command',[])
    r.require(len(cmd)==8 and digest(Path(cmd[0]))==record.get('runner_sha256') and
        cmd[1:]==[str(inputs/'index'),str(path/'requests.tsv'),str(path/'device'),str(record['device']),str(vendors),str(k),str(shards)],'验收调用程序/命令改变')
    r.require(digest(path/'device.log')==record.get('device_log_sha256') and digest(path/'requests.tsv')==record.get('plan_sha256'),'验收日志/计划改变')
    r.require(fused.check_session(path/'device',index,records,record['device'],k,shards)==record.get('session'),'验收设备记录改变')
    actual={p.relative_to(path/'device').as_posix():digest(p) for p in (path/'device').rglob('*.bin')}
    r.require(actual==record.get('output_sha256'),'既有融合验收输出改变')
    for i,name in enumerate(fused.SEQUENCE):
        fused.check_output(inputs,name,path/'device'/f'call{i}_{name}',index,records,k,shards,references[name])
    return {'report':str(path/'report.json'),'sha256':digest(path/'report.json'),'installation':record['installation']}


def decision(rows):
    """Predeclared operational gate, not a statistical significance claim."""
    r.require(len(rows)==6,'固定六轮；不按观测结果扩轮/选最低值')
    result={}; passed=True
    for name in ('A','B'):
        a=[x['baseline']['summary'][name]['median_batch_ms'] for x in rows]
        b=[x['candidate']['summary'][name]['median_batch_ms'] for x in rows]
        ap=[x['baseline']['summary'][name]['p95_batch_ms'] for x in rows]
        bp=[x['candidate']['summary'][name]['p95_batch_ms'] for x in rows]
        win=sum(y<x for x,y in zip(a,b)); reduction=1-statistics.median(b)/statistics.median(a)
        ok=win==6 and reduction>=.05 and statistics.median(bp)<=statistics.median(ap)
        result[name]={'baseline_medians_ms':a,'fused_medians_ms':b,'baseline_p95_ms':ap,'fused_p95_ms':bp,
            'faster_rounds':win,'median_reduction_fraction':reduction,'passes':ok}
        passed &= ok
    return {'keep_fused_candidate':bool(passed),'cases':result,
        'rule':'A and B each:6/6 faster; median-of-round-medians reduction>=5%; median-of-round-p95 not worse. Operational gate only.'}


def run_fused(directory,inputs,index,records,cache,raw,runner,vendors,device,shards,capq,capwords,warmup,repeats):
    directory.mkdir(parents=True)
    k=records['A']['k']; actual=directory/'actual'
    cmd=[str(runner),str(inputs/'index'),str(raw/'A.txt'),str(raw/'B.txt'),str(actual),str(device),str(vendors),
        str(k),str(shards),str(capq),str(capwords),str(warmup),str(repeats),
        str(inputs/'A/expected_scores.bin'),str(inputs/'B/expected_scores.bin')]
    env=os.environ.copy();env['ASCEND_CUSTOM_OPP_PATH']=':'.join(str(vendors/name) for name in ('bm25_fused_score_topk','bm25_topk_merge'))
    print('实际融合完整Search：'+json.dumps(cmd,ensure_ascii=False),flush=True)
    fused.score.execute(cmd,directory/'run.log',env)
    session,summary=fc.check_session(actual,'npu',index,records,shards,1,device,capq,capwords,warmup,repeats,0)
    verified,hashes=checks.check_checkpoints(actual,'npu',cache,shards)
    row={'directory':str(directory),'summary':summary,'session':session,'checks':verified,'output_sha256':hashes,
        'command':cmd,'runner_sha256':digest(runner),'log_sha256':digest(directory/'run.log')}
    r.write_json(directory/'report.json',row)
    return {'directory':str(directory),'summary':summary,'report_sha256':digest(directory/'report.json')}


def cross(row,cache,shards,other='baseline'):
    for checkpoint in checks.CHECKPOINTS:
        a=Path(row[other]['directory'])/'actual'/checkpoint
        b=Path(row['candidate']['directory'])/'actual'/checkpoint
        r.require((a/'top_ids.bin').read_bytes()==(b/'top_ids.bin').read_bytes(),'全部查询严格编号与对照不同')
        for file in ('scores.bin','top_scores.bin'):
            unused,x=fused.chain.read(a/file,'f',cache['A']['q']*cache['A']['n'] if file=='scores.bin' else cache['A']['q']*cache['A']['k'])
            unused,y=fused.chain.read(b/file,'f',len(x));r.compare(x,y)


def run(a):
    inputs=a.input.resolve();output=a.output.resolve();vendors=a.vendors.resolve();project=a.project.resolve();merge=a.merge_project.resolve()
    r.require(not output.exists(),'闭环结果目录已存在；不覆盖或重启')
    # First gate: the current in-flight small acceptance is READ ONLY, never re-submitted.
    small=acceptance(a.fused_acceptance.resolve(),a.small_input.resolve(),vendors,project,merge,3,2)
    index,records,snapshot=fixture.validate_inputs(inputs);q=records['A']['num_queries'];k=records['A']['k']
    r.require((index['num_docs'],q,k,a.shards)==(262144,8,10,32),'本轮固定256K/Q8/K10/S32，不扩大调参范围')
    cache=checks.cache_inputs(inputs,index,records)
    baseline_runner=a.baseline_runner.resolve();runner=a.runner.resolve();cpu_runner=a.cpu_runner.resolve()
    for p in (baseline_runner,runner,cpu_runner):r.require(p.is_file(),'主机程序缺失：'+str(p))
    # CMake embeds the exact source map used by WriteCommon; reject a stale executable before any device work.
    binary=runner.read_bytes()
    r.require(all(sha.encode('ascii') in binary for sha in checks.sources(fc.HOST_SOURCES).values()),
        '融合Search二进制缺少当前源码指纹，请先配置编译；不启动设备或测速')
    old=prior.validate_selected(a.baseline_acceptance.resolve(),inputs,index,records,snapshot,cache,baseline_runner,vendors,
        {'score':a.range_project.resolve(),'topk':a.compact_project.resolve(),'merge':merge})
    output.mkdir(parents=True)
    report={'status':'running','input_sha256':snapshot,'small_acceptance':small,'per_round':[],'cpu_rounds':[],
        'device':a.device,'kernel_parameters':{'threads':128,'shards':32,'dynamic_ub_bytes':65536},
        'timer_policy':checks.TIMER,'scope':'Fixed synthetic256K/Q8; complete Search; not universal or full-card utilization proof.',
        'decision_rule':'A and B each:6/6 faster; median reduction>=5%; median p95 no worse; otherwise retain old implementation.',
        'resources':bench.host_resources()}
    r.write_json(output/'report.json',report)
    names=set(fc.HOST_SOURCES)|set(fused.TOOL_SOURCES)|set(prior.TOOL_SOURCES)|{
        'standalone/close_fused_search.py','standalone/fused_search_checks.py'}
    names.update('npu_prototype/fused_score_topk/'+i for i in fused.generator.SOURCES)
    frozen=checks.sources(tuple(sorted(names)))
    binaries={str(p):digest(p) for p in (runner,baseline_runner,cpu_runner)}
    try:
        # Only the changed fused path on the target workload; no old suite rerun.
        report['workload_acceptance']=acceptance(output/'workload_acceptance',inputs,vendors,project,merge,k,32,True,a.device)
        raw=output/'raw_queries';raw.mkdir()
        for name in ('A','B'):search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
        hashes={str(p):digest(p) for p in raw.iterdir()};runtime=output/'baseline_runtime/vendors'
        mapping=prior.selected.previous.create_runtime(runtime,old['targets'])
        capq=max(x['num_queries'] for x in records.values());capwords=max(x['query_term_count'] for x in records.values())
        def check_frozen():
            scale.unchanged(snapshot,'运行中输入改变');scale.unchanged(hashes,'原始查询改变')
            r.require(checks.sources(tuple(frozen))==frozen,'源码冻结被破坏')
            scale.unchanged(binaries,'主机程序改变')
            r.require(fused.installed(vendors,project,merge)==small['installation'],'融合安装被替换')
            r.require(prior.selected.previous.validate_runtime(runtime,old['targets'])==mapping,'旧版运行库映射改变')
        for i in range(6):
            row={'round':i+1,'process_order':['baseline','candidate'] if i%2==0 else ['candidate','baseline']}
            print('新旧完整Search第'+str(i+1)+'/6轮：'+str(row['process_order']),flush=True)
            for role in row['process_order']:
                check_frozen()
                if role=='baseline':row[role]=bench.run_one(output/f'round{i+1}'/role,'npu',inputs,index,records,cache,raw,
                    baseline_runner,runtime,old['installation'],1,1,32,a.device,capq,capwords,5,30)
                else:row[role]=run_fused(output/f'round{i+1}'/role,inputs,index,records,cache,raw,runner,vendors,a.device,32,capq,capwords,5,30)
            cross(row,cache,32);report['per_round'].append(row);r.write_json(output/'report.json',report)
        report['decision']=decision(report['per_round']);r.write_json(output/'report.json',report)
        if report['decision']['keep_fused_candidate']:
            print('融合满足预定改善门槛，继续与CPU32线程/32文章分区同期六轮；不重新扫描CPU。',flush=True)
            for i in range(6):
                row={'round':i+1,'process_order':['cpu','candidate'] if i%2==0 else ['candidate','cpu']}
                for role in row['process_order']:
                    check_frozen()
                    if role=='cpu':row[role]=bench.run_one(output/f'cpu_round{i+1}'/role,'cpu',inputs,index,records,cache,raw,
                        cpu_runner,runtime,old['installation'],32,32,32,a.device,capq,capwords,5,30)
                    else:row[role]=run_fused(output/f'cpu_round{i+1}'/role,inputs,index,records,cache,raw,runner,vendors,a.device,32,capq,capwords,5,30)
                    if role=='cpu':r.require(row[role]['actual_threads']==32,'CPU实际线程不足32，不接受降配对手')
                cross(row,cache,32,'cpu');report['cpu_rounds'].append(row);r.write_json(output/'report.json',report)
            report['cpu_comparison']=decision([dict(x,baseline=x['cpu']) for x in report['cpu_rounds']])
        check_frozen();scale.unchanged(old['evidence_sha256'],'旧验收证据改变')
        report['status']=STATUS;r.write_json(output/'report.json',report)
        for name,row in report['decision']['cases'].items():print(name+': '+json.dumps(row,ensure_ascii=False),flush=True)
        for name,row in report.get('cpu_comparison',{}).get('cases',{}).items():
            print('CPU32/融合同期 '+name+': '+json.dumps(row,ensure_ascii=False),flush=True)
        print('结论：'+('保留融合候选；CPU比较另列，不承诺已超过CPU。' if report['decision']['keep_fused_candidate'] else '未达到预定保留门槛，保留旧版；停止继续堆融合优化。'),flush=True)
    except (Exception,KeyboardInterrupt) as e:
        report.update(status='failed',first_error=str(e));r.write_json(output/'report.json',report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    defaults={'input':'results/search_256k_q8_input1','fused-acceptance':'results/fused_chain_small_s2_ubfix1',
        'small-input':'results/query_score_small_input1','baseline-acceptance':'results/search_topk_compact_256k_q8_pair1',
        'runner':'results/fused_search_host_build/bm25_fused_search_benchmark',
        'baseline-runner':'results/search_benchmark_host_build/bm25_npu_search_benchmark',
        'cpu-runner':'results/search_benchmark_host_build/bm25_cpu_search_benchmark',
        'vendors':'results/custom_opp/vendors','project':'results/Bm25FusedScoreTopKUbFix1',
        'merge-project':'results/Bm25TopKMerge','range-project':'results/Bm25QueryScoreRange','compact-project':'results/Bm25TopKPositiveCompact'}
    for name,value in defaults.items():p.add_argument('--'+name,type=Path,default=ROOT/value)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=5)
    p.add_argument('--shards',type=int,choices=(32,),default=32)
    a=p.parse_args()
    try:
        r.require(sys.platform=='linux','请在A5 Linux执行');run(a)
    except (Exception,KeyboardInterrupt) as e:print('FAILED：'+str(e)+'；首错即停，不测速或CPU替代。',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
