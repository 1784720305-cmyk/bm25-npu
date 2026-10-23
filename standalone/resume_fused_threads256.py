"""Repair candidate OPP lookup paths and reuse completed 256-thread acceptance. No build/install/new kernels."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import fused_threads256 as previous

close=previous.close
fused=previous.fused
r=previous.r
digest=previous.digest
STATUS='fused_t256_canonical_opp_search_comparison_finished'


def opp_paths(vendors):
    """Match the already-passed diagnostic: API aliases may stay, but CANN receives real installed vendor paths."""
    return [str((vendors/name).resolve(strict=True)) for name in ('bm25_fused_score_topk','bm25_topk_merge')]


def measure(directory,inputs,index,records,cache,raw,runner,vendors,device,capq,capwords,warmup=5,repeats=30):
    r.require(not directory.exists(),'单轮目录已存在，不覆盖')
    r.require(r.integer(warmup,1,10000) and r.integer(repeats,10,100000),
        '现有已验收调用器要求预热至少1轮、重复至少10轮；不改变主机程序')
    paths=opp_paths(vendors);directory.mkdir(parents=True)
    command=[str(runner),str(inputs/'index'),str(raw/'A.txt'),str(raw/'B.txt'),str(directory/'actual'),str(device),str(vendors),
        '10','32',str(capq),str(capwords),str(warmup),str(repeats),
        str(inputs/'A/expected_scores.bin'),str(inputs/'B/expected_scores.bin')]
    env=os.environ.copy();env['ASCEND_CUSTOM_OPP_PATH']=':'.join(paths)
    report={'status':'running','command':command,'ascend_custom_opp_paths':paths,'runner_sha256':digest(runner),
        'scope':'Same accepted binary and complete Search timer; only OPP environment paths resolved before process launch.'}
    r.write_json(directory/'report.json',report)
    try:
        print('完整Search使用实际安装路径：'+env['ASCEND_CUSTOM_OPP_PATH'],flush=True)
        fused.score.execute(command,directory/'run.log',env)
        session,summary=close.fc.check_session(directory/'actual','npu',index,records,32,1,device,capq,capwords,warmup,repeats,0)
        verified,hashes=close.checks.check_checkpoints(directory/'actual','npu',cache,32)
        r.require(digest(runner)==report['runner_sha256'],'测量程序发生变化')
        report.update(status='canonical_opp_complete_search_and_checkpoints_passed',session=session,summary=summary,
            checks=verified,output_sha256=hashes,log_sha256=digest(directory/'run.log'))
        r.write_json(directory/'report.json',report)
        for case,value in summary.items():
            print(f'{case}完整Search：中位数={value["median_batch_ms"]:.6g}ms，p95={value["p95_batch_ms"]:.6g}ms。',flush=True)
        return {'directory':str(directory),'summary':summary,'report_sha256':digest(directory/'report.json')}
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));r.write_json(directory/'report.json',report);raise


def validate_previous(path,inputs,project,vendors,merge,runner,diag_runner,cpu_runner,device,index,records,snapshot,installed):
    """Recheck provenance/bytes of the passed acceptance; do not submit its requests again."""
    data=json.loads((path/'report.json').read_text(encoding='utf-8'))
    r.require(data.get('status')=='failed' and data.get('device')==device and data.get('input_sha256')==snapshot and
        data.get('baseline_installation')==installed,'需要同设备/输入/正式安装的已停止旧试验')
    sources=data.get('frozen_source_sha256',{})
    r.require(sources and close.checks.sources(tuple(sources))==sources,'旧试验冻结源码改变；不跳过验收')
    expected_binaries={str(p):digest(p) for p in (runner,diag_runner,cpu_runner)}
    r.require(data.get('runner_sha256')==expected_binaries,'旧试验使用的程序已改变')
    close.scale.unchanged(data.get('prior_proofs_sha256',{}),'既有证据发生变化')
    candidate=previous.candidate_install(vendors/previous.VENDOR,path/'project',project)
    r.require(candidate==data.get('candidate_installation'),'256线程源码、接口或安装产物改变')
    runtime=path/'runtime/vendors'
    r.require(opp_paths(runtime)==[str((vendors/previous.VENDOR).resolve()),str((vendors/'bm25_topk_merge').resolve())],
        '旧运行目录映射改变')
    accepted=json.loads((path/'acceptance/report.json').read_text(encoding='utf-8'))
    r.require(accepted==data.get('acceptance') and accepted.get('status')=='t256_actual_workload_correctness_passed',
        '256线程五批设备验收未通过或已改变；不继续计时')
    command=accepted.get('command',[])
    expected=[str(diag_runner),str(inputs/'index'),str(path/'acceptance/requests.tsv'),str(path/'acceptance/device'),
        str(device),str(runtime),'10','32']
    r.require(command==expected,'既有设备验收命令不是当前输入/程序/256线程运行目录')
    plan='\n'.join(f'call{i}_{name}\t{inputs/name}' for i,name in enumerate(fused.SEQUENCE))+'\n'
    r.require((path/'acceptance/requests.tsv').read_bytes()==plan.encode(),'既有查询计划改变')
    session_path=path/'acceptance/device/device_session.json'
    r.require(json.loads(session_path.read_text(encoding='utf-8'))==accepted['session'],'既有设备会话记录改变')
    session=accepted['session']
    expected_session={'status':'device_fused_chain_debug_modes_and_rankings_checked','device':device,
        'docs':index['num_docs'],'k':10,'shards':32,'pipeline_stages_per_mode':2,'index_upload_batches':1,
        'index_readback_byte_identical':True}
    r.require(all(type(session.get(k)) is type(v) and session[k]==v for k,v in expected_session.items()) and
        len(session.get('requests',[]))==5 and len(accepted.get('checks',[]))==5,'既有五批验收的设备/尺寸/记录不完整')
    r.require(accepted['session'].get('host_source_sha256')==close.checks.sources(fused.HOST_SOURCES),
        '既有验收不是当前主机源码')
    files={p.relative_to(path/'acceptance/device').as_posix():digest(p) for p in (path/'acceptance/device').rglob('*.bin')}
    wanted={f'call{i}_{name}/{mode}/{file}' for i,name in enumerate(fused.SEQUENCE)
        for mode,extra in (('debug1','scores.bin'),('debug0','debug_placeholder.bin')) for file in fused.CANDIDATES+(extra,)}
    r.require(set(files)==wanted and files==accepted.get('output_sha256') and digest(path/'acceptance/device.log')==accepted.get('log_sha256'),
        '既有设备结果或执行日志改变')
    proofs=dict(data['prior_proofs_sha256'])
    proofs.update({str(path/'acceptance/device'/name):sha for name,sha in files.items()})
    for relative in ('report.json','acceptance/report.json','acceptance/device.log','acceptance/requests.tsv',
                     'acceptance/device/device_session.json','comparison.json','round1/candidate/run.log'):
        file=path/relative
        if file.is_file():proofs[str(file)]=digest(file)
    return data,candidate,runtime,proofs


def compare(output,inputs,index,records,cache,raw,runner,cpu_runner,baseline,candidate,device,check):
    capq=max(x['num_queries'] for x in records.values());capwords=max(x['query_term_count'] for x in records.values())
    report={'status':'running','per_round':[],'cpu_rounds':[],'timer_policy':close.checks.TIMER,
        'old_partial_rounds_included':False,'parameters':{'device':device,'baseline_threads':128,'candidate_threads':256,
        'shards':32,'warmup':5,'repeats':30,'rounds':6}}
    def save():r.write_json(output/'comparison.json',report)
    def measured(directory,vendors,warmup=5,repeats=30):
        check();result=measure(directory,inputs,index,records,cache,raw,runner,vendors,device,capq,capwords,warmup,repeats)
        check();return result
    save()
    # Only the previously failing Search interface gets a short check. No compile/install/old five-request suite.
    print('先检查曾失败的完整Search入口：使用现有调用器最小1轮预热/10轮A+B；结果不用于性能比较。',flush=True)
    report['path_preflight']=measured(output/'candidate_path_check',candidate,1,10);save()
    for i in range(6):
        row={'round':i+1,'process_order':['baseline','candidate'] if i%2==0 else ['candidate','baseline']}
        print('同周期128/256线程完整Search第'+str(i+1)+'/6轮：'+str(row['process_order']),flush=True)
        for role in row['process_order']:
            row[role]=measured(output/f'round{i+1}'/role,baseline if role=='baseline' else candidate)
        close.cross(row,cache,32);report['per_round'].append(row);save()
    report['decision']=close.decision(report['per_round']);save()
    if report['decision']['keep_fused_candidate']:
        for i in range(6):
            row={'round':i+1,'process_order':['cpu','candidate'] if i%2==0 else ['candidate','cpu']}
            print('同期CPU32/256线程第'+str(i+1)+'/6轮：'+str(row['process_order']),flush=True)
            for role in row['process_order']:
                check()
                directory=output/f'cpu_round{i+1}'/role
                if role=='candidate':row[role]=measured(directory,candidate)
                else:
                    # Shared harness argument is unused for CPU: no device workspace or fictitious transfer.
                    row[role]=close.bench.run_one(directory,'cpu',inputs,index,records,cache,raw,cpu_runner,baseline,
                        {'topk_acceptance':{'workspace_bytes':0}},32,32,32,device,capq,capwords,5,30)
                    r.require(row[role]['actual_threads']==32,'实际CPU线程不足32')
                check()
            close.cross(row,cache,32,'cpu');report['cpu_rounds'].append(row);save()
        report['cpu_comparison']=close.decision([dict(x,baseline=x['cpu']) for x in report['cpu_rounds']])
    report['status']=STATUS;save();return report


def run(a):
    inputs,old,output,project,vendors,merge,runner,diag_runner,cpu_runner,acceptance=(getattr(a,n).resolve()
        for n in ('input','previous','output','project','vendors','merge_project','runner','diagnostic_runner','cpu_runner','acceptance'))
    r.require(not output.exists() and output.is_relative_to((ROOT/'results').resolve()) and
        not output.is_relative_to(old) and not old.is_relative_to(output),'需要独立新结果目录，不覆盖/嵌套旧试验')
    index,records,snapshot,installed,prior_proofs=previous.prior.evidence(inputs,acceptance,runner,vendors,project,merge)
    r.require((index['num_docs'],records['A']['num_queries'],records['A']['k'])==(262144,8,10),'只继续当前256K/Q8/K10/S32试验')
    old_record,candidate,runtime,proofs=validate_previous(old,inputs,project,vendors,merge,runner,diag_runner,cpu_runner,
        a.device,index,records,snapshot,installed)
    proofs.update(prior_proofs)
    sources=dict(old_record['frozen_source_sha256']);sources['standalone/resume_fused_threads256.py']=digest(Path(__file__).resolve())
    binaries=old_record['runner_sha256']
    output.mkdir(parents=True)
    report={'status':'running','previous_directory':str(old),'reused_acceptance':str(old/'acceptance/report.json'),
        'proof_sha256':proofs,'source_sha256':sources,'runner_sha256':binaries,'input_sha256':snapshot,
        'candidate_installation':candidate,'baseline_installation':installed,'device':a.device,
        'scope':'Only canonical OPP environment fix. Existing build/install/5-request device acceptance reused. '
        'Fresh complete Search pairs are required; old interrupted timings are excluded. No new kernels/parameters/CPU changes.'}
    r.write_json(output/'report.json',report)
    try:
        cache=close.checks.cache_inputs(inputs,index,records);raw=output/'raw_queries';raw.mkdir()
        for name in ('A','B'):close.search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
        raw_hash={str(p):digest(p) for p in raw.iterdir()}
        def unchanged():
            for value in (proofs,binaries,snapshot,raw_hash):close.scale.unchanged(value,'既有证据/输入/程序/查询改变')
            r.require(close.checks.sources(tuple(sources))==sources and fused.installed(vendors,project,merge)==installed and
                previous.candidate_install(vendors/previous.VENDOR,old/'project',project)==candidate and
                opp_paths(runtime)==[str((vendors/previous.VENDOR).resolve()),str((vendors/'bm25_topk_merge').resolve())],
                '冻结源码/安装或运行库映射改变')
        unchanged()
        print('复用已通过的256线程设备验收；不编译、不安装、不重跑A/B/A/空/B。只检查修复的Search入口及重新成对测量。',flush=True)
        report['comparison']=compare(output,inputs,index,records,cache,raw,runner,cpu_runner,vendors,runtime,a.device,unchanged)
        unchanged();report['status']=STATUS;r.write_json(output/'report.json',report)
        for name,value in report['comparison']['decision']['cases'].items():print(name+' 128/256线程：'+json.dumps(value,ensure_ascii=False),flush=True)
        for name,value in report['comparison'].get('cpu_comparison',{}).get('cases',{}).items():
            print(name+' CPU32/256线程：'+json.dumps(value,ensure_ascii=False),flush=True)
        print('结论：'+('256线程满足稳定改善门槛；CPU结果另列，正式128线程安装仍保留。' if
            report['comparison']['decision']['keep_fused_candidate'] else '256线程未达到保留门槛，继续使用128线程版，不追加线程扫描。'),flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));r.write_json(output/'report.json',report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    defaults={'previous':'results/fused_t256_close1','input':'results/search_256k_q8_input1',
        'project':'results/Bm25FusedScoreTopKUbFix1','vendors':'results/custom_opp/vendors','merge-project':'results/Bm25TopKMerge',
        'runner':'results/fused_search_host_build/bm25_fused_search_benchmark',
        'diagnostic-runner':'results/fused_chain_host_build/bm25_fused_chain_runner',
        'cpu-runner':'results/search_benchmark_host_build/bm25_cpu_search_benchmark','acceptance':'results/fused_search_256k_q8_close1'}
    for name,value in defaults.items():p.add_argument('--'+name,type=Path,default=ROOT/value)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=5)
    a=p.parse_args()
    try:r.require(sys.platform=='linux' and r.integer(a.device,0,r.LIMIT),'需要A5 Linux和合法设备号');run(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；首错停止，不重建旧工程或继续测速。',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
