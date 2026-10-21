"""One frozen fused Search profile; reuse completed acceptance, no old suite/CPU benchmark or kernel changes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import close_fused_search as close, fused_search_checks as fc
from standalone import search_benchmark_checks as checks, search_scale_fixture as fixture, run_search as search
from standalone import resident_index as r, run_search_scale as scale

STATUS='frozen_fused_profile_application_and_checkpoints_passed'
TOOLS=('standalone/profile_fused_search.py','standalone/analyze_fused_search_profile.py','standalone/fused_resource_report.py')
digest=close.digest


def evidence(inputs,acceptance,runner,vendors,project,merge_project):
    index,records,snapshot=fixture.validate_inputs(inputs)
    r.require((index['num_docs'],records['A']['num_queries'],records['A']['k'])==(262144,8,10),'仅当前256K/Q8/K10工作负载')
    report=json.loads((acceptance/'report.json').read_text(encoding='utf-8'))
    r.require(report.get('status')==close.STATUS and report.get('input_sha256')==snapshot and
        report.get('decision',{}).get('keep_fused_candidate') is True and len(report.get('per_round',[]))==6,
        '需要当前输入已完成且决定保留融合版的六轮报告；不重启验收')
    installation=close.fused.installed(vendors,project,merge_project)
    proof=report['workload_acceptance']; path=acceptance/'workload_acceptance'
    r.require(proof.get('report')==str(path/'report.json') and digest(path/'report.json')==proof.get('sha256') and
        proof.get('installation')==installation,'已通过设备验收/安装来源改变')
    accepted=json.loads((path/'report.json').read_text(encoding='utf-8'))
    r.require(accepted.get('status')==close.fused.STATUS and accepted.get('input_sha256')==snapshot and
        accepted.get('installation')==installation and (accepted.get('k'),accepted.get('shards'))==(10,32) and
        accepted.get('host_source_sha256')==checks.sources(close.fused.HOST_SOURCES),
        '既有正常规模融合验收不符')
    # Hash existing proof, no resubmission or recomputation of the accepted device suite.
    files={str(acceptance/'report.json'):digest(acceptance/'report.json'),str(path/'report.json'):digest(path/'report.json')}
    r.require(digest(path/'device.log')==accepted.get('device_log_sha256'),'既有设备验收日志改变')
    files[str(path/'device.log')]=digest(path/'device.log')
    expected_outputs={f'call{i}_{name}/{mode}/{file}' for i,name in enumerate(close.fused.SEQUENCE)
        for mode,files in (('debug1',close.fused.CANDIDATES+('scores.bin',)),('debug0',close.fused.CANDIDATES+('debug_placeholder.bin',)))
        for file in files}
    r.require(set(accepted['output_sha256'])==expected_outputs,'既有验收输出文件集合缺失')
    for relative,sha in accepted['output_sha256'].items():
        actual=path/'device'/relative
        r.require(actual.resolve().is_relative_to((path/'device').resolve()) and digest(actual)==sha,'既有设备验收输出改变')
        files[str(actual)]=sha
    linked=report['per_round'][0]['candidate']; directory=acceptance/'round1/candidate'
    r.require(linked.get('directory')==str(directory) and digest(directory/'report.json')==linked.get('report_sha256'),
        '已通过的融合Search单轮报告改变')
    measured=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    r.require(measured.get('runner_sha256')==digest(runner) and measured['command'][0]==str(runner) and
        measured.get('session',{}).get('host_source_sha256')==checks.sources(fc.HOST_SOURCES),'实际Search程序或源码改变')
    binary=runner.read_bytes()
    r.require(all(sha.encode('ascii') in binary for sha in checks.sources(fc.HOST_SOURCES).values()),'Search二进制不是当前已验收源码')
    files[str(directory/'report.json')]=digest(directory/'report.json')
    return index,records,snapshot,installation,files


def application(inputs,acceptance,output,runner,vendors,project,merge_project,device=5):
    r.require(not output.exists(),'采集应用目录已存在，不覆盖/重启')
    index,records,snapshot,installation,prior=evidence(inputs,acceptance,runner,vendors,project,merge_project)
    cache=checks.cache_inputs(inputs,index,records)
    output.mkdir(parents=True);raw=output/'raw_queries';raw.mkdir()
    for name in ('A','B'):search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
    record={'status':'running','input_directory':str(inputs),'index':index,'records':records,'device':device,
        'shards':32,'capacity_queries':8,'capacity_words':max(x['query_term_count'] for x in records.values()),
        'warmup':5,'repeats':10,'installation':installation,'vendors':str(vendors),'project':str(project),
        'merge_project':str(merge_project),'input_sha256':snapshot,'prior_sha256':prior,
        'runner_sha256':digest(runner),'host_source_sha256':checks.sources(fc.HOST_SOURCES),
        'tool_source_sha256':checks.sources(TOOLS),'raw_query_sha256':{str(p):digest(p) for p in raw.iterdir()},
        'scope':'Only frozen fused+merge; current ordinary benchmark retained. Profiler overhead is not a new speed result.'}
    r.write_json(output/'report.json',record)
    try:
        record['measurement']=close.run_fused(output/'measurement',inputs,index,records,cache,raw,runner,vendors,
            device,32,8,record['capacity_words'],5,10)
        for proof in (snapshot,prior,record['raw_query_sha256']):scale.unchanged(proof,'采集时已有证据/输入改变')
        r.require(close.fused.installed(vendors,project,merge_project)==installation and digest(runner)==record['runner_sha256']
            and checks.sources(fc.HOST_SOURCES)==record['host_source_sha256'] and checks.sources(TOOLS)==record['tool_source_sha256'],
            '采集时安装/源码/调用器改变')
        record['status']=STATUS;r.write_json(output/'report.json',record)
        print('融合采集应用完成：39条两段链/78条设备任务，其中仅20次正式A/B Search用于分析；其余为冷启动、检查及预热。',flush=True)
    except (Exception,KeyboardInterrupt) as e:
        record.update(status='failed',first_error=str(e));r.write_json(output/'report.json',record);raise


def validate_record(output):
    data=json.loads((output/'report.json').read_text(encoding='utf-8'))
    r.require(data.get('status')==STATUS and data.get('host_source_sha256')==checks.sources(fc.HOST_SOURCES)
        and data.get('tool_source_sha256')==checks.sources(TOOLS),'采集应用未通过或源码已改变')
    r.require(close.fused.installed(Path(data['vendors']),Path(data['project']),Path(data['merge_project']))==data['installation'],
        '采集实际算子改变')
    for proof in (data['input_sha256'],data['prior_sha256'],data['raw_query_sha256']):scale.unchanged(proof,'采集证据改变')
    linked=data['measurement']; directory=output/'measurement'
    r.require(linked.get('directory')==str(directory) and digest(directory/'report.json')==linked.get('report_sha256'),'采集单轮来源改变')
    one=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    inputs=Path(data['input_directory']); cmd=one.get('command',[])
    r.require(len(cmd)==15 and digest(Path(cmd[0]))==one.get('runner_sha256')==data['runner_sha256'] and
        cmd[1:]==[str(inputs/'index'),str(output/'raw_queries/A.txt'),str(output/'raw_queries/B.txt'),str(directory/'actual'),
        str(data['device']),data['vendors'],'10','32','8',str(data['capacity_words']),'5','10',
        str(inputs/'A/expected_scores.bin'),str(inputs/'B/expected_scores.bin')]
        and digest(directory/'run.log')==one.get('log_sha256'),'采集命令/调用器/日志改变')
    session,summary=fc.check_session(directory/'actual','npu',data['index'],data['records'],32,1,data['device'],8,data['capacity_words'],5,10,0)
    r.require(session==one.get('session') and summary==one.get('summary')==linked.get('summary'),'采集样本/汇总改变')
    hashes={label+'/'+file:digest(directory/'actual'/label/file) for label in checks.CHECKPOINTS for file in scale.chain.FILES}
    r.require(hashes==one.get('output_sha256'),'采集检查点缺失或改变')
    return data,session


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','_application'))
    defaults={'input':'results/search_256k_q8_input1','acceptance':'results/fused_search_256k_q8_close1',
        'runner':'results/fused_search_host_build/bm25_fused_search_benchmark','vendors':'results/custom_opp/vendors',
        'project':'results/Bm25FusedScoreTopKUbFix1','merge-project':'results/Bm25TopKMerge'}
    for name,value in defaults.items():p.add_argument('--'+name,type=Path,default=ROOT/value)
    p.add_argument('--device',type=int,default=5);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        r.require(sys.platform=='linux','实采仅A5 Linux执行')
        paths=[a.input,a.acceptance,a.output,a.runner,a.vendors,a.project,a.merge_project]
        inputs,acceptance,output,runner,vendors,project,merge_project=[x.resolve() for x in paths]
        if a.mode=='_application':application(inputs,acceptance,output,runner,vendors,project,merge_project,a.device);return 0
        r.require(not output.exists() and r.integer(a.device,0,2147483647),'结果已存在或卡号非法，不覆盖/重启')
        # Fail before creating a profiler directory if acceptance or current artifacts are wrong.
        evidence(inputs,acceptance,runner,vendors,project,merge_project)
        profiler=shutil.which('msprof');r.require(profiler is not None,'msprof不在当前环境；不调整安装')
        help_run=subprocess.run([profiler,'--help'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        help_text=help_run.stdout.decode('utf-8',errors='replace')
        options=('--runtime-api','--task-time','--ai-core','--aic-metrics','--python-path')
        r.require(help_run.returncode==0 and all(x in help_text for x in options),'实际msprof帮助未确认既有采集选项；停止，不猜参数')
        output.mkdir(parents=True);(output/'msprof_help.txt').write_bytes(help_run.stdout)
        related=[line.strip() for line in help_text.splitlines() if any(word in line.lower() for word in ('simt','dcache','atomic'))]
        print('实际工具帮助中的SIMT/缓存相关项：'+json.dumps(related[:20],ensure_ascii=False)+'；本轮仅启用已确认的PipeUtilization。',flush=True)
        command=[profiler,'--output='+str(output/'profiler'),'--runtime-api=on','--task-time=l1','--ai-core=on',
            '--aic-metrics=PipeUtilization','--python-path='+sys.executable,sys.executable,'-B',str(Path(__file__).resolve()),
            '_application','--input',str(inputs),'--acceptance',str(acceptance),'--output',str(output/'application'),
            '--runner',str(runner),'--vendors',str(vendors),'--project',str(project),'--merge-project',str(merge_project),'--device',str(a.device)]
        env=os.environ.copy();env['ASCEND_CUSTOM_OPP_PATH']=':'.join(str(vendors/x) for x in ('bm25_fused_score_topk','bm25_topk_merge'))
        print('一次短融合设备采集；既有编译/CPU/正式六轮不重跑：'+json.dumps(command,ensure_ascii=False),flush=True)
        close.fused.score.execute(command,output/'msprof.log',env)
        from standalone import analyze_fused_search_profile as analysis, fused_resource_report as resource
        result=analysis.analyze(output/'profiler',output/'application')
        r.write_json(output/'device_analysis.json',result);analysis.print_result(result)
        resources=resource.run(project,output/'resource_report',runner,vendors,merge_project)
        r.write_json(output/'report.json',{'status':'fused_device_profile_and_resource_inspection_completed',
            'device_analysis_sha256':digest(output/'device_analysis.json'),'resources':resources,
            'scope':'Diagnosis only. Existing0.24ms full Search baseline retained; accepted install untouched.'})
        print('短采集及资源检查完成。只分享终端摘要，黄区文件不导出。',flush=True)
    except (Exception,KeyboardInterrupt) as e:
        print('FAILED：'+str(e)+'；首错停止；不安装、不回退、不覆盖既有结果。',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
