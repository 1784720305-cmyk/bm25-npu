"""One short profile of the RETAINED 256-thread fused Search. No build/install/CPU run/new kernel."""
import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import fused_local32 as provenance, analyze_fused_search_profile as rows

accepted=provenance.accepted
t=accepted.previous
close=t.close
r=t.r
digest=t.digest
APP_STATUS='retained_t256_profile_application_and_checkpoints_passed'
STATUS='retained_t256_profile_diagnosis_completed'
VARIANT='retained_fused_FP32_SIMT256_local64_S32'
TOOLS=tuple(sorted(set(t.fused.TOOL_SOURCES)|set(t.fused.HOST_SOURCES)|set(close.fc.HOST_SOURCES)|
    set(close.checks.CPU_SOURCES)|{'standalone/profile_fused_t256.py','standalone/analyze_fused_search_profile.py',
    'standalone/analyze_pipeline_profile.py','standalone/resume_fused_threads256.py','standalone/fused_local32.py',
    'standalone/fused_threads256.py','standalone/profile_fused_search.py','standalone/close_fused_search.py',
    'standalone/fused_search_checks.py','standalone/search_benchmark_checks.py','standalone/benchmark_search.py'}))


def evidence(a):
    """Hash/read existing acceptance only; it is not resubmitted to the device."""
    index,records,snapshot,installation,proofs=t.prior.evidence(
        a.input,a.acceptance,a.runner,a.vendors,a.project,a.merge_project)
    old=json.loads((a.previous/'report.json').read_text(encoding='utf-8'))
    prior_device=old.get('device');r.require(r.integer(prior_device,0,r.LIMIT),'既有验收卡号非法')
    old,candidate,runtime,previous=accepted.validate_previous(a.previous,a.input,a.project,a.vendors,a.merge_project,
        a.runner,a.diagnostic_runner,a.cpu_runner,prior_device,index,records,snapshot,installation)
    proofs.update(previous);proofs.update(provenance.completed_evidence(a.completed,candidate,snapshot))
    proofs.update(old['runner_sha256'])
    return index,records,snapshot,installation,candidate,runtime,proofs,old.get('candidate_vf_reports',[])


def unchanged(a,base,tool_hashes):
    index,records,snapshot,installation,candidate,runtime,proofs,resources=base
    for group in (snapshot,proofs):close.scale.unchanged(group,'采集时输入/既有验收证据改变')
    r.require(close.checks.sources(TOOLS)==tool_hashes and
        t.fused.installed(a.vendors,a.project,a.merge_project)==installation and
        t.candidate_install(a.vendors/t.VENDOR,a.previous/'project',a.project)==candidate and
        accepted.opp_paths(runtime)==[str((a.vendors/t.VENDOR).resolve()),str((a.vendors/'bm25_topk_merge').resolve())],
        '采集时源码、已验收安装或运行库映射改变')


def application(a):
    r.require(not a.output.exists(),'采集应用目录已存在，不覆盖/重启')
    base=evidence(a)
    index,records,snapshot,installation,candidate,runtime,proofs,resources=base
    cache=close.checks.cache_inputs(a.input,index,records)
    hashes=close.checks.sources(TOOLS);binary=digest(a.runner)
    a.output.mkdir(parents=True);raw=a.output/'raw_queries';raw.mkdir()
    for name in ('A','B'):close.search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
    capq=max(x['num_queries'] for x in records.values());capwords=max(x['query_term_count'] for x in records.values())
    report={'status':'running','kernel_variant':VARIANT,'simt_threads':256,'device':a.device,
        'index':index,'records':records,'input_directory':str(a.input),'input_sha256':snapshot,'prior_sha256':proofs,
        'shards':32,'capacity_queries':capq,'capacity_words':capwords,'warmup':5,'repeats':10,
        'candidate_installation':candidate,'baseline_installation':installation,'vendors':str(a.vendors),
        'original_project':str(a.project),'candidate_project':str(a.previous/'project'),
        'merge_project':str(a.merge_project),'runtime':str(runtime),'runner':str(a.runner),'runner_sha256':binary,
        'host_source_sha256':close.checks.sources(close.fc.HOST_SOURCES),'tool_source_sha256':hashes,
        'raw_query_sha256':{str(p):digest(p) for p in raw.iterdir()},'reused_vf_reports':resources,
        'scope':'Retained256/local64 only. Profiling overhead is diagnostic; old acceptance/CPU/resource builds are not rerun.'}
    r.write_json(a.output/'report.json',report)
    try:
        unchanged(a,base,hashes)
        report['measurement']=accepted.measure(a.output/'measurement',a.input,index,records,cache,raw,
            a.runner,runtime,a.device,capq,capwords,5,10)
        unchanged(a,base,hashes)
        close.scale.unchanged(report['raw_query_sha256'],'采集查询改变')
        r.require(digest(a.runner)==binary,'采集调用器改变')
        report['status']=APP_STATUS;r.write_json(a.output/'report.json',report)
        print('256线程短采集应用完成：35次公开Search+4条额外调试链；只分析20次正式A/B调用。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));r.write_json(a.output/'report.json',report);raise


def validate_record(run):
    data=json.loads((run/'report.json').read_text(encoding='utf-8'))
    r.require(data.get('status')==APP_STATUS and data.get('simt_threads')==256 and
        data.get('kernel_variant')==VARIANT and data.get('tool_source_sha256')==close.checks.sources(TOOLS) and
        data.get('host_source_sha256')==close.checks.sources(close.fc.HOST_SOURCES),'采集记录/256线程来源/源码不符')
    vendors=Path(data['vendors']);original=Path(data['original_project']);project=Path(data['candidate_project'])
    r.require(t.fused.installed(vendors,original,Path(data['merge_project']))==data['baseline_installation'] and
        t.candidate_install(vendors/t.VENDOR,project,original)==data['candidate_installation'] and
        accepted.opp_paths(Path(data['runtime']))==[str((vendors/t.VENDOR).resolve()),str((vendors/'bm25_topk_merge').resolve())],
        '采集安装不是已验收256线程版，或发生替换')
    for group in (data['input_sha256'],data['prior_sha256'],data['raw_query_sha256']):
        close.scale.unchanged(group,'采集已有证据/输入改变')
    directory=run/'measurement';linked=data['measurement']
    r.require(linked.get('directory')==str(directory) and digest(directory/'report.json')==linked.get('report_sha256'),
        '采集单轮报告来源改变')
    one=json.loads((directory/'report.json').read_text(encoding='utf-8'));inputs=Path(data['input_directory'])
    cmd=[data['runner'],str(inputs/'index'),str(run/'raw_queries/A.txt'),str(run/'raw_queries/B.txt'),
        str(directory/'actual'),str(data['device']),data['runtime'],'10','32',str(data['capacity_queries']),
        str(data['capacity_words']),'5','10',str(inputs/'A/expected_scores.bin'),str(inputs/'B/expected_scores.bin')]
    r.require(one.get('status')=='canonical_opp_complete_search_and_checkpoints_passed' and one.get('command')==cmd and
        digest(Path(data['runner']))==one.get('runner_sha256')==data['runner_sha256'] and
        digest(directory/'run.log')==one.get('log_sha256') and
        one.get('ascend_custom_opp_paths')==accepted.opp_paths(Path(data['runtime'])),'采集命令/调用器/实际OPP路径/日志改变')
    session,summary=close.fc.check_session(directory/'actual','npu',data['index'],data['records'],32,1,
        data['device'],data['capacity_queries'],data['capacity_words'],5,10,0)
    r.require(session==one.get('session') and summary==one.get('summary')==linked.get('summary'),'采集样本或汇总改变')
    wanted={label+'/'+file for label in close.checks.CHECKPOINTS for file in close.scale.chain.FILES}
    r.require(set(one.get('output_sha256',{}))==wanted,'采集结果文件不完整')
    for relative,sha in one['output_sha256'].items():
        r.require(digest(directory/'actual'/relative)==sha,'采集正确性检查点改变：'+relative)
    return data,session


def analyze_rows(root,run,record,session):
    """Same two-operator CSV checks as the accepted128 analyzer; record is validated separately for256."""
    csvtools=rows.csvtools;operators=rows.OPERATORS;plan=rows.phases(session)
    r.require(record.get('simt_threads')==256 and record.get('kernel_variant')==VARIANT,'不能把其他版本标成256线程')
    paths=[csvtools.find_one(root,name) for name in ('op_summary*.csv','task_time*.csv','api_statistic*.csv')]
    summary=csvtools.read_table(paths[0],('Device_id','Task ID','Stream ID','OP Type','Task Type',
        'Task Start Time(us)','Task Duration(us)','Block Num','Input Shapes'))
    r.require(not any(x['OP Type'].startswith('Bm25') and x['OP Type'] not in operators for x in summary),'混入其他BM25流程')
    own=[x for x in summary if x['OP Type'] in operators]
    r.require(len(own)==2*len(plan),'设备任务数量与实际采集应用不符；不猜缺失轮次')
    r.require({x['Device_id'].strip() for x in own}=={str(record['device'])} and
        len({x['Stream ID'].strip() for x in own})==1 and all(x['Task Type']=='AI_VECTOR_CORE' for x in own),
        '设备、单Stream或任务类型不符')
    for item in own:
        item['_start']=csvtools.number(item['Task Start Time(us)'],'start')
        item['_duration']=csvtools.number(item['Task Duration(us)'],'duration');item['_blocks']=int(item['Block Num'])
        r.require(item['_duration']>0 and item['_blocks']>0,'设备耗时/工作块非法')
    own.sort(key=lambda x:x['_start'])
    r.require(len({(x['Stream ID'].strip(),x['Task ID'].strip()) for x in own})==len(own),'设备任务ID重复')
    tasks=csvtools.read_table(paths[1],('Device_id','kernel_name','kernel_type','stream_id','task_id',
        'task_time(us)','task_start(us)','task_stop(us)'))
    lookup={}
    for task in tasks:
        key=tuple(task[x].strip() for x in ('Device_id','stream_id','task_id'))
        r.require(key not in lookup,'task_time任务ID重复');lookup[key]=task
    tolerance=Decimal('0.002')
    for item in own:
        key=tuple(item[x].strip() for x in ('Device_id','Stream ID','Task ID'))
        r.require(key in lookup,'设备任务未在task_time找到');task=lookup[key]
        start=csvtools.number(task['task_start(us)'],'task start');stop=csvtools.number(task['task_stop(us)'],'task stop')
        duration=csvtools.number(task['task_time(us)'],'task duration')
        r.require(task['kernel_type']==item['Task Type'] and task['kernel_name'].startswith(item['OP Type']+'_') and
            stop>=start and abs(start-item['_start'])<=tolerance and abs(duration-item['_duration'])<=tolerance and
            abs(stop-start-duration)<=tolerance,'两张设备表的任务名称/时间不符')
    index=record['index'];records=record['records'];measured=[];selected=[]
    for pos,(phase,name,debug) in enumerate(plan):
        chain=own[2*pos:2*pos+2]
        r.require(tuple(x['OP Type'] for x in chain)==operators,'融合/合并任务次序不符')
        q=record['capacity_queries'] if phase=='cold' else records[name]['num_queries']
        words=1 if phase=='cold' else max(1,records[name]['query_term_count'])
        n,v,m,s,k=index['num_docs'],index['vocab_size'],index['posting_count'],record['shards'],records['A']['k']
        shapes=(((v+1,),(m,),(m,),(n,),(v,),(q+1,),(words,)),((q,s,k),(q,s,k)))
        for item,expected in zip(chain,shapes):
            r.require(csvtools.input_shapes(item['Input Shapes'])==expected,'实际设备输入形状不符')
        if 'Output Shapes' in chain[0]:
            r.require(csvtools.input_shapes(chain[0]['Output Shapes'])==((q,s,k),(q,s,k),((q,n) if debug else (1,))),
                '普通/调试输出形状错位')
        gap=chain[1]['_start']-chain[0]['_start']-chain[0]['_duration']
        r.require(gap>=0,'单Stream设备任务重叠')
        if phase!='measured':continue
        sample=session['samples'][len(measured)]
        r.require(sample['case']==name,'主机/设备正式调用错位')
        span=chain[1]['_start']+chain[1]['_duration']-chain[0]['_start']
        wall=Decimal(str(sample['call_wall_ms']))*1000
        r.require(wall>=span,'设备链跨度超过对应Search时间，不猜测匹配')
        measured.append({'sample_index':len(measured),'case':name,'cycle':sample['cycle'],
            'task_ids':[x['Task ID'].strip() for x in chain],
            'operator_us':{x['OP Type']:float(x['_duration']) for x in chain},
            'kernel_sum_us':float(sum((x['_duration'] for x in chain),Decimal(0))),
            'device_chain_span_us':float(span),'intra_chain_gap_sum_us':float(gap),
            'profiled_complete_search_us':float(wall),'outside_device_chain_elapsed_us':float(wall-span)})
        for item in chain:item['_case']=name
        selected.extend(chain)
    r.require(len(measured)==len(session['samples']),'正式主机/设备样本数不符')
    cases={}
    for name in ('A','B'):
        group=[x for x in measured if x['case']==name]
        cases[name]={'operators':{op:{**rows.stats([x['operator_us'][op] for x in group]),
            'block_counts':sorted({x['_blocks'] for x in selected if x['OP Type']==op and x['_case']==name})} for op in operators},
            **{key:rows.stats([x[key] for x in group]) for key in ('kernel_sum_us','device_chain_span_us',
                'intra_chain_gap_sum_us','profiled_complete_search_us','outside_device_chain_elapsed_us')}}
    api=csvtools.read_table(paths[2],('Device_id','Level','API Name','Time(us)','Count','Avg(us)','Min(us)','Max(us)'))
    return {'status':'retained_t256_device_rows_and_call_samples_cross_checked','kernel_variant':VARIANT,
        'device':record['device'],'docs':index['num_docs'],'queries':records['A']['num_queries'],'k':records['A']['k'],
        'simt_threads':256,'all_search_calls':len(plan)-4,'all_device_chains':len(plan),'own_operator_rows':len(own),
        'measured_search_calls':len(measured),'excluded_cold_calls':1,'excluded_before_checkpoints':2,
        'excluded_after_checkpoints':2,'excluded_debug_chains':4,'excluded_warmup_calls':2*session['warmup_cycles'],
        'cases':cases,'matched_samples':measured,'optional_raw_counters_by_task':[
            {'task_id':x['Task ID'].strip(),'case':x['_case'],'operator':x['OP Type'],
                'counters':{key:value for key,value in x.items() if key.startswith(('aiv_','aic_','cube_'))}} for x in selected],
        'whole_application_api_totals_not_steady_state':[{'name':x['API Name'],'level':x['Level'],'count':int(x['Count']),
            'total_us':float(csvtools.number(x['Time(us)'],'API total'))} for x in api],
        'source_sha256':{str(p):digest(p) for p in paths+[run/'report.json',Path(__file__)]},
        'scope':'Profile-only diagnosis. No new speedup result. Per-call wall minus SAME-call device span; '
        'outside time includes packing/transfers/prepare/submit/sync/cleanup/profiler, not pure memcpy. '
        'Raw pipeline ratios are not SIMT occupancy/whole-card utilization or a score-vs-selection breakdown.'}


def print_result(result):
    print(f'保留256线程版：核对{result["own_operator_rows"]}条设备任务，只统计{result["measured_search_calls"]}次正式Search。',flush=True)
    for name,item in result['cases'].items():
        print(name+'：',flush=True)
        for op,value in item['operators'].items():
            print(f'  {op} 中位数={value["median_us"]:.6g}us，p95={value["p95_us"]:.6g}us，块数={value["block_counts"]}',flush=True)
        print('  同次设备链跨度中位数={:.6g}us；采集下完整Search={:.6g}us；逐次链外耗时中位数={:.6g}us。'.format(
            item['device_chain_span_us']['median_us'],item['profiled_complete_search_us']['median_us'],
            item['outside_device_chain_elapsed_us']['median_us']),flush=True)
    print('链外时间不是纯传输；融合时间不能直接拆成评分/选择。工作块和vec_ratio不代表整卡利用率。普通Search基准仍保留，未重编译/安装/运行CPU。',flush=True)


def run(a):
    r.require(not a.output.exists(),'采集目录已存在，不覆盖/重启')
    r.require(a.output.is_relative_to((ROOT/'results').resolve()) and all(not a.output.is_relative_to(p) and
        not p.is_relative_to(a.output) for p in (a.input,a.previous,a.completed,a.project,a.vendors,a.acceptance)),
        '需要results内独立新目录')
    base=evidence(a);hashes=close.checks.sources(TOOLS)
    profiler=shutil.which('msprof');r.require(profiler is not None,'当前环境没有msprof；停止，不改变环境')
    help_run=subprocess.run([profiler,'--help'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    text=help_run.stdout.decode('utf-8',errors='replace')
    r.require(help_run.returncode==0 and all(flag in text for flag in
        ('--runtime-api','--task-time','--ai-core','--aic-metrics','--python-path')),'实际msprof帮助未确认既有选项；不猜参数')
    a.output.mkdir(parents=True);(a.output/'msprof_help.txt').write_bytes(help_run.stdout)
    report={'status':'running','device':a.device,'kernel_variant':VARIANT,'tool_source_sha256':hashes,
        'reused_vf_reports':base[7],'scope':'One short retained256 profile only; no kernel/host/CPU changes or resource rebuild.'}
    r.write_json(a.output/'report.json',report)
    try:
        command=[profiler,'--output='+str(a.output/'profiler'),'--runtime-api=on','--task-time=l1','--ai-core=on',
            '--aic-metrics=PipeUtilization','--python-path='+sys.executable,sys.executable,'-u','-B',str(Path(__file__).resolve()),'_application']
        for name in ('input','previous','completed','project','vendors','merge_project','runner','diagnostic_runner','cpu_runner','acceptance'):
            command.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
        command.extend(['--output',str(a.output/'application'),'--device',str(a.device)])
        env=os.environ.copy();env['ASCEND_CUSTOM_OPP_PATH']=':'.join(accepted.opp_paths(base[5]))
        report.update(command=command,profiler_sha256=digest(Path(profiler)));r.write_json(a.output/'report.json',report)
        print('只采集已验收256线程版；重复的五批验收、CPU六轮和资源编译全部跳过。',flush=True)
        print('工具帮助中的SIMT/缓存相关项：'+json.dumps([line.strip() for line in text.splitlines()
            if any(word in line.lower() for word in ('simt','dcache','atomic'))][:20],ensure_ascii=False)+
            '；仅使用已核对的PipeUtilization，不推测更细计数器。',flush=True)
        t.fused.score.execute(command,a.output/'msprof.log',env)
        unchanged(a,base,hashes)
        record,session=validate_record(a.output/'application')
        result=analyze_rows(a.output/'profiler',a.output/'application',record,session)
        r.write_json(a.output/'device_analysis.json',result);print_result(result)
        report.update(status=STATUS,device_analysis_sha256=digest(a.output/'device_analysis.json'))
        r.write_json(a.output/'report.json',report)
        print('采集完成；只需分享以上终端摘要，文件留在黄区。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));r.write_json(a.output/'report.json',report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','_application'))
    defaults={'previous':'results/fused_t256_close1','completed':'results/fused_t256_compare2','input':'results/search_256k_q8_input1',
        'project':'results/Bm25FusedScoreTopKUbFix1','vendors':'results/custom_opp/vendors','merge-project':'results/Bm25TopKMerge',
        'runner':'results/fused_search_host_build/bm25_fused_search_benchmark',
        'diagnostic-runner':'results/fused_chain_host_build/bm25_fused_chain_runner',
        'cpu-runner':'results/search_benchmark_host_build/bm25_cpu_search_benchmark','acceptance':'results/fused_search_256k_q8_close1'}
    for name,value in defaults.items():p.add_argument('--'+name,type=Path,default=ROOT/value)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=5)
    a=p.parse_args()
    try:
        r.require(sys.platform=='linux' and r.integer(a.device,0,r.LIMIT),'需要A5 Linux和合法卡号')
        for name,value in vars(a).items():
            if isinstance(value,Path):setattr(a,name,value.resolve())
        if a.mode=='_application':application(a)
        else:run(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；首错停止，不重建、不回退、不猜测轮次。',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
