"""Profile the completed graph-directory binary once; no build/install/CPU run or kernel changes."""
import argparse
from collections import defaultdict
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import graph_search as g, analyze_fused_search_profile as rows

r=g.r
csvtools=rows.csvtools
OPERATORS=rows.OPERATORS
STATUS='accepted_graph_directory_profile_matched_not_performance_acceptance'
TOOLS=('standalone/profile_graph_directory.py','standalone/analyze_fused_search_profile.py',
       'standalone/analyze_pipeline_profile.py')


def evidence(completed):
    report=json.loads((completed/'report.json').read_text(encoding='utf-8'))
    r.require(report.get('status') in (g.PASS,g.MISS) and len(report.get('per_round',[]))==6,
        '需要已经完整结束的 Graph/目录闭环；不重启验收')
    frozen=report.get('host_and_tool_sha256',{})
    r.require(set(g.HOST_SOURCES)<=set(frozen) and frozen==g.checks.sources(tuple(frozen)),
        '既有闭环源码已经改变；不能继续使用其二进制')
    g.simd.close.scale.unchanged(report['runtime_sha256'],'已验收运行库/主机二进制改变')
    g.simd.close.scale.unchanged(report['input_sha256'],'已验收输入改变')
    linked=report['per_round'][0]['graph_directory'];path=Path(linked['directory'])
    r.require(path==completed/'round1/graph_directory' and g.digest(path/'report.json')==linked['report_sha256'],
        '既有单轮来源不符')
    one=json.loads((path/'report.json').read_text(encoding='utf-8'));cmd=one['command']
    r.require(len(cmd)==17 and cmd[-2:]==['1','1'] and cmd[7:13]==['10','32','8','24','5','30'],
        '不是原256K/Q8/K10、固定24词容量的 Graph/目录调用')
    r.require(g.digest(Path(cmd[0]))==one['runner_sha256'],'实际已验收二进制不符')
    session=one['session']
    r.require(session.get('host_source_sha256')==g.checks.sources(g.HOST_SOURCES) and
        session.get('graph_capture') is True and session.get('directory_index') is True and
        session.get('normal_prepare_calls')==0 and session.get('replays')==75,'不是已验收固定计划或重放计数不符')
    reference=path/'actual'
    for name,sha in one['output_sha256'].items():
        r.require(g.digest(reference/name)==sha,'既有已核对的完整输出改变：'+name)
    proofs={str(completed/'report.json'):g.digest(completed/'report.json'),str(path/'report.json'):g.digest(path/'report.json')}
    proofs.update(report['runtime_sha256']);proofs.update(report['input_sha256'])
    return report,one,reference,proofs


def validate_application(output,previous,command):
    session=json.loads((output/'benchmark_session.json').read_text(encoding='utf-8'))
    old=previous['session']
    fixed=('status','backend','docs','queries','k','partitions','capacity_queries','capacity_words',
        'input_contract','timer_policy','index_load_batches','index_bytes','cold_synthetic_empty_queries',
        'checkpoint_search_calls','diagnostic_checkpoints_outside_measurement','index_readback_byte_identical',
        'graph_capture','directory_index','packet_input_bytes','packet_output_bytes','directory_bytes',
        'device','index_tensor_h2d_calls','pipeline_stages','normal_full_score_elements',
        'debug_placeholder_elements','diagnostic_extra_chains','debug_index_d2h_calls','intermediate_host_roundtrips',
        'workspace_capacity_bytes','host_source_sha256','measured_debug_download_calls','measured_poison_calls','measured_file_io_calls')
    r.require(all(session.get(key)==old[key] and type(session.get(key)) is type(old[key]) for key in fixed),
        '采集应用与已验收程序的固定尺寸/边界/来源不符')
    wanted={'warmup_cycles':5,'warmup_search_calls':10,'warmup_topk_bit_checks':10,'measured_cycles':10,
        'measured_search_calls':20,'measured_topk_bit_checks':20,'replays':35,'normal_search_calls':35,'normal_prepare_calls':0}
    r.require(all(type(session.get(k)) is type(v) and session[k]==v for k,v in wanted.items()),'采集调用数量不符')
    r.require(command[11:13]==['5','10'] and command[-2:]==['1','1'],'采集命令不是同一 Graph/目录路径')
    samples=session.get('samples',[]);r.require(len(samples)==20,'正式采集样本缺失')
    for pos,item in enumerate(samples):
        cycle,slot=divmod(pos,2);case='B' if slot^(cycle&1) else 'A'
        r.require(item['cycle']==cycle and item['case']==case and g.math.isfinite(item['internal_search_ms']) and
            g.math.isfinite(item['call_wall_ms']) and 0<item['internal_search_ms']<=item['call_wall_ms'] and
            item['query_h2d_bytes']==session['packet_input_bytes'] and item['topk_d2h_bytes']==session['packet_output_bytes'],
            '采集正式顺序/完整边界/实际传输字节不符')
    return session


def analyze_tables(summary,tasks,session,device):
    """Replay may reuse task IDs AND use a different stream for each captured node.
    Match each execution occurrence by device/stream/task/start, never task ID alone.
    """
    own=[dict(row) for row in summary if row['OP Type'] in OPERATORS]
    r.require(not any(row['OP Type'].startswith('Bm25') and row['OP Type'] not in OPERATORS for row in summary),
        '混入其他 BM25 算子')
    plan=rows.phases(session)
    r.require(len(own)==2*len(plan),f'设备执行次数不符：预期{2*len(plan)}，实际{len(own)}；不猜缺失迭代')
    r.require({x['Device_id'].strip() for x in own}=={str(device)} and all(x['Task Type']=='AI_VECTOR_CORE' for x in own),
        '设备或核类型不符')
    for item in own:
        item['_start']=csvtools.number(item['Task Start Time(us)'],'start')
        item['_duration']=csvtools.number(item['Task Duration(us)'],'duration')
        item['_blocks']=int(item['Block Num'])
        r.require(item['_duration']>0 and item['_blocks']>0,'设备时间/块数非法')
    own.sort(key=lambda x:x['_start'])
    lookup=defaultdict(list)
    for i,task in enumerate(tasks):
        key=tuple(task[field].strip() for field in ('Device_id','stream_id','task_id'))
        lookup[key].append((i,task))
    used=set();tolerance=Decimal('.002')
    for item in own:
        key=tuple(item[field].strip() for field in ('Device_id','Stream ID','Task ID'))
        candidates=[(i,t) for i,t in lookup[key] if i not in used and
            abs(csvtools.number(t['task_start(us)'],'task start')-item['_start'])<=tolerance]
        r.require(len(candidates)==1,'执行实例未唯一匹配 task_time（重复 Task ID 必须用时间区分）')
        i,task=candidates[0];used.add(i)
        start=csvtools.number(task['task_start(us)'],'task start');stop=csvtools.number(task['task_stop(us)'],'task stop')
        duration=csvtools.number(task['task_time(us)'],'task duration')
        r.require(task['kernel_type']==item['Task Type'] and task['kernel_name'].startswith(item['OP Type']+'_') and
            stop>=start and abs(duration-item['_duration'])<=tolerance and abs(stop-start-duration)<=tolerance,
            '两张设备表的类型/名称/时长不符')
    n,q,k,s=session['docs'],session['queries'],session['k'],session['partitions']
    measured=[]
    for pos,(phase,name,debug) in enumerate(plan):
        chain=own[2*pos:2*pos+2]
        r.require(tuple(x['OP Type'] for x in chain)==OPERATORS,'评分选择/合并执行次序不符')
        shapes=csvtools.input_shapes(chain[0]['Input Shapes'])
        r.require(len(shapes)==7 and shapes[3:]==((n,),(shapes[4][0],),(q+1,),(max(1,session['capacity_words']),)) and
            shapes[0]==(shapes[4][0]*(s+1),) and shapes[1]==shapes[2],
            'Graph 固定形状或目录合同不符，包括冷启动也必须为固定词容量')
        vocab=shapes[4][0]
        posting_bytes=session['index_bytes']-session['directory_bytes']-4*n-4*vocab
        r.require(session['directory_bytes']==vocab*(s+1)*8 and posting_bytes>=0 and posting_bytes%8==0 and
            shapes[1]==(posting_bytes//8,),'采集词表/倒排尺寸与已验收索引字节合同不同')
        r.require(csvtools.input_shapes(chain[1]['Input Shapes'])==((q,s,k),(q,s,k)),'合并输入形状不符')
        if 'Output Shapes' in chain[0]:
            r.require(csvtools.input_shapes(chain[0]['Output Shapes'])==((q,s,k),(q,s,k),((q,n) if debug else (1,))),
                '普通重放与额外调试链输出错位')
        gap=chain[1]['_start']-chain[0]['_start']-chain[0]['_duration']
        r.require(gap>=0,'有依赖的两个任务时间重叠，不能按本调用顺序对应')
        if phase!='measured':continue
        sample=session['samples'][len(measured)]
        r.require(sample['case']==name,'主机与设备调用错位')
        span=chain[1]['_start']+chain[1]['_duration']-chain[0]['_start']
        wall=Decimal(str(sample['call_wall_ms']))*1000
        r.require(wall.is_finite() and wall>=span,'对应主机调用非法或短于设备链，不能相减')
        measured.append({'case':name,'cycle':sample['cycle'],'operator_us':{x['OP Type']:float(x['_duration']) for x in chain},
            'device_chain_span_us':float(span),'intra_chain_gap_us':float(gap),'profiled_complete_search_us':float(wall),
            'outside_compute_chain_us':float(wall-span),'streams':[x['Stream ID'] for x in chain],
            'task_ids':[x['Task ID'] for x in chain]})
    r.require(len(measured)==len(session['samples']),'正式调用匹配不完整')
    cases={}
    for name in ('A','B'):
        selected=[x for x in measured if x['case']==name]
        cases[name]={'operators':{op:rows.stats([x['operator_us'][op] for x in selected]) for op in OPERATORS},
            **{field:rows.stats([x[field] for x in selected]) for field in
               ('device_chain_span_us','intra_chain_gap_us','profiled_complete_search_us','outside_compute_chain_us')}}
    return {'status':STATUS,'own_operator_execution_rows':len(own),'measured_search_calls':len(measured),
        'cases':cases,'matched_samples':measured,'distinct_stream_ids':sorted({x['Stream ID'] for x in own}),
        'reused_task_id_occurrences':len(own)-len({(x['Stream ID'],x['Task ID']) for x in own}),
        'ignored_other_task_rows':len(tasks)-len(used),
        'scope':'Same-call profile wall minus COMPUTE span, then statistics. Outside includes transfer, host, runtime and profiling; not pure memcpy. No ordinary/profiled subtraction, occupancy or score/selection split.'}


def run(completed,output):
    r.require(sys.platform=='linux','采集仅在 A5 Linux 执行')
    r.require(not output.exists() and output.is_relative_to(ROOT/'results') and not output.is_relative_to(completed) and
        not completed.is_relative_to(output),'需要独立新 results 目录，不覆盖既有闭环')
    report,one,reference,proofs=evidence(completed)
    profiler=shutil.which('msprof');r.require(profiler is not None,'msprof 未找到，不调整环境')
    help_run=subprocess.run([profiler,'--help'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    help_text=help_run.stdout.decode(errors='replace')
    required=('--runtime-api','--task-time','--ai-core','--aic-metrics','--python-path')
    r.require(help_run.returncode==0 and all(x in help_text for x in required),'实际 msprof 帮助未确认既有选项')
    output.mkdir(parents=True);(output/'msprof_help.txt').write_bytes(help_run.stdout)
    command=list(one['command']);command[4]=str(output/'actual');command[12]='10'
    record={'status':'running','completed':str(completed),'prior_sha256':proofs,'command':command,
        'tool_sha256':g.checks.sources(TOOLS),'ordinary_cpu_gate':report['decision'],'ordinary_npu_summary':one['summary']}
    r.write_json(output/'report.json',record)
    try:
        collect=[profiler,'--output='+str(output/'profiler'),'--runtime-api=on','--task-time=l1','--ai-core=on',
            '--aic-metrics=PipeUtilization','--python-path='+sys.executable]+command
        g.execute(collect,output/'msprof.log',g.environment(Path(command[6])))
        session=validate_application(output/'actual',one,command)
        # Compare only already checked bytes, not another CPU benchmark or oracle rerun.
        for label in g.checks.CHECKPOINTS:
            for file in g.simd.close.scale.chain.FILES:
                r.require((output/'actual'/label/file).read_bytes()==(reference/label/file).read_bytes(),
                    '采集检查点与既有完整验收输出不同：'+label+'/'+file)
        g.simd.close.scale.unchanged(proofs,'采集期间旧输入/二进制/验收改变')
        summary=csvtools.find_one(output/'profiler','op_summary*.csv');tasks=csvtools.find_one(output/'profiler','task_time*.csv')
        result=analyze_tables(csvtools.read_table(summary,('Device_id','Task ID','Stream ID','OP Type','Task Type',
            'Task Start Time(us)','Task Duration(us)','Block Num','Input Shapes')),
            csvtools.read_table(tasks,('Device_id','kernel_name','kernel_type','stream_id','task_id',
            'task_time(us)','task_start(us)','task_stop(us)')),session,report['device'])
        result['csv_sha256']={str(p):g.digest(p) for p in (summary,tasks)}
        r.write_json(output/'device_analysis.json',result)
        record.update(status=STATUS,session=session,device_analysis_sha256=g.digest(output/'device_analysis.json'))
        r.write_json(output/'report.json',record)
        print('已匹配78条计算任务，只统计20次正式Search；允许Graph重用Task ID及不同执行Stream。',flush=True)
        for name,item in result['cases'].items():
            print(name+'：'+json.dumps(item,ensure_ascii=False),flush=True)
        print('仅诊断；不替代普通六轮基准，不据此宣称超过CPU。融合算子尚不能拆出评分/选择各占多少。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        record.update(status='failed',first_error=str(error));r.write_json(output/'report.json',record);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completed',type=Path,default=ROOT/'results/graph_directory_close1')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.completed.resolve(),a.output.resolve())
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；保留原始采集，不重编译/安装/改算法/重跑CPU。',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
