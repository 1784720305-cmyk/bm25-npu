"""Measure four stages inside a diagnostic copy; reuse accepted outputs, never run CPU."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import register_select as prior,fused_stage_sources as source

g=prior.g;r=g.r;digest=g.digest;simd=g.simd
STAGES=('score','pack','threshold','collect')
STATUS='diagnostic_device_stage_records_checked_not_performance_acceptance'
OWN=('standalone/fused_stage_sources.py','standalone/fused_stage_profile.py',
 'npu_prototype/fused_stage_host/stage_runner.cpp','npu_prototype/fused_stage_host/CMakeLists.txt',
 'npu_prototype/fused_chain_host/fused_runtime.h','common/shard_directory.h')


def analyze(raw,q=8,shards=32):
    tasks=q*shards
    r.require(len(raw)==tasks*source.WORDS*8,'分段记录长度不符')
    blocks=defaultdict(list);all_tasks=[]
    for task,record in enumerate(struct.iter_unpack('<8Q',raw)):
        magic,ident,block,*times=record
        r.require(magic==source.MAGIC and ident==task and block<tasks,'设备分段记录标识不符')
        r.require(all(0<a<b for a,b in zip(times,times[1:])),'分段时钟缺失、未严格递增或零时长')
        values={stage:(times[i+1]-times[i])/1000 for i,stage in enumerate(STAGES)}
        row={'task':task,'block':block,'start':times[0],'end':times[-1],'stage_us':values}
        blocks[block].append(row);all_tasks.append(row)
    block_count=len(blocks)
    r.require(set(blocks)==set(range(block_count)) and all(row['block']==row['task']%block_count for row in all_tasks),
        '分段记录不符合原grid-stride调度')
    sums=[]
    for block,rows in blocks.items():
        rows.sort(key=lambda x:x['task'])
        r.require(all(a['end']<=b['start'] for a,b in zip(rows,rows[1:])),'同一逻辑块内时钟重叠')
        stages={stage:sum(row['stage_us'][stage] for row in rows) for stage in STAGES}
        sums.append({'block':block,'tasks':len(rows),'stage_us':stages,'last_tick':rows[-1]['end'],
            'work_sum_us':sum(stages.values()),'block_span_us':(rows[-1]['end']-rows[0]['start'])/1000})
    last=max(sums,key=lambda row:row['last_tick'])
    total={stage:sum(row['stage_us'][stage] for row in all_tasks) for stage in STAGES}
    return {'blocks':block_count,'tasks':tasks,'task_stage_us':{stage:prior.profile.rows.stats([row['stage_us'][stage] for row in all_tasks]) for stage in STAGES},
        'all_block_work_us':total,'all_block_work_fraction':{stage:total[stage]/sum(total.values()) for stage in STAGES},
        'last_finishing_block':last,'observed_work_span_us':(max(row['end'] for row in all_tasks)-min(row['start'] for row in all_tasks))/1000,
        'scope':'1GHz system counter on950. Stage includes its VF dispatch/completion barrier. Score/pack have an extra diagnostic split that may change register allocation. Excludes initial buffer setup and final record stores. Not ordinary Search latency or score-only instruction cycles.'}


def evidence(path):
    old=json.loads((path/'report.json').read_text(encoding='utf-8'))
    r.require(old.get('status') in (prior.PASS,prior.MISS) and len(old.get('per_round',[]))==6,'需要刚完成的寄存器版本六轮结果')
    r.require(old['source_sha256']==g.checks.sources(prior.OWN),'已验收寄存器版本源码改变')
    proofs={str(path/'report.json'):digest(path/'report.json')}
    for key in ('previous_sha256','candidate_sha256'):
        simd.close.scale.unchanged(old[key],'既有寄存器版本来源改变');proofs.update(old[key])
    link=old['per_round'][0]['graph_directory'];one_path=Path(link['directory'])/'report.json'
    r.require(digest(one_path)==link['report_sha256'],'原完整Search单轮报告改变')
    proofs[str(one_path)]=digest(one_path)
    one=json.loads(one_path.read_text(encoding='utf-8'));reference=Path(link['directory'])/'actual'
    r.require(one['command'][7:13]==['10','32','8','24','5','30'] and one['command'][-2:]==['1','1'],'原固定Search合同改变')
    for name,sha in one['output_sha256'].items():
        p=reference/name;r.require(digest(p)==sha,'原正确性输出改变');proofs[str(p)]=sha
    return old,one,reference,proofs


def build(a):
    vendor=a.vendors/source.VENDOR;r.require(not vendor.exists(),'诊断vendor已存在；不覆盖')
    target=a.output/'project';metadata=source.prepare(a.original,target)
    (a.output/'compiler_logs').mkdir();env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(a.output/'compiler_logs')
    try:g.execute(['bash','build.sh'],a.output/'resource_build.log',env,target)
    except Exception:
        # Expose real compiler diagnostics in the same terminal summary, not just exit2.
        print((a.output/'resource_build.log').read_text(encoding='utf-8',errors='replace')[-24000:],flush=True);raise
    packages=list((target/'build_out').glob('*.run'));r.require(len(packages)==1,'诊断包不唯一')
    listing=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (a.output/'package_list.log').write_bytes(listing.stdout)
    r.require(listing.returncode==0 and set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listing.stdout.decode(errors='replace')))=={source.VENDOR},'不是独立诊断安装包')
    old=simd.t.resource.cache(a.original/'build_out/CMakeCache.txt');new=simd.t.resource.cache(target/'build_out/CMakeCache.txt')
    r.require(old and new.get('vendor_name')==source.VENDOR and {k:v for k,v in old.items() if k!='vendor_name'}==
        {k:v for k,v in new.items() if k!='vendor_name'},'诊断编译目标/优化等级改变')
    g.execute(['bash',packages[0],'--install-path='+str(a.vendors.parent)],a.output/'install.log',timeout=300)
    header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    declarations=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name,args in simd.fused.PROTOTYPES.items():
        found=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',declarations,flags=re.S)
        r.require(len(found)==1 and re.sub(r'\s+','',found[0])==re.sub(r'\s+','',args),'诊断ABI改变')
    for name,raw in source.candidate_sources(a.original).items():
        r.require((target/name).read_bytes()==raw,'诊断工程源码改变：'+name)
    objects=simd.fused.chain.artifacts(vendor,target,'bm25_fused_score_top_k',header.name)
    logs={str(p):digest(p) for p in (a.output/'compiler_logs').rglob('*') if p.is_file()}
    return {'metadata':metadata,'vendor':str(vendor),'device_objects':objects,'compiler_log_sha256':logs}


def run(a):
    r.require(sys.platform=='linux','仅在A5执行；本机不生成设备数据')
    r.require(not a.output.exists() and a.output.is_relative_to(ROOT/'results') and all(not a.output.is_relative_to(p) and
        not p.is_relative_to(a.output) for p in (a.accepted,a.original,a.vendors)),'需要独立新results目录')
    old,one,reference,proofs=evidence(a.accepted)
    r.require(old['candidate']['metadata']['source_sha256']=={name:simd.hashlib.sha256(raw).hexdigest()
        for name,raw in source.base.candidate_sources(a.original).items()},'诊断来源不是刚测的算子')
    hist={p.resolve() for p in a.toolkit.rglob('kernel_reg_compute_histograms_intf.h') if p.is_file()}
    r.require(len(hist)==1,'实际工具包路径不唯一')
    clock=next(iter(hist)).parent.parent/'kernel_operator_sys_var_intf.h'
    r.require(clock.is_file() and re.search(r'\bGetSystemCycle\s*\(',clock.read_text(encoding='utf-8')),'实际950时钟API未确认')
    proofs[str(clock)]=digest(clock);own=g.checks.sources(OWN)
    a.output.mkdir(parents=True)
    report={'status':'running','accepted_sha256':proofs,'source_sha256':own,'device':old['device'],
        'ordinary_cpu_decision_unchanged':old['decision'],'clock_hz':1000000000,
        'clock_source':'https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md'}
    def save():r.write_json(a.output/'report.json',report)
    def check():
        simd.close.scale.unchanged(proofs,'诊断期间旧证据改变')
        r.require(g.checks.sources(OWN)==own,'诊断源码改变')
        if 'installation_sha256' in report:simd.close.scale.unchanged(report['installation_sha256'],'诊断工程/安装改变')
    save()
    try:
        report['candidate']=build(a);check();save()
        vendors=g.aliases(a.output/'runtime/vendors',a.vendors/source.VENDOR,Path(one['command'][6])/'bm25_topk_merge')
        build_path=a.output/'host_build'
        for command,log in ((['cmake','-S',ROOT/'npu_prototype/fused_stage_host','-B',build_path,
            '-DTOOLKIT_HOME='+str(a.toolkit),'-DBM25_VENDOR_DIR='+str(vendors)],'host_configure.log'),
            (['cmake','--build',build_path,'-j','4'],'host_build.log')):
            g.execute(command,a.output/log);check()
        runner=build_path/'bm25_fused_stage_diagnostic';inputs=Path(one['command'][1]).parent
        plan=a.output/'requests.tsv';plan.write_bytes(''.join(f'{name}\t{inputs/name}\n' for name in ('A','B')).encode())
        files=[runner,plan]+[p for p in (a.vendors/source.VENDOR).rglob('*') if p.is_file() and p.suffix in ('.o','.so','.h','.json')]
        files += [a.output/'project'/name for name in source.candidate_sources(a.original)]
        report['installation_sha256']={str(p.resolve()):digest(p) for p in files};check();save()
        g.execute([runner,inputs/'index',plan,reference,vendors,old['device'],a.output/'actual'],a.output/'device.log',g.environment(vendors))
        session=json.loads((a.output/'actual/report.json').read_text(encoding='utf-8'))
        r.require(session.get('status')=='stage_records_outputs_and_guards_checked' and session.get('calls')==30 and
            session.get('measured_calls')==20 and session.get('host_sha256')==digest(ROOT/OWN[2]),'设备分段核对报告不符')
        report['session']=session;report['cases']={};records={}
        for name in ('A','B'):
            samples=[]
            for i in range(10):
                p=a.output/'actual'/f'{name}_{i}.bin';samples.append(analyze(p.read_bytes()));records[str(p)]=digest(p)
            report['cases'][name]={'samples':samples,
                'median_task_stage_us':{stage:prior.profile.rows.stats([s['task_stage_us'][stage]['median_us'] for s in samples]) for stage in STAGES},
                'last_finishing_block_stage_us':{stage:prior.profile.rows.stats([s['last_finishing_block']['stage_us'][stage] for s in samples]) for stage in STAGES},
                'all_block_work_fraction':{stage:sum(s['all_block_work_fraction'][stage] for s in samples)/10 for stage in STAGES},
                'observed_work_span_us':prior.profile.rows.stats([s['observed_work_span_us'] for s in samples])}
            compact={k:v for k,v in report['cases'][name].items() if k!='samples'}
            print(name+' 融合内部诊断：'+json.dumps(compact,ensure_ascii=False),flush=True)
        report.update(status=STATUS,record_sha256=records);check();save()
        print('只诊断融合内部四阶段；结果位与既有输出一致、保护区通过。没有运行CPU，不是性能达标证据。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        for log in ('device.log','host_build.log','host_configure.log'):
            path=a.output/log
            if path.is_file():
                print('最近阶段日志 '+str(path)+'：\n'+path.read_text(encoding='utf-8',errors='replace')[-24000:],flush=True)
                break
        report.update(status='failed',first_error=str(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,path in {'accepted':'results/register_select_close2','original':'results/Bm25FusedScoreTopKUbFix1','vendors':'results/custom_opp/vendors'}.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    p.add_argument('--toolkit',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    try:run(a)
    except (Exception,KeyboardInterrupt) as error:print('FAILED：'+str(error),file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
