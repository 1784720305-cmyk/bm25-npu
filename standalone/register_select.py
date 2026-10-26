"""One NPU-operator candidate; existing host and BOTH existing CPU binaries remain frozen."""
import argparse
from decimal import Decimal,InvalidOperation
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import register_select_sources as source, profile_graph_directory as profile

g=profile.g
r=g.r
simd=g.simd
digest=g.digest
OWN=('standalone/register_select.py','standalone/register_select_sources.py',source.HEADER)
PASS='npu_register_selector_complete_search_faster_than_frozen_cpu'
MISS='npu_register_selector_correct_but_cpu_target_not_met'


def inspect_profile(path):
    """Read counters already collected for the exact current operator. No new device run."""
    report=json.loads((path/'report.json').read_text(encoding='utf-8'))
    r.require(report.get('status')==profile.STATUS and report.get('tool_sha256')==g.checks.sources(profile.TOOLS),
        '需要已完成且来源未变的当前Graph目录版短采集')
    analysis=path/'device_analysis.json'
    r.require(digest(analysis)==report['device_analysis_sha256'],'短采集分析改变')
    data=json.loads(analysis.read_text(encoding='utf-8'))
    g.simd.close.scale.unchanged(data['csv_sha256'],'原始采集CSV改变')
    summary=profile.csvtools.find_one(path/'profiler','op_summary*.csv')
    rows=profile.csvtools.read_table(summary,('OP Type','Task Start Time(us)'))
    own=sorted([row for row in rows if row['OP Type'] in profile.OPERATORS],
        key=lambda x:profile.csvtools.number(x['Task Start Time(us)'],'start'))
    phases=profile.rows.phases(report['session']);r.require(len(own)==2*len(phases),'当前采集执行行缺失')
    counters={}
    for op in profile.OPERATORS:
        selected=[own[2*i+j] for i,(phase,case,debug) in enumerate(phases) if phase=='measured'
            for j in range(2) if own[2*i+j]['OP Type']==op]
        fields=[name for name in selected[0] if name.lower().startswith(('aiv_','aic_','vec_','scalar_','mte','ub_','main_mem','dcache','simt','icache'))]
        counters[op]={}
        for field in fields:
            values=[]
            for row in selected:
                try:value=Decimal(row[field].strip())
                except (InvalidOperation,AttributeError):continue
                if value.is_finite() and value>=0:values.append(float(value))
            statistics=profile.rows.stats(values) if values else None
            if statistics:statistics={key.removesuffix('_us'):value for key,value in statistics.items()}
            counters[op][field]={'available':len(values),'raw_statistics':statistics,
                'raw_examples':[row[field] for row in selected[:3]],'units':'exact CSV field units; not normalized or interpreted as card occupancy'}
    return {'source_sha256':data['csv_sha256'],'device_time_cases':data['cases'],'counter_fields':counters,
        'interpretation':'Pipeline ratios are instruction-cycle ratios, not SIMT occupancy or whole-card utilization. Missing/zero unsupported counters prove neither inactivity nor full use.'}


def management_snapshot(output,label):
    executable=__import__('shutil').which('npu-smi')
    if not executable:return {'status':'npu_smi_unavailable'}
    records=[]
    # Discover physical/logical mapping; ACL logical device5 is not assumed to be NPU ID5.
    for suffix,args in (('mapping',['info','-m']),('devices',['info','-l']),('overview',['info'])):
        log=output/(label+'_'+suffix+'.log')
        try:
            child=subprocess.run([executable]+args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15)
            log.write_bytes(child.stdout)
            records.append({'command':[executable]+args,'returncode':child.returncode,'log_sha256':digest(log)})
        except (OSError,subprocess.TimeoutExpired) as error:records.append({'command':args,'unavailable':str(error)})
    return {'records':records,'scope':'Untimed management snapshots only. A 100us request may be invisible to seconds-level sampling; no utilization conclusion. Not run concurrently with msprof.'}


def actual_headers(toolkit):
    base=simd.actual_headers(toolkit)
    paths={p.resolve() for p in toolkit.rglob('kernel_reg_compute_histograms_intf.h') if p.is_file()}
    r.require(len(paths)==1,'实际Reg头文件目录不唯一')
    parent=next(iter(paths)).parent
    names=('Compare','Arange','Reduce','Duplicate','Select','Min','Max','Sub','ShiftLefts','Or')
    found={name:[] for name in names};files={}
    for path in parent.glob('*.h'):
        text=path.read_text(encoding='utf-8',errors='replace');matches=[]
        for i,line in enumerate(text.splitlines(),1):
            for name in names:
                if re.search(r'\b'+name+r'\s*\(',line) and not line.lstrip().startswith('//'):
                    found[name].append(str(path));matches.append({'line':i,'declaration':line.strip()})
        if matches:files[str(path)]={'sha256':digest(path),'declarations':matches}
    r.require(all(found.values()),'寄存器控制所需实际API缺失：'+','.join(n for n,x in found.items() if not x))
    return {'retained_api':base,'new_api_files':files,'scope':'Actual declarations recorded; target compiler still must check every overload/type.'}


def measurement_command(previous,destination,role,vendors):
    command=list(previous['command'])
    if role=='graph_directory':
        r.require(len(command)==17 and command[7:13]==['10','32','8','24','5','30'] and command[-2:]==['1','1'],
            '既有NPU命令合同改变')
        command[6]=str(vendors)
    else:
        r.require(role in ('cpu','cpu_directory') and len(command)==12 and
            command[5:]==['32','10','32','8','24','5','30'],'冻结CPU命令合同改变')
    command[4]=str(destination/'actual')
    return command


def build_install(a):
    target=a.output/'project';vendor=a.vendors/source.VENDOR
    r.require(not vendor.exists(),'候选vendor已存在，不覆盖：'+str(vendor))
    metadata=source.prepare(a.original,target)
    logs=a.output/'compiler_logs';logs.mkdir();env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(logs)
    g.execute(['bash','build.sh'],a.output/'resource_build.log',env,target)
    packages=list((target/'build_out').glob('*.run'));r.require(len(packages)==1,'独立候选安装包不唯一')
    listing=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (a.output/'package_list.log').write_bytes(listing.stdout)
    names=set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listing.stdout.decode(errors='replace')))
    r.require(listing.returncode==0 and names=={source.VENDOR},'安装包不是唯一独立候选')
    old=simd.t.resource.cache(a.original/'build_out/CMakeCache.txt');new=simd.t.resource.cache(target/'build_out/CMakeCache.txt')
    r.require(old and new.get('vendor_name')==source.VENDOR and
        {k:v for k,v in old.items() if k!='vendor_name'}=={k:v for k,v in new.items() if k!='vendor_name'},'候选构建等级或目标改变')
    g.execute(['bash',packages[0],'--install-path='+str(a.vendors.parent)],a.output/'install.log',timeout=300)
    header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name,parameters in simd.fused.PROTOTYPES.items():
        declarations=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,flags=re.S)
        r.require(len(declarations)==1 and re.sub(r'\s+','',declarations[0])==re.sub(r'\s+','',parameters),'实际生成ABI改变：'+name)
    for name,raw in source.candidate_sources(a.original).items():r.require((target/name).read_bytes()==raw,'候选源码改变')
    objects=simd.fused.chain.artifacts(vendor,target,'bm25_fused_score_top_k',header.name)
    logs,resources=simd.compiler_resources(a.output)
    return {'vendor':str(vendor),'metadata':metadata,'device_objects':objects,
        'header_sha256':digest(header),'api_sha256':digest(vendor/'op_api/lib/libcust_opapi.so'),
        'compiler_logs':logs,'simt_resources':resources,'simd_resource_fields_not_reported':'UNKNOWN; no spill/occupancy inference from stack size'}


def run(a):
    r.require(sys.platform=='linux','需要A5 Linux；不在本机生成NPU性能结果')
    r.require(not a.output.exists() and a.output.is_relative_to(ROOT/'results') and all(
        not a.output.is_relative_to(p) and not p.is_relative_to(a.output) for p in (a.completed,a.profile,a.accepted,a.original,a.vendors)),
        '需要独立新results目录，不覆盖或嵌套已有证据')
    completed,one,reference,proofs=profile.evidence(a.completed)
    inspected=inspect_profile(a.profile)
    collected=json.loads((a.profile/'report.json').read_text(encoding='utf-8'))
    r.require(collected['prior_sha256']==proofs,'已有采集不属于同一已验收组合')
    proofs.update(inspected['source_sha256'])
    proofs.update({str(a.profile/name):digest(a.profile/name) for name in ('report.json','device_analysis.json')})
    old=json.loads((a.accepted/'report.json').read_text(encoding='utf-8'))
    selector=a.accepted/'selector_host_build/bm25_fused_simd_selector_check'
    r.require(digest(selector)==old['selector_checks']['runner_sha256'],'已验收原始FP32选择检查器不符')
    r.require(digest(a.accepted/'report.json')==completed['accepted_report_sha256'],'原SIMD验收来源改变')
    proofs[str(a.accepted/'report.json')]=digest(a.accepted/'report.json')
    baseline_source={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in g.source.candidate_sources(a.original).items()}
    r.require(baseline_source==completed['directory_installation']['metadata']['source_sha256'],'当前目录算子来源不同')
    source_hash=g.checks.sources(OWN)
    a.output.mkdir(parents=True)
    report={'status':'running','device':completed['device'],'scope':'NPU threshold-control-only candidate. Existing graph/host/index/CPU binaries frozen.',
        'source_sha256':source_hash,'previous_sha256':proofs,'existing_pipeline_counter_inspection':inspected,
        'cpu_policy':'No CPU source/flag/config optimization or rebuild; reuse both compiled32thread32shard CPU opponents.',
        'per_round':[]}
    def save():r.write_json(a.output/'report.json',report)
    def unchanged():
        g.simd.close.scale.unchanged(proofs,'既有二进制/输入/验收改变')
        r.require(g.checks.sources(OWN)==source_hash,'候选源码或工具改变')
        if 'candidate_sha256' in report:g.simd.close.scale.unchanged(report['candidate_sha256'],'候选安装/工程/实际头文件改变')
    save()
    try:
        print('读取已有PipeUtilization，正式20次；字段原单位，不等于整卡利用率：'+
            json.dumps(inspected['counter_fields'],ensure_ascii=False),flush=True)
        report['management_before']=management_snapshot(a.output,'before')
        report['actual_headers']=actual_headers(a.toolkit);save()
        report['candidate']=build_install(a);save()
        vendors=g.aliases(a.output/'runtime/vendors',a.vendors/source.VENDOR,Path(one['command'][6])/'bm25_topk_merge')
        files=[p for p in (a.vendors/source.VENDOR).rglob('*') if p.is_file() and p.suffix in ('.so','.o','.json','.h')]
        files += [a.output/'project'/name for name in source.candidate_sources(a.original)]
        files += [selector]+[Path(p) for p in report['actual_headers']['new_api_files']]
        files += [Path(p) for p in report['actual_headers']['retained_api']['files']]
        report['candidate_sha256']={str(p.resolve()):digest(p) for p in files};unchanged();save()
        # Existing raw-score checker: S=1, so directory shape V*(1+1) equals its 2-entry tensor.
        raw=a.output/'selector_checks'
        g.execute([selector,vendors,completed['device'],raw],a.output/'selector_device.log',g.environment(vendors))
        checked=json.loads((raw/'report.json').read_text(encoding='utf-8'))
        r.require(checked.get('status')=='actual_simd_selector_guards_and_strict_bit_rankings_passed' and checked.get('calls')==54,
            '改变的寄存器选择未通过真实FP32/正负零/容量/同分/保护区验收')
        r.require(checked.get('checked_rows',0)>0 and checked.get('host_sha256')==old['selector_checks']['host_sha256'],
            '选择器检查器来源或检查行数不符')
        report['raw_selector_checks']=checked;save();unchanged()
        edges=a.output/'edges';edges.mkdir();plan=g.edge_inputs(edges)
        checker=a.completed/'directory_header_build/bm25_plan_check'
        g.execute([checker,edges/'index',plan,a.output/'chain_checks',completed['device'],vendors,10,2,2,128,1,1],
            a.output/'chain_device.log',g.environment(vendors))
        report['chain_checks']=json.loads((a.output/'chain_checks/report.json').read_text(encoding='utf-8'))
        r.require(report['chain_checks'].get('calls')==8 and report['chain_checks'].get('status')==
            'actual_planned_chain_full_scores_rankings_guards_and_new_inputs_passed','新算子Graph链验收失败')
        save();unchanged()
        inputs=Path(one['command'][1]).parent
        index,records,snapshot=g.fixture.validate_inputs(inputs)
        cache=g.checks.cache_inputs(inputs,index,records)
        cpu_prior={}
        for role in ('cpu','cpu_directory'):
            linked=completed['per_round'][0][role];path=Path(linked['directory'])/'report.json'
            r.require(digest(path)==linked['report_sha256'],'既有CPU单轮来源改变')
            cpu_prior[role]=json.loads(path.read_text(encoding='utf-8'))
        def measure(destination,role):
            unchanged();destination.mkdir(parents=True)
            command=measurement_command(one if role=='graph_directory' else cpu_prior[role],destination,role,vendors)
            env=g.environment(vendors) if role=='graph_directory' else os.environ.copy()
            g.execute(command,destination/'run.log',env)
            result=g.check_measurement(destination/'actual',index,records,cache,role,32,completed['device'],24)
            result.update(command=command,runner_sha256=digest(Path(command[0])),log_sha256=digest(destination/'run.log'))
            r.write_json(destination/'report.json',result);unchanged()
            print(role+' 完整Search：'+json.dumps(result['summary'],ensure_ascii=False),flush=True)
            return {'directory':str(destination),'summary':result['summary'],'report_sha256':digest(destination/'report.json')}
        for i in range(6):
            roles=('cpu','graph_directory','cpu_directory') if i%2==0 else ('cpu_directory','graph_directory','cpu')
            row={'round':i+1,'process_order':list(roles)}
            for role in roles:row[role]=measure(a.output/f'round{i+1}'/role,role)
            report['per_round'].append(row);save()
        report['decision']=g.summary_gate(report['per_round'])
        report['management_after']=management_snapshot(a.output,'after')
        report['status']=PASS if report['decision']['performance_target_met'] else MISS
        unchanged();save()
        print('最终CPU门槛：'+json.dumps(report['decision'],ensure_ascii=False),flush=True)
        print('已达到约定完整Search目标。' if report['status']==PASS else '性能目标仍未达到；不作为达标交付。',flush=True)
        return 0 if report['status']==PASS else 2
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,path in {'completed':'results/graph_directory_close1','profile':'results/graph_directory_profile1',
        'accepted':'results/fused_simd_close2','original':'results/Bm25FusedScoreTopKUbFix1','vendors':'results/custom_opp/vendors'}.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    p.add_argument('--toolkit',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    try:return run(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不生成CPU替代结果、不改CPU或正式算子。',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
