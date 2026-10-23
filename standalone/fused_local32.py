"""One A5 SIMT candidate: bounded local selection counters use 32 bits; retain 256 threads and FP32."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resume_fused_threads256 as accepted

t=accepted.previous
close=t.close
fused=t.fused
r=t.r
digest=t.digest
VENDOR='bm25_fused_score_topk_local32'
STATUS='fused_local32_complete_search_comparison_finished'
KERNEL='op_kernel/bm25_fused_score_top_k.cpp'
SPLIT='// Three disjoint regions: full local scores, score metadata, accepted selector scratch.'
# ONLY UB-local positions change. Global output row/query, task/address arithmetic,
# resident int64 index offsets, LowerBound and the entire scoring VF remain unchanged.
REPLACEMENTS=(
    ('uint64_t base, uint64_t position, bool compact','uint32_t base, uint32_t position, bool compact'),
    ('uint64_t position, bool compact','uint32_t position, bool compact'),
    ('uint64_t base, uint64_t query','uint32_t base, uint64_t query'),
    ('const uint64_t base = 0;','const uint32_t base = 0;'),
    ('for (uint64_t doc = thread;','for (uint32_t doc = thread;'),
    ('for (uint64_t doc = static_cast<uint64_t>(winner) + kThreads;',
     'for (uint32_t doc = winner + kThreads;'),
    ('const uint64_t sourceBegin','const uint32_t sourceBegin'),
    ('const uint64_t sourceEnd','const uint32_t sourceEnd'),
    ('const uint64_t sourceStep','const uint32_t sourceStep'),
    ('uint64_t equalPosition = 0;','uint32_t equalPosition = 0;'),
    ('for (uint64_t doc = sourceBegin;','for (uint32_t doc = sourceBegin;'),
    ('for (uint64_t doc = equalPosition + sourceStep;',
     'for (uint32_t doc = equalPosition + sourceStep;'))


def narrow_kernel(raw):
    text=raw.decode('utf-8')
    r.require(text.count(SPLIT)==1,'无法确定局部选择/评分源码边界')
    local,scoring=text.split(SPLIT)
    for before,after in REPLACEMENTS:
        expected=3 if before=='for (uint64_t doc = sourceBegin;' else 1
        r.require(local.count(before)==expected,'冻结的256线程局部计数源码不符：'+before)
        local=local.replace(before,after)
    # Both global-row declarations deliberately keep uint64_t. No other wide local counter remains.
    r.require(re.findall(r'uint64_t\s+(\w+)',local)==['query']*2,'存在未说明的64位声明或全局行号改变')
    return (local+SPLIT+scoring).encode('utf-8')


def candidate_sources(original):
    source=t.candidate_sources(original)
    source[KERNEL]=narrow_kernel(source[KERNEL])
    return source


def prepare(original,target):
    """Reuse the exact CANN template/copy guards; do not edit any accepted project."""
    source=candidate_sources(original)
    metadata=t.prepare(original,target)
    (target/KERNEL).write_bytes(source[KERNEL])
    preset=target/'CMakePresets.json';data=json.loads(preset.read_text(encoding='utf-8'))
    for item in data.get('configurePresets',[]):
        value=item.get('cacheVariables',{}).get('vendor_name')
        if isinstance(value,dict):
            r.require(value.get('value')==t.VENDOR,'副本vendor配置不符');value['value']=VENDOR
        elif value is not None:
            r.require(value==t.VENDOR,'副本vendor配置不符');item['cacheVariables']['vendor_name']=VENDOR
    r.write_json(preset,data)
    metadata.update(status='local32_generated_not_compiled_or_device_verified',vendor_name=VENDOR,
        local_selection_index_bits=32,source_sha256={n:hashlib.sha256(v).hexdigest() for n,v in source.items()},
        scope='Only bounded UB-local Top-K counters/positions uint64->uint32; maximum local width8192, '
        'increment<=256 and compact capacity1024. Scoring, global64-bit index/address arithmetic, '
        'FP32 bits, tie order, 256 threads, UB layout, barriers and algorithm unchanged.')
    r.write_json(target/'bm25_project.json',metadata)
    return metadata


def candidate_install(vendor,project,original):
    expected=candidate_sources(original)
    data=json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    r.require(vendor.resolve(strict=True).name==VENDOR and data.get('vendor_name')==VENDOR and
        data.get('simt_threads')==256 and data.get('local_selection_index_bits')==32 and
        data.get('explicit_ub_bytes')==50848 and data.get('dynamic_ub_bytes')==65536,
        '候选vendor/局部下标/线程/UB配置不符')
    for name,raw in expected.items():
        r.require((project/name).read_bytes()==raw and digest(project/name)==data['source_sha256'][name],
            '候选并非唯一局部32位计数改动：'+name)
    header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name,parameters in fused.PROTOTYPES.items():
        found=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,flags=re.S)
        r.require(len(found)==1 and re.sub(r'\s+','',found[0])==re.sub(r'\s+','',parameters),
            '实际新接口与已核对ABI不同；停止，不调用：'+name)
    objects=fused.chain.artifacts(vendor,project,'bm25_fused_score_top_k',header.name)
    return {'vendor':str(vendor.resolve()),'source_sha256':data['source_sha256'],
        'header_sha256':digest(header),'api_sha256':digest(vendor/'op_api/lib/libcust_opapi.so'),
        'metadata_sha256':digest(project/'bm25_project.json'),'device_objects':objects}


def completed_evidence(path,candidate,snapshot):
    """Read prior completed decision/provenance only; never repeat accepted device requests."""
    data=json.loads((path/'report.json').read_text(encoding='utf-8'))
    comparison=json.loads((path/'comparison.json').read_text(encoding='utf-8'))
    r.require(data.get('status')==accepted.STATUS and data.get('comparison')==comparison and
        comparison.get('status')==accepted.STATUS and data.get('candidate_installation')==candidate and
        data.get('input_sha256')==snapshot,'需要当前256线程已完成的完整Search报告')
    r.require(comparison.get('decision')==close.decision(comparison.get('per_round',[])) and
        comparison['decision']['keep_fused_candidate'],'256线程未通过既定保留门槛')
    r.require(len(comparison.get('cpu_rounds',[]))==6 and comparison.get('cpu_comparison')==
        close.decision([dict(x,baseline=x['cpu']) for x in comparison['cpu_rounds']]),
        '此前256线程与CPU同期比较未完成或汇总改变')
    r.require(close.checks.sources(tuple(data['source_sha256']))==data['source_sha256'],'已验收源码改变')
    proofs={str(path/name):digest(path/name) for name in ('report.json','comparison.json')}
    for row in comparison['per_round']+comparison.get('cpu_rounds',[]):
        for role in row['process_order']:
            record=row[role];file=Path(record['directory'])/'report.json'
            r.require(digest(file)==record['report_sha256'],'既有完整Search子报告改变')
            proofs[str(file)]=record['report_sha256']
    return proofs


def paired(output,inputs,index,records,cache,raw,runner,cpu_runner,baseline,candidate,device,check):
    capq=max(x['num_queries'] for x in records.values());capwords=max(x['query_term_count'] for x in records.values())
    report={'status':'running','per_round':[],'cpu_rounds':[],'timer_policy':close.checks.TIMER,
        'parameters':{'baseline':'accepted_t256_local64','candidate':'t256_local32','shards':32,
            'device':device,'warmup':5,'repeats':30,'rounds':6}}
    def save():r.write_json(output/'comparison.json',report)
    def measured(directory,vendors):
        check();result=accepted.measure(directory,inputs,index,records,cache,raw,runner,vendors,device,capq,capwords)
        check();return result
    save()
    for i in range(6):
        row={'round':i+1,'process_order':['baseline','candidate'] if i%2==0 else ['candidate','baseline']}
        print('256线程局部64/32位完整Search第'+str(i+1)+'/6轮：'+str(row['process_order']),flush=True)
        for role in row['process_order']:
            row[role]=measured(output/f'round{i+1}'/role,baseline if role=='baseline' else candidate)
        close.cross(row,cache,32);report['per_round'].append(row);save()
    report['decision']=close.decision(report['per_round']);save()
    if report['decision']['keep_fused_candidate']:
        for i in range(6):
            row={'round':i+1,'process_order':['cpu','candidate'] if i%2==0 else ['candidate','cpu']}
            print('同期CPU32/局部32位第'+str(i+1)+'/6轮：'+str(row['process_order']),flush=True)
            for role in row['process_order']:
                check();directory=output/f'cpu_round{i+1}'/role
                if role=='candidate':row[role]=measured(directory,candidate)
                else:
                    row[role]=close.bench.run_one(directory,'cpu',inputs,index,records,cache,raw,cpu_runner,baseline,
                        {'topk_acceptance':{'workspace_bytes':0}},32,32,32,device,capq,capwords,5,30)
                    r.require(row[role]['actual_threads']==32,'实际CPU线程不足32')
                check()
            close.cross(row,cache,32,'cpu');report['cpu_rounds'].append(row);save()
        report['cpu_comparison']=close.decision([dict(x,baseline=x['cpu']) for x in report['cpu_rounds']])
    report['status']=STATUS;save();return report


def build_install(original,target,vendors,output):
    metadata=prepare(original,target)
    logs=output/'compiler_logs';logs.mkdir();env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(logs)
    print('只编译独立local32候选；日志：'+str(output/'resource_build.log')+'；不要重复启动。',flush=True)
    with (output/'resource_build.log').open('wb') as stream:
        child=subprocess.run(['bash','build.sh'],cwd=target,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    r.require(child.returncode==0,'编译失败；查看 '+str(output/'resource_build.log'))
    old=t.resource.cache(original/'build_out/CMakeCache.txt');new=t.resource.cache(target/'build_out/CMakeCache.txt')
    r.require(old and new.get('vendor_name')==VENDOR and
        {k:v for k,v in old.items() if k!='vendor_name'}=={k:v for k,v in new.items() if k!='vendor_name'},
        '候选优化等级/计算目标与已验收版不同')
    log_records,resources=t.resource.compiler_reports(output)
    print('新候选VF资源：'+json.dumps(resources,ensure_ascii=False)+'；未知项不记为零，不据此代替Search实测。',flush=True)
    packages=list((target/'build_out').glob('*.run'));r.require(len(packages)==1,'候选安装包不唯一')
    listed=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (output/'package_list.log').write_bytes(listed.stdout)
    names=set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listed.stdout.decode('utf-8',errors='replace')))
    r.require(listed.returncode==0 and names=={VENDOR},'安装包vendor不符，停止安装')
    with (output/'install.log').open('wb') as stream:
        child=subprocess.run(['bash',str(packages[0]),'--install-path='+str(vendors.parent)],
            stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    r.require(child.returncode==0,'安装失败；查看 '+str(output/'install.log'))
    installed=candidate_install(vendors/VENDOR,target,original)
    print('实际生成的两个接口声明与既有ABI一致；复用已验收的调用器，不猜接口。',flush=True)
    return {'metadata':metadata,'compiler_log_records':log_records,'vf_reports':resources,'installation':installed}


def run(a):
    inputs,old,completed,output,project,vendors,merge,runner,diag_runner,cpu_runner,acceptance=(getattr(a,n).resolve()
        for n in ('input','previous','completed','output','project','vendors','merge_project','runner','diagnostic_runner','cpu_runner','acceptance'))
    r.require(not output.exists() and output.is_relative_to((ROOT/'results').resolve()) and
        all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in (old,completed,project,merge,inputs,vendors)),
        '需要独立results新目录，不覆盖或嵌套旧试验')
    index,records,snapshot,installed,proofs=t.prior.evidence(inputs,acceptance,runner,vendors,project,merge)
    r.require((index['num_docs'],records['A']['num_queries'],records['A']['k'])==(262144,8,10),'只比较当前256K/Q8/K10/S32')
    prior,baseline,baseline_runtime,old_proofs=accepted.validate_previous(old,inputs,project,vendors,merge,runner,
        diag_runner,cpu_runner,a.device,index,records,snapshot,installed)
    proofs.update(old_proofs);proofs.update(completed_evidence(completed,baseline,snapshot))
    target_vendor=vendors/VENDOR;r.require(not target_vendor.exists(),'独立候选vendor已存在，不覆盖')
    frozen=dict(prior['frozen_source_sha256'])
    frozen.update(close.checks.sources(('standalone/resume_fused_threads256.py','standalone/fused_local32.py')))
    binaries={str(p):digest(p) for p in (runner,diag_runner,cpu_runner)}
    output.mkdir(parents=True)
    report={'status':'running','device':a.device,'input_sha256':snapshot,'proof_sha256':proofs,
        'source_sha256':frozen,'runner_sha256':binaries,'baseline_installation':baseline,
        'baseline_vf_reports':prior.get('candidate_vf_reports',[]),
        'scope':'Only local bounded integer width in the accepted256-thread selector. No CPU/host/formula/algorithm/'
        'UB/thread/transfer/timer changes. Build+read actual ABI+changed-path device correctness+six full Search '
        'pairs; CPU32/32 only if retained. No old acceptance requests repeated. First failure stops.'}
    r.write_json(output/'report.json',report)
    try:
        def baseline_unchanged():
            for group in (proofs,binaries,snapshot):close.scale.unchanged(group,'既有证据/输入/程序改变')
            r.require(close.checks.sources(tuple(frozen))==frozen and fused.installed(vendors,project,merge)==installed and
                t.candidate_install(vendors/t.VENDOR,old/'project',project)==baseline and
                accepted.opp_paths(baseline_runtime)==[str((vendors/t.VENDOR).resolve()),str((vendors/'bm25_topk_merge').resolve())],
                '冻结源码、128/256安装或运行库映射改变')
        baseline_unchanged()
        report['candidate']=build_install(project,output/'project',vendors,output)
        runtime=output/'runtime/vendors';runtime.mkdir(parents=True)
        for name,value in (('bm25_fused_score_topk',target_vendor),('bm25_topk_merge',vendors/'bm25_topk_merge')):
            (runtime/name).symlink_to(value,target_is_directory=True)
        def unchanged():
            baseline_unchanged()
            r.require(candidate_install(target_vendor,output/'project',project)==report['candidate']['installation'] and
                accepted.opp_paths(runtime)==[str(target_vendor.resolve()),str((vendors/'bm25_topk_merge').resolve())],
                '候选源码/安装/别名路径改变')
        unchanged();r.write_json(output/'report.json',report)
        cache=close.checks.cache_inputs(inputs,index,records)
        print('只验收改动的local32算子：当前负载A/B/A/空/B，debug1/debug0；旧256线程验收只读取证据。',flush=True)
        report['acceptance']=t.diagnostic(inputs,output/'acceptance',diag_runner,runtime,a.device,index,records,cache)
        report['acceptance'].update(status='local32_changed_path_correctness_passed',
            scope='New local32 selector on unchanged256-thread configuration; all five requests and both debug modes checked.')
        r.write_json(output/'acceptance/report.json',report['acceptance'])
        unchanged();r.write_json(output/'report.json',report)
        raw=output/'raw_queries';raw.mkdir()
        for name in ('A','B'):close.search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
        raw_hash={str(p):digest(p) for p in raw.iterdir()}
        def check_all():unchanged();close.scale.unchanged(raw_hash,'原始查询改变')
        report['comparison']=paired(output,inputs,index,records,cache,raw,runner,cpu_runner,baseline_runtime,runtime,a.device,check_all)
        check_all();report['status']=STATUS;r.write_json(output/'report.json',report)
        for name,value in report['comparison']['decision']['cases'].items():
            print(name+'局部64/32位：'+json.dumps(value,ensure_ascii=False),flush=True)
        for name,value in report['comparison'].get('cpu_comparison',{}).get('cases',{}).items():
            print(name+'同期CPU32/local32：'+json.dumps(value,ensure_ascii=False),flush=True)
        print('结论：'+('local32达到既定改善门槛，保留候选；CPU结果另列。' if
            report['comparison']['decision']['keep_fused_candidate'] else
            '放弃local32候选，继续保留已验收的256线程版；不追加参数扫描。'),flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));r.write_json(output/'report.json',report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    defaults={'previous':'results/fused_t256_close1','completed':'results/fused_t256_compare2',
        'input':'results/search_256k_q8_input1','project':'results/Bm25FusedScoreTopKUbFix1',
        'vendors':'results/custom_opp/vendors','merge-project':'results/Bm25TopKMerge',
        'runner':'results/fused_search_host_build/bm25_fused_search_benchmark',
        'diagnostic-runner':'results/fused_chain_host_build/bm25_fused_chain_runner',
        'cpu-runner':'results/search_benchmark_host_build/bm25_cpu_search_benchmark','acceptance':'results/fused_search_256k_q8_close1'}
    for name,value in defaults.items():p.add_argument('--'+name,type=Path,default=ROOT/value)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=5)
    a=p.parse_args()
    try:r.require(sys.platform=='linux' and r.integer(a.device,0,r.LIMIT),'请在A5 Linux使用合法设备号');run(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；首错停止，不继续计时或生成CPU替代输出。',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
