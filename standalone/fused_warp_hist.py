"""One A5 candidate: per-warp UB histograms in the first radix pass; full Search decides retention."""
import argparse
from array import array
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import profile_fused_t256 as profile

t=profile.t
close=t.close
fused=t.fused
accepted=profile.accepted
r=t.r
digest=t.digest
VENDOR='bm25_fused_score_topk_warp_hist'
STATUS='fused_first_pass_warp_hist_search_comparison_finished'
SOURCE_ROOT='npu_prototype/fused_warp_hist/'
SOURCE_FILES=('op_kernel/bm25_fused_score_top_k.cpp','op_host/bm25_fused_score_top_k.cpp')
REFERENCE='https://gitcode.com/cann/cann-samples/blob/d4f652e821a3ffc3282400771e1e785c874bb26c/Samples/2_Performance/simt_histogram_story/src/1_basic_simt.asc'
REFERENCE_SHA256='085cb178102d1adc8f20b1bb8c67827f35bd4cc3f4644e9cc0119f546ba85295'


def candidate_sources(original):
    source=t.candidate_sources(original)
    for name in SOURCE_FILES:source[name]=(ROOT/SOURCE_ROOT/name).read_bytes()
    # Scoring arithmetic, query-term order, offsets and all global outputs stay unchanged.
    old=t.candidate_sources(original)['op_kernel/bm25_fused_score_top_k.cpp'].decode()
    new=source['op_kernel/bm25_fused_score_top_k.cpp'].decode()
    marker='constexpr float kMaxFloat = 3.402823466e+38F;'
    r.require(old.count(marker)==new.count(marker)==1 and old.split(marker)[1]==new.split(marker)[1],
        '候选改动超出局部选择/UB布局：评分或入口发生变化')
    return source


def prepare(original,target):
    source=candidate_sources(original);metadata=t.prepare(original,target)
    for name,raw in source.items():(target/name).write_bytes(raw)
    file=target/'CMakePresets.json';data=json.loads(file.read_text(encoding='utf-8'))
    for item in data.get('configurePresets',[]):
        variables=item.get('cacheVariables',{});value=variables.get('vendor_name')
        if isinstance(value,dict):
            r.require(value.get('value')==t.VENDOR,'副本vendor配置不符');value['value']=VENDOR
        elif value is not None:
            r.require(value==t.VENDOR,'副本vendor配置不符');variables['vendor_name']=VENDOR
    r.write_json(file,data)
    metadata.update(status='warp_hist_prepared_not_compiled_or_device_verified',vendor_name=VENDOR,
        explicit_ub_bytes=108416,dynamic_ub_bytes=131072,selection_scratch_bytes=74304,
        first_pass_histograms=8,histogram_stride_words=2056,private_histogram_words=14392,
        source_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in source.items()},
        reference_source={'url':REFERENCE,'sha256':REFERENCE_SHA256,'scope':'Read-only layout reference; own VF adaptation, no runtime dependency.'},
        scope='Only first radix-pass UB histogram privatization + required UB allocation. 256threads, FP32 scoring, '
        'three radix widths, later passes, compaction, tie handling, global addresses, host timer and CPU unchanged. '
        'Extra clear/reduction and potential DCache capacity cost included; no assumed speedup.')
    r.write_json(target/'bm25_project.json',metadata);return metadata


def candidate_install(vendor,project,original):
    data=json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    wanted={'vendor_name':VENDOR,'simt_threads':256,'explicit_ub_bytes':108416,'dynamic_ub_bytes':131072,
        'selection_scratch_bytes':74304,'first_pass_histograms':8,'histogram_stride_words':2056,'private_histogram_words':14392}
    r.require(vendor.resolve(strict=True).name==VENDOR and all(data.get(k)==v for k,v in wanted.items()),'候选布局/vendor元数据不符')
    for name,raw in candidate_sources(original).items():
        r.require((project/name).read_bytes()==raw and digest(project/name)==data['source_sha256'][name],'候选计算源码发生变化：'+name)
    header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name,parameters in fused.PROTOTYPES.items():
        found=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,flags=re.S)
        r.require(len(found)==1 and re.sub(r'\s+','',found[0])==re.sub(r'\s+','',parameters),'实际接口与已核对ABI不同，停止：'+name)
    objects=fused.chain.artifacts(vendor,project,'bm25_fused_score_top_k',header.name)
    return {'vendor':str(vendor.resolve()),'source_sha256':data['source_sha256'],'header_sha256':digest(header),
        'api_sha256':digest(vendor/'op_api/lib/libcust_opapi.so'),'metadata_sha256':digest(project/'bm25_project.json'),
        'device_objects':objects}


def profile_evidence(path,inputs,runner,baseline):
    data=json.loads((path/'report.json').read_text(encoding='utf-8'))
    r.require(data.get('status')==profile.STATUS and data.get('kernel_variant')==profile.VARIANT and
        data.get('tool_source_sha256')==close.checks.sources(profile.TOOLS),'需要已完成的256线程短采集；不重新采集')
    file=path/'device_analysis.json';r.require(digest(file)==data.get('device_analysis_sha256'),'已通过采集分析改变')
    result=json.loads(file.read_text(encoding='utf-8'))
    record,session=profile.validate_record(path/'application')
    r.require(record['input_directory']==str(inputs) and record['runner']==str(runner) and
        record['candidate_installation']==baseline and result.get('kernel_variant')==profile.VARIANT and
        result.get('status')=='retained_t256_device_rows_and_call_samples_cross_checked' and
        result.get('own_operator_rows')==78 and result.get('measured_search_calls')==20,'采集输入/实际程序/算子/任务数量不符')
    close.scale.unchanged(result['source_sha256'],'已采集CSV或分析来源改变')
    proofs=dict(result['source_sha256']);proofs.update({str(path/'report.json'):digest(path/'report.json'),str(file):digest(file)})
    return proofs,result['cases']


def first_pass_counts(cache):
    """Untimed address-count model from checked FP32 oracle scores, not a hardware counter or timing."""
    r.require(sys.byteorder=='little','与已有调用器相同，只接受小端FP32环境')
    result={}
    for name in ('A','B'):
        item=cache[name];width=item['n']//32
        raw=array('I');raw.frombytes(array('f',item['scores']).tobytes())
        shared=[];private=[];source=[];skipped=0
        for begin in range(0,len(raw),width):
            row=raw[begin:begin+width]
            positive=[(doc,bits) for doc,bits in enumerate(row) if bits&0x7fffffff]
            compact=width>1024 and width<=256*64 and len(positive)<=1024
            if compact and len(positive)<=item['k']:skipped+=1;continue
            inputs=positive if compact else enumerate(row)
            whole=Counter();warps=Counter();count=0
            for doc,bits in inputs:
                r.require(bits&0x7fffffff<0x7f800000 and (bits<0x80000000 or bits==0x80000000),'统计发现非法分数')
                bucket=((~(0 if bits&0x7fffffff==0 else bits))&0xffffffff)>>21
                whole[bucket]+=1;warps[(doc%256//32,bucket)]+=1;count+=1
            shared.append(max(whole.values()));private.append(max(warps.values()));source.append(count)
        result[name]={'blocks':len(shared),'early_return_blocks':skipped,
            'median_first_pass_source_values':statistics.median(source) if source else 0,
            'median_hottest_shared_counter_updates':statistics.median(shared) if shared else 0,
            'median_hottest_private_counter_updates':statistics.median(private) if private else 0,
            'scope':'Source-derived counts only. Total atomic count unchanged; no UB-bank/serialization/latency or speedup inference.'}
    return result


def build_install(original,target,vendors,output):
    metadata=prepare(original,target)
    logs=output/'compiler_logs';logs.mkdir();env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(logs)
    print('只编译独立首轮Warp直方图候选；日志：'+str(output/'resource_build.log'),flush=True)
    with (output/'resource_build.log').open('wb') as stream:
        child=subprocess.run(['bash','build.sh'],cwd=target,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    r.require(child.returncode==0,'编译失败；查看 '+str(output/'resource_build.log'))
    old=t.resource.cache(original/'build_out/CMakeCache.txt');new=t.resource.cache(target/'build_out/CMakeCache.txt')
    r.require(old and new.get('vendor_name')==VENDOR and {k:v for k,v in old.items() if k!='vendor_name'}==
        {k:v for k,v in new.items() if k!='vendor_name'},'候选构建等级/计算目标不符')
    logs,resources=t.resource.compiler_reports(output)
    print('候选实际VF资源：'+json.dumps(resources,ensure_ascii=False)+'；缺失保持UNKNOWN。',flush=True)
    packages=list((target/'build_out').glob('*.run'));r.require(len(packages)==1,'候选安装包不唯一')
    listing=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (output/'package_list.log').write_bytes(listing.stdout)
    names=set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listing.stdout.decode('utf-8',errors='replace')))
    r.require(listing.returncode==0 and names=={VENDOR},'安装包不只包含独立候选vendor')
    with (output/'install.log').open('wb') as stream:
        child=subprocess.run(['bash',str(packages[0]),'--install-path='+str(vendors.parent)],stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    r.require(child.returncode==0,'安装失败；查看 '+str(output/'install.log'))
    installed=candidate_install(vendors/VENDOR,target,original)
    print('已读取实际生成头文件：两阶段函数声明与已核对ABI一致，复用现有调用器。',flush=True)
    return {'metadata':metadata,'compiler_log_records':logs,'vf_reports':resources,'installation':installed}


def paired(output,a,index,records,cache,raw,baseline,candidate,check):
    capq=max(x['num_queries'] for x in records.values());capwords=max(x['query_term_count'] for x in records.values())
    report={'status':'running','per_round':[],'cpu_rounds':[],'timer_policy':close.checks.TIMER,
        'parameters':{'baseline':'retained_t256','candidate':'t256_first_pass_warp_hist','shards':32,
            'device':a.device,'warmup':5,'repeats':30,'rounds':6}}
    def save():r.write_json(output/'comparison.json',report)
    def measure(directory,vendors):
        check();value=accepted.measure(directory,a.input,index,records,cache,raw,a.runner,vendors,a.device,capq,capwords)
        check();return value
    save()
    for i in range(6):
        row={'round':i+1,'process_order':['baseline','candidate'] if i%2==0 else ['candidate','baseline']}
        print('原256/首轮Warp直方图完整Search第'+str(i+1)+'/6轮：'+str(row['process_order']),flush=True)
        for role in row['process_order']:row[role]=measure(output/f'round{i+1}'/role,baseline if role=='baseline' else candidate)
        close.cross(row,cache,32);report['per_round'].append(row);save()
    report['decision']=close.decision(report['per_round']);save()
    if report['decision']['keep_fused_candidate']:
        for i in range(6):
            row={'round':i+1,'process_order':['cpu','candidate'] if i%2==0 else ['candidate','cpu']}
            print('同期CPU32/首轮Warp直方图第'+str(i+1)+'/6轮：'+str(row['process_order']),flush=True)
            for role in row['process_order']:
                check();directory=output/f'cpu_round{i+1}'/role
                if role=='candidate':row[role]=measure(directory,candidate)
                else:
                    row[role]=close.bench.run_one(directory,'cpu',a.input,index,records,cache,raw,a.cpu_runner,baseline,
                        {'topk_acceptance':{'workspace_bytes':0}},32,32,32,a.device,capq,capwords,5,30)
                    r.require(row[role]['actual_threads']==32,'实际CPU线程不足32')
                check()
            close.cross(row,cache,32,'cpu');report['cpu_rounds'].append(row);save()
        report['cpu_comparison']=close.decision([dict(x,baseline=x['cpu']) for x in report['cpu_rounds']])
    report['status']=STATUS;save();return report


def run(a):
    r.require(not a.output.exists() and a.output.is_relative_to((ROOT/'results').resolve()) and all(
        not a.output.is_relative_to(p) and not p.is_relative_to(a.output) for p in
        (a.input,a.previous,a.completed,a.project,a.vendors,a.acceptance,a.profile)),'需要results内独立新目录，不覆盖/嵌套旧试验')
    base=profile.evidence(a);index,records,snapshot,original,baseline,runtime,proofs,resources=base
    profile_proofs,cases=profile_evidence(a.profile,a.input,a.runner,baseline);proofs.update(profile_proofs)
    source_files=tuple(SOURCE_ROOT+name for name in SOURCE_FILES)
    p_hashes=close.checks.sources(profile.TOOLS)
    frozen=close.checks.sources(profile.TOOLS+('standalone/fused_warp_hist.py','standalone/fused_resource_report.py')+source_files)
    target_vendor=a.vendors/VENDOR;r.require(not target_vendor.exists(),'候选vendor已存在，不覆盖')
    a.output.mkdir(parents=True)
    report={'status':'running','device':a.device,'input_sha256':snapshot,'proof_sha256':proofs,'source_sha256':frozen,
        'baseline_installation':baseline,'baseline_vf_reports':resources,'profiled_baseline_cases':cases,
        'hypothesis':'Per-warp first-pass UB counters reduce cross-warp same-address atomic contention. '
        'Atomic bottleneck is not proven by the fused timing; extra clear/reduce and DCache capacity cost may dominate.',
        'reference_source':{'url':REFERENCE,'sha256':REFERENCE_SHA256},
        'scope':'One operator/SIMT candidate, not CPU/host/precision/query changes. Profile reused, not rerun. '
        'Changed candidate correctness before six alternating full Search pairs. ConditionalCPU32/32 only after retention.'}
    r.write_json(a.output/'report.json',report)
    try:
        def baseline_unchanged():
            profile.unchanged(a,base,p_hashes)
            close.scale.unchanged(profile_proofs,'既有采集证据改变')
            r.require(close.checks.sources(tuple(frozen))==frozen,'实验冻结源码改变')
        baseline_unchanged()
        cache=close.checks.cache_inputs(a.input,index,records)
        report['first_pass_address_count_model']=first_pass_counts(cache)
        for name,value in report['first_pass_address_count_model'].items():
            print(name+' 首轮计数地址模型（非硬件计数/收益）：'+json.dumps(value,ensure_ascii=False),flush=True)
        r.write_json(a.output/'report.json',report)
        report['candidate']=build_install(a.project,a.output/'project',a.vendors,a.output)
        candidate=a.output/'runtime/vendors';candidate.mkdir(parents=True)
        for name,value in (('bm25_fused_score_topk',target_vendor),('bm25_topk_merge',a.vendors/'bm25_topk_merge')):
            (candidate/name).symlink_to(value,target_is_directory=True)
        def unchanged():
            baseline_unchanged()
            r.require(candidate_install(target_vendor,a.output/'project',a.project)==report['candidate']['installation'] and
                accepted.opp_paths(candidate)==[str(target_vendor.resolve()),str((a.vendors/'bm25_topk_merge').resolve())],
                '新候选源码/安装/运行库映射改变')
        unchanged();r.write_json(a.output/'report.json',report)
        print('只验收改变的首轮直方图候选：当前负载A/B/A/空/B，debug1/debug0；原版不重复验收。',flush=True)
        report['acceptance']=t.diagnostic(a.input,a.output/'acceptance',a.diagnostic_runner,candidate,a.device,index,records,cache)
        report['acceptance'].update(status='first_pass_warp_hist_actual_correctness_passed',
            scope='Own per-warp first-pass counts; FP32 full scores, local/final strict rankings and guards checked.')
        r.write_json(a.output/'acceptance/report.json',report['acceptance']);unchanged();r.write_json(a.output/'report.json',report)
        raw=a.output/'raw_queries';raw.mkdir()
        for name in ('A','B'):close.search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
        raw_hash={str(p):digest(p) for p in raw.iterdir()}
        def check_all():unchanged();close.scale.unchanged(raw_hash,'原始查询改变')
        report['comparison']=paired(a.output,a,index,records,cache,raw,runtime,candidate,check_all)
        check_all();report['status']=STATUS;r.write_json(a.output/'report.json',report)
        for name,value in report['comparison']['decision']['cases'].items():
            print(name+' 原256/首轮Warp直方图：'+json.dumps(value,ensure_ascii=False),flush=True)
        for name,value in report['comparison'].get('cpu_comparison',{}).get('cases',{}).items():
            print(name+' 同期CPU32/候选：'+json.dumps(value,ensure_ascii=False),flush=True)
        print('结论：'+('首轮Warp直方图达到原定保留门槛，CPU结果另列。' if
            report['comparison']['decision']['keep_fused_candidate'] else
            '放弃首轮Warp直方图候选，继续保留原256线程版；不增加扫描或变体。'),flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));r.write_json(a.output/'report.json',report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    defaults={'previous':'results/fused_t256_close1','completed':'results/fused_t256_compare2','profile':'results/fused_t256_profile1',
        'input':'results/search_256k_q8_input1','project':'results/Bm25FusedScoreTopKUbFix1','vendors':'results/custom_opp/vendors',
        'merge-project':'results/Bm25TopKMerge','runner':'results/fused_search_host_build/bm25_fused_search_benchmark',
        'diagnostic-runner':'results/fused_chain_host_build/bm25_fused_chain_runner',
        'cpu-runner':'results/search_benchmark_host_build/bm25_cpu_search_benchmark','acceptance':'results/fused_search_256k_q8_close1'}
    for name,value in defaults.items():p.add_argument('--'+name,type=Path,default=ROOT/value)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=5)
    a=p.parse_args()
    try:
        r.require(sys.platform=='linux' and r.integer(a.device,0,r.LIMIT),'需要A5 Linux和合法设备号')
        for name,value in vars(a).items():
            if isinstance(value,Path):setattr(a,name,value.resolve())
        run(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；首错停止，不继续测速或生成CPU替代输出。',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
