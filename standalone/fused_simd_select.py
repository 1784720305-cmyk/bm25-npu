"""One mixed SIMT-score/SIMD-selection candidate; fail-fast affected checks then complete Search pairs."""
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
from standalone import fused_warp_hist as old_trial, fused_simd_sources as source

profile=old_trial.profile
t=profile.t
close=t.close
fused=t.fused
accepted=profile.accepted
r=t.r
digest=t.digest
profile_evidence=old_trial.profile_evidence
VENDOR='bm25_fused_score_topk_simd'
STATUS='fused_simd_selection_complete_search_comparison_finished'
OWN=('standalone/fused_simd_select.py','standalone/fused_simd_sources.py',source.HEADER,
     'npu_prototype/fused_simd_check_host/selector_check.cpp',
     'npu_prototype/fused_simd_check_host/CMakeLists.txt')


def actual_headers(toolkit):
    """Read actual basic API declarations before compiling. Compiler checks full overloads/types.

    Discovery is not permission to substitute another API. An absent declaration stops this run.
    Public source commit and actual beta.2 toolkit are intentionally recorded separately.
    """
    paths={p.resolve() for p in toolkit.rglob('kernel_reg_compute_histograms_intf.h') if p.is_file()}
    r.require(len(paths)==1,'实际Reg Histograms接口头文件不唯一或不存在；停止，不猜替代接口')
    histogram=next(iter(paths));directory=histogram.parent
    names=('Histograms','LoadAlign','StoreAlign','DeInterleave','Interleave','Compares','Pack',
           'ShiftRights','Not','Duplicate','And','UpdateMask','CreateMask')
    found={name:[] for name in names};records={}
    for p in sorted(directory.glob('*.h')):
        if p.stat().st_size>2*1024*1024:continue
        text=p.read_text(encoding='utf-8',errors='replace')
        selected=[]
        for number,line in enumerate(text.splitlines(),1):
            for name in names:
                if re.search(r'\b'+name+r'\s*\(',line) and not line.lstrip().startswith('//'):
                    found[name].append(str(p));selected.append({'line':number,'declaration':line.strip()})
        if selected:records[str(p)]={'sha256':digest(p),'declarations':selected}
    r.require(all(found.values()),'实际接口声明不全，停止：'+','.join(k for k,v in found.items() if not v))
    print('已读取实际Reg基础接口：'+str(histogram),flush=True)
    for row in records[str(histogram)]['declarations']:
        print(str(row['line'])+': '+row['declaration'],flush=True)
    return {'status':'actual_declarations_read_compiler_overload_check_still_required',
            'files':records,'api_names':list(names),'namespace_used':'AscendC::Reg',
            'scope':'Read actual CANN headers only. Not evidence of device correctness or performance.'}


def prepare(original,target):
    sources=source.candidate_sources(original);metadata=t.prepare(original,target)
    for name,raw in sources.items():(target/name).write_bytes(raw)
    preset=target/'CMakePresets.json';data=json.loads(preset.read_text(encoding='utf-8'))
    count=0
    for item in data.get('configurePresets',[]):
        variables=item.get('cacheVariables',{});value=variables.get('vendor_name')
        if isinstance(value,dict) and value.get('value')==t.VENDOR:value['value']=VENDOR;count+=1
        elif value==t.VENDOR:variables['vendor_name']=VENDOR;count+=1
    r.require(count>0,'副本vendor配置不符');r.write_json(preset,data)
    (target/'LICENSE-CANN').write_bytes((ROOT/'npu_prototype/fused_simd_select/LICENSE-CANN').read_bytes())
    metadata.update(status='simd_selection_source_prepared_not_device_verified',vendor_name=VENDOR,
        source_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in sources.items()},
        explicit_ub_bytes=50848,dynamic_ub_bytes=65536,selection_scratch_bytes=16736,
        simt_threads=256,radix_byte_passes=4,radix_bins=256,positive_capacity=1024,
        scope='Same SIMT FP32 term order. Own Reg SIMD CDF on positive compact input; original selector fallback '
              'for <=K/dense and strict real-ID collector/final-K sort. Two SIMT VFs and up to four SIMD '
              'scopes per shard, including V_S/S_V synchronization. No full-score GM in Search/debug0. '
              'debug2 is an isolated raw-score selector check, never public Search.')
    r.write_json(target/'bm25_project.json',metadata);return metadata


def candidate_install(vendor,project,original):
    data=json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    wanted={'vendor_name':VENDOR,'simt_threads':256,'explicit_ub_bytes':50848,'dynamic_ub_bytes':65536,
            'selection_scratch_bytes':16736,'radix_byte_passes':4,'radix_bins':256,'positive_capacity':1024}
    r.require(vendor.resolve(strict=True).name==VENDOR and all(data.get(k)==v for k,v in wanted.items()),
              '候选vendor/布局元数据不符')
    for name,raw in source.candidate_sources(original).items():
        r.require((project/name).read_bytes()==raw and digest(project/name)==data['source_sha256'][name],
                  '候选计算源码改变：'+name)
    header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name,parameters in fused.PROTOTYPES.items():
        declarations=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,flags=re.S)
        r.require(len(declarations)==1 and re.sub(r'\s+','',declarations[0])==re.sub(r'\s+','',parameters),
                  '实际生成接口与已核对ABI不同，停止：'+name)
    objects=fused.chain.artifacts(vendor,project,'bm25_fused_score_top_k',header.name)
    return {'vendor':str(vendor.resolve()),'source_sha256':data['source_sha256'],'header_sha256':digest(header),
            'api_sha256':digest(vendor/'op_api/lib/libcust_opapi.so'),
            'metadata_sha256':digest(project/'bm25_project.json'),'device_objects':objects}


def compiler_resources(output):
    logs=[];reports=[]
    pattern=re.compile(r'Function properties for\s+(\S+)\s*:\s*Stack size:\s*(\d+)\s*bytes,\s*Used register number:\s*(\d+)')
    for path in sorted((output/'compiler_logs').rglob('*')):
        if not path.is_file() or path.stat().st_size>16*1024*1024:continue
        text=path.read_text(encoding='utf-8',errors='replace')
        if not any(name in text for name in ('FusedScorePackSimt','FusedCollectSimt','HistogramCdf')):continue
        record={'path':path.relative_to(output).as_posix(),'sha256':digest(path)};logs.append(record)
        for match in pattern.finditer(text):
            if any(n in match[1] for n in ('FusedScorePackSimt','FusedCollectSimt','HistogramCdf')):
                reports.append({'path':record['path'],'function':match[1],'stack_size_bytes':int(match[2]),
                                'used_register_number':int(match[3])})
    return logs,reports


def build_install(original,target,vendors,output):
    metadata=prepare(original,target)
    logs=output/'compiler_logs';logs.mkdir();env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(logs)
    print('编译唯一独立候选；原256线程版不修改。日志：'+str(output/'resource_build.log'),flush=True)
    with (output/'resource_build.log').open('wb') as f:
        child=subprocess.run(['bash','build.sh'],cwd=target,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
    r.require(child.returncode==0,'编译失败，停止；查看 '+str(output/'resource_build.log'))
    old=t.resource.cache(original/'build_out/CMakeCache.txt');new=t.resource.cache(target/'build_out/CMakeCache.txt')
    r.require(old and new.get('vendor_name')==VENDOR and {k:v for k,v in old.items() if k!='vendor_name'}==
              {k:v for k,v in new.items() if k!='vendor_name'},'编译目标/优化等级变化')
    logs,resources=compiler_resources(output)
    print('实际候选VF资源：'+json.dumps(resources,ensure_ascii=False)+'；未输出的SIMT/SIMD字段保持UNKNOWN，不由栈大小推断溢出。',flush=True)
    packages=list((target/'build_out').glob('*.run'));r.require(len(packages)==1,'安装包不唯一')
    listing=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (output/'package_list.log').write_bytes(listing.stdout)
    names=set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listing.stdout.decode(errors='replace')))
    r.require(listing.returncode==0 and names=={VENDOR},'安装包不是独立候选vendor')
    with (output/'install.log').open('wb') as f:
        child=subprocess.run(['bash',str(packages[0]),'--install-path='+str(vendors.parent)],stdout=f,
                             stderr=subprocess.STDOUT,timeout=300)
    r.require(child.returncode==0,'安装失败，停止；查看 '+str(output/'install.log'))
    installed=candidate_install(vendors/VENDOR,target,original)
    print('已读取实际生成头文件：两阶段声明与既有调用器完全一致。',flush=True)
    return {'metadata':metadata,'compiler_log_records':logs,'vf_reports':resources,'installation':installed}


def selector_check(a,vendors,check):
    build=a.output/'selector_host_build'
    for command,log in ((['cmake','-S',str(ROOT/'npu_prototype/fused_simd_check_host'),'-B',str(build),
            '-DTOOLKIT_HOME='+str(a.toolkit),'-DBM25_VENDOR_DIR='+str(vendors)],'selector_configure.log'),
            (['cmake','--build',str(build),'-j','4'],'selector_build.log')):
        check();fused.score.execute(command,a.output/log,os.environ.copy());check()
    runner=build/'bm25_fused_simd_selector_check'
    env=os.environ.copy();env['ASCEND_CUSTOM_OPP_PATH']=':'.join(accepted.opp_paths(vendors))
    check();fused.score.execute([str(runner),str(vendors),str(a.device),str(a.output/'selector_checks')],
                               a.output/'selector_device.log',env);check()
    record=json.loads((a.output/'selector_checks/report.json').read_text(encoding='utf-8'))
    r.require(record.get('status')=='actual_simd_selector_guards_and_strict_bit_rankings_passed' and
              record.get('calls')==54 and record.get('checked_rows',0)>0 and
              record.get('host_sha256')==digest(ROOT/OWN[3]),'设备选择器检查报告不完整')
    return dict(record,runner_sha256=digest(runner),log_sha256=digest(a.output/'selector_device.log'))


def paired(output,a,index,records,cache,raw,baseline,candidate,check):
    capq=max(x['num_queries'] for x in records.values());capwords=max(x['query_term_count'] for x in records.values())
    report={'status':'running','per_round':[],'cpu_rounds':[],'timer_policy':close.checks.TIMER,
        'parameters':{'baseline':'retained_t256','candidate':'t256_simd_byte_selection','shards':32,
            'device':a.device,'warmup':5,'repeats':30,'rounds':6}}
    def save():r.write_json(output/'comparison.json',report)
    def measure(directory,vendors):
        check();value=accepted.measure(directory,a.input,index,records,cache,raw,a.runner,vendors,a.device,capq,capwords)
        check();return value
    save()
    for i in range(6):
        row={'round':i+1,'process_order':['baseline','candidate'] if i%2==0 else ['candidate','baseline']}
        print('原256/SIMD局部选择完整Search第'+str(i+1)+'/6轮：'+str(row['process_order']),flush=True)
        for role in row['process_order']:row[role]=measure(output/f'round{i+1}'/role,baseline if role=='baseline' else candidate)
        close.cross(row,cache,32);report['per_round'].append(row);save()
    report['decision']=close.decision(report['per_round']);save()
    if report['decision']['keep_fused_candidate']:
        for i in range(6):
            row={'round':i+1,'process_order':['cpu','candidate'] if i%2==0 else ['candidate','cpu']}
            print('同期CPU32/SIMD局部选择第'+str(i+1)+'/6轮：'+str(row['process_order']),flush=True)
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
    source_files=OWN
    p_hashes=close.checks.sources(profile.TOOLS)
    frozen=close.checks.sources(tuple(sorted(set(profile.TOOLS)|set(source_files)|{'standalone/fused_warp_hist.py','standalone/fused_resource_report.py'})))
    target_vendor=a.vendors/VENDOR;r.require(not target_vendor.exists(),'候选vendor已存在，不覆盖')
    a.output.mkdir(parents=True)
    report={'status':'running','device':a.device,'input_sha256':snapshot,'proof_sha256':proofs,'source_sha256':frozen,
        'baseline_installation':baseline,'baseline_vf_reports':resources,'profiled_baseline_cases':cases,
        'hypothesis':'Replace compact-input SIMT histogram atomics/scans with bounded 950 Reg hardware CDF. '
        'Extra SIMD/SIMT transitions, synchronization and fourth pass are included; no promised benefit.',
        'reference_source':'standalone/final_optimization_sources_20261023.json',
        'scope':'One mixed NPU candidate. Same score FP32/order/index/query-ID Search timer/CPU. '
        'Actual basic API declarations, generated ABI, changed selector edges and current workload '
        'before six full Search pairs; CPU32/32 only after stable retention.'}
    r.write_json(a.output/'report.json',report)
    try:
        def baseline_unchanged():
            profile.unchanged(a,base,p_hashes)
            close.scale.unchanged(profile_proofs,'既有采集证据改变')
            r.require(close.checks.sources(tuple(frozen))==frozen,'实验冻结源码改变')
        baseline_unchanged()
        cache=close.checks.cache_inputs(a.input,index,records)
        report['actual_headers']=actual_headers(a.toolkit)
        r.write_json(a.output/'report.json',report)
        report['candidate']=build_install(a.project,a.output/'project',a.vendors,a.output)
        candidate=a.output/'runtime/vendors';candidate.mkdir(parents=True)
        for name,value in (('bm25_fused_score_topk',target_vendor),('bm25_topk_merge',a.vendors/'bm25_topk_merge')):
            (candidate/name).symlink_to(value,target_is_directory=True)
        header_hash={name:row['sha256'] for name,row in report['actual_headers']['files'].items()}
        def unchanged():
            baseline_unchanged()
            close.scale.unchanged(header_hash,'实际Reg接口头文件改变')
            r.require(candidate_install(target_vendor,a.output/'project',a.project)==report['candidate']['installation'] and
                accepted.opp_paths(candidate)==[str(target_vendor.resolve()),str((a.vendors/'bm25_topk_merge').resolve())],
                '新候选源码/安装/运行库映射改变')
        unchanged();r.write_json(a.output/'report.json',report)
        report['selector_checks']=selector_check(a,candidate,unchanged)
        r.write_json(a.output/'report.json',report)
        print('只验收改变的SIMD选择候选：当前负载A/B/A/空/B，debug1/debug0；原版不重复验收。',flush=True)
        report['acceptance']=t.diagnostic(a.input,a.output/'acceptance',a.diagnostic_runner,candidate,a.device,index,records,cache)
        report['acceptance'].update(status='simd_selection_actual_current_workload_correctness_passed',
            scope='Own SIMD CDF and exact real-ID collector; FP32 full scores, local/final strict rankings and guards checked.')
        r.write_json(a.output/'acceptance/report.json',report['acceptance']);unchanged();r.write_json(a.output/'report.json',report)
        raw=a.output/'raw_queries';raw.mkdir()
        for name in ('A','B'):close.search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
        raw_hash={str(p):digest(p) for p in raw.iterdir()}
        def check_all():unchanged();close.scale.unchanged(raw_hash,'原始查询改变')
        report['comparison']=paired(a.output,a,index,records,cache,raw,runtime,candidate,check_all)
        check_all();report['status']=STATUS;r.write_json(a.output/'report.json',report)
        for name,value in report['comparison']['decision']['cases'].items():
            print(name+' 原256/SIMD局部选择：'+json.dumps(value,ensure_ascii=False),flush=True)
        for name,value in report['comparison'].get('cpu_comparison',{}).get('cases',{}).items():
            print(name+' 同期CPU32/候选：'+json.dumps(value,ensure_ascii=False),flush=True)
        print('结论：'+('SIMD局部选择达到原定保留门槛，CPU结果另列。' if
            report['comparison']['decision']['keep_fused_candidate'] else
            '放弃SIMD局部选择候选，继续保留原256线程版；不增加扫描或变体。'),flush=True)
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
    p.add_argument('--toolkit',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'));
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
