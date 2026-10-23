"""One isolated 128-vs-256 SIMT experiment; same formula, selector, ABI and complete Search boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import close_fused_search as close, profile_fused_search as prior
from standalone import fused_resource_report as resource

fused=close.fused
r=close.r
digest=fused.digest
VENDOR='bm25_fused_score_topk_t256'
STATUS='fused_t256_complete_search_comparison_finished'
THREAD_REFERENCE='https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simt_programming/memory_hierarchy.html'


def replace_once(text,old,new):
    r.require(text.count(old)==1,'已冻结源码/模板不符，不自动适配：'+old)
    return text.replace(old,new,1)


def baseline_resources(path,project,runner):
    """Reuse the completed resource report as immutable evidence, without rebuilding/reprofiling it."""
    record=json.loads((path/'report.json').read_text(encoding='utf-8'))
    r.require(record.get('status')=='resource_diagnostic_compile_completed' and
        record.get('accepted_artifacts_unchanged') is True and record.get('accepted_runner_sha256')==digest(runner),
        '需要已完成且对应当前正式程序的128线程资源报告')
    sources=record.get('accepted_source_sha256',{})
    names=set(fused.generator.SOURCES)|{'op_kernel/CMakeLists.txt','CMakePresets.json','build.sh'}
    r.require(set(sources)==names and all(digest(project/n)==sources[n] for n in names),'128线程资源报告来源改变')
    values={(item['function'],item['stack_size_bytes'],item['used_register_number'])
        for item in record.get('fused_vf_reports',[]) if 'FusedScoreTopKSimt' in item.get('function','')}
    r.require(len(values)==1 and next(iter(values))[1:]==(68,114),'资源报告与本轮68字节栈/114寄存器依据不符')
    proof={str(path/'report.json'):digest(path/'report.json')}
    for item in record.get('compiler_log_records',[]):
        log=(path/item['path']).resolve(strict=True)
        r.require(log.is_relative_to(path.resolve()) and digest(log)==item['sha256'],'既有编译阶段日志来源改变')
        proof[str(log)]=item['sha256']
    return {'stack_size_bytes':68,'used_register_number':114,'attributed_functions':len(values),'proof_sha256':proof}


def candidate_sources(project):
    """Only launch width and its dependent warp-total storage change; no new selection algorithm."""
    sources={name:(project/name).read_bytes() for name in fused.generator.SOURCES}
    kernel=sources['op_kernel/bm25_fused_score_top_k.cpp'].decode('utf-8')
    for old,new in (
        ('constexpr uint32_t kThreads = 128;','constexpr uint32_t kThreads = 256;'),
        ('constexpr uint32_t kControl = kScan + 8;','constexpr uint32_t kControl = kScan + 2*kWarps;'),
        ('kSelectScratchBytes == 16704','kSelectScratchBytes == 16736'),
        ('kBufferBytes == 50816','kBufferBytes == 50848')):
        kernel=replace_once(kernel,old,new)
    kernel=kernel.replace('scan totals[8]','scan totals[16]').replace('// 16704','// 16736')
    kernel=kernel.replace('All 128 threads','All 256 threads').replace('four explicit UB warp totals','eight explicit UB warp totals')
    kernel=kernel.replace('Merge four warp minima','Merge eight warp minima').replace('all 128 threads','all 256 threads')
    kernel=kernel.replace('50,816 bytes','50,848 bytes')
    host=sources['op_host/bm25_fused_score_top_k.cpp'].decode('utf-8')
    host=replace_once(host,'constexpr uint32_t kBufferBytes = 50816;','constexpr uint32_t kBufferBytes = 50848;')
    host=host.replace('50,816-byte','50,848-byte')
    sources['op_kernel/bm25_fused_score_top_k.cpp']=kernel.encode('utf-8')
    sources['op_host/bm25_fused_score_top_k.cpp']=host.encode('utf-8')
    return sources


def prepare(project,target):
    project=project.resolve(strict=True);target=target.resolve()
    r.require(not target.exists() and target.is_relative_to((ROOT/'results').resolve()) and
        not target.is_relative_to(project) and not project.is_relative_to(target),'新工程必须独立放在results下，不覆盖/嵌套原工程')
    source=candidate_sources(project)
    cmake=(project/'op_kernel/CMakeLists.txt').read_text(encoding='utf-8')
    hooks=list(re.finditer(r'(?im)^[ \t]*if\s*\([^\n]*CMAKE_BUILD_TYPE[^\n]*Debug[^\n]*\)\s*\n'
        r'[ \t]*npu_op_kernel_options\s*\(\s*ascendc_kernels\s+ALL\s+OPTIONS\b',cmake))
    r.require(len(hooks)==1 and '--cce-res-usage' not in cmake,'需要已核对的原版CANN9.2模板，不复用资源副本')
    for directory,dirs,files in os.walk(project,followlinks=False):
        dirs[:]=[name for name in dirs if name not in resource.EXCLUDED]
        for name in dirs+files:r.require(not (Path(directory)/name).is_symlink(),'工程包含符号链接，停止复制')
    shutil.copytree(project,target,ignore=lambda unused,names:[n for n in names if n in resource.EXCLUDED])
    for path in target.rglob('*'):
        if path.is_file() and (path.suffix in ('.cmake','.sh') or path.name=='CMakeLists.txt'):
            r.require(str(project) not in path.read_text(encoding='utf-8',errors='replace'),'副本构建脚本硬编码原工程路径')
    for name,data in source.items():(target/name).write_bytes(data)
    added='npu_op_kernel_options(ascendc_kernels ALL OPTIONS --cce-res-usage)\n'
    (target/'op_kernel/CMakeLists.txt').write_bytes((cmake[:hooks[0].start()]+added+cmake[hooks[0].start():]).encode('utf-8'))
    preset=target/'CMakePresets.json';data=json.loads(preset.read_text(encoding='utf-8'));count=0
    for item in data.get('configurePresets',[]):
        variables=item.get('cacheVariables',{})
        if 'ASCEND_COMPUTE_UNIT' not in variables:continue
        value=variables.get('vendor_name')
        if isinstance(value,dict):value['value']=VENDOR
        else:variables['vendor_name']={'type':'STRING','value':VENDOR}
        count+=1
    r.require(count>0,'工程缺少已核对的vendor配置')
    r.write_json(preset,data)
    metadata=json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    metadata.update(status='t256_generated_not_compiled_or_device_verified',vendor_name=VENDOR,
        simt_threads=256,warps=8,explicit_ub_bytes=50848,selection_scratch_bytes=16736,
        baseline_project=str(project),source_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in source.items()},
        scope='Only threads128->256 and dependent warp-total storage +32bytes; original FP32/order/selection/ABI/64KiB dynamic UB retained.')
    r.write_json(target/'bm25_project.json',metadata)
    return metadata


def candidate_install(vendor,project,baseline):
    expected=candidate_sources(baseline)
    data=json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    r.require(vendor.resolve(strict=True).name==VENDOR and data.get('vendor_name')==VENDOR and
        data.get('simt_threads')==256 and data.get('explicit_ub_bytes')==50848 and data.get('dynamic_ub_bytes')==65536,
        '候选vendor/线程/UB配置不符')
    for name,raw in expected.items():
        r.require((project/name).read_bytes()==raw and digest(project/name)==data['source_sha256'][name],
            '候选计算源码不是唯一已核对的256线程改动：'+name)
    header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name,parameters in fused.PROTOTYPES.items():
        found=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,flags=re.S)
        r.require(len(found)==1 and re.sub(r'\s+','',found[0])==re.sub(r'\s+','',parameters),
            '实际候选接口与已核对ABI不一致；不启动设备：'+name)
    objects=fused.chain.artifacts(vendor,project,'bm25_fused_score_top_k',header.name)
    return {'vendor':str(vendor.resolve()),'source_sha256':data['source_sha256'],
        'header_sha256':digest(header),'api_sha256':digest(vendor/'op_api/lib/libcust_opapi.so'),
        'metadata_sha256':digest(project/'bm25_project.json'),'device_objects':objects}


def diagnostic(inputs,output,runner,vendors,device,index,records,cache):
    output.mkdir();lines=[]
    for i,name in enumerate(fused.SEQUENCE):
        path=str(inputs/name);r.require(not any(c in path for c in '\r\n\t'),'查询路径含控制字符')
        lines.append(f'call{i}_{name}\t{path}')
    plan=output/'requests.tsv';plan.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    command=[str(runner),str(inputs/'index'),str(plan),str(output/'device'),str(device),str(vendors),'10','32']
    env=os.environ.copy();env['ASCEND_CUSTOM_OPP_PATH']=':'.join(str((vendors/n).resolve(strict=True))
        for n in ('bm25_fused_score_topk','bm25_topk_merge'))
    print('只验收改变的256线程路径：当前256K/Q8，A/B/A/空/B，每批debug1/debug0；不重复旧版设备验收。',flush=True)
    fused.score.execute(command,output/'device.log',env)
    session=fused.check_session(output/'device',index,records,device,10,32)
    checked=[]
    for i,name in enumerate(fused.SEQUENCE):
        checked.append(fused.check_output(inputs,name,output/'device'/f'call{i}_{name}',index,records,10,32,cache[name]['scores']))
    hashes={p.relative_to(output/'device').as_posix():digest(p) for p in (output/'device').rglob('*.bin')}
    result={'status':'t256_actual_workload_correctness_passed','session':session,'checks':checked,
        'output_sha256':hashes,'command':command,'log_sha256':digest(output/'device.log')}
    r.write_json(output/'report.json',result)
    return result


def paired(output,inputs,index,records,cache,raw,runner,cpu_runner,baseline_runtime,candidate_runtime,device,check_frozen):
    """Same accepted executable/checkers/timer, balanced process order; CPU only after a retained improvement."""
    capq=max(x['num_queries'] for x in records.values());capwords=max(x['query_term_count'] for x in records.values())
    report={'status':'running','per_round':[],'cpu_rounds':[],'timer_policy':close.checks.TIMER,
        'parameters':{'baseline_threads':128,'candidate_threads':256,'device':device,'shards':32,'warmup':5,'repeats':30}}
    def save():r.write_json(output/'comparison.json',report)
    def candidate(directory):
        return close.run_fused(directory,inputs,index,records,cache,raw,runner,candidate_runtime,device,32,capq,capwords,5,30)
    save()
    for i in range(6):
        row={'round':i+1,'process_order':['baseline','candidate'] if i%2==0 else ['candidate','baseline']}
        print('完整Search 128/256线程，第'+str(i+1)+'/6轮：'+str(row['process_order']),flush=True)
        for role in row['process_order']:
            check_frozen();directory=output/f'round{i+1}'/role
            row[role]=(candidate(directory) if role=='candidate' else
                close.run_fused(directory,inputs,index,records,cache,raw,runner,baseline_runtime,device,32,capq,capwords,5,30))
        close.cross(row,cache,32);check_frozen();report['per_round'].append(row);save()
    report['decision']=close.decision(report['per_round']);save()
    if report['decision']['keep_fused_candidate']:
        print('256线程达到预定门槛，继续同期CPU32线程/32文章分区对照；CPU源码和设置不改。',flush=True)
        for i in range(6):
            row={'round':i+1,'process_order':['cpu','candidate'] if i%2==0 else ['candidate','cpu']}
            for role in row['process_order']:
                check_frozen();directory=output/f'cpu_round{i+1}'/role
                # The shared CPU harness accepts a workspace parameter but never uses NPU memory on CPU.
                row[role]=(candidate(directory) if role=='candidate' else close.bench.run_one(directory,'cpu',inputs,
                    index,records,cache,raw,cpu_runner,baseline_runtime,{'topk_acceptance':{'workspace_bytes':0}},
                    32,32,32,device,capq,capwords,5,30))
                if role=='cpu':r.require(row[role]['actual_threads']==32,'CPU实际线程不足32')
            close.cross(row,cache,32,'cpu');check_frozen();report['cpu_rounds'].append(row);save()
        report['cpu_comparison']=close.decision([dict(x,baseline=x['cpu']) for x in report['cpu_rounds']])
    report['status']=STATUS;save()
    return report


def run(a):
    inputs,output,project,vendors,merge,runner,diag_runner,cpu_runner,acceptance=(getattr(a,n).resolve()
        for n in ('input','output','project','vendors','merge_project','runner','diagnostic_runner','cpu_runner','acceptance'))
    r.require(not output.exists() and output.is_relative_to((ROOT/'results').resolve()),'需要results下新目录，不覆盖/重启')
    r.require(r.integer(a.device,0,r.LIMIT) and all(p.is_file() for p in (runner,diag_runner,cpu_runner)),'设备号/已有主机程序不符')
    index,records,snapshot,installed,proofs=prior.evidence(inputs,acceptance,runner,vendors,project,merge)
    resources=baseline_resources(a.baseline_resources.resolve(),project,runner)
    proofs.update(resources['proof_sha256'])
    r.require((index['num_docs'],records['A']['num_queries'],records['A']['k'])==(262144,8,10),'只验收当前256K/Q8/K10/S32工作负载')
    binary=diag_runner.read_bytes()
    r.require(all(sha.encode('ascii') in binary for sha in close.checks.sources(fused.HOST_SOURCES).values()),'诊断程序不是现有已验收源码')
    target_vendor=vendors/VENDOR;r.require(not target_vendor.exists(),'候选vendor已存在，不覆盖原试验')
    frozen=close.checks.sources(tuple(sorted(set(fused.TOOL_SOURCES)|set(fused.HOST_SOURCES)|set(close.fc.HOST_SOURCES)|set(close.checks.CPU_SOURCES)|{
        'standalone/fused_threads256.py','standalone/close_fused_search.py','standalone/fused_resource_report.py',
        *('npu_prototype/fused_score_topk/'+n for n in fused.generator.SOURCES)})))
    binaries={str(p):digest(p) for p in (runner,diag_runner,cpu_runner)}
    output.mkdir(parents=True);report={'status':'running','device':a.device,'baseline_installation':installed,
        'baseline_resources':resources,
        'prior_proofs_sha256':proofs,'frozen_source_sha256':frozen,'runner_sha256':binaries,'input_sha256':snapshot,
        'thread_configuration_reference':THREAD_REFERENCE,
        'scope':'One SIMT launch-width candidate, not a new algorithm or broad parameter sweep. Full Search includes raw-ID packing/upload and CPU Top-K receipt.'}
    r.write_json(output/'report.json',report)
    try:
        def baseline_unchanged():
            for evidence in (snapshot,proofs,binaries):close.scale.unchanged(evidence,'输入/既有证据/主机程序改变')
            r.require(close.checks.sources(tuple(frozen))==frozen and fused.installed(vendors,project,merge)==installed,
                '冻结源码或128线程正式安装改变')
        target=output/'project';report['candidate_metadata']=prepare(project,target)
        logs=output/'compiler_logs';logs.mkdir();env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(logs)
        print('独立256线程工程：'+str(target)+'；独立vendor='+VENDOR+'；只增加32字节临时UB，仍声明64KiB动态UB。',flush=True)
        print('开始编译，输出写入 '+str(output/'resource_build.log')+'；完成后自动读取实际接口，不要重复启动。',flush=True)
        with (output/'resource_build.log').open('wb') as stream:
            child=subprocess.run(['bash','build.sh'],cwd=target,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
        r.require(child.returncode==0,'候选编译失败；查看 '+str(output/'resource_build.log'))
        old_options=resource.cache(project/'build_out/CMakeCache.txt');new_options=resource.cache(target/'build_out/CMakeCache.txt')
        r.require(old_options and new_options.get('vendor_name')==VENDOR,'候选实际构建vendor不符')
        r.require({k:v for k,v in old_options.items() if k!='vendor_name'}==
            {k:v for k,v in new_options.items() if k!='vendor_name'},'候选优化等级/计算目标与128线程版不同')
        report['compiler_log_records'],report['candidate_vf_reports']=resource.compiler_reports(output)
        print('256线程实际VF资源：'+json.dumps(report['candidate_vf_reports'],ensure_ascii=False)+'；缺失仍为UNKNOWN。',flush=True)
        packages=list((target/'build_out').glob('*.run'));r.require(len(packages)==1,'候选安装包不唯一')
        listed=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        (output/'package_list.log').write_bytes(listed.stdout)
        vendor_names=set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listed.stdout.decode('utf-8',errors='replace')))
        r.require(listed.returncode==0 and vendor_names=={VENDOR},'安装包不只包含独立候选vendor，停止安装')
        print('安装独立候选vendor；日志：'+str(output/'install.log')+'；128线程vendor随后检查未变。',flush=True)
        with (output/'install.log').open('wb') as stream:
            child=subprocess.run(['bash',str(packages[0]),'--install-path='+str(vendors.parent)],
                stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        r.require(child.returncode==0,'候选安装失败；查看 '+str(output/'install.log'))
        candidate=candidate_install(target_vendor,target,project);report['candidate_installation']=candidate
        print('已读取实际生成接口，两个函数声明与已核对ABI一致；现有主机二进制可复用。',flush=True)
        runtime=output/'runtime/vendors';runtime.mkdir(parents=True)
        for name,value in (('bm25_fused_score_topk',target_vendor),('bm25_topk_merge',vendors/'bm25_topk_merge')):
            (runtime/name).symlink_to(value,target_is_directory=True)
        def unchanged():
            baseline_unchanged()
            r.require(candidate_install(target_vendor,target,project)==candidate and
                (runtime/'bm25_fused_score_topk').resolve()==target_vendor.resolve() and
                (runtime/'bm25_topk_merge').resolve()==(vendors/'bm25_topk_merge').resolve(),'候选或运行库映射改变')
        unchanged();cache=close.checks.cache_inputs(inputs,index,records)
        report['acceptance']=diagnostic(inputs,output/'acceptance',diag_runner,runtime,a.device,index,records,cache)
        unchanged();r.write_json(output/'report.json',report)
        raw=output/'raw_queries';raw.mkdir()
        for name in ('A','B'):close.search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
        raw_hash={str(p):digest(p) for p in raw.iterdir()}
        def check_all():unchanged();close.scale.unchanged(raw_hash,'原始查询改变')
        report['comparison']=paired(output,inputs,index,records,cache,raw,runner,cpu_runner,vendors,runtime,a.device,check_all)
        check_all();report['status']=STATUS;r.write_json(output/'report.json',report)
        for name,row in report['comparison']['decision']['cases'].items():print(name+' 128/256线程：'+json.dumps(row,ensure_ascii=False),flush=True)
        for name,row in report['comparison'].get('cpu_comparison',{}).get('cases',{}).items():
            print(name+' CPU32/256线程：'+json.dumps(row,ensure_ascii=False),flush=True)
        print('结论：'+('256线程达到保留门槛；CPU同期结果另列，不自动替换正式128线程安装。'
            if report['comparison']['decision']['keep_fused_candidate'] else '256线程未稳定改善，保留已验收128线程版，不追加线程扫参。'),flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));r.write_json(output/'report.json',report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    defaults={'input':'results/search_256k_q8_input1','project':'results/Bm25FusedScoreTopKUbFix1',
        'vendors':'results/custom_opp/vendors','merge-project':'results/Bm25TopKMerge',
        'runner':'results/fused_search_host_build/bm25_fused_search_benchmark',
        'diagnostic-runner':'results/fused_chain_host_build/bm25_fused_chain_runner',
        'cpu-runner':'results/search_benchmark_host_build/bm25_cpu_search_benchmark',
        'acceptance':'results/fused_search_256k_q8_close1','baseline-resources':'results/fused_kernel_resources3'}
    for name,value in defaults.items():p.add_argument('--'+name,type=Path,default=ROOT/value)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=5)
    a=p.parse_args()
    try:r.require(sys.platform=='linux','请在已有CANN的A5 Linux运行');run(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；首错停止，不继续测速或生成CPU替代输出。',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
