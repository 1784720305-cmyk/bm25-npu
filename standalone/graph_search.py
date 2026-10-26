"""One A5 closure: fixed async/graph plans + index directory; exact checks and honest CPU gate."""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import shard_directory_sources as source, search_benchmark_checks as checks
from standalone import search_scale_fixture as fixture, resident_index as r, run_search as search
from standalone import benchmark_search as oldbench

simd=source.simd
digest=simd.digest
HOST_SOURCES=(
 'common/fixed_query_packet.h','common/shard_directory.h',
 'cpu_reference/directory_search/cpu_directory_search.h','cpu_reference/index_search/cpu_index_search.h',
 'cpu_reference/topk_document_shards.h','npu_prototype/search_host/search_inputs.h',
 'npu_prototype/fused_chain_host/fused_runtime.h','npu_prototype/fused_search_host/fused_api.h',
 'npu_prototype/search_benchmark_host/benchmark_common.h',
 'npu_prototype/graph_search_host/CMakeLists.txt','npu_prototype/graph_search_host/planned_search.h',
 'npu_prototype/graph_search_host/planned_benchmark.cpp','npu_prototype/graph_search_host/plan_check.cpp',
 'npu_prototype/graph_search_host/portable_check.cpp','npu_prototype/graph_search_host/cpu_directory_benchmark.cpp')
TOOLS=('standalone/graph_search.py','standalone/shard_directory_sources.py')
PASS='actual_graph_directory_full_search_faster_than_cpu'
MISS='actual_graph_directory_correct_but_cpu_target_not_met'


def execute(command,log,env=None,cwd=None,timeout=900):
    print('执行：'+json.dumps([str(x) for x in command],ensure_ascii=False),flush=True)
    with log.open('wb') as out:
        child=subprocess.run([str(x) for x in command],cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=timeout)
    r.require(child.returncode==0,'阶段失败：'+str(log)+'；退出码='+str(child.returncode))


def aliases(path,fused,merge):
    path.mkdir(parents=True)
    for name,target in (('bm25_fused_score_topk',fused),('bm25_topk_merge',merge)):
        (path/name).symlink_to(target.resolve(strict=True),target_is_directory=True)
    return path


def environment(vendors):
    env=os.environ.copy()
    env['ASCEND_CUSTOM_OPP_PATH']=':'.join(str((vendors/name).resolve(strict=True)) for name in ('bm25_fused_score_topk','bm25_topk_merge'))
    return env


def install_directory(a):
    original=a.original; target=a.output/'directory_project'; output=a.output/'directory_build'; output.mkdir()
    vendor=a.vendors/source.VENDOR
    r.require(not vendor.exists(),'独立目录 vendor 已存在，停止防止覆盖：'+str(vendor))
    metadata=source.prepare(original,target)
    env=os.environ.copy(); logs=output/'compiler_logs'; logs.mkdir(); env['ASCENDC_BUILD_LOG_DIR']=str(logs)
    execute(['bash','build.sh'],output/'build.log',env,target)
    packages=list((target/'build_out').glob('*.run')); r.require(len(packages)==1,'目录算子安装包不唯一')
    listing=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (output/'package_list.log').write_bytes(listing.stdout)
    names=set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listing.stdout.decode(errors='replace')))
    r.require(listing.returncode==0 and names=={source.VENDOR},'目录安装包 vendor 不符')
    execute(['bash',packages[0],'--install-path='+str(a.vendors.parent)],output/'install.log',timeout=300)
    header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name,parameters in simd.fused.PROTOTYPES.items():
        declarations=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,flags=re.S)
        r.require(len(declarations)==1 and re.sub(r'\s+','',declarations[0])==re.sub(r'\s+','',parameters),'实际目录 ABI 不符：'+name)
    for name,raw in source.candidate_sources(original).items():
        r.require((target/name).read_bytes()==raw,'目录源码发生变化：'+name)
    artifacts=simd.fused.chain.artifacts(vendor,target,'bm25_fused_score_top_k',header.name)
    return {'vendor':str(vendor),'metadata':metadata,'header_sha256':digest(header),
        'api_sha256':digest(vendor/'op_api/lib/libcust_opapi.so'),'device_objects':artifacts}


def edge_inputs(output):
    # Small independent data, built from document TF/DF rather than a device output.
    documents=[[d%7,(d*3)%11,2,2]+([4,4] if d%5==0 else []) for d in range(96)]
    index,values=r.build_index(documents,16)
    r.save_arrays(output/'index',index,values)
    cases={'A':[[2,4,2],[1,9]],'B':[[15],[3,3,5]],'empty':[[],[]],
           'unknown':[[99,99],[INT32_MAX_MINUS_ONE]],'words64':[[2]*64,[4]*64]}
    sequence=('A','B','A','empty','unknown','words64','empty','B')
    lines=[]
    for i,name in enumerate(sequence):
        path=output/f'call{i}_{name}'
        record,arrays=r.make_queries(index,cases[name],10)
        r.save_arrays(path,record,arrays)
        scores=r.diagnostic_scores(index,values,cases[name])
        (path/'expected_scores.bin').write_bytes(r.encode(array('f',scores)))
        lines.append(f'call{i}_{name}\t{path}\n')
    plan=output/'requests.tsv'; plan.write_bytes(''.join(lines).encode())
    return plan


INT32_MAX_MINUS_ONE=2**31-2


def summary_gate(rows):
    r.require(len(rows)==6,'必须完成预先固定的六轮')
    result={};passed=True
    for case in ('A','B'):
        cpu=[min(row['cpu']['summary'][case]['median_batch_ms'],row['cpu_directory']['summary'][case]['median_batch_ms']) for row in rows]
        npu=[row['graph_directory']['summary'][case]['median_batch_ms'] for row in rows]
        cp=[min(row['cpu']['summary'][case]['p95_batch_ms'],row['cpu_directory']['summary'][case]['p95_batch_ms']) for row in rows]
        np=[row['graph_directory']['summary'][case]['p95_batch_ms'] for row in rows]
        wins=sum(a<b for a,b in zip(npu,cpu))
        good=wins==6 and statistics.median(np)<statistics.median(cp)
        result[case]={'fastest_cpu_medians_ms':cpu,'graph_directory_medians_ms':npu,
            'fastest_cpu_p95_ms':cp,'graph_directory_p95_ms':np,'wins':wins,
            'cpu_over_npu':statistics.median(cpu)/statistics.median(npu),'passed':good,
            'engineering_55us_target_met':statistics.median(npu)<=.055}
        passed &= good
    return {'performance_target_met':passed,'cases':result}


def check_measurement(path,index,records,cache,role,threads,device,capwords):
    if role=='baseline':
        data,summary=simd.close.fc.check_session(path,'npu',index,records,32,threads,device,
            records['A']['num_queries'],capwords,5,30,None)
        validation,hashes=checks.check_checkpoints(path,'npu',cache,32)
        return {'summary':summary,'checks':validation,'output_sha256':hashes,'session':data}
    data=json.loads((path/'benchmark_session.json').read_text(encoding='utf-8'))
    q=records['A']['num_queries']; k=records['A']['k']; is_npu=role in ('async','graph','graph_directory')
    graph=role in ('graph','graph_directory'); directory=role in ('graph_directory','cpu_directory')
    directory_bytes=index['vocab_size']*33*8 if role=='graph_directory' else 0
    wanted={'status':'search_benchmark_calls_completed_checks_pending','backend':'npu' if is_npu else 'cpu',
        'docs':index['num_docs'],'queries':q,'k':k,'partitions':32,'capacity_queries':q,'capacity_words':capwords,
        'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory','timer_policy':checks.TIMER,
        'index_load_batches':1,'checkpoint_search_calls':4,'warmup_cycles':5,'warmup_search_calls':10,
        'measured_cycles':30,'measured_search_calls':60,'measured_topk_bit_checks':60,
        'measured_debug_download_calls':0,'measured_poison_calls':0,'measured_file_io_calls':0,
        'index_readback_byte_identical':True,'diagnostic_checkpoints_outside_measurement':True}
    if is_npu:
        wanted.update(graph_capture=graph,directory_index=directory,device=device,directory_bytes=directory_bytes,
            pipeline_stages=2,normal_full_score_elements=0,intermediate_host_roundtrips=0)
        expected_index_bytes=index['index_bytes']+directory_bytes-(index['vocab_size']+1)*8 if directory else index['index_bytes']
        wanted['index_bytes']=expected_index_bytes
        r.require(data.get('normal_prepare_calls')==(0 if graph else 2*data.get('normal_search_calls',-1)),
            'Graph 热路径仍准备执行器，或异步计数不符')
        r.require(data.get('replays')==(data.get('normal_search_calls') if graph else 0),'真实重放调用数不符')
    else:
        wanted.update(requested_threads=threads,actual_threads=threads,index_bytes=index['index_bytes'],
            fp32_flags='-O3 -march=native -ffp-contract=off -fno-fast-math')
    r.require(all(type(data.get(key)) is type(value) and data[key]==value for key,value in wanted.items()),'完整 Search 计时/尺寸/索引/路径记录不符：'+role)
    if role!='cpu': r.require(data.get('host_source_sha256')==checks.sources(HOST_SOURCES),'新二进制编译来源不符')
    else: r.require(data.get('host_source_sha256')==checks.sources(checks.CPU_SOURCES),'原 CPU 二进制不是当前已验收源码')
    samples=data.get('samples',[]);r.require(len(samples)==60,'样本数量不符')
    for pos,item in enumerate(samples):
        cycle,slot=divmod(pos,2); case='B' if slot^(cycle&1) else 'A'
        r.require(item.get('cycle')==cycle and item.get('case')==case and
            type(item.get('call_wall_ms')) in (int,float) and math.isfinite(item['call_wall_ms']) and item['call_wall_ms']>0 and
            type(item.get('internal_search_ms')) in (int,float) and math.isfinite(item['internal_search_ms']) and item['internal_search_ms']>0 and
            item['call_wall_ms']>=item['internal_search_ms'],'时序/完整边界非法')
        if is_npu:
            align=lambda n:(n+31)//32*32
            expected=(align(align((q+1)*8)+max(1,capwords)*4),align(align(q*k*4)+q*k*4))
        else: expected=(0,0)
        r.require((item.get('query_h2d_bytes'),item.get('topk_d2h_bytes'))==expected,'实际传输 arena 字节不符')
    validation,hashes=checks.check_checkpoints(path,'npu' if is_npu else 'cpu',cache,32)
    summary={name:checks.summarize([s['call_wall_ms'] for s in samples if s['case']==name],q) for name in ('A','B')}
    return {'summary':summary,'checks':validation,'output_sha256':hashes,'session':data}


def run(a):
    r.require(sys.platform=='linux','需在 A5 Linux/CANN 环境执行，本机不生成替代性能结果')
    r.require(not a.output.exists() and a.output.is_relative_to(ROOT/'results'),'需要 results 下独立新输出目录')
    r.require(a.device>=0 and a.threads>=32,'设备非法或 CPU 线程低于既有32线程对手')
    index,records,snapshot=fixture.validate_inputs(a.input)
    r.require((index['num_docs'],records['A']['num_queries'],records['A']['k'])==(262144,8,10),'首版固定原256K/Q8/K10工作负载')
    previous=json.loads((a.accepted/'report.json').read_text(encoding='utf-8'))
    r.require(previous.get('status')==simd.STATUS and previous.get('comparison',{}).get('decision',{}).get('keep_fused_candidate') is True,
        '需要已通过的 SIMD 闭环，不重跑旧算子验收')
    r.require(previous.get('input_sha256')==snapshot,'SIMD 验收输入不同')
    installed=simd.candidate_install(a.vendors/simd.VENDOR,a.accepted/'project',a.original)
    r.require(installed==previous['candidate']['installation'],'已验收 SIMD 安装或源码已改变')
    a.output.mkdir(parents=True)
    frozen=checks.sources(tuple(sorted(set(HOST_SOURCES+TOOLS+simd.OWN))))
    report={'status':'running','device':a.device,'input_sha256':snapshot,'host_and_tool_sha256':frozen,
        'accepted_report_sha256':digest(a.accepted/'report.json'),'accepted_installation':installed,
        'resources':oldbench.host_resources(),'per_round':[],'diagnostic_comparison':[],
        'scope':'New pinned fixed-capacity plans, real per-call query transfer and CPU-available results. No cached answer/CPU fallback. Performance target is six rounds faster than both CPU variants.'}
    def save():r.write_json(a.output/'report.json',report)
    def unchanged():
        r.require(checks.sources(tuple(frozen))==frozen,'执行期间冻结源码改变')
        simd.close.scale.unchanged(snapshot,'执行期间输入改变')
        r.require(simd.candidate_install(a.vendors/simd.VENDOR,a.accepted/'project',a.original)==installed,'已验收算子改变')
        if 'runtime_sha256' in report:
            simd.close.scale.unchanged(report['runtime_sha256'],'执行期间运行库/主机二进制改变')
    save()
    try:
        base=aliases(a.output/'baseline/vendors',a.vendors/simd.VENDOR,a.vendors/'bm25_topk_merge')
        build=a.output/'host_build'
        execute(['cmake','-S',ROOT/'npu_prototype/graph_search_host','-B',build,'-DTOOLKIT_HOME='+str(a.toolkit),
            '-DBM25_VENDOR_DIR='+str(base)],a.output/'configure.log')
        execute(['cmake','--build',build,'-j','4'],a.output/'host_build.log')
        execute([build/'bm25_plan_portable_check'],a.output/'portable_check.log')
        report['directory_installation']=install_directory(a);save()
        directory=aliases(a.output/'directory/vendors',a.vendors/source.VENDOR,a.vendors/'bm25_topk_merge')
        # Identical generated signatures must be checked in the actual target C++ compiler too.
        dirbuild=a.output/'directory_header_build'
        execute(['cmake','-S',ROOT/'npu_prototype/graph_search_host','-B',dirbuild,'-DTOOLKIT_HOME='+str(a.toolkit),
            '-DBM25_VENDOR_DIR='+str(directory)],a.output/'directory_configure.log')
        execute(['cmake','--build',dirbuild,'--target','bm25_plan_check','-j','4'],a.output/'directory_header_build.log')
        runtime_files=[build/name for name in ('bm25_planned_search_benchmark','bm25_plan_check',
            'bm25_directory_cpu_benchmark','bm25_plan_portable_check')]
        runtime_files += [dirbuild/'bm25_plan_check',a.cpu_runner,a.npu_runner]
        for vendor in (a.vendors/source.VENDOR,a.vendors/'bm25_topk_merge'):
            runtime_files += [p for p in vendor.rglob('*') if p.is_file() and p.suffix in ('.so','.o','.json','.h')]
        report['runtime_sha256']={str(p.resolve()):digest(p) for p in runtime_files}
        unchanged();save()
        edges=a.output/'edges';edges.mkdir();plan=edge_inputs(edges)
        report['device_checks']=[]
        for role,graph,isdirectory in (('async',0,0),('graph',1,0),('graph_directory',1,1)):
            vendors=directory if isdirectory else base; dest=a.output/('check_'+role)
            execute([dirbuild/'bm25_plan_check' if isdirectory else build/'bm25_plan_check',edges/'index',plan,dest,
                a.device,vendors,10,2,2,128,graph,isdirectory],a.output/('check_'+role+'.log'),environment(vendors))
            result=json.loads((dest/'report.json').read_text(encoding='utf-8'))
            r.require(result.get('status')=='actual_planned_chain_full_scores_rankings_guards_and_new_inputs_passed' and result.get('calls')==8,'设备边界验收报告不完整')
            report['device_checks'].append(result);save()
        cache=checks.cache_inputs(a.input,index,records)
        raw=a.output/'raw_queries';raw.mkdir()
        for name in ('A','B'):search.write_queries(raw/(name+'.txt'),cache[name]['queries'])
        capwords=max(x['query_term_count'] for x in records.values());q=records['A']['num_queries'];k=records['A']['k']
        def measure(path,role):
            unchanged();path.mkdir(parents=True); actual=path/'actual'
            cpu=role in ('cpu','cpu_directory');isdir=role=='graph_directory';vendors=directory if isdir else base
            runner=(a.cpu_runner if role=='cpu' else build/'bm25_directory_cpu_benchmark') if cpu else (
                a.npu_runner if role=='baseline' else build/'bm25_planned_search_benchmark')
            command=[runner,a.input/'index',raw/'A.txt',raw/'B.txt',actual]
            if cpu: command += [a.threads,k,32,q,capwords,5,30]
            else:
                command += [a.device,vendors,k,32,q,capwords,5,30,a.input/'A/expected_scores.bin',a.input/'B/expected_scores.bin']
                if role!='baseline':command += [int(role!='async'),int(isdir)]
            execute(command,path/'run.log',os.environ.copy() if cpu else environment(vendors))
            result=check_measurement(actual,index,records,cache,role,a.threads,a.device,capwords)
            result.update(command=[str(x) for x in command],runner_sha256=digest(runner),log_sha256=digest(path/'run.log'))
            r.write_json(path/'report.json',result);unchanged()
            print(role+' 完整Search：'+json.dumps(result['summary'],ensure_ascii=False),flush=True)
            return {'directory':str(path),'summary':result['summary'],'report_sha256':digest(path/'report.json')}
        # A single attribution comparison; it is not used to decide CPU success.
        for role in ('baseline','async','graph'):
            report['diagnostic_comparison'].append(measure(a.output/'attribution'/role,role));save()
        for i in range(6):
            roles=['cpu','graph_directory','cpu_directory'] if i%2==0 else ['cpu_directory','graph_directory','cpu']
            row={'round':i+1,'process_order':roles}
            for role in roles:row[role]=measure(a.output/f'round{i+1}'/role,role)
            report['per_round'].append(row);save()
        report['decision']=summary_gate(report['per_round']);report['status']=PASS if report['decision']['performance_target_met'] else MISS
        unchanged();save()
        print('最终结论：'+json.dumps(report['decision'],ensure_ascii=False),flush=True)
        print('性能验收通过。' if report['status']==PASS else '性能目标未达到；保留完整失败报告，不作为达标交付。',flush=True)
        return 0 if report['status']==PASS else 2
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    defaults={'input':'results/search_256k_q8_input1','accepted':'results/fused_simd_close2',
        'original':'results/Bm25FusedScoreTopKUbFix1','vendors':'results/custom_opp/vendors',
        'cpu-runner':'results/search_benchmark_host_build/bm25_cpu_search_benchmark',
        'npu-runner':'results/fused_search_host_build/bm25_fused_search_benchmark'}
    for name,value in defaults.items():p.add_argument('--'+name,type=Path,default=ROOT/value)
    p.add_argument('--toolkit',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=5);p.add_argument('--threads',type=int,default=32)
    a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    try:return run(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不回退CPU生成NPU结果。',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
