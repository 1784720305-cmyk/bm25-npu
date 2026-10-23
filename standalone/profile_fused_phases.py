"""Isolated retained256 SIMT clock diagnosis: score versus local selection, no optimization or Search benchmark."""
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
from standalone import profile_fused_t256 as profile
from standalone import fused_warp_hist as previous_trial

t=profile.t
close=t.close
fused=t.fused
r=t.r
digest=t.digest
VENDOR='bm25_fused_score_topk_phase'
STATUS='retained_t256_instrumented_phase_cycles_checked'
CLOCK_SHA='80bf6faab5c6ef2428617336fb0a121873ae12cbd90a39c1edb39946c159f8fa'
CLOCK_REFERENCE='https://asc.gitcode.com/api/Utils-API/tuning_interface/clock.html'
HOST=('npu_prototype/fused_phase_host/CMakeLists.txt','npu_prototype/fused_phase_host/phase_runner.cpp',
    'npu_prototype/fused_chain_host/fused_runtime.h')
TOOLS=tuple(sorted(set(profile.TOOLS)|set(HOST)|{'standalone/profile_fused_phases.py',
    'standalone/fused_warp_hist.py','standalone/fused_resource_report.py'}))


def diagnostic_sources(original):
    """Keep all scoring/selection arithmetic and barriers; only debug2 exports four uint64 stamps/task."""
    source=t.candidate_sources(original)
    name='op_kernel/bm25_fused_score_top_k.cpp';kernel=source[name].decode('utf-8')
    changes=[
        ('#include "simt_api/device_sync_functions.h"',
         '#include "simt_api/device_sync_functions.h"\n#include "utils/debug/asc_time.h"'),
        ('    auto *bounds = reinterpret_cast<__ubuf__ int64_t *>(control+8);',
         '    auto *bounds = reinterpret_cast<__ubuf__ int64_t *>(control+8);\n'
         '    // Diagnostic only: control[2..7] is disjoint from flags[0..1], bounds[8..11], selector.\n'
         '    auto *phase = reinterpret_cast<__ubuf__ uint64_t *>(control+2);'),
        ('        const uint32_t last = first+width;\n        if (thread == 0) {',
         '        const uint32_t last = first+width;\n'
         '        if (debug == 2 && thread == 0) {\n'
         '            phase[0] = __asc_simt_vf::clock();\n'
         '            phase[1] = __asc_simt_vf::clock();\n'
         '        }\n        if (thread == 0) {'),
        ('        const bool bad = control[0] != 0;\n        asc_syncthreads();',
         '        const bool bad = control[0] != 0;\n'
         '        if (debug == 2 && thread == 0) phase[2] = __asc_simt_vf::clock();\n'
         '        asc_syncthreads();'),
        ('        if (debug != 0) {\n            const uint64_t base =',
         '        if (debug == 1) {\n            const uint64_t base ='),
        ('        asc_syncthreads(); // grid-stride next task cannot overwrite scores while selection is reading',
         '        asc_syncthreads(); // grid-stride next task cannot overwrite scores while selection is reading\n'
         '        if (debug == 2 && thread == 0) {\n'
         '            const uint64_t finished = __asc_simt_vf::clock();\n'
         '            auto *trace = reinterpret_cast<__gm__ uint64_t *>(debugScores);\n'
         '            trace[task*4] = phase[0]; trace[task*4+1] = phase[1];\n'
         '            trace[task*4+2] = phase[2]; trace[task*4+3] = finished;\n'
         '        }')]
    for before,after in changes:kernel=t.replace_once(kernel,before,after)
    # Removal of precisely the inserted diagnostics must restore every retained kernel byte.
    restored=kernel
    for before,after in reversed(changes):restored=t.replace_once(restored,after,before)
    r.require(restored.encode()==source[name],'计时副本改变了评分/选择算法')
    source[name]=kernel.encode()
    name='op_host/bm25_fused_score_top_k.cpp';host=source[name].decode()
    for before,after in (
        ('(*debug != 0 && *debug != 1)', '(*debug != 0 && *debug != 1 && *debug != 2)'),
        ('    dims = {static_cast<uint32_t>(queries),',
         '    if (*debug == 2 && queries > kLimit/((*shards)*8)) return false;\n'
         '    dims = {static_cast<uint32_t>(queries),'),
        ('    if (dims.debug != 0) return shape.GetDimNum() == 2',
         '    if (dims.debug == 2) return shape.GetDimNum() == 1 &&\n'
         '        shape.GetDim(0) == static_cast<int64_t>(dims.queries)*dims.shards*8;\n'
         '    if (dims.debug != 0) return shape.GetDimNum() == 2'),
        ('        else if (dims.debug != 0) *output = gert::Shape({dims.queries, dims.docs});',
         '        else if (dims.debug == 2) *output = gert::Shape({static_cast<int64_t>(dims.queries)*dims.shards*8});\n'
         '        else if (dims.debug != 0) *output = gert::Shape({dims.queries, dims.docs});')):
        host=t.replace_once(host,before,after)
    source[name]=host.encode()
    return source


def clock_header(toolkit):
    paths={p.resolve() for p in toolkit.rglob('asc_time.h') if p.is_file()}
    matches=[p for p in paths if digest(p)==CLOCK_SHA]
    r.require(len(matches)==1,'实际asc_time.h与已收到的950声明/指纹不同，停止，不换用猜测接口')
    path=matches[0];text=path.read_text(encoding='utf-8')
    r.require(re.search(r'namespace\s+__asc_simt_vf\s*\{[^}]*inline\s+uint64_t\s+clock\s*\(void\)',text,re.S)
        is not None and 'asc_time_simt_impl.h' in text,'实际头文件缺少已核对SIMT clock声明')
    impl={p.resolve() for p in toolkit.rglob('asc_time_simt_impl.h') if p.is_file()}
    r.require(impl and len(impl)<=8,'找不到SIMT clock实现头文件，或来源不明确')
    result={'actual_header':str(path),'actual_header_sha256':digest(path),
        'implementation_headers':{str(p):digest(p) for p in sorted(impl)},
        'api':'__asc_simt_vf::clock(void) -> uint64_t','units':'cycles',
        'reference':CLOCK_REFERENCE,'evidence':'Actual yellow-zone 9.2.0 header supplied by user, not inferred from master docs.'}
    print('实际950 SIMT clock声明/头文件SHA256通过；只记录周期，不猜频率。',flush=True)
    return result


def prepare(original,target):
    source=diagnostic_sources(original);metadata=t.prepare(original,target)
    for name,raw in source.items():(target/name).write_bytes(raw)
    file=target/'CMakePresets.json';presets=json.loads(file.read_text(encoding='utf-8'))
    changed=0
    for item in presets.get('configurePresets',[]):
        variables=item.get('cacheVariables',{});value=variables.get('vendor_name')
        if isinstance(value,dict) and value.get('value')==t.VENDOR:value['value']=VENDOR;changed+=1
        elif value==t.VENDOR:variables['vendor_name']=VENDOR;changed+=1
    r.require(changed>0,'副本没有已核对的vendor配置');r.write_json(file,presets)
    metadata.update(status='phase_diagnostic_prepared_not_compiled_or_device_verified',vendor_name=VENDOR,
        source_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in source.items()},
        clock_header_sha256=CLOCK_SHA,debug2_output='FLOAT 1D Q*S*8 elements, four raw uint64 timestamps/task',
        scope='Isolated instrumentation only. Retained256 FP32/order/selector/barriers/50848byte UB/64KiB declaration '
        'unchanged; debug2 uses previously unused control[2..7] for three stamps and an independently sized output. '
        'Extra clock/branches/UB and post-task GM stores can change registers, scheduling and cache. Not Search performance.')
    r.write_json(target/'bm25_project.json',metadata);return metadata


def installation(vendor,project,original):
    record=json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    r.require(vendor.resolve(strict=True).name==VENDOR and record.get('vendor_name')==VENDOR and
        record.get('simt_threads')==256 and record.get('explicit_ub_bytes')==50848 and
        record.get('dynamic_ub_bytes')==65536,'计时副本vendor/线程/UB不符')
    for name,raw in diagnostic_sources(original).items():
        r.require((project/name).read_bytes()==raw and digest(project/name)==record['source_sha256'][name],
            '计时副本源码或指纹改变：'+name)
    header=vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name,parameters in fused.PROTOTYPES.items():
        matches=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,re.S)
        r.require(len(matches)==1 and re.sub(r'\s+','',matches[0])==re.sub(r'\s+','',parameters),
            '实际生成ABI改变，不编译调用器：'+name)
    objects=fused.chain.artifacts(vendor,project,'bm25_fused_score_top_k',header.name)
    return {'vendor':str(vendor.resolve()),'metadata_sha256':digest(project/'bm25_project.json'),
        'source_sha256':record['source_sha256'],'header_sha256':digest(header),
        'api_sha256':digest(vendor/'op_api/lib/libcust_opapi.so'),'device_objects':objects}


def build_install(a):
    target=a.output/'project';metadata=prepare(a.project,target)
    logs=a.output/'compiler_logs';logs.mkdir();env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(logs)
    print('编译独立计时副本；正式256线程vendor及程序不修改。日志：'+str(a.output/'resource_build.log'),flush=True)
    with (a.output/'resource_build.log').open('wb') as stream:
        child=subprocess.run(['bash','build.sh'],cwd=target,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    r.require(child.returncode==0,'计时副本编译失败；查看 '+str(a.output/'resource_build.log'))
    old=t.resource.cache(a.project/'build_out/CMakeCache.txt');new=t.resource.cache(target/'build_out/CMakeCache.txt')
    r.require(old and new.get('vendor_name')==VENDOR and {k:v for k,v in old.items() if k!='vendor_name'}==
        {k:v for k,v in new.items() if k!='vendor_name'},'诊断编译等级/目标不同')
    logs,resources=t.resource.compiler_reports(a.output)
    print('计时副本VF资源：'+json.dumps(resources,ensure_ascii=False)+'；缺失保持UNKNOWN。',flush=True)
    packages=list((target/'build_out').glob('*.run'));r.require(len(packages)==1,'诊断安装包不唯一')
    listing=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (a.output/'package_list.log').write_bytes(listing.stdout)
    names=set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listing.stdout.decode('utf-8',errors='replace')))
    r.require(listing.returncode==0 and names=={VENDOR},'安装包不只含独立诊断vendor')
    with (a.output/'install.log').open('wb') as stream:
        child=subprocess.run(['bash',str(packages[0]),'--install-path='+str(a.vendors.parent)],
            stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    r.require(child.returncode==0,'独立诊断安装失败；查看 '+str(a.output/'install.log'))
    actual=installation(a.vendors/VENDOR,target,a.project)
    print('实际生成的两阶段aclnn接口已读取，与已核对ABI相同；随后编译独立诊断调用器。',flush=True)
    return {'metadata':metadata,'compiler_log_records':logs,'vf_reports':resources,'installation':actual}


def summarize(values):
    r.require(values and all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in values),'周期样本非法')
    ordered=sorted(values)
    return {'median':statistics.median(values),'p95':ordered[math.ceil(len(ordered)*.95)-1],
        'min':ordered[0],'max':ordered[-1],'count':len(values)}


def trace_rows(raw,tasks=256):
    r.require(len(raw)==tasks*32,'SIMT周期记录尺寸不完整')
    ticks=array('Q');ticks.frombytes(raw);r.require(ticks.itemsize==8,'需要64位无符号数组')
    if sys.byteorder!='little':ticks.byteswap()
    result=[]
    for task in range(tasks):
        t0,t1,t2,t3=ticks[task*4:task*4+4]
        r.require(t0!=(1<<64)-1 and t0<=t1<t2<t3 and t3!=(1<<64)-1,'周期缺失/回绕/次序非法：task='+str(task))
        score=t2-t1;selection=t3-t2
        result.append({'task':task,'query':task//32,'shard':task%32,'clock_read_pair_cycles':t1-t0,
            'score_cycles':score,'selection_cycles':selection,'score_and_selection_cycles':score+selection,
            'selection_fraction':selection/(score+selection)})
    return result


def analyze(actual,session,index,records,cache,device):
    wanted={'status':'phase_cycles_outputs_and_guards_checked','device':device,'queries':8,'docs':262144,'k':10,
        'shards':32,'warmup':5,'repeats':10,'index_upload_batches':1,'index_tensor_h2d_calls':5,
        'index_upload_bytes':index['index_bytes'],'index_readback_byte_identical':True,'debug1_calls':2,
        'clock_mode_calls':32,'saved_trace_calls':20,'every_clock_call_rank_bits_match_debug1':True,
        'all_output_guards_passed':True,'full_score_guard_check_policy':'preflight_and_after_all_clock_calls',
        'measured_calls_full_score_poison_or_download':False,'host_source_sha256':close.checks.sources(HOST)}
    r.require(set(session)==set(wanted) and all(type(session[key]) is type(value) and session[key]==value
        for key,value in wanted.items()),'诊断调用次数/源码/索引/设备/保护区记录不符')
    r.require((index['num_docs'],records['A']['num_queries'],records['B']['num_queries'],records['A']['k'],
        records['B']['k'])==(262144,8,8,10,10),'只分析当前固定负载')
    expected={'A','B','device_session.json'}|{f'cycle{i}_{name}.bin' for i in range(10) for name in ('A','B')}
    r.require({p.name for p in actual.iterdir()}==expected,'诊断输出文件集合不完整或混入其他运行')
    correctness={};samples=[];cases={}
    for name in ('A','B'):
        r.require({p.name for p in (actual/name).iterdir()}==set(fused.chain.FILES),'独立公式检查点文件不完整')
        correctness[name]=close.checks.check_output(actual/name,cache[name],'npu',32)
        one=[]
        for cycle in range(10):
            rows=trace_rows((actual/f'cycle{cycle}_{name}.bin').read_bytes())
            score=sum(x['score_cycles'] for x in rows);selection=sum(x['selection_cycles'] for x in rows)
            # Shares are computed within each SAME call, before taking medians.
            sample={'case':name,'cycle':cycle,'score_sum_cycles':score,'selection_sum_cycles':selection,
                'selection_work_fraction':selection/(score+selection),'tasks':rows}
            samples.append(sample);one.append(sample)
        cases[name]={
            'per_task_score_cycles':summarize([x['score_cycles'] for s in one for x in s['tasks']]),
            'per_task_selection_cycles':summarize([x['selection_cycles'] for s in one for x in s['tasks']]),
            'clock_read_pair_cycles':summarize([x['clock_read_pair_cycles'] for s in one for x in s['tasks']]),
            'per_call_selection_work_fraction':summarize([s['selection_work_fraction'] for s in one]),
            'per_call_score_sum_cycles':[s['score_sum_cycles'] for s in one],
            'per_call_selection_sum_cycles':[s['selection_sum_cycles'] for s in one],
            'per_call_selection_work_fractions':[s['selection_work_fraction'] for s in one]}
    return {'status':STATUS,'correctness':correctness,'cases':cases,'samples':samples,
        'scope':'Instrumented thread0 per-task elapsed cycles only, with original block barriers. Scoring includes '
        'score zeroing, query/index bounds, posting reads, FP32 accumulation and score-side barriers. Selection '
        'includes the existing pre-selection reader barrier, exact SelectLocal and final block barrier. '
        'Read-pair delta includes its UB store/scheduling, is NOT pure clock-call overhead and is NOT subtracted. '
        'Full-score poison/download only in preflight and final guard check, outside warmup/recording; small '
        'candidate/trace poison and all-rank byte checks still occur between calls and may affect cache state. '
        'Per-call shares sum parallel task-cycle work, NOT device critical-path duration/utilization; never convert '
        'cycles to us without verified clock semantics/frequency, or multiply this share by old139us. '
        'No claim of instruction-exclusive timing, original binary performance, complete Search or speedup.'}


def run(a):
    r.require(not a.output.exists() and a.output.is_relative_to((ROOT/'results').resolve()) and all(
        not a.output.is_relative_to(p) and not p.is_relative_to(a.output) for p in
        (a.input,a.previous,a.completed,a.project,a.vendors,a.acceptance,a.profile)),
        '需要results下独立新目录，不覆盖/嵌套旧结果')
    r.require(not (a.vendors/VENDOR).exists(),'计时副本vendor已存在，不覆盖')
    base=profile.evidence(a);index,records,snapshot,old128,baseline,runtime,proofs,resources=base
    r.require((index['num_docs'],records['A']['num_queries'],records['B']['num_queries'],records['A']['k'],
        records['B']['k'])==(262144,8,8,10,10),'本轮仅诊断当前256K/Q8/K10/S32')
    prof_proofs,cases=previous_trial.profile_evidence(a.profile,a.input,a.runner,baseline)
    header=clock_header(a.toolkit);frozen=close.checks.sources(TOOLS);p_hashes=close.checks.sources(profile.TOOLS)
    a.output.mkdir(parents=True)
    report={'status':'running','device':a.device,'input_sha256':snapshot,'proof_sha256':proofs,
        'source_sha256':frozen,'clock':header,'retained256_installation':baseline,'retained256_vf_reports':resources,
        'reused_coarse_profile_cases':cases,'scope':'One isolated phase diagnostic, not an optimization. '
        'No CPU run, ordinary Search benchmark, old acceptance replay, msprof run or environment changes.'}
    r.write_json(a.output/'report.json',report)
    try:
        def unchanged():
            profile.unchanged(a,base,p_hashes);close.scale.unchanged(prof_proofs,'既有采集证据改变')
            r.require(close.checks.sources(TOOLS)==frozen,'诊断冻结源码改变')
            close.scale.unchanged({header['actual_header']:header['actual_header_sha256'],**header['implementation_headers']},
                '实际SIMT clock头文件改变')
        unchanged();cache=close.checks.cache_inputs(a.input,index,records)
        report['diagnostic']=build_install(a);unchanged();r.write_json(a.output/'report.json',report)
        vendors=a.output/'runtime/vendors';vendors.mkdir(parents=True)
        for alias,value in (('bm25_fused_score_topk',a.vendors/VENDOR),('bm25_topk_merge',a.vendors/'bm25_topk_merge')):
            (vendors/alias).symlink_to(value,target_is_directory=True)
        build=a.output/'host_build'
        for command,name in ((['cmake','-S',str(ROOT/'npu_prototype/fused_phase_host'),'-B',str(build),
            '-DTOOLKIT_HOME='+str(a.toolkit),'-DBM25_VENDOR_DIR='+str(vendors)],'host_configure.log'),
            (['cmake','--build',str(build),'-j','2'],'host_build.log')):
            fused.score.execute(command,a.output/name,os.environ.copy())
        runner=build/'bm25_fused_phase_runner'
        binary=runner.read_bytes()
        r.require(all(sha.encode() in binary for sha in close.checks.sources(HOST).values()),'诊断调用器缺少实际源码指纹')
        plan=a.output/'requests.tsv';lines=[]
        for name in ('A','B'):
            path=str(a.input/name);r.require(not any(c in path for c in '\t\r\n'),'输入路径含控制字符')
            lines.append(name+'\t'+path)
        plan.write_bytes(('\n'.join(lines)+'\n').encode())
        command=[str(runner),str(a.input/'index'),str(plan),str(a.output/'actual'),str(a.device),str(vendors)]
        env=os.environ.copy();env['ASCEND_CUSTOM_OPP_PATH']=':'.join(profile.accepted.opp_paths(vendors))
        report.update(command=command,runner_sha256=digest(runner),plan_sha256=digest(plan),
            ascend_custom_opp_paths=profile.accepted.opp_paths(vendors));r.write_json(a.output/'report.json',report)
        print('当前A/B各一次完整分数/严格排名；clock模式每次与之逐字节核对，预热5轮、记录10轮。',flush=True)
        fused.score.execute(command,a.output/'device.log',env)
        unchanged();r.require(digest(runner)==report['runner_sha256'] and digest(plan)==report['plan_sha256'],
            '诊断调用器或请求计划改变')
        r.require(installation(a.vendors/VENDOR,a.output/'project',a.project)==report['diagnostic']['installation'],
            '诊断实际安装/源码改变')
        actual=a.output/'actual';session=json.loads((actual/'device_session.json').read_text(encoding='utf-8'))
        result=analyze(actual,session,index,records,cache,a.device)
        r.write_json(a.output/'phase_analysis.json',result)
        report.update(status=STATUS,analysis_sha256=digest(a.output/'phase_analysis.json'),
            output_sha256={p.relative_to(actual).as_posix():digest(p) for p in actual.rglob('*') if p.is_file()},
            log_sha256=digest(a.output/'device.log'))
        r.write_json(a.output/'report.json',report)
        for name,item in result['cases'].items():
            print(name+' 分片周期：评分中位数={:.6g}、p95={:.6g}；局部选K中位数={:.6g}、p95={:.6g}。'.format(
                item['per_task_score_cycles']['median'],item['per_task_score_cycles']['p95'],
                item['per_task_selection_cycles']['median'],item['per_task_selection_cycles']['p95']),flush=True)
            print('  各轮局部选K任务周期份额：'+str([round(x*100,2) for x in item['per_call_selection_work_fractions']])+ '%',flush=True)
            print('  连续读钟间隔（含UB存储/调度，未扣除）：'+str(item['clock_read_pair_cycles']),flush=True)
        print('当前负载诊断与严格排名通过；正式256线程版保持不变。以上是带计时副本的分片周期，'
            '不能换算旧139us或完整Search加速比；请结合本次寄存器/栈变化判断扰动。文件留在黄区，只需终端摘要。',flush=True)
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
    p.add_argument('--toolkit',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=5)
    a=p.parse_args()
    try:
        r.require(sys.platform=='linux' and r.integer(a.device,0,r.LIMIT),'需要A5 Linux和合法设备号')
        for name,value in vars(a).items():
            if isinstance(value,Path):setattr(a,name,value.resolve())
        run(a)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；首错停止，不继续测速、不回退CPU。',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
