"""Grow article count at fixed Q8/three words/K10; frozen CPUs, bounded NPU tree, honest crossover gate."""
import argparse
from array import array
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import large_search_sources as source,large_corpus as corpus,fused_stage_profile as accepted
g=accepted.g;r=g.r;simd=g.simd;digest=g.digest
SIZES=(262144,1048576,4194304,10000000)
HOST=(
 'common/large_search_plan.h','common/fixed_query_packet.h','common/shard_directory.h',
 'cpu_reference/topk_document_shards.h','npu_prototype/search_host/search_inputs.h',
 'npu_prototype/fused_chain_host/fused_runtime.h','npu_prototype/fused_search_host/fused_api.h',
 'npu_prototype/graph_search_host/planned_search.h','npu_prototype/search_benchmark_host/benchmark_common.h',
 'npu_prototype/large_search_host/CMakeLists.txt','npu_prototype/large_search_host/benchmark.cpp',
 'npu_prototype/large_search_host/large_search.h','npu_prototype/large_search_host/merge_check.h')
OWN=('standalone/large_search.py','standalone/large_search_sources.py','standalone/large_corpus.py',
 'standalone/fused_stage_profile.py','standalone/register_select.py','standalone/register_select_sources.py',
 'standalone/shard_directory_sources.py','standalone/fused_simd_sources.py','standalone/fused_threads256.py',
 'npu_prototype/fused_simd_select/simd_selection.h','npu_prototype/fused_reg_select/register_threshold.h')


def layout(n,v,m,q=8,k=10,words=24):
    require=r.require;limit=2**31-1
    require(all(type(x) is int for x in (n,v,m,q,k,words)) and 1<=k<=32 and k<=n<=limit and
        1<=v<limit and 1<=m<=limit and 1<=q<=10000 and 0<=words<=16777216,'Invalid retained index/query limits')
    leaves=(n+8191)//8192;padded=leaves*8192
    require(leaves<=65536 and padded<=limit and q*padded<=limit,'Padded INT32 bounds exceeded')
    lists=[leaves]
    while True:
        lists.append((lists[-1]+31)//32)
        if lists[-1]==1:break
    align=lambda n:(n+31)//32*32
    inp=align(align((q+1)*8)+max(1,words)*4);out=align(align(q*k*4)+q*k*4)
    index=8*m+4*padded+4*v+8*v*(leaves+1);candidates=sum(q*s*k*8 for s in lists)
    return {'docs':n,'padded_docs':padded,'leaves':leaves,'list_counts':lists,'pipeline_stages':len(lists),
        'resident_index_bytes':index,'directory_bytes':8*v*(leaves+1),'candidate_bytes':candidates,
        'packet_input_bytes':inp,'packet_output_bytes':out,
        'known_device_bytes_upper_bound':index+candidates+inp+out+4+(8+2*len(lists))*128,
        'debug_score_bytes':q*padded*4+128,'cpu_peak_known_bytes':2*(8*m+4*n+4*v+8*(v+1))+8*q*n+8*v*33,
        'scope':'Device workspace/capture/runtime allocations excluded here; actual executable checks free memory and every actual workspace.'}


def gate(rows):
    r.require(len(rows)==6,'Exactly six predeclared rounds required')
    result={};passed=True
    for case in ('A','B'):
        med=[min(row[cpu]['summary'][case]['median_batch_ms'] for cpu in ('cpu','cpu_directory')) for row in rows]
        np=[row['npu']['summary'][case]['median_batch_ms'] for row in rows]
        cp=[min(row[cpu]['summary'][case]['p95_batch_ms'] for cpu in ('cpu','cpu_directory')) for row in rows]
        pp=[row['npu']['summary'][case]['p95_batch_ms'] for row in rows]
        wins=sum(a<b for a,b in zip(np,med));ok=wins==6 and statistics.median(pp)<statistics.median(cp)
        result[case]={'cpu_medians_ms':med,'npu_medians_ms':np,'cpu_p95_ms':cp,'npu_p95_ms':pp,'wins':wins,
            'cpu_over_npu':statistics.median(med)/statistics.median(np),'passed':ok};passed &= ok
    return {'performance_target_met_at_this_scale':passed,'cases':result}


def crossover(tiers):
    measured=[x for x in tiers if x.get('status')=='six_rounds_checked']
    wins=[x['docs'] for x in measured if x['docs']>=1000000 and x['decision']['performance_target_met_at_this_scale']]
    nonmonotonic=any(a['decision']['performance_target_met_at_this_scale'] and not b['decision']['performance_target_met_at_this_scale']
        for a,b in zip(measured,measured[1:]))
    first=min(wins) if wins else None
    earlier=[x['docs'] for x in measured if first and x['docs']<first and not x['decision']['performance_target_met_at_this_scale']]
    return {'large_scale_hardware_target_met':bool(wins),'first_tested_large_winning_docs':first,
        'tested_large_winning_docs':wins,'previous_tested_losing_docs':max(earlier) if earlier else None,
        'nonmonotonic_tested_results':nonmonotonic,
        'all_planned_scales_measured':len(measured)==len(tiers),
        'scope':'Only tested article counts, workload and frozen opponents. No interpolation, all-small-scale requirement or MT business-data acceptance.'}


def build_operator(a,kind,original):
    vendor=source.FUSED if kind=='fused' else source.MERGE
    r.require(not (a.vendors/vendor).exists(),'Independent large vendor exists; do not overwrite: '+vendor)
    output=a.output/(kind+'_build');output.mkdir();project=output/'project';metadata=source.prepare(original,project,kind)
    logs=output/'compiler_logs';logs.mkdir();env=os.environ.copy();env['ASCENDC_BUILD_LOG_DIR']=str(logs)
    try:g.execute(['bash','build.sh'],output/'build.log',env,project,timeout=1800)
    except Exception:print((output/'build.log').read_text(encoding='utf-8',errors='replace')[-24000:],flush=True);raise
    old=simd.t.resource.cache(original/'build_out/CMakeCache.txt');new=simd.t.resource.cache(project/'build_out/CMakeCache.txt')
    r.require(old and new.get('vendor_name')==vendor and {k:v for k,v in old.items() if k!='vendor_name'}==
        {k:v for k,v in new.items() if k!='vendor_name'},'Operator target/optimization settings changed')
    packages=list((project/'build_out').glob('*.run'));r.require(len(packages)==1,'Unique operator package required')
    listing=subprocess.run(['bash',str(packages[0]),'--list'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (output/'package_list.log').write_bytes(listing.stdout)
    r.require(listing.returncode==0 and set(re.findall(r'vendors/([A-Za-z0-9_]+)/',listing.stdout.decode(errors='replace')))=={vendor},'Not an isolated package')
    g.execute(['bash',packages[0],'--install-path='+str(a.vendors.parent)],output/'install.log',timeout=300)
    op='bm25_fused_score_top_k' if kind=='fused' else 'bm25_top_k_merge'
    header=a.vendors/vendor/'op_api/include'/('aclnn_'+op+'.h')
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    prototypes=simd.fused.PROTOTYPES if kind=='fused' else {
        'aclnnBm25TopKMergeGetWorkspaceSize':'constaclTensor*partialScores,constaclTensor*partialIds,int64_tk,int64_tdocCount,constaclTensor*topScoresOut,constaclTensor*topIdsOut,uint64_t*workspaceSize,aclOpExecutor**executor',
        'aclnnBm25TopKMerge':'void*workspace,uint64_tworkspaceSize,aclOpExecutor*executor,aclrtStreamstream'}
    for name,args in prototypes.items():
        found=re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,flags=re.S)
        r.require(len(found)==1 and re.sub(r'\s+','',found[0])==re.sub(r'\s+','',args),'Actual generated ABI changed: '+name)
    generated=source.fused_sources(original) if kind=='fused' else source.merge_sources(original)
    for name,raw in generated.items():r.require((project/name).read_bytes()==raw,'Generated source changed')
    objects=simd.fused.chain.artifacts(a.vendors/vendor,project,op,header.name)
    return {'vendor':vendor,'project':str(project),'metadata':metadata,'device_objects':objects}


def same_stream(a,b):
    if a.stat().st_size!=b.stat().st_size:return False
    with a.open('rb') as left,b.open('rb') as right:
        while True:
            chunk=left.read(8*1024*1024)
            if chunk!=right.read(len(chunk)):return False
            if not chunk:return True


def verify_run(path,inputs,description,index,role,device,warmup=5,repeats=30):
    data=json.loads((path/'benchmark_session.json').read_text(encoding='utf-8'))
    n=description['docs'];q=description['queries'];k=description['k'];npu=role=='npu';plan=layout(n,index['vocab_size'],index['posting_count'],q,k,q*3)
    wanted={'status':'search_benchmark_calls_completed_checks_pending','backend':'npu' if npu else 'cpu',
        'docs':n,'queries':q,'k':k,'partitions':plan['leaves'] if npu else 32,'capacity_queries':q,'capacity_words':q*3,
        'timer_policy':g.checks.TIMER,'warmup_cycles':warmup,'measured_cycles':repeats,'measured_search_calls':2*repeats,
        'measured_topk_bit_checks':2*repeats,'checkpoint_search_calls':4,'index_readback_byte_identical':True,
        'measured_file_io_calls':0,'measured_debug_download_calls':0,'measured_poison_calls':0,
        'diagnostic_checkpoints_outside_measurement':True,'index_load_batches':1}
    if npu:
        wanted.update(large_search_contract='bounded8192_global_ids_tree32_v1',device=device,padded_docs=plan['padded_docs'],
            leaf_docs=8192,merge_fan_in=32,list_counts=plan['list_counts'],pipeline_stages=plan['pipeline_stages'],
            graph_capture=True,normal_full_score_elements=0,intermediate_host_roundtrips=0,normal_prepare_calls=0,
            normal_search_calls=5+2*warmup+2*repeats,replays=5+2*warmup+2*repeats,directory_bytes=plan['directory_bytes'],index_bytes=plan['resident_index_bytes'],
            packet_input_bytes=plan['packet_input_bytes'],packet_output_bytes=plan['packet_output_bytes'],
            known_device_bytes_upper_bound=plan['known_device_bytes_upper_bound'])
    else:wanted.update(requested_threads=32,actual_threads=32,index_bytes=index['index_bytes'],
        fp32_flags='-O3 -march=native -ffp-contract=off -fno-fast-math')
    r.require(all(data.get(key)==value and type(data.get(key)) is type(value) for key,value in wanted.items()),'Measured session contract mismatch: '+role)
    expected_sources=g.checks.sources(HOST if npu else g.checks.CPU_SOURCES if role=='cpu' else g.HOST_SOURCES)
    r.require(data.get('host_source_sha256')==expected_sources,'Compiled host source mismatch')
    samples=data['samples'];r.require(len(samples)==2*repeats,'Incomplete measured samples')
    for pos,item in enumerate(samples):
        cycle,slot=divmod(pos,2);case='B' if slot^(cycle&1) else 'A'
        r.require(item['case']==case and item['cycle']==cycle and math.isfinite(item['call_wall_ms']) and
            math.isfinite(item['internal_search_ms']) and 0<item['internal_search_ms']<=item['call_wall_ms'],'Sample order/boundary invalid')
        expected=(plan['packet_input_bytes'],plan['packet_output_bytes']) if npu else (0,0)
        r.require((item['query_h2d_bytes'],item['topk_d2h_bytes'])==expected,'Transfer boundary changed')
    output_hash={}
    for checkpoint in g.checks.CHECKPOINTS:
        case=checkpoint[-1];actual=path/checkpoint;gold=inputs/case
        r.require((actual/'top_ids.bin').read_bytes()==(gold/'expected_top_ids.bin').read_bytes(),'Independent strict final ranking differs')
        scores=array('f');scores.frombytes((actual/'top_scores.bin').read_bytes());ref=array('f');ref.frombytes((gold/'expected_top_scores.bin').read_bytes())
        r.require(len(scores)==q*k and all(math.isfinite(v) and v>=0 for v in scores),'Invalid TopK scores')
        if npu:
            r.compare(scores,ref)
            proof=json.loads((actual/'tree_check.json').read_text(encoding='utf-8'))
            r.require(proof['status']=='all_scores_all_leaf_lists_all_merge_levels_and_strict_gold_ids_passed' and proof['levels']==plan['pipeline_stages'],
                'Full scores/all tree levels not checked')
            r.require((actual/'scores.bin').stat().st_size==n*q*4,'Trimmed real-document diagnostic size differs')
            for level,count in enumerate(plan['list_counts']):
                for suffix in ('scores','ids'):
                    r.require((actual/f'level_{level}_{suffix}.bin').stat().st_size==q*count*k*4,'Retained tree-level output size differs')
        else:
            r.require(same_stream(actual/'scores.bin',gold/'expected_scores.bin') and
                (actual/'top_scores.bin').read_bytes()==(gold/'expected_top_scores.bin').read_bytes(),'CPU ordered FP32 formula/selected bits differ')
        for file in actual.iterdir():
            if file.is_file():output_hash[str(file.relative_to(path))]=corpus.digest(file)
    for case in ('A','B'):
        r.require(same_stream(path/('before_'+case)/'scores.bin',path/('after_'+case)/'scores.bin'),
            'Full score bits changed after intervening queries')
    return {'session':data,'benchmark_session_sha256':corpus.digest(path/'benchmark_session.json'),
        'output_sha256':output_hash,'summary':{case:g.checks.summarize([x['call_wall_ms'] for x in samples if x['case']==case],q) for case in ('A','B')}}


def host_available_bytes():
    text=Path('/proc/meminfo').read_text();match=re.search(r'^MemAvailable:\s+(\d+) kB$',text,re.M)
    r.require(match is not None,'Actual host MemAvailable unavailable');return int(match[1])*1024


def run(a):
    r.require(sys.platform=='linux' and not a.output.exists() and a.output.is_relative_to(ROOT/'results'),'A5 Linux and fresh results directory required')
    old,one,unused,proofs=accepted.evidence(a.accepted)
    r.require(old['candidate']['metadata']['source_sha256']=={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in source.base.candidate_sources(a.original).items()},'Accepted scoring/selection source changed')
    cpus={}
    for role in ('cpu','cpu_directory'):
        linked=old['per_round'][0][role];p=Path(linked['directory'])/'report.json'
        r.require(digest(p)==linked['report_sha256'],'Frozen CPU report changed');item=json.loads(p.read_text(encoding='utf-8'))
        r.require(digest(Path(item['command'][0]))==item['runner_sha256'],'Frozen CPU binary changed')
        cpus[role]=item['command'][0];proofs[str(p)]=digest(p)
    for name,raw in source.merge_sources(a.merge_original).items():
        # Generator derives from the actual installed original template, checked again at build.
        r.require(bool(raw),'Missing merge source')
    anchor=json.loads((a.anchor/'source/metadata.json').read_text(encoding='utf-8'))
    r.require(anchor['dataset']['case']=='performance_uniform_short_v1' and anchor['dataset']['stress_queries_injected'] is False,'Original synthetic family unconfirmed')
    seed=anchor['dataset']['seed'];vocab=anchor['dataset']['vocab_size']
    raw={'A':Path(one['command'][2]),'B':Path(one['command'][3])};cases={name:corpus.load_raw(path) for name,path in raw.items()}
    r.require(all(len(rows)==8 and all(len(row)==3 for row in rows) for rows in cases.values()),'FixedQ8/three-word original queries required')
    a.output.mkdir(parents=True);own=g.checks.sources(OWN+HOST)
    report={'status':'running','target':'Stable NPU advantage at tested large article counts; small-scale losses allowed.',
        'selector_decision':'Keep verified register selector as leaf baseline; defer further threshold tuning and old phase diagnostic.',
        'fixed_query_cases':cases,'source_sha256':own,'previous_sha256':proofs,'device':old['device'],'tiers':[],
        'dataset_scope':'User confirmed no MT-approved corpus; synthetic family by default, optional supplied JSONL explicitly identified without business generalization.',
        'cpu_policy':'Reuse both accepted32thread/32shard optimized binaries unchanged; faster opponent per round. No CPU implementation rebuild or optimization.'}
    def save():r.write_json(a.output/'report.json',report)
    def check():
        simd.close.scale.unchanged(proofs,'Accepted evidence/runtime changed')
        r.require(g.checks.sources(OWN+HOST)==own,'Large-search sources changed')
        if 'installation_sha256' in report:simd.close.scale.unchanged(report['installation_sha256'],'New installation/runner changed')
    save()
    try:
        cpuinfo=Path('/proc/cpuinfo').read_bytes();(a.output/'cpu_info.txt').write_bytes(cpuinfo)
        report['host_context']={'cpu_info_sha256':corpus.digest(a.output/'cpu_info.txt'),
            'logical_cpu_count':os.cpu_count(),'allowed_cpu_affinity':sorted(os.sched_getaffinity(0)),
            'openmp_environment':{key:value for key,value in os.environ.items() if key.startswith(('OMP_','GOMP_'))}}
        report['management_before']=accepted.prior.management_snapshot(a.output,'large_before');save()
        report['operators']={}
        for kind,path in (('fused',a.original),('merge',a.merge_original)):
            report['operators'][kind]=build_operator(a,kind,path);save();check()
        vendors=g.aliases(a.output/'runtime/vendors',a.vendors/source.FUSED,a.vendors/source.MERGE)
        build=a.output/'host_build'
        for cmd,log in ((['cmake','-S',ROOT/'npu_prototype/large_search_host','-B',build,
            '-DTOOLKIT_HOME='+str(a.toolkit),'-DBM25_VENDOR_DIR='+str(vendors)],'host_configure.log'),
            (['cmake','--build',build,'-j','4'],'host_build.log')):g.execute(cmd,a.output/log);check()
        runner=build/'bm25_large_search_benchmark'
        files=[runner]+[p for kind in (source.FUSED,source.MERGE) for p in (a.vendors/kind).rglob('*') if p.is_file() and p.suffix in ('.so','.o','.h','.json')]
        for kind,record in report['operators'].items():
            project=Path(record['project'])
            files += [project/name for name in record['metadata']['source_sha256']]
            files += [project/name for name in ('bm25_project.json','CMakePresets.json','op_kernel/CMakeLists.txt','build.sh')]
        report['installation_sha256']={str(p.resolve()):digest(p) for p in files};check();save()
        env=g.environment(vendors)
        g.execute([runner,'--memory',old['device'],a.output/'device_memory.json'],a.output/'memory.log',env)
        memory=json.loads((a.output/'device_memory.json').read_text(encoding='utf-8'));report['device_memory_probe']=memory;save()
        selector=a.accepted.parent/'fused_simd_close2/selector_host_build/bm25_fused_simd_selector_check'
        r.require(str(selector.resolve()) in old['candidate_sha256'] and digest(selector)==old['candidate_sha256'][str(selector.resolve())],'Retained raw-selector checker provenance missing')
        g.execute([selector,vendors,old['device'],a.output/'selector_checks'],a.output/'selector.log',env)
        selector_report=json.loads((a.output/'selector_checks/report.json').read_text(encoding='utf-8'))
        r.require(selector_report['status']=='actual_simd_selector_guards_and_strict_bit_rankings_passed' and selector_report['calls']==54,'Affected global-ID leaf selector failed')
        g.execute([runner,'--merge-check',vendors,old['device'],a.output/'merge_checks'],a.output/'merge_check.log',env)
        merge_report=json.loads((a.output/'merge_checks/report.json').read_text(encoding='utf-8'))
        r.require(merge_report['status']=='actual_global_id_merge_tree_boundaries_guards_and_bits_passed' and merge_report['calls']==36,'Affected tree boundary checks failed')
        report['affected_checks']={'leaf':selector_report,'merge':merge_report};save();check()
        small=a.output/'uneven_tail_input'
        small_description=corpus.prepare(small,8193,vocab,seed,cases)
        small_description,small_index=corpus.validate(small)
        small_actual=a.output/'uneven_tail_actual'
        g.execute([runner,small/'index',small/'A.txt',small/'B.txt',small_actual,old['device'],vendors,10,8,24,1,10,
            small/'A/expected_scores.bin',small/'B/expected_scores.bin'],a.output/'uneven_tail.log',env)
        tail_result=verify_run(small_actual,small,small_description,small_index,'npu',old['device'],1,10)
        report['affected_checks']['uneven_tail']={'docs':8193,'padded_docs':16384,'checked':True,
            'session':tail_result['session'],'output_sha256':tail_result['output_sha256']};save();check()
        for n in a.sizes:
            print(f'Grow ARTICLE DATABASE: N={n}; fixedQ8/three words/K10. Not a batch-throughput substitution.',flush=True)
            # Upper bounds before expensive generation; actual counts and budgets replace them below.
            upper=layout(n,vocab,min(48*n,2**31-1));available=host_available_bytes()
            if upper['cpu_peak_known_bytes']+512*1024**2>=available*4//5:
                report['tiers'].append({'docs':n,'status':'not_measured_host_budget','available_host_bytes':available,
                    'conservative_plan':upper,'reason':'Retained CPU and reference RAM estimate exceeds80% of MemAvailable'});save();continue
            required_disk=(n*8*4*4*3*6)+(n*48*8+n*4+n*8*8)+512*1024**2
            disk_free=shutil.disk_usage(a.output).free
            if required_disk>=disk_free:
                report['tiers'].append({'docs':n,'status':'not_measured_disk_budget','free_disk_bytes':disk_free,
                    'required_disk_upper_bound':required_disk,'reason':'Insufficient disk for inputs and immutable full checkpoints'});save();continue
            inputs=a.output/'inputs'/str(n)
            description=corpus.prepare(inputs,n,vocab,seed,cases,corpus=a.corpus_jsonl)
            description,index=corpus.validate(inputs);plan=layout(n,vocab,index['posting_count'])
            tier={'docs':n,'status':'prepared','workload':description,'memory_plan':plan,'per_round':[]}
            report['tiers'].append(tier);save()
            actual_ram=host_available_bytes()
            if plan['cpu_peak_known_bytes']+512*1024**2>=actual_ram*4//5:
                tier.update(status='not_measured_host_budget',available_host_bytes=actual_ram,reason='Actual posting-count memory budget exceeds80% of MemAvailable');save();continue
            # Known normal+diagnostic memory plus conservative1GiB workspace reserve.
            # Actual runtime independently checks every measured workspace and allocation.
            if plan['known_device_bytes_upper_bound']+plan['debug_score_bytes']+1024**3>memory['free']*4//5:
                tier.update(status='not_measured_device_budget',reason='Known memory+debug+1GiB reserve exceeds80% of probed free memory');save();continue
            def measure(role,destination):
                destination.mkdir(parents=True);actual=destination/'actual'
                if role=='npu':command=[runner,inputs/'index',inputs/'A.txt',inputs/'B.txt',actual,old['device'],vendors,10,8,24,5,30,
                    inputs/'A/expected_scores.bin',inputs/'B/expected_scores.bin']
                else:command=[cpus[role],inputs/'index',inputs/'A.txt',inputs/'B.txt',actual,32,10,32,8,24,5,30]
                check();g.execute(command,destination/'run.log',env if role=='npu' else os.environ.copy(),timeout=3600)
                result=verify_run(actual,inputs,description,index,role,old['device']);check()
                result.update(command=[str(x) for x in command],runner_sha256=digest(Path(command[0])))
                r.write_json(destination/'report.json',result)
                print(f'N={n} {role} complete Search: '+json.dumps(result['summary'],ensure_ascii=False),flush=True)
                return {'directory':str(destination),'summary':result['summary'],'report_sha256':digest(destination/'report.json')}
            for i in range(6):
                corpus.validate(inputs);roles=('cpu','npu','cpu_directory') if i%2==0 else ('cpu_directory','npu','cpu')
                row={'round':i+1,'process_order':list(roles)}
                for role in roles:row[role]=measure(role,a.output/f'N{n}'/f'round{i+1}'/role)
                tier['per_round'].append(row);save()
            corpus.validate(inputs);tier.update(status='six_rounds_checked',decision=gate(tier['per_round']));save()
            print(f'N={n} same-boundary CPU/NPU decision: '+json.dumps(tier['decision'],ensure_ascii=False),flush=True)
        report['crossover']=crossover(report['tiers'])
        report['status']='actual_large_search_target_met_on_declared_data' if report['crossover']['large_scale_hardware_target_met'] else 'actual_large_search_no_winning_scale_proven'
        check();save();print('Tested scale crossover: '+json.dumps(report['crossover'],ensure_ascii=False),flush=True)
        return 0 if report['crossover']['large_scale_hardware_target_met'] else 2
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',first_error=str(error));save()
        logs=list(a.output.rglob('*.log'));recent=max(logs,key=lambda p:p.stat().st_mtime) if logs else None
        if recent:print('Latest stage log '+str(recent)+'\n'+recent.read_text(encoding='utf-8',errors='replace')[-20000:],flush=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,path in {'accepted':'results/register_select_close2','original':'results/Bm25FusedScoreTopKUbFix1',
        'merge-original':'results/Bm25TopKMerge','vendors':'results/custom_opp/vendors','anchor':'results/search_256k_q8_input1'}.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    p.add_argument('--toolkit',type=Path,default=Path('/usr/local/Ascend/cann-9.2.0'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=list(SIZES));p.add_argument('--corpus-jsonl',type=Path);a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    try:
        r.require(a.sizes==sorted(set(a.sizes)) and len(a.sizes)>=2 and a.sizes[0]>=262144 and a.sizes[-1]>=1000000,'Increasing database tiers, not Q, required')
        return run(a)
    except (Exception,KeyboardInterrupt) as error:print('FAILED: '+str(error)+'; no fabricated hardware results or CPU substitution.',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
