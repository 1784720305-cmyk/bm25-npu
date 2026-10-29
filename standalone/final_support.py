"""Final full-Search result/byte/timing checks; no historical experiment imports."""
from array import array
import json,math,os,re,statistics,subprocess
from pathlib import Path
from types import SimpleNamespace
from standalone import resident_index as r,large_corpus as corpus
ROOT=Path(__file__).resolve().parents[1]
digest=corpus.digest
TIMER='caller steady_clock before Search through returned CPU Top-K'
CHECKPOINTS=('before_A','before_B','after_A','after_B')
CPU_SOURCES=('npu_prototype/search_host/search_inputs.h', 'npu_prototype/search_benchmark_host/benchmark_common.h', 'cpu_reference/index_search/cpu_index_search.h', 'cpu_reference/topk_document_shards.h', 'npu_prototype/search_benchmark_host/cpu_benchmark.cpp')
DIRECTORY_SOURCES=('npu_prototype/search_host/search_inputs.h', 'npu_prototype/search_benchmark_host/benchmark_common.h', 'cpu_reference/index_search/cpu_index_search.h', 'cpu_reference/directory_search/cpu_directory_search.h', 'cpu_reference/topk_document_shards.h', 'common/shard_directory.h', 'npu_prototype/graph_search_host/cpu_directory_benchmark.cpp')
NPU_SOURCES=('common/large_search_plan.h', 'common/fixed_query_packet.h', 'common/shard_directory.h', 'cpu_reference/topk_document_shards.h', 'npu_prototype/search_host/search_inputs.h', 'npu_prototype/search_benchmark_host/benchmark_common.h', 'npu_prototype/fused_chain_host/fused_runtime.h', 'npu_prototype/fused_search_host/fused_api.h', 'npu_prototype/graph_search_host/planned_search.h', 'npu_prototype/large_search_host/benchmark.cpp', 'npu_prototype/large_search_host/large_search.h', 'npu_prototype/large_search_host/merge_check.h')
def summarize(values, q):
    r.require(values and all((isinstance(value,(int,float)) and math.isfinite(value) and value>0) for value in values), '计时样本为空或非法')
    ordered = sorted(values)
    return {'samples':len(values), 'median_batch_ms':statistics.median(values),
            'p95_batch_ms':ordered[math.ceil(.95*len(values))-1], 'mean_batch_ms':statistics.mean(values),
            'search_time_throughput_qps':q*1000/statistics.mean(values)}

g=SimpleNamespace(checks=SimpleNamespace(TIMER=TIMER,CHECKPOINTS=CHECKPOINTS,summarize=summarize))

def check_compiled_sources(data,role):
 required=NPU_SOURCES if role=='npu' else CPU_SOURCES if role=='cpu' else DIRECTORY_SOURCES
 actual=data.get('host_source_sha256',{})
 r.require(all(actual.get(name)==digest(ROOT/name) for name in required),'Compiled core source mismatch')
 # Fresh final builds also bind their unified CMake recipe. Historical accepted
 # binaries carry extra historical entries, retained as provenance, not re-executed.
 if 'CMakeLists.txt' in actual:r.require(actual['CMakeLists.txt']==digest(ROOT/'CMakeLists.txt'),'Final build recipe changed')
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

def same_stream(a,b):
    if a.stat().st_size!=b.stat().st_size:return False
    with a.open('rb') as left,b.open('rb') as right:
        while True:
            chunk=left.read(8*1024*1024)
            if chunk!=right.read(len(chunk)):return False
            if not chunk:return True

def verify_run(path,inputs,description,index,role,device,warmup=5,repeats=30,threads=32):
    data=json.loads((path/'benchmark_session.json').read_text(encoding='utf-8'))
    n=description['docs'];q=description['queries'];k=description['k'];npu=role=='npu';plan=layout(n,index['vocab_size'],index['posting_count'],q,k,q*3)
    wanted={'status':'search_benchmark_calls_completed_checks_pending','backend':'npu' if npu else 'cpu',
        'docs':n,'queries':q,'k':k,'partitions':plan['leaves'] if npu else threads,'capacity_queries':q,'capacity_words':q*3,
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
    else:wanted.update(requested_threads=threads,actual_threads=threads,index_bytes=index['index_bytes'],
        fp32_flags='-O3 -march=native -ffp-contract=off -fno-fast-math')
    r.require(all(data.get(key)==value and type(data.get(key)) is type(value) for key,value in wanted.items()),'Measured session contract mismatch: '+role)
    check_compiled_sources(data,role)
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
def cpu_gate(rows,roles):
    r.require(len(rows)==6,'Six complete rounds required');cases={}
    for case in ('A','B'):
        cpu=[min(row[role]['summary'][case]['median_batch_ms'] for role in roles) for row in rows]
        npu=[row['candidate']['summary'][case]['median_batch_ms'] for row in rows]
        cp95=[min(row[role]['summary'][case]['p95_batch_ms'] for role in roles) for row in rows]
        np95=[row['candidate']['summary'][case]['p95_batch_ms'] for row in rows]
        wins=sum(a<b for a,b in zip(npu,cpu))
        cases[case]={'cpu_medians_ms':cpu,'npu_medians_ms':npu,'cpu_p95_ms':cp95,'npu_p95_ms':np95,
            'wins':wins,'cpu_over_npu':statistics.median(cpu)/statistics.median(npu),
            'passed':wins==6 and statistics.median(np95)<statistics.median(cp95)}
    return {'performance_target_met_at_this_scale':all(v['passed'] for v in cases.values()),'cpu_roles':list(roles),'cases':cases}
