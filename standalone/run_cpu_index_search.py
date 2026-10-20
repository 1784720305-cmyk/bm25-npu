"""同一整库索引＋原始词编号的C++ CPU Search验收；可与已有NPU Search结果直接核对。"""
import argparse
from array import array
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, run_search as npu, run_resident_chain as chain, query_score_fixture as fixture

HOST_SOURCES=('cpu_reference/index_search/CMakeLists.txt','cpu_reference/index_search/cpu_index_search.h',
    'cpu_reference/index_search/cpu_search_runner.cpp','cpu_reference/topk_document_shards.h',
    'npu_prototype/search_host/search_inputs.h')
TOOL_SOURCES=tuple(sorted(set(npu.TOOL_SOURCES)|{'standalone/run_cpu_index_search.py'}))
FILES=('scores.bin','top_scores.bin','top_ids.bin')
TIMER_POLICY='CPU Search entry before request validation through index lookup/FP32 scoring/Top-K/required checks/CPU result'
digest=chain.digest


def validate_npu_reference(path,inputs,index,records,snapshot,k):
    print('核对已有NPU Search报告及结果文件；此步不调用设备。',flush=True)
    report=json.loads((path/'report.json').read_text(encoding='utf-8'))
    r.require(report.get('status')=='search_boundary_device_calls_and_all_checks_passed' and
              report.get('index_id')==index['index_id'] and type(report.get('k')) is int and report['k']==k and
              report.get('input_sha256')==snapshot,'NPU参考未通过或不属于同一索引/查询/K；不能套用旧计算实验')
    r.require(r.integer(report.get('device'),0,2147483647) and r.integer(report.get('shards'),1,32) and
              r.integer(report.get('capacity_queries'),1,10000) and r.integer(report.get('capacity_words'),0,16777216),
              'NPU参考设备或容量记录非法')
    r.require(report.get('host_source_sha256')=={name:digest(ROOT/name) for name in npu.HOST_SOURCES} and
              report.get('tool_source_sha256')=={name:digest(ROOT/name) for name in npu.TOOL_SOURCES},'NPU参考对应的源码/检查工具已改变')
    command=report.get('command',[])
    workspace=report['installation']['topk_acceptance']['workspace_bytes']
    r.require(len(command)==11 and digest(Path(command[0]))==report.get('runner_sha256') and
              command[1:5]==[str(inputs/'index'),str(path/'requests.tsv'),str(path/'device'),str(report['device'])] and
              command[6:]==[str(k),str(report['shards']),str(report['capacity_queries']),str(report['capacity_words']),str(workspace)],
              'NPU参考可执行程序/实际命令缺失、改变或不属于本输入')
    r.require(digest(path/'device.log')==report.get('device_log_sha256'),'NPU参考日志改变')
    outputs={f'call{i}_{name}/{file}':digest(path/'device'/f'call{i}_{name}'/file)
             for i,name in enumerate(chain.SEQUENCE) for file in chain.FILES}
    r.require(outputs==report.get('output_sha256'),'NPU参考完整输出指纹改变')
    for mapping in (report.get('raw_query_sha256',{}),snapshot):
        r.require(mapping and all(digest(Path(p))==value for p,value in mapping.items()),'NPU参考原始查询或输入已改变')
    npu.check_session(path/'device',index,records,report['device'],k,report['shards'],
                      report['capacity_queries'],report['capacity_words'],workspace)
    for i,name in enumerate(chain.SEQUENCE):
        chain.check_outputs(inputs/'index',inputs/name,
                            path/'device'/f'call{i}_{name}',k,report['shards'])
    return {'directory':str(path),'report_sha256':digest(path/'report.json'),'device':report['device'],
            'scope':'same-input accepted raw-query-ID NPU Search correctness reference; no performance ratio'}


def check_cpu_outputs(index_dir,query_dir,actual,k):
    index,values=r.validate_index(index_dir); record,queries=r.validate_queries(query_dir,index)
    q,n=record['num_queries'],index['num_docs']
    fixture.check(index_dir,query_dir,actual/'scores.bin')
    raw,full=chain.read(actual/'scores.bin','f',q*n)
    top_raw,top=chain.read(actual/'top_scores.bin','f',q*k)
    unused,ids=chain.read(actual/'top_ids.bin','i',q*k)
    expected=r.diagnostic_scores(index,values,queries)
    r.require(raw==r.encode(expected),'C++CPU完整分数与同输入FP32有序公式不逐字节一致')
    r.require(all(math.isfinite(x) and x>=0 for x in top),'CPU Top-K含非法分数')
    for row in range(q):
        base=row*n; chosen=sorted(range(n),key=lambda i:(-expected[base+i],i))[:k]
        r.require(list(ids[row*k:(row+1)*k])==chosen,'CPU Top-K编号或同分规则不符')
        wanted=b''.join(raw[(base+i)*4:(base+i+1)*4] for i in chosen)
        r.require(top_raw[row*k*4:(row+1)*k*4]==wanted,'CPU Top-K未保留对应文章分数原字节')
    return {'queries':q,'docs':n,'k':k,'full_fp32_reference_bits_identical':True,'strict_ids_and_selected_score_bits':True}


def compare_npu(cpu_dir,npu_dir,q,n,k):
    errors={}
    for name,count in (('scores.bin',q*n),('top_scores.bin',q*k)):
        unused,cpu=chain.read(cpu_dir/name,'f',count); unused,device=chain.read(npu_dir/name,'f',count)
        r.require(all(math.isfinite(x) and x>=0 for x in device),'NPU参考含非法分数')
        errors[name]=r.compare(cpu,device)
    r.require((cpu_dir/'top_ids.bin').read_bytes()==(npu_dir/'top_ids.bin').read_bytes(),'新CPU与NPU的最终编号/顺序不一致')
    return {'strict_ids_identical':True,'max_abs_score_errors':errors}


def check_session(output,index,records,threads,k,shards,capq,capwords):
    data=json.loads((output/'cpu_search_session.json').read_text(encoding='utf-8'))
    wanted={'status':'cpu_index_search_calls_completed_independent_check_pending','docs':index['num_docs'],
        'k':k,'document_shards':shards,'requested_threads':threads,'capacity_queries':capq,'capacity_words':capwords,
        'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory','timer_policy':TIMER_POLICY,
        'index_storage':'host_memory','query_index_loads':1,'index_bytes':index['index_bytes'],
        'host_index_byte_identical_to_source':True,'debug_source_reread_batches':1,
        'cold_bootstrap_empty_queries':capq,'cpu_needs_backend_word_packing':False,
        'diagnostic_poison_before_calls':True,'diagnostic_files_outside_Search':True,
        'fp32_flags':'-O3 -march=native -ffp-contract=off -fno-fast-math'}
    r.require(all(type(data.get(key)) is type(value) and data[key]==value for key,value in wanted.items()),
              'CPU Search输入/计时边界/索引/线程/容量记录不符')
    r.require(r.integer(data.get('actual_threads'),1,threads) and npu.finite_positive(data.get('cold_index_load_threads_capacity_bootstrap_ms')),
              'CPU实际线程数或冷启动诊断时间非法')
    r.require(data.get('host_source_sha256')=={name:digest(ROOT/name) for name in HOST_SOURCES},'CPU程序不是当前源码编译')
    r.require(len(data.get('requests',[]))==5,'五次CPU Search记录缺失')
    for i,(name,call) in enumerate(zip(chain.SEQUENCE,data['requests'])):
        source=records[name]; q=source['num_queries']
        wanted={'label':f'call{i}_{name}','queries':q,'words':source['query_term_count'],'topk_output_bytes':8*q*k,
            'debug_full_scores_bytes':4*q*index['num_docs'],'index_lookup_inside_Search':True,
            'topk_ready_before_stop':True,'debug_outside_Search':True}
        r.require(all(type(call.get(key)) is type(value) and call[key]==value for key,value in wanted.items()) and
                  npu.finite_positive(call.get('search_call_ms')),'CPU请求/计时/输出尺寸记录不符')
    for a,b in ((0,2),(1,4)):
        for file in FILES:
            r.require((output/f'call{a}_{chain.SEQUENCE[a]}'/file).read_bytes()==
                      (output/f'call{b}_{chain.SEQUENCE[b]}'/file).read_bytes(),'CPU重复查询分数/编号字节不一致')
    return data


def run(inputs,output,runner,threads,k,shards,capq,capwords,reference=None):
    inputs,output,runner=(p.resolve() for p in (inputs,output,runner))
    r.require(not output.exists(),'CPU结果目录已存在，请换新名字，不覆盖')
    r.require(runner.is_file() and r.integer(threads,1,65536) and r.integer(capq,1,10000) and r.integer(capwords,0,16777216),
              'CPU调用器缺失或线程/容量非法')
    index,records,snapshot=chain.score.validate_inputs(inputs)
    r.require(r.integer(k,1,min(index['num_docs'],32)) and r.integer(shards,1,min(threads,index['num_docs'])) and
              capq*index['num_docs']<=2147483647 and all(x['num_queries']<=capq and x['query_term_count']<=capwords for x in records.values()),
              'CPU索引/请求尺寸、K或文章分块不支持')
    prior=validate_npu_reference(reference.resolve(),inputs,index,records,snapshot,k) if reference else None
    output.mkdir(parents=True)
    report={'status':'running','index_id':index['index_id'],'requested_threads':threads,'k':k,'document_shards':shards,
        'capacity_queries':capq,'capacity_words':capwords,'input_sha256':snapshot,'runner_sha256':digest(runner),
        'host_source_sha256':{name:digest(ROOT/name) for name in HOST_SOURCES},
        'tool_source_sha256':{name:digest(ROOT/name) for name in TOOL_SOURCES},'npu_reference':prior,
        'scope':'Same public raw-ID/resident raw-TF index CPU Search diagnostic; no CANN runtime and no old prepared-array timings; no speedup claim.'}
    r.write_json(output/'report.json',report)
    try:
        raw=output/'raw_queries'; raw.mkdir(); raw_hashes={}
        for name in records:
            unused,queries=r.validate_queries(inputs/name,index)
            file=raw/(name+'.txt'); npu.write_queries(file,queries); raw_hashes[str(file)]=digest(file)
        plan=output/'requests.tsv'; lines=[]
        for i,name in enumerate(chain.SEQUENCE):
            file=str(raw/(name+'.txt')); r.require(not any(c in file for c in '\t\r\n'),'原始查询路径含控制字符')
            lines.append(f'call{i}_{name}\t{file}')
        plan.write_bytes(('\n'.join(lines)+'\n').encode('utf-8')); raw_hashes[str(plan)]=digest(plan)
        command=[str(runner),str(inputs/'index'),str(plan),str(output/'cpu'),str(threads),str(k),str(shards),str(capq),str(capwords),'raw_query_ids_v1']
        report.update(command=command,raw_query_sha256=raw_hashes); r.write_json(output/'report.json',report)
        print('实际C++CPU Search：原始词编号→常驻索引查找/FP32计算/已有多线程Top-K；不读取期望答案，不伪造上传开销。',flush=True)
        print(json.dumps(command,ensure_ascii=False),flush=True)
        chain.score.execute(command,output/'cpu.log',os.environ.copy())
        session=check_session(output/'cpu',index,records,threads,k,shards,capq,capwords)
        checks=[]; cross=[]; hashes={}
        for i,name in enumerate(chain.SEQUENCE):
            label=f'call{i}_{name}'; actual=output/'cpu'/label
            checks.append(check_cpu_outputs(inputs/'index',inputs/name,actual,k))
            if prior: cross.append(compare_npu(actual,Path(prior['directory'])/'device'/label,records[name]['num_queries'],index['num_docs'],k))
            for file in FILES: hashes[f'{label}/{file}']=digest(actual/file)
            print(label+'：CPU完整公式和严格排名通过'+('，与已有NPU Search编号/分数直接核对通过。' if prior else '。'),flush=True)
        r.require(all(digest(Path(p))==value for mapping in (snapshot,raw_hashes) for p,value in mapping.items()),'CPU运行期间输入/查询/参考改变')
        r.require(digest(runner)==report['runner_sha256'] and all(digest(ROOT/name)==value for key in
                  ('host_source_sha256','tool_source_sha256') for name,value in report[key].items()),'CPU运行期间调用器或源码改变')
        if prior: r.require(validate_npu_reference(Path(prior['directory']),inputs,index,records,snapshot,k)==prior,'NPU参考运行期间改变')
        report.update(status='cpu_index_search_and_independent_checks_passed',session=session,checks=checks,npu_cross_checks=cross,
                      output_sha256=hashes,cpu_log_sha256=digest(output/'cpu.log'))
        r.write_json(output/'report.json',report)
        print('五次CPU索引Search和结果检查通过；单次诊断，不用这五次与此前NPU日志计算稳态加速比。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        r.write_json(output/'report.json',report); raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/query_score_small_input1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'results/cpu_index_search_build/bm25_cpu_index_search_runner')
    p.add_argument('--threads',type=int,default=4); p.add_argument('--k',type=int,default=3)
    p.add_argument('--document-shards',type=int,default=2)
    p.add_argument('--max-queries',type=int,default=65); p.add_argument('--max-words',type=int,default=512)
    p.add_argument('--npu-reference',type=Path)
    args=p.parse_args()
    try: run(args.input,args.output,args.runner,args.threads,args.k,args.document_shards,args.max_queries,args.max_words,args.npu_reference)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不用旧CPU展开数组程序或备用结果替代。',file=sys.stderr); return 1
    return 0


if __name__=='__main__': sys.exit(main())
