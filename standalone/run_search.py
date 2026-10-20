"""实际调用原始查询词编号Search入口，核对边界及排名；首次诊断，不作加速结论。"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, run_resident_chain as chain

HOST_SOURCES=('npu_prototype/search_host/CMakeLists.txt','npu_prototype/search_host/search_inputs.h',
    'npu_prototype/search_host/npu_search.h','npu_prototype/search_host/search_runner.cpp',
    'npu_prototype/resident_chain_host/resident_chain_common.h','npu_prototype/query_score_host/query_score_runtime.h')
TOOL_SOURCES=tuple(sorted(set(chain.TOOL_SOURCES)|{'standalone/run_search.py'}))
TIMER_POLICY='Search entry before validation/packing through CPU Top-K download/required checks/request cleanup'
digest=chain.digest


def write_queries(path,queries):
    r.require(isinstance(queries,list) and 1<=len(queries)<=10000 and
        all(isinstance(row,list) and all(r.integer(w,0,2147483646) for w in row) for row in queries),
        '原始查询须为非空二维INT32词编号列表；每条可为空，保留重复及未知词')
    lines=['BM25_QUERY_IDS_V1',str(len(queries))]
    lines.extend(' '.join([str(len(row))]+[str(w) for w in row]) for row in queries)
    path.write_bytes(('\n'.join(lines)+'\n').encode('ascii'))


def finite_positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def check_session(path,index,records,device,k,shards,capq,capwords,top_workspace):
    data=json.loads((path/'search_session.json').read_text(encoding='utf-8'))
    wanted={'status':'device_search_calls_completed_independent_check_pending','device':device,
        'docs':index['num_docs'],'k':k,'shards':shards,'capacity_queries':capq,'capacity_words':capwords,
        'index_upload_batches':1,'index_tensor_h2d_calls':5,'index_upload_bytes':index['index_bytes'],
        'index_readback_byte_identical':True,'pipeline_stages':3,'intermediate_host_roundtrips':0,
        'input_contract':'raw_grouped_INT32_word_IDs_in_CPU_memory','timer_policy':TIMER_POLICY,
        'debug_index_d2h_calls':5,'debug_index_d2h_bytes':index['index_bytes']}
    r.require(all(type(data.get(key)) is type(value) and data[key]==value for key,value in wanted.items()),
              '实际Search输入/边界/设备/索引复用记录不符')
    r.require(data.get('host_source_sha256')=={name:digest(ROOT/name) for name in HOST_SOURCES},
              'Search调用器不是当前源码编译，请重新配置编译新主机')
    r.require(finite_positive(data.get('cold_index_load_runtime_upload_capacity_bootstrap_ms')),'冷启动诊断时间非法')
    r.require(data.get('cold_bootstrap')=={'synthetic_empty_queries':capq,'query_h2d_calls':2,
        'query_h2d_bytes':8*(capq+1)+4,'topk_d2h_calls':2,'topk_d2h_bytes':8*capq*k},'合成空容量预启动未单列')
    workspaces=data.get('workspace_capacity_bytes',[])
    r.require(len(workspaces)==3 and all(type(n) is int and n>=0 for n in workspaces) and
              workspaces[1]==top_workspace,'Search工作区与已验收Top-K不符')
    r.require(len(data.get('requests',[]))==5,'五批实际Search调用缺失')
    poison=4*capq*index['num_docs']+8*capq*shards*k+8*capq*k
    for i,(name,call) in enumerate(zip(chain.SEQUENCE,data['requests'])):
        record=records[name]; q=record['num_queries']; logical=record['query_input_bytes']
        wanted={'label':f'call{i}_{name}','queries':q,'words':record['query_term_count'],
            'logical_query_bytes':logical,'query_h2d_calls':2,
            'query_h2d_bytes':logical+(4 if record['query_term_count']==0 else 0),
            'topk_d2h_calls':2,'topk_d2h_bytes':8*q*k,'debug_poison_h2d_calls':5,
            'debug_poison_h2d_bytes':poison,'debug_score_candidate_d2h_calls':3,
            'debug_score_candidate_d2h_bytes':4*q*index['num_docs']+8*q*shards*k,
            'packing_inside_Search':True,'topk_received_before_stop':True,'debug_outside_Search':True}
        r.require(all(type(call.get(key)) is type(value) and call[key]==value for key,value in wanted.items()) and
                  finite_positive(call.get('search_call_ms')),'Search单次时间/传输/整理/停表记录不符：'+str(i))
    for first,last in ((0,2),(1,4)):
        for file in chain.FILES:
            r.require((path/f'call{first}_{chain.SEQUENCE[first]}'/file).read_bytes()==
                      (path/f'call{last}_{chain.SEQUENCE[last]}'/file).read_bytes(),'重复Search输出字节不同：'+file)
    return data


def run(inputs,output,runner,vendors,score_project,top_project,merge_project,acceptance,device,k,shards,capq,capwords):
    inputs,output,runner,vendors=(p.resolve() for p in (inputs,output,runner,vendors))
    r.require(not output.exists(),'结果目录已存在，请换新名字，不覆盖')
    r.require(runner.is_file() and r.integer(device,0,2147483647),'Search调用器缺失或设备号非法')
    r.require(r.integer(capq,1,10000) and r.integer(capwords,0,16777216),'服务容量非法')
    index,records,snapshot=chain.score.validate_inputs(inputs)
    chain.dimensions(index['num_docs'],k,shards)
    r.require(capq*index['num_docs']<=2147483647 and all(
        item['num_queries']<=capq and item['query_term_count']<=capwords for item in records.values()),'验收请求超过声明容量')
    installation=chain.validate_install(vendors,score_project,top_project,merge_project,acceptance)
    output.mkdir(parents=True)
    report={'status':'running','device':device,'index_id':index['index_id'],'k':k,'shards':shards,
        'capacity_queries':capq,'capacity_words':capwords,'installation':installation,'input_sha256':snapshot,
        'runner_sha256':digest(runner),'host_source_sha256':{name:digest(ROOT/name) for name in HOST_SOURCES},
        'tool_source_sha256':{name:digest(ROOT/name) for name in TOOL_SOURCES},
        'scope':'First raw-query-ID Search boundary/correctness diagnostic; five calls with outside-timer debug poison/download; not steady-state or a speedup claim.'}
    r.write_json(output/'report.json',report)
    try:
        raw=output/'raw_queries'; raw.mkdir(); raw_hashes={}
        for name in records:
            unused,queries=r.validate_queries(inputs/name,index)
            file=raw/(name+'.txt'); write_queries(file,queries); raw_hashes[str(file)]=digest(file)
        lines=[]
        for i,name in enumerate(chain.SEQUENCE):
            file=str(raw/(name+'.txt'))
            r.require(not any(c in file for c in '\t\r\n'),'原始查询路径含控制字符')
            lines.append(f'call{i}_{name}\t{file}')
        plan=output/'requests.tsv'; plan.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
        raw_hashes[str(plan)]=digest(plan)
        top_workspace=installation['topk_acceptance']['workspace_bytes']
        command=[str(runner),str(inputs/'index'),str(plan),str(output/'device'),str(device),str(vendors),
                 str(k),str(shards),str(capq),str(capwords),str(top_workspace)]
        report.update(command=command,raw_query_sha256=raw_hashes); r.write_json(output/'report.json',report)
        env=os.environ.copy(); env['ASCEND_CUSTOM_OPP_PATH']=':'.join(str((vendors/name).resolve(strict=True)) for name in
            ('bm25_query_score','bm25_topk_radix_collect','bm25_topk_merge'))
        print('实际原始词编号Search：计时内整理/上传/三段计算/同步/Top-K下载；诊断污染与中间结果下载在计时外。',flush=True)
        print(json.dumps(command,ensure_ascii=False),flush=True)
        chain.score.execute(command,output/'device.log',env)
        session=check_session(output/'device',index,records,device,k,shards,capq,capwords,top_workspace)
        checks=[]; hashes={}
        for i,name in enumerate(chain.SEQUENCE):
            label=f'call{i}_{name}'; actual=output/'device'/label
            checks.append(chain.check_outputs(inputs/'index',inputs/name,actual,k,shards))
            for file in chain.FILES: hashes[f'{label}/{file}']=digest(actual/file)
            print(label+'：Search返回结果及计时外完整分数/候选/严格排名核对通过。',flush=True)
        r.require(all(digest(Path(path))==value for mapping in (snapshot,raw_hashes) for path,value in mapping.items()),
                  '运行期间索引/查询/参考或原始词编号文件改变')
        r.require(chain.validate_install(vendors,score_project,top_project,merge_project,acceptance)==installation and
                  digest(runner)==report['runner_sha256'] and all(digest(ROOT/name)==value for key in
                  ('host_source_sha256','tool_source_sha256') for name,value in report[key].items()),'运行期间安装/调用器/源码改变')
        report.update(status='search_boundary_device_calls_and_all_checks_passed',session=session,checks=checks,
            output_sha256=hashes,device_log_sha256=digest(output/'device.log'))
        r.write_json(output/'report.json',report)
        print('五次原始词编号Search及全部结果检查通过；已记录单次完整边界诊断时间，尚无稳态或CPU加速结论。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        r.write_json(output/'report.json',report)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/query_score_small_input1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'results/search_host_build/bm25_search_runner')
    p.add_argument('--vendors',type=Path,default=ROOT/'results/custom_opp/vendors')
    p.add_argument('--score-project',type=Path,default=ROOT/'results/Bm25QueryScore')
    p.add_argument('--topk-project',type=Path,default=ROOT/'results/Bm25TopKRadixCollect')
    p.add_argument('--merge-project',type=Path,default=ROOT/'results/Bm25TopKMerge')
    p.add_argument('--topk-acceptance',type=Path,default=ROOT/'results/radix_collect_topk_validation1')
    p.add_argument('--device',type=int,default=2); p.add_argument('--k',type=int,default=3); p.add_argument('--shards',type=int,default=2)
    p.add_argument('--max-queries',type=int,default=65); p.add_argument('--max-words',type=int,default=512)
    args=p.parse_args()
    try: run(args.input,args.output,args.runner,args.vendors,args.score_project,args.topk_project,args.merge_project,
             args.topk_acceptance,args.device,args.k,args.shards,args.max_queries,args.max_words)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不生成CPU替代设备结果或成功报告。',file=sys.stderr); return 1
    return 0


if __name__=='__main__': sys.exit(main())
