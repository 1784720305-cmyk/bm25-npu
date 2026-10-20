"""实际常驻索引评分→已有局部Top-K→合并的五批正确性验收，不测search，不回退CPU。"""
import argparse
from array import array
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, query_score_fixture as fixture, run_query_score as score
from standalone import validate_topk_radix_collect as top, validate_topk_merge as merge
from standalone.prepare_sharded_runtime import validate_radix_acceptance

SEQUENCE = score.SEQUENCE
FILES = ('scores.bin','partial_scores.bin','partial_ids.bin','top_scores.bin','top_ids.bin')
HOST_SOURCES = ('npu_prototype/resident_chain_host/CMakeLists.txt',
                'npu_prototype/resident_chain_host/resident_chain_common.h',
                'npu_prototype/resident_chain_host/resident_chain_runner.cpp',
                'npu_prototype/query_score_host/query_score_runtime.h')
TOOL_SOURCES = tuple(sorted(set(score.TOOL_SOURCES) | {
    'standalone/run_resident_chain.py','standalone/validate_topk_radix_collect.py',
    'standalone/validate_topk_merge.py','standalone/prepare_sharded_runtime.py',
    'standalone/prepare_topk_parallel_runtime.py','standalone/topk_fixture.py'}))
digest = score.digest


def dimensions(n,k,shards):
    r.require(r.integer(k,1,32) and r.integer(shards,1,32) and n%shards == 0 and n//shards >= k,
              '已有合并要求文章数能整除分块数、每块至少K篇；K和分块数均在1..32，不自动补文章')


def artifacts(vendor,project,stem,header):
    info = json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    r.require(info.get('compute_unit') == 'ascend950','需要实际950构建产物')
    for path in (vendor/'op_api/include'/header,vendor/'op_api/lib/libcust_opapi.so'):
        matches = list((project/'build_out').rglob(path.name))
        r.require(path.stat().st_size > 0 and matches and any(digest(p) == digest(path) for p in matches),
                  '安装头文件/API库与实际构建产物不匹配：'+path.name)
    objects = sorted((vendor/f'op_impl/ai_core/tbe/kernel/ascend950/{stem}').glob('*.o'))
    r.require(objects,'缺少实际950设备目标文件：'+stem)
    hashes = {}
    for obj in objects:
        matches = list((project/'build_out').rglob(obj.name))
        r.require(matches and all(digest(p) == digest(obj) for p in matches),'950设备目标文件与构建产物不匹配')
        hashes[obj.relative_to(vendor).as_posix()] = digest(obj)
    return hashes


def validate_install(vendors,score_project,top_project,merge_project,acceptance):
    vendors = vendors.resolve(strict=True)
    score_install = score.validate_install(vendors/'bm25_query_score',score_project)
    top_install = top.validate_install(vendors/'bm25_topk_radix_collect',top_project)
    accepted = validate_radix_acceptance(acceptance,top_install,top)
    merge_vendor = vendors/'bm25_topk_merge'
    r.require(merge_vendor.resolve(strict=True).name == 'bm25_topk_merge','需要实际独立合并vendor')
    merge_install = merge.validate_install(merge_vendor,merge_project)
    top_objects = artifacts(vendors/'bm25_topk_radix_collect',top_project,'bm25_top_k','aclnn_bm25_top_k.h')
    merge_objects = artifacts(merge_vendor,merge_project,'bm25_top_k_merge','aclnn_bm25_top_k_merge.h')
    return {'score':score_install,'topk':top_install,'merge':merge_install,'topk_acceptance':accepted,
            'topk_device_objects':top_objects,'merge_device_objects':merge_objects,
            'scope':'installed artifacts and prior radix_collect acceptance verified; new three-stage device session pending'}


def read(path,kind,count):
    raw = path.read_bytes()
    r.require(len(raw) == count*4,'设备输出尺寸不符：'+str(path))
    values = array(kind); values.frombytes(raw)
    r.require(values.itemsize == 4,'本机数组类型宽度不符')
    if sys.byteorder != 'little': values.byteswap()
    return raw,values


def check_outputs(index_dir,query_dir,actual,k,shards,cpu_scores=None):
    index,values = r.validate_index(index_dir)
    record,queries = r.validate_queries(query_dir,index)
    n,q = index['num_docs'],record['num_queries']
    dimensions(n,k,shards)
    fixture.check(index_dir,query_dir,actual/'scores.bin')
    full_raw,full = read(actual/'scores.bin','f',q*n)
    part_raw,parts = read(actual/'partial_scores.bin','f',q*shards*k)
    unused,local = read(actual/'partial_ids.bin','i',q*shards*k)
    top_raw,tops = read(actual/'top_scores.bin','f',q*k)
    unused,ids = read(actual/'top_ids.bin','i',q*k)
    r.require(all(math.isfinite(x) and x >= 0 for x in full) and
              all(math.isfinite(x) and x >= 0 for x in parts) and
              all(math.isfinite(x) and x >= 0 for x in tops),'设备输出含负分/NaN/Inf')
    # Slow checker from the SAME FP32 index/ordered formula, never a performance opponent.
    if cpu_scores is None: cpu_scores = r.diagnostic_scores(index,values,queries)
    error = r.compare(full,cpu_scores)
    width = n//shards
    for row in range(q):
        base = row*n
        for shard in range(shards):
            start = base+shard*width
            chosen = sorted(range(width),key=lambda i:(-full[start+i],i))[:k]
            offset = (row*shards+shard)*k
            r.require(list(local[offset:offset+k]) == chosen,'局部Top-K编号/同分规则不符')
            expected = b''.join(full_raw[(start+i)*4:(start+i+1)*4] for i in chosen)
            r.require(part_raw[offset*4:(offset+k)*4] == expected,'局部候选分数未保留原始设备FP32字节')
        chosen = sorted(range(n),key=lambda i:(-full[base+i],i))[:k]
        cpu_chosen = sorted(range(n),key=lambda i:(-cpu_scores[base+i],i))[:k]
        r.require(list(ids[row*k:(row+1)*k]) == chosen,'最终编号与设备完整分数排序不符')
        r.require(chosen == cpu_chosen,'设备与同输入FP32 CPU参考排名不同；停止，需定位分数舍入/算法差异')
        expected = b''.join(full_raw[(base+i)*4:(base+i+1)*4] for i in chosen)
        r.require(top_raw[row*k*4:(row+1)*k*4] == expected,'最终分数未保留原始设备FP32字节')
    return {'queries':q,'docs':n,'k':k,'shards':shards,'max_abs_same_fp32_reference_error':error,
            'all_local_and_global_ids_strict':True,'selected_score_bits_preserved':True,
            'same_input_cpu_fp32_ids_strict':True}


def check_session(path,index,records,k,shards,device,accepted_workspace):
    session = json.loads((path/'device_session.json').read_text(encoding='utf-8'))
    expected = {'status':'device_resident_chain_completed_reference_and_ranking_checked','device':device,
                'docs':index['num_docs'],'k':k,'shards':shards,'pipeline_stages':3,'intermediate_host_roundtrips':0,
                'index_upload_batches':1,'index_tensor_h2d_calls':5,'index_upload_bytes':index['index_bytes'],
                'index_readback_byte_identical':True,'index_device_allocations_reused':True,
                'debug_index_d2h_calls':5,'debug_index_d2h_bytes':index['index_bytes']}
    r.require(all(type(session.get(key)) is type(value) and session[key] == value for key,value in expected.items()),
              '三段会话未证明当前设备/尺寸/索引一次上传/回读不变/无中途回传')
    r.require(session.get('host_source_sha256') == {name:digest(ROOT/name) for name in HOST_SOURCES},
              '主机可执行程序不是当前源码编译，请重新配置编译')
    r.require(len(session.get('requests',[])) == len(SEQUENCE),'实际五批查询记录缺失')
    maxq = max(item['num_queries'] for item in records.values())
    poison = 4*maxq*index['num_docs']+8*maxq*shards*k+8*maxq*k
    for i,(name,item) in enumerate(zip(SEQUENCE,session['requests'])):
        source = records[name]; q = source['num_queries']
        expected = {'label':f'call{i}_{name}','queries':q,'logical_word_count':source['query_term_count'],
                    'logical_query_bytes':source['query_input_bytes'],
                    'query_h2d_bytes':source['query_input_bytes']+(4 if source['query_term_count'] == 0 else 0),
                    'query_tensor_h2d_calls':2,'topk_d2h_bytes':8*q*k,'topk_d2h_calls':2,
                    'debug_score_and_candidate_d2h_bytes':4*q*index['num_docs']+8*q*shards*k,
                    'debug_score_and_candidate_d2h_calls':3,'debug_poison_h2d_calls':5,'debug_poison_h2d_bytes':poison,
                    'outputs_poisoned':True,'fresh_executors':3,'chain_synchronizations':1,'debug_tensor_cleanup_synchronizations':1,
                    'full_reference_checked':True,'exact_device_ranking_checked':True,'repeated_all_output_bits_checked':i in (2,4)}
        r.require(all(type(item.get(key)) is type(value) and item[key] == value for key,value in expected.items()),
                  '请求尺寸/字节/污染/新executor/同步/重复检查记录不符：'+str(i))
        workspace = item.get('workspace_bytes',[])
        r.require(len(workspace) == 3 and all(type(w) is int and w >= 0 for w in workspace) and
                  workspace[1] == accepted_workspace,'局部Top-K工作区与已验收版本不符')
    for first,last in ((0,2),(1,4)):
        for name in FILES:
            r.require((path/f'call{first}_{SEQUENCE[first]}'/name).read_bytes() ==
                      (path/f'call{last}_{SEQUENCE[last]}'/name).read_bytes(),'重复查询输出字节不同：'+name)
    return session


def run(inputs,output,runner,vendors,score_project,top_project,merge_project,acceptance,device,k,shards):
    inputs,output,runner,vendors = (p.resolve() for p in (inputs,output,runner,vendors))
    r.require(not output.exists(),'结果目录已存在，请换新名字，不覆盖')
    r.require(runner.is_file() and r.integer(device,0,2**31-1),'实际调用器缺失或设备号非法')
    index,records,snapshot = score.validate_inputs(inputs)
    dimensions(index['num_docs'],k,shards)
    installation = validate_install(vendors,score_project,top_project,merge_project,acceptance)
    workspace = installation['topk_acceptance']['workspace_bytes']
    output.mkdir(parents=True)
    report = {'status':'running','device':device,'k':k,'shards':shards,'index_id':index['index_id'],
              'installation':installation,'input_sha256':snapshot,'runner_sha256':digest(runner),
              'host_source_sha256':{name:digest(ROOT/name) for name in HOST_SOURCES},
              'tool_source_sha256':{name:digest(ROOT/name) for name in TOOL_SOURCES},
              'scope':'Resident index + query IDs through actual Top-K; diagnostic only, no search timing or speedup.'}
    r.write_json(output/'report.json',report)
    try:
        lines = []
        for i,name in enumerate(SEQUENCE):
            path = str(inputs/name)
            r.require(not any(c in path for c in '\t\r\n'),'输入路径含控制字符')
            lines.append(f'call{i}_{name}\t{path}')
        plan = output/'requests.tsv'; plan.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
        command = [str(runner),str(inputs/'index'),str(plan),str(output/'device'),str(device),str(vendors),str(k),str(shards),str(workspace)]
        report['command'] = command; r.write_json(output/'report.json',report)
        env = os.environ.copy()
        env['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(str((vendors/name).resolve(strict=True)) for name in
                ('bm25_query_score','bm25_topk_radix_collect','bm25_topk_merge'))
        print('实际三段设备会话：整库一次上传，A/B/A/全空/B；中间分数不回CPU重传，诊断下载在整条链之后。',flush=True)
        print(json.dumps(command,ensure_ascii=False),flush=True)
        score.execute(command,output/'device.log',env)
        session = check_session(output/'device',index,records,k,shards,device,workspace)
        unused,arrays = r.validate_index(inputs/'index')
        cpu = {}
        for name in set(SEQUENCE):
            unused,queries = r.validate_queries(inputs/name,index)
            cpu[name] = r.diagnostic_scores(index,arrays,queries)
        checks,hashes = [],{}
        for i,name in enumerate(SEQUENCE):
            label = f'call{i}_{name}'; actual = output/'device'/label
            checks.append(check_outputs(inputs/'index',inputs/name,actual,k,shards,cpu[name]))
            for file in FILES: hashes[f'{label}/{file}'] = digest(actual/file)
            print(f'{label}通过：完整分数、局部候选、最终严格排名及同输入FP32 CPU编号核对。',flush=True)
        r.require(all(digest(Path(path)) == value for path,value in snapshot.items()),'运行中输入/核对答案改变')
        r.require(validate_install(vendors,score_project,top_project,merge_project,acceptance) == installation,
                  '运行中实际安装或独立Top-K验收改变')
        r.require(digest(runner) == report['runner_sha256'] and
                  all(digest(ROOT/name) == value for key in ('host_source_sha256','tool_source_sha256') for name,value in report[key].items()),
                  '运行中调用器/源码/验收工具改变')
        report.update(status='resident_chain_device_session_and_all_rankings_passed',session=session,
                      checks=checks,output_sha256=hashes,device_log_sha256=digest(output/'device.log'))
        r.write_json(output/'report.json',report)
        print('常驻索引三段会话五批全部通过，索引不变、重复字节一致；尚未验收正式search性能。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        r.write_json(output/'report.json',report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/query_score_small_input1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'results/resident_chain_host_build/bm25_resident_chain_runner')
    p.add_argument('--vendors',type=Path,default=ROOT/'results/custom_opp/vendors')
    p.add_argument('--score-project',type=Path,default=ROOT/'results/Bm25QueryScore')
    p.add_argument('--topk-project',type=Path,default=ROOT/'results/Bm25TopKRadixCollect')
    p.add_argument('--merge-project',type=Path,default=ROOT/'results/Bm25TopKMerge')
    p.add_argument('--topk-acceptance',type=Path,default=ROOT/'results/radix_collect_topk_validation1')
    p.add_argument('--device',type=int,default=2)
    p.add_argument('--k',type=int,default=3)
    p.add_argument('--shards',type=int,default=2)
    args = p.parse_args()
    try: run(args.input,args.output,args.runner,args.vendors,args.score_project,args.topk_project,args.merge_project,
             args.topk_acceptance,args.device,args.k,args.shards)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不生成CPU替代设备结果或成功报告。',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__': sys.exit(main())
