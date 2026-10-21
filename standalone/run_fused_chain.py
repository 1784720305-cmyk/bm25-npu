"""按已核对实际ABI运行融合/合并；五批各debug1/debug0，正确性诊断，不测Search或回退CPU。"""
import argparse
from array import array
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import resident_index as r, run_query_score as score, query_score_fixture as fixture
from standalone import run_resident_chain as chain, search_scale_fixture as normal
from standalone import create_fused_score_topk as generator, validate_topk_merge as merge

STATUS = 'fused_chain_actual_debug_modes_and_all_rankings_passed'
HOST_SOURCES = ('npu_prototype/fused_chain_host/CMakeLists.txt',
    'npu_prototype/fused_chain_host/fused_runtime.h', 'npu_prototype/fused_chain_host/fused_chain_runner.cpp')
TOOL_SOURCES = tuple(sorted(set(score.TOOL_SOURCES) | set(chain.TOOL_SOURCES) | {
    'standalone/run_fused_chain.py', 'standalone/create_fused_score_topk.py',
    'standalone/search_scale_fixture.py'}))
SEQUENCE = score.SEQUENCE
CANDIDATES = ('partial_scores.bin', 'partial_ids.bin', 'top_scores.bin', 'top_ids.bin')
GROUPS = ('group_scores.bin', 'group_ids.bin')
PROTOTYPES = {
    'aclnnBm25FusedScoreTopKGetWorkspaceSize':
        'const aclTensor *termOffsets, const aclTensor *postingDocIds, const aclTensor *postingTf, '
        'const aclTensor *docNorm, const aclTensor *idf, const aclTensor *queryOffsets, const aclTensor *queryTermIds, '
        'int64_t k, int64_t shards, int64_t debug, const aclTensor *partialScoresOut, const aclTensor *partialIdsOut, '
        'const aclTensor *debugScoresOut, uint64_t *workspaceSize, aclOpExecutor **executor',
    'aclnnBm25FusedScoreTopK': 'void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream'}
digest = score.digest


def dimensions(n, k, shards):
    r.require(r.integer(n,1,r.LIMIT) and r.integer(k,1,32) and type(shards) is int and
        (1 <= shards <= 32 or shards == 64) and n%shards == 0 and k <= n//shards <= 8192,
        '融合要求K1..32，S1..32或64，等长每段K..8192篇；不截断或补文章')


def validate_install(vendor, project):
    vendor, project = vendor.resolve(strict=True), project.resolve(strict=True)
    r.require(vendor.name == generator.VENDOR, '需要实际独立融合vendor')
    data = json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    expected = {'operator':'fused_score_topk', 'vendor_name':generator.VENDOR, 'compute_unit':'ascend950',
        'max_terms_per_query':64, 'max_docs_per_shard':8192, 'max_shards':64,
        'explicit_ub_bytes':50816, 'simt_local_bytes':32768, 'debug_disabled_score_elements':1}
    r.require(all(type(data.get(key)) is type(value) and data[key] == value for key,value in expected.items()) and
        set(data.get('source_sha256',{})) == set(generator.SOURCES) and
        data.get('selection_source_sha256') == digest(ROOT/'npu_prototype/topk_positive_compact/op_kernel/bm25_top_k.cpp'),
        '融合工程/片上容量/复用选择源码身份不符')
    for name in generator.SOURCES:
        r.require(digest(project/name) == data['source_sha256'][name] == digest(ROOT/'npu_prototype/fused_score_topk'/name),
            '实际编译工程不是当前融合源码：'+name)
    header = vendor/'op_api/include/aclnn_bm25_fused_score_top_k.h'
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', header.read_text(encoding='utf-8'), flags=re.S)
    for name, parameters in PROTOTYPES.items():
        found = re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;', text, flags=re.S)
        r.require(len(found) == 1 and re.sub(r'\s+','',found[0]) == re.sub(r'\s+','',parameters),
            '实际生成接口不是2026-10-21已核对声明：'+name)
    objects = chain.artifacts(vendor,project,'bm25_fused_score_top_k',header.name)
    return {'vendor':str(vendor), 'header_sha256':digest(header),
        'api_sha256':digest(vendor/'op_api/lib/libcust_opapi.so'), 'project_metadata_sha256':digest(project/'bm25_project.json'),
        'source_sha256':data['source_sha256'], 'device_object_sha256':objects}


def installed(vendors, project, merge_project):
    return {'fused':validate_install(vendors/generator.VENDOR,project),
        'merge':merge.validate_install(vendors/'bm25_topk_merge',merge_project),
        'merge_device_objects':chain.artifacts(vendors/'bm25_topk_merge',merge_project,'bm25_top_k_merge','aclnn_bm25_top_k_merge.h')}


def validate_inputs(inputs):
    preparation = json.loads((inputs/'preparation_report.json').read_text(encoding='utf-8'))
    validator = normal.validate_inputs if preparation.get('contract') == normal.CONTRACT else score.validate_inputs
    index, records, hashes = validator(inputs)
    index2, values = r.validate_index(inputs/'index'); r.require(index2 == index,'索引在核对时改变')
    references = {}
    for name in records:
        current, queries = r.validate_queries(inputs/name,index)
        r.require(current == records[name] and all(len(row) <= 64 for row in queries),'每查询最多64词，不截断')
        references[name] = r.diagnostic_scores(index,values,queries)
    r.require(sum(records[name]['num_queries']*index['num_docs'] for name in SEQUENCE) <= 16777216,
        '五批完整分数诊断超过16777216元素预算；不启动设备')
    return index, records, hashes, references


def check_output(inputs, label, actual, index, records, k, shards, cpu):
    query = records[label]; n, q, width = index['num_docs'], query['num_queries'], index['num_docs']//shards
    extra = GROUPS if shards == 64 else ()
    on, off = actual/'debug1', actual/'debug0'
    r.require({p.name for p in on.iterdir()} == set(CANDIDATES+extra+('scores.bin',)) and
        {p.name for p in off.iterdir()} == set(CANDIDATES+extra+('debug_placeholder.bin',)), '调试开关输出文件集合不符')
    fixture.check(inputs/'index',inputs/label,on/'scores.bin')
    full_raw, full = chain.read(on/'scores.bin','f',q*n)
    part_raw, parts = chain.read(on/'partial_scores.bin','f',q*shards*k)
    unused, local = chain.read(on/'partial_ids.bin','i',q*shards*k)
    top_raw, top = chain.read(on/'top_scores.bin','f',q*k)
    unused, ids = chain.read(on/'top_ids.bin','i',q*k)
    r.require(all(math.isfinite(x) and x >= 0 for values in (full,parts,top) for x in values), '分数含负值/NaN/Inf')
    error = r.compare(full,cpu)
    def exact_range(start, length, actual_ids, raw, offset):
        wanted = sorted(range(length),key=lambda doc:(-full[start+doc],doc))[:k]
        r.require(list(actual_ids[offset:offset+k]) == wanted, '局部/分组合并/全局排名或同分编号不符')
        expected = b''.join(full_raw[(start+doc)*4:(start+doc+1)*4] for doc in wanted)
        r.require(raw[offset*4:(offset+k)*4] == expected, '候选/合并分数未保留原始设备FP32字节')
        return wanted
    if shards == 64:
        group_raw, group = chain.read(on/'group_scores.bin','f',q*2*k)
        unused, group_ids = chain.read(on/'group_ids.bin','i',q*2*k)
        r.require(all(math.isfinite(x) and x >= 0 for x in group),'分组分数非法')
    for row in range(q):
        for part in range(shards): exact_range(row*n+part*width,width,local,part_raw,(row*shards+part)*k)
        if shards == 64:
            for group in range(2): exact_range(row*n+group*(n//2),n//2,group_ids,group_raw,(row*2+group)*k)
        chosen = exact_range(row*n,n,ids,top_raw,row*k)
        r.require(chosen == sorted(range(n),key=lambda doc:(-cpu[row*n+doc],doc))[:k], '同输入有序FP32 CPU与NPU最终编号不同')
    for name in CANDIDATES+extra: r.require((on/name).read_bytes() == (off/name).read_bytes(),'debug开关导致候选/合并结果变化：'+name)
    r.require((off/'debug_placeholder.bin').read_bytes() == b'\0'*4,'关闭debug时的设备占位不是正零')
    return {'queries':q,'docs':n,'k':k,'shards':shards,'max_abs_same_fp32_error':error,
        'all_local_group_global_ids_strict':True,'selected_device_score_bits_preserved':True,'debug_modes_candidate_topk_byte_equal':True}


def check_session(actual, index, records, device, k, shards):
    session = json.loads((actual/'device_session.json').read_text(encoding='utf-8'))
    stages = 3 if shards == 64 else 2; maxq = max(x['num_queries'] for x in records.values()); n = index['num_docs']
    expected = {'status':'device_fused_chain_debug_modes_and_rankings_checked','device':device,'docs':n,'k':k,'shards':shards,
        'pipeline_stages_per_mode':stages,'intermediate_host_roundtrips':0,'index_upload_batches':1,'index_tensor_h2d_calls':5,
        'index_upload_bytes':index['index_bytes'],'index_readback_byte_identical':True,'index_device_allocations_reused':True,
        'debug_index_d2h_calls':5,'debug_index_d2h_bytes':index['index_bytes'],
        'host_source_sha256':{name:digest(ROOT/name) for name in HOST_SOURCES}}
    r.require(all(type(session.get(key)) is type(value) and session[key] == value for key,value in expected.items()) and
        len(session.get('requests',[])) == 5,'融合会话未证明当前源码/设备/整库一次上传/索引不变')
    poison = 8*maxq*n+8*maxq*shards*k+8*maxq*k+(16*maxq*k if shards == 64 else 0)+64
    for ordinal,(label,call) in enumerate(zip(SEQUENCE,session['requests'])):
        record = records[label]; q, count = record['num_queries'], record['query_term_count']
        expected = {'label':f'call{ordinal}_{label}','queries':q,'logical_word_count':count,'mode_order':[1,0],
            'query_h2d_bytes_per_mode':record['query_input_bytes']+(4 if count == 0 else 0),'query_tensor_h2d_calls_per_mode':2,
            'fresh_executors_per_mode':stages,'chain_synchronizations_per_mode':1,'tensor_cleanup_synchronizations_per_mode':1,
            'debug_poison_h2d_calls_per_mode':8 if shards == 64 else 6,'debug_poison_h2d_bytes_per_mode':poison,
            'placeholder_guard_bytes':4*(maxq*n+16),'debug_enabled_score_elements':q*n,'debug_disabled_score_elements':1,
            'outputs_poisoned':True,'logical_tail_guards_unchanged':True,'disabled_full_buffer_unchanged':True,
            'disabled_placeholder_guard_unchanged':True,'debug_modes_all_candidate_group_topk_bits_equal':True,
            'full_reference_checked':True,'exact_device_ranking_checked':True,'repeated_all_output_bits_checked':ordinal in (2,4)}
        r.require(all(type(call.get(key)) is type(value) and call[key] == value for key,value in expected.items()),'融合模式/保护/传输记录不符：'+str(ordinal))
        workspace = call.get('workspace_bytes_by_mode',[])
        r.require(len(workspace) == 2 and all(isinstance(row,list) and len(row) == stages and
            all(type(value) is int and value >= 0 for value in row) for row in workspace),'工作区记录非法')
    for first,last in ((0,2),(1,4)):
        for mode in ('debug1','debug0'):
            for path in (actual/f'call{first}_{SEQUENCE[first]}'/mode).iterdir():
                r.require(path.read_bytes() == (actual/f'call{last}_{SEQUENCE[last]}'/mode/path.name).read_bytes(),'重复查询输出字节改变')
    return session


def run(inputs, output, runner, vendors, project, merge_project, device=2, k=3, shards=2):
    inputs, output, runner, vendors, project, merge_project = (p.resolve() for p in (inputs,output,runner,vendors,project,merge_project))
    r.require(not output.exists() and runner.is_file() and r.integer(device,0,r.LIMIT),'结果已存在/程序缺失/设备号非法')
    index, records, snapshot, references = validate_inputs(inputs); dimensions(index['num_docs'],k,shards)
    installation = installed(vendors,project,merge_project)
    output.mkdir(parents=True)
    report = {'status':'running','device':device,'k':k,'shards':shards,'index':index,'records':records,'input_sha256':snapshot,
        'installation':installation,'runner_sha256':digest(runner),'host_source_sha256':{name:digest(ROOT/name) for name in HOST_SOURCES},
        'tool_source_sha256':{name:digest(ROOT/name) for name in TOOL_SOURCES},
        'scope':'Diagnostic only: actual debug1/debug0, poisoned outputs and guards; no complete Search timing or speedup.'}
    r.write_json(output/'report.json',report)
    try:
        lines = []
        for ordinal,name in enumerate(SEQUENCE):
            path = str(inputs/name); r.require(not any(c in path for c in '\t\r\n'),'输入路径含控制字符')
            lines.append(f'call{ordinal}_{name}\t{path}')
        plan = output/'requests.tsv'; plan.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
        report['plan_sha256'] = digest(plan)
        command = [str(runner),str(inputs/'index'),str(plan),str(output/'device'),str(device),str(vendors),str(k),str(shards)]
        report['command'] = command; r.write_json(output/'report.json',report)
        env = os.environ.copy(); env['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(str((vendors/name).resolve(strict=True)) for name in (generator.VENDOR,'bm25_topk_merge'))
        print('实际融合设备会话：索引一次上传，A/B/A/空/B，每批debug1后debug0；完整公式/严格排名/关闭debug保护。',flush=True)
        print(json.dumps(command,ensure_ascii=False),flush=True)
        score.execute(command,output/'device.log',env)
        session = check_session(output/'device',index,records,device,k,shards)
        verified, hashes = [], {}
        for ordinal,name in enumerate(SEQUENCE):
            label = f'call{ordinal}_{name}'; directory = output/'device'/label
            verified.append(check_output(inputs,name,directory,index,records,k,shards,references[name]))
            for path in sorted(directory.rglob('*')):
                if path.is_file(): hashes[path.relative_to(output/'device').as_posix()] = digest(path)
            print(label+'：全部分数、局部/分组/最终编号及FP32字节、debug开关核对通过。',flush=True)
        r.require(all(digest(Path(path)) == sha for path,sha in snapshot.items()) and digest(plan) == report['plan_sha256'], '运行中输入/查询计划改变')
        r.require(installed(vendors,project,merge_project) == installation and digest(runner) == report['runner_sha256'] and
            all(digest(ROOT/name) == sha for key in ('host_source_sha256','tool_source_sha256') for name,sha in report[key].items()),'运行中实际算子/调用器/源码改变')
        report.update(status=STATUS,session=session,checks=verified,output_sha256=hashes,device_log_sha256=digest(output/'device.log'))
        r.write_json(output/'report.json',report)
        print('五批共10条实际模式链及全部核对通过；仍是正确性诊断，不代表完整Search性能达标。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        r.write_json(output/'report.json',report); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/query_score_small_input1'); p.add_argument('--output',type=Path,required=True)
    for name, value in (('runner','results/fused_chain_host_build/bm25_fused_chain_runner'),('vendors','results/custom_opp/vendors'),
                        ('project','results/Bm25FusedScoreTopK'),('merge-project','results/Bm25TopKMerge')):
        p.add_argument('--'+name,type=Path,default=ROOT/value)
    for name,value in (('device',2),('k',3),('shards',2)): p.add_argument('--'+name,type=int,default=value)
    a = p.parse_args()
    try:
        r.require(sys.platform == 'linux','实际调用器必须在A5 Linux运行')
        run(a.input,a.output,a.runner,a.vendors,a.project,a.merge_project,a.device,a.k,a.shards)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不生成CPU替代输出或成功性能结论。',file=sys.stderr); return 1
    return 0


if __name__ == '__main__': sys.exit(main())
