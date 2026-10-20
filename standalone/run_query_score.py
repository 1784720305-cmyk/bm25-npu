"""真实ACLNN评分验收：整库上传一次、A/B/A/空/B同进程换查询；不测search、不回退CPU。"""
import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, query_score_fixture as fixture

SOURCES = ('op_host/bm25_query_score.cpp','op_host/bm25_query_score_tiling.h','op_kernel/bm25_query_score.cpp')
HOST_SOURCES = ('npu_prototype/query_score_host/CMakeLists.txt',
                'npu_prototype/query_score_host/query_score_runtime.h',
                'npu_prototype/query_score_host/query_score_runner.cpp')
TOOL_SOURCES = ('standalone/run_query_score.py','standalone/query_score_fixture.py',
                'standalone/resident_index.py','standalone/accumulate_fixture.py','check_result.py')
SEQUENCE = ('A','B','A','empty','B')
# Verbatim parameter names/types confirmed from the installed header on 2026-10-20.
PROTOTYPES = {
    'aclnnBm25QueryScoreGetWorkspaceSize':
        'const aclTensor *termOffsets, const aclTensor *postingDocIds, const aclTensor *postingTf, '
        'const aclTensor *docNorm, const aclTensor *idf, const aclTensor *queryOffsets, '
        'const aclTensor *queryTermIds, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor',
    'aclnnBm25QueryScore':'void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def validate_install(vendor, project):
    vendor, project = vendor.resolve(strict=True), project.resolve(strict=True)
    r.require(vendor.name == 'bm25_query_score','需要独立bm25_query_score安装目录')
    record = json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    r.require(record.get('operator') == 'query_score' and record.get('vendor_name') == vendor.name and
              record.get('compute_unit') == 'ascend950' and set(record.get('source_sha256',{})) == set(SOURCES),
              '需要当前独立950评分工程')
    for name in SOURCES:
        r.require(digest(project/name) == record['source_sha256'][name] == digest(ROOT/'npu_prototype/query_score'/name),
                  '工程与当前评分源码不一致：'+name)
    header = vendor/'op_api/include/aclnn_bm25_query_score.h'
    text = re.sub(r'/\*.*?\*/|//[^\n]*','',header.read_text(encoding='utf-8'),flags=re.S)
    for name, params in PROTOTYPES.items():
        matches = re.findall(r'\baclnnStatus\s+'+name+r'\s*\((.*?)\)\s*;',text,flags=re.S)
        r.require(len(matches) == 1 and re.sub(r'\s+','',matches[0]) == re.sub(r'\s+','',params),
                  '实际生成接口与已核对声明不同：'+name)
    library = (vendor/'op_api/lib/libcust_opapi.so').resolve(strict=True)
    r.require(library.stat().st_size > 0,'安装API库为空')
    for installed in (header,library):
        candidates = list((project/'build_out').rglob(installed.name))
        # CMake may change ELF RPATH between build and install trees; require an exact
        # matching build/package copy, not byte equality of every differently staged copy.
        r.require(candidates and any(digest(p) == digest(installed) for p in candidates),
                  '安装头文件/API与构建产物不符：'+installed.name)
    objects = sorted((vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_query_score').glob('*.o'))
    r.require(objects,'未找到实际安装的950评分设备目标文件')
    object_hashes = {}
    for obj in objects:
        candidates = list((project/'build_out').rglob(obj.name))
        r.require(candidates and all(digest(p) == digest(obj) for p in candidates),'安装设备目标文件与构建产物不符：'+obj.name)
        object_hashes[obj.relative_to(vendor).as_posix()] = digest(obj)
    return {'vendor':str(vendor),'header_sha256':digest(header),'api_sha256':digest(library),
            'project_source_sha256':record['source_sha256'],'device_object_sha256':object_hashes}


def validate_inputs(directory):
    index, unused = r.validate_index(directory/'index')
    source, vocab, documents, cases = fixture.small_definition()
    expected_index, unused = r.build_index(documents,len(vocab))
    r.require(index['index_id'] == expected_index['index_id'],'本阶段只验收原始小语料，索引不匹配')
    preparation = json.loads((directory/'preparation_report.json').read_text(encoding='utf-8'))
    r.require(preparation.get('status') == 'host_query_score_fixtures_checked_device_not_run' and
              preparation.get('index_id') == index['index_id'] and set(preparation.get('cases',{})) == set(cases) and
              preparation.get('source_contract_sha256') == digest(ROOT/'fixtures/compute_contract_small.json'),
              '小样本准备状态或来源不匹配')
    records = {}
    for name, original in cases.items():
        path = directory/name
        record, queries = r.validate_queries(path,index)
        wanted = [[vocab.get(t,len(vocab)) for t in row] for row in original]
        r.require(queries == wanted and record['files'] == preparation['cases'][name]['query_files'],'验收查询不是原A/B/全空输入')
        expected = r.encode(array('f',fixture.independent_scores(source['documents'],original)))
        r.require((path/'expected_scores.bin').read_bytes() == expected and
                  digest(path/'expected_scores.bin') == preparation['cases'][name]['expected_sha256'], '期望文件与独立原始语料公式不符')
        reference = json.loads((path/'reference.json').read_text(encoding='utf-8'))
        r.require(reference.get('index_id') == index['index_id'] and reference.get('query_files') == record['files'] and
                  reference.get('num_queries') == record['num_queries'] and reference.get('num_docs') == index['num_docs'] and
                  reference.get('expected_sha256') == r.digest(expected) and
                  reference.get('reference') == 'independent_original_corpus_formula', '参考说明与独立期望/索引/查询不匹配')
        records[name] = record
    r.require(records['A']['files'] != records['B']['files'] and
              (directory/'A/expected_scores.bin').read_bytes() != (directory/'B/expected_scores.bin').read_bytes(),
              'A/B必须具有不同输入和不同文章分数')
    snapshot = {}
    for parent in (directory/'index',)+(tuple(directory/name for name in cases)):
        for path in parent.iterdir():
            if path.is_file(): snapshot[str(path.resolve())] = digest(path)
    snapshot[str((directory/'preparation_report.json').resolve())] = digest(directory/'preparation_report.json')
    return index, records, snapshot


def check_session(output, index, records):
    device = json.loads((output/'device_session.json').read_text(encoding='utf-8'))
    r.require(device.get('status') == 'device_score_session_completed_reference_checked' and
              type(device.get('index_upload_batches')) is int and device['index_upload_batches'] == 1 and
              type(device.get('index_tensor_h2d_calls')) is int and device['index_tensor_h2d_calls'] == 5 and
              type(device.get('index_upload_bytes')) is int and device['index_upload_bytes'] == index['index_bytes'] and
              device.get('index_readback_byte_identical') is True and device.get('index_device_allocations_reused') is True and
              type(device.get('debug_index_d2h_calls')) is int and device['debug_index_d2h_calls'] == 5 and
              type(device.get('debug_index_d2h_bytes')) is int and device['debug_index_d2h_bytes'] == index['index_bytes'] and
              len(device.get('requests',[])) == len(SEQUENCE), '实际会话记录未证明整库一次上传/回读不变/五次调用')
    for i,(name,call) in enumerate(zip(SEQUENCE,device['requests'])):
        record = records[name]
        logical = record['query_input_bytes']
        physical = logical+(4 if record['query_term_count'] == 0 else 0)
        wanted = {'label':f'call{i}_{name}','queries':record['num_queries'],
                  'logical_word_count':record['query_term_count'],'logical_query_bytes':logical,
                  'query_h2d_bytes':physical,'query_tensor_h2d_calls':2,
                  'score_d2h_bytes':record['num_queries']*index['num_docs']*4,
                  'score_d2h_calls':1,'debug_poison_h2d_calls':1,
                  'debug_poison_h2d_bytes':max(x['num_queries'] for x in records.values())*index['num_docs']*4,
                  'output_poisoned':True,'fresh_executor':True,'full_reference_checked':True,
                  'repeated_query_bits_checked':i in (2,4)}
        r.require(all(type(call.get(key)) is type(value) and call[key] == value for key,value in wanted.items()) and
                  type(call.get('workspace_bytes')) is int and call['workspace_bytes'] >= 0,'会话调用内容/字节/复用核对失败：'+str(i))
    r.require((output/'call0_A_scores.bin').read_bytes() == (output/'call2_A_scores.bin').read_bytes() and
              (output/'call1_B_scores.bin').read_bytes() == (output/'call4_B_scores.bin').read_bytes(),'A/B重复分数字节不一致')
    return device


def execute(command, logfile, env):
    with logfile.open('w',encoding='utf-8',newline='\n') as log:
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                   encoding='utf-8',errors='replace',env=env)
        try:
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            code = process.wait()
            if code: raise subprocess.CalledProcessError(code,command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
            raise
        finally:
            process.stdout.close()


def run(directory, output, runner, vendor, project, device):
    directory, output, runner, vendor, project = (p.resolve() for p in (directory,output,runner,vendor,project))
    r.require(not output.exists(),'输出目录已存在，不覆盖；请用新名字')
    r.require(runner.is_file() and r.integer(device,0,2**31-1),'调用器缺失或卡号非法')
    installation = validate_install(vendor,project)
    index, records, snapshot = validate_inputs(directory)
    output.mkdir(parents=True)
    report = {'status':'running','device':device,'index_id':index['index_id'],'installation':installation,
              'input_sha256':snapshot,'runner_sha256':digest(runner),
              'host_source_sha256':{name:digest(ROOT/name) for name in HOST_SOURCES},
              'tool_source_sha256':{name:digest(ROOT/name) for name in TOOL_SOURCES},
              'scope':'Same-process score correctness only; no Top-K, search timing or speedup claim.'}
    r.write_json(output/'report.json',report)
    try:
        lines = []
        for i,name in enumerate(SEQUENCE):
            path = str(directory/name)
            r.require('\t' not in path and '\n' not in path and '\r' not in path,'查询目录路径含非法控制字符')
            lines.append(f'call{i}_{name}\t{path}')
        plan = output/'requests.tsv'
        plan.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
        command = [str(runner),str(directory/'index'),str(plan),str(output/'device'),str(device),str(vendor)]
        report['command'] = command
        r.write_json(output/'report.json',report)
        env = os.environ.copy()
        env['ASCEND_CUSTOM_OPP_PATH'] = str(vendor) # Child only; no changes to the parent shell/global CANN.
        print('实际同进程NPU评分：整库上传一次，A/B/A/全空/B，全部输出poison及独立公式核对。',flush=True)
        print(json.dumps(command,ensure_ascii=False),flush=True)
        execute(command,output/'device.log',env)
        session = check_session(output/'device',index,records)
        r.require(session.get('device') == device,'实际设备号不符')
        r.require(session.get('host_source_sha256') == report['host_source_sha256'], '调用器编译源码不是当前版本，请重新配置并编译主机程序')
        hashes = {}
        for i,name in enumerate(SEQUENCE):
            actual = output/'device'/f'call{i}_{name}_scores.bin'
            fixture.check(directory/'index',directory/name,actual)
            hashes[actual.name] = digest(actual)
        r.require(all(digest(Path(path)) == value for path,value in snapshot.items()),'运行期间输入/参考文件改变')
        r.require(installation == validate_install(vendor,project) and digest(runner) == report['runner_sha256'] and
                  all(digest(ROOT/name) == value for name,value in report['host_source_sha256'].items()),'运行期间安装库/源码/调用器改变')
        r.require(all(digest(ROOT/name) == value for name,value in report['tool_source_sha256'].items()),'运行期间验收工具源码改变')
        report.update(status='device_score_session_and_independent_formula_checks_passed',
                      session=session,output_sha256=hashes,device_log_sha256=digest(output/'device.log'))
        r.write_json(output/'report.json',report)
        print('五次实际评分和独立公式检查通过；整库一次上传、设备索引回读不变、A/B重复字节一致。尚未验证Top-K或search性能。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error) or type(error).__name__)
        r.write_json(output/'report.json',report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/query_score_small_input1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'results/query_score_host_build/bm25_query_score_runner')
    p.add_argument('--vendor',type=Path,default=ROOT/'results/custom_opp/vendors/bm25_query_score')
    p.add_argument('--project',type=Path,default=ROOT/'results/Bm25QueryScore')
    p.add_argument('--device',type=int,default=2)
    args = p.parse_args()
    try: run(args.input,args.output,args.runner,args.vendor,args.project,args.device)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不生成CPU替代设备分数或成功记录。',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
