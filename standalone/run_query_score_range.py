"""独立连续分区评分验收；复用已编译的原接口调用器，不修改原Search或计时。"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, query_score_fixture as fixture
from standalone import run_query_score as original
from standalone.run_query_score import digest, execute, check_session, validate_inputs

# Both generated declarations were read from the actual range build on 2026-10-20;
# the seven input tensors and both signatures match the accepted original ABI.
SOURCES = original.SOURCES
HOST_SOURCES = original.HOST_SOURCES
PROTOTYPES = original.PROTOTYPES
SEQUENCE = original.SEQUENCE
TOOL_SOURCES = original.TOOL_SOURCES+('standalone/run_query_score_range.py',
                                     'standalone/create_query_score_range.py')
MAX_TERMS = 64


def enforce_query_limits(directory,index,records):
    maxima = {}
    for name,record in records.items():
        current,queries = r.validate_queries(directory/name,index)
        r.require(current == record,'查询记录在支持范围检查时改变：'+name)
        maxima[name] = max(map(len,queries),default=0)
        r.require(maxima[name] <= MAX_TERMS,
                  f'实验版每条查询最多{MAX_TERMS}词：{name}出现{maxima[name]}词；不截断、不启动设备')
    return maxima


def validate_install(vendor, project):
    vendor, project = vendor.resolve(strict=True), project.resolve(strict=True)
    r.require(vendor.name == 'bm25_query_score_range','需要独立bm25_query_score_range安装目录')
    record = json.loads((project/'bm25_project.json').read_text(encoding='utf-8'))
    r.require(record.get('operator') == 'query_score_range' and record.get('vendor_name') == vendor.name and
              record.get('compute_unit') == 'ascend950' and
              type(record.get('max_terms_per_query')) is int and record['max_terms_per_query'] == MAX_TERMS and
              record.get('formula') == 'same_FP32_query_time_formula_and_original_term_order' and set(record.get('source_sha256',{})) == set(SOURCES),
              '需要当前独立950评分工程')
    for name in SOURCES:
        r.require(digest(project/name) == record['source_sha256'][name] == digest(ROOT/'npu_prototype/query_score_range'/name),
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
            'project_metadata_sha256':digest(project/'bm25_project.json'),
            'project_source_sha256':record['source_sha256'],'device_object_sha256':object_hashes}


def run(directory, output, runner, vendor, project, device):
    directory, output, runner, vendor, project = (p.resolve() for p in (directory,output,runner,vendor,project))
    r.require(not output.exists(),'输出目录已存在，不覆盖；请用新名字')
    r.require(runner.is_file() and r.integer(device,0,2**31-1),'调用器缺失或卡号非法')
    installation = validate_install(vendor,project)
    index, records, snapshot = validate_inputs(directory)
    lengths = enforce_query_limits(directory,index,records)
    output.mkdir(parents=True)
    report = {'status':'running','device':device,'index_id':index['index_id'],'installation':installation,
              'input_sha256':snapshot,'runner_sha256':digest(runner),
              'host_source_sha256':{name:digest(ROOT/name) for name in HOST_SOURCES},
              'tool_source_sha256':{name:digest(ROOT/name) for name in TOOL_SOURCES},
              'variant':'contiguous_range_forward_cursor_v1','max_terms_per_query':MAX_TERMS,
              'actual_max_query_terms':lengths,
              'scope':'Range score correctness only; reused unchanged host; no Top-K, Search timing or speedup.'}
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
        print('实验版实际同进程NPU评分（每查询最多64词）：整库上传一次，A/B/A/全空/B，全部输出poison及独立公式核对。',flush=True)
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
        report.update(status='range_device_score_session_and_independent_formula_checks_passed',
                      session=session,output_sha256=hashes,device_log_sha256=digest(output/'device.log'))
        r.write_json(output/'report.json',report)
        print('实验版五次实际评分和独立公式检查通过；整库一次上传、设备索引回读不变、A/B重复字节一致。尚未验证Top-K或search性能。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error) or type(error).__name__)
        r.write_json(output/'report.json',report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/query_score_small_input1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'results/query_score_host_build/bm25_query_score_runner')
    p.add_argument('--vendor',type=Path,default=ROOT/'results/custom_opp/vendors/bm25_query_score_range')
    p.add_argument('--project',type=Path,default=ROOT/'results/Bm25QueryScoreRange')
    p.add_argument('--device',type=int,default=2)
    args = p.parse_args()
    try: run(args.input,args.output,args.runner,args.vendor,args.project,args.device)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不生成CPU替代设备分数或成功记录。',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
