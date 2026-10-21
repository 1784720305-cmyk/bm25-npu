"""真实执行三组连续分区/64词边界；复用原调用器，不测Top-K或Search性能。"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, query_score_fixture as fixture
from standalone import run_query_score_range as score, query_score_range_boundary_fixture as boundaries
from standalone.run_query_score_range import digest,execute,check_session,enforce_query_limits,validate_install

HOST_SOURCES=score.HOST_SOURCES
TOOL_SOURCES=score.TOOL_SOURCES+('standalone/query_score_range_boundary_fixture.py',
                               'standalone/validate_query_score_range_boundaries.py')
MAX_TERMS=score.MAX_TERMS
SEQUENCE=score.SEQUENCE
validate_inputs=boundaries.validate_case


def run(directory, output, runner, vendor, project, device):
    directory, output, runner, vendor, project = (p.resolve() for p in (directory,output,runner,vendor,project))
    r.require(not output.exists(),'输出目录已存在，不覆盖；请用新名字')
    r.require(runner.is_file() and r.integer(device,0,2**31-1),'调用器缺失或卡号非法')
    installation = validate_install(vendor,project)
    index, records, snapshot = validate_inputs(directory)
    lengths = enforce_query_limits(directory,index,records)
    rejected = boundaries.check_rejection(directory,index)
    output.mkdir(parents=True)
    report = {'status':'running','device':device,'index_id':index['index_id'],'installation':installation,
              'input_sha256':snapshot,'runner_sha256':digest(runner),
              'host_source_sha256':{name:digest(ROOT/name) for name in HOST_SOURCES},
              'tool_source_sha256':{name:digest(ROOT/name) for name in TOOL_SOURCES},
              'variant':'contiguous_range_forward_cursor_v1','max_terms_per_query':MAX_TERMS,
              'actual_max_query_terms':lengths,'host_rejection':rejected,
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
        report.update(status='range_boundary_device_score_session_and_independent_formula_checks_passed',
                      session=session,output_sha256=hashes,device_log_sha256=digest(output/'device.log'))
        r.write_json(output/'report.json',report)
        print('实验版五次实际评分和独立公式检查通过；整库一次上传、设备索引回读不变、A/B重复字节一致。尚未验证Top-K或search性能。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error) or type(error).__name__)
        r.write_json(output/'report.json',report)
        raise


def run_suite(inputs,output,runner,vendor,project,device):
    inputs,output=inputs.resolve(),output.resolve()
    r.require(not output.exists(),'输出目录已存在，不覆盖')
    r.require(r.integer(device,0,2**31-1),'设备号非法')
    suite=json.loads((inputs/'suite.json').read_text(encoding='utf-8'))
    r.require(suite=={'kind':boundaries.VERSION,'cases':list(boundaries.CASES),
        'status':'host_range_boundary_suite_prepared_device_not_run','device_verified':False},'输入不是指定边界套件')
    suite_hash=digest(inputs/'suite.json')
    output.mkdir(parents=True)
    report={'status':'running','kind':boundaries.VERSION,'device':device,'cases':[],
            'suite_sha256':suite_hash,'scope':'Score-only device correctness; host rejects 65 terms; no performance claim.'}
    r.write_json(output/'report.json',report)
    try:
        for name,(n,q) in boundaries.CASES.items():
            print(f'开始{name}：N={n}，A查询数={q}，64词、A/B/A/空/B。',flush=True)
            preparation=json.loads((inputs/name/'preparation_report.json').read_text(encoding='utf-8'))
            r.require(preparation.get('case')==name,'目录与套件指定用例不符')
            run(inputs/name,output/name,runner,vendor,project,device)
            record=json.loads((output/name/'report.json').read_text(encoding='utf-8'))
            r.require(record.get('status')=='range_boundary_device_score_session_and_independent_formula_checks_passed' and
                record['session']['requests'][0]['queries']==q and record['actual_max_query_terms']['A']==64 and
                record['actual_max_query_terms']['B']==64,'边界子会话未完整通过')
            report['cases'].append({'name':name,'num_docs':n,'queries_A':q,
                'report_sha256':digest(output/name/'report.json')})
            r.write_json(output/'report.json',report)
        r.require(digest(inputs/'suite.json')==suite_hash,'套件说明在运行期间改变')
        report['status']='all_three_range_boundary_device_sessions_passed'
        r.write_json(output/'report.json',report)
        print('三组15次实际评分/64词/重复/全空/索引不变检查通过；65词仅验证主机拒绝。未验收Top-K、完整Search或性能。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        r.write_json(output/'report.json',report); raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/query_score_range_boundaries_input1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'results/query_score_host_build/bm25_query_score_runner')
    p.add_argument('--vendor',type=Path,default=ROOT/'results/custom_opp/vendors/bm25_query_score_range')
    p.add_argument('--project',type=Path,default=ROOT/'results/Bm25QueryScoreRange')
    p.add_argument('--device',type=int,default=2)
    args=p.parse_args()
    try: run_suite(args.input,args.output,args.runner,args.vendor,args.project,args.device)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止后续用例，不生成CPU替代结果或通过汇总。',file=sys.stderr); return 1
    return 0


if __name__=='__main__': sys.exit(main())
