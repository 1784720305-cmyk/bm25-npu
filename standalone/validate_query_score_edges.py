"""复用已安装评分算子/调用器，逐组实际执行分块边界会话；失败即停，不测性能。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone import resident_index as r, query_score_edge_fixture as edges, run_query_score as tool


def run(inputs,output,runner,vendor,project,device):
    inputs,output = inputs.resolve(),output.resolve()
    r.require(not output.exists(),'验收输出目录已存在，请换新名字，不覆盖')
    r.require(r.integer(device,0,2**31-1),'设备号非法')
    suite = json.loads((inputs/'suite.json').read_text(encoding='utf-8'))
    r.require(suite == {'kind':edges.VERSION,'sizes':list(edges.SIZES),
                       'status':'host_edge_suite_prepared_device_not_run','device_verified':False},
              '输入不是当前确定性边界套件')
    suite_hash = tool.digest(inputs/'suite.json')
    output.mkdir(parents=True)
    report = {'status':'running','device':device,'kind':edges.VERSION,'cases':[],
              'suite_sha256':suite_hash,'scope':'Score correctness only; no Top-K, search timing or speedup.'}
    try:
        for n in edges.SIZES:
            print(f'开始实际NPU评分边界验收：N={n}，A/B/A/全空/B，同一会话索引只上传一次。',flush=True)
            name = f'n{n}'
            preparation = json.loads((inputs/name/'preparation_report.json').read_text(encoding='utf-8'))
            r.require(preparation.get('fixture_profile') == edges.profile(n),'边界目录与套件指定文章数不符')
            tool.run(inputs/name,output/name,runner,vendor,project,device)
            record = json.loads((output/name/'report.json').read_text(encoding='utf-8'))
            index,unused = r.validate_index(inputs/name/'index')
            r.require(record.get('status') == 'device_score_session_and_independent_formula_checks_passed' and
                      record.get('device') == device and record.get('index_id') == index['index_id'] and index['num_docs'] == n,
                      '子会话未完成实际设备及独立公式验收')
            report['cases'].append({'name':name,'num_docs':n,'index_id':index['index_id'],
                                    'report_sha256':tool.digest(output/name/'report.json')})
            r.write_json(output/'report.json',report)
        r.require(tool.digest(inputs/'suite.json') == suite_hash,'运行中输入套件说明改变')
        report['status'] = 'all_five_device_score_edge_sessions_passed'
        r.write_json(output/'report.json',report)
        print('五组边界会话全部通过：25次实际评分、独立公式及索引复用检查。尚未验收Top-K或search性能。',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        r.write_json(output/'report.json',report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/query_score_edges_input1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner',type=Path,default=ROOT/'results/query_score_host_build/bm25_query_score_runner')
    p.add_argument('--vendor',type=Path,default=ROOT/'results/custom_opp/vendors/bm25_query_score')
    p.add_argument('--project',type=Path,default=ROOT/'results/Bm25QueryScore')
    p.add_argument('--device',type=int,default=2)
    args = p.parse_args()
    try: run(args.input,args.output,args.runner,args.vendor,args.project,args.device)
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止后续设备用例，不生成CPU替代结果或通过汇总。',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__': sys.exit(main())
