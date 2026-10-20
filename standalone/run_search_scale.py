"""Run the accepted CPU/NPU raw-ID Search executables on one normal corpus; correctness only."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import search_scale_fixture as fixture, run_cpu_index_search as cpu
from standalone import run_search as npu, resident_index as r, run_resident_chain as chain

TOOL_SOURCES = tuple(sorted(set(cpu.TOOL_SOURCES) | {
    'standalone/search_scale_fixture.py', 'standalone/run_search_scale.py'}))
HOST_SOURCES = tuple(sorted(set(cpu.HOST_SOURCES) | set(npu.HOST_SOURCES)))
STATUS = 'normal_scale_cpu_npu_raw_id_search_and_all_checks_passed'
digest = chain.digest


def unchanged(mapping, message):
    r.require(all(digest(Path(path)) == value for path, value in mapping.items()), message)


def run(inputs, output, cpu_runner, npu_runner, vendors, score_project, top_project, merge_project,
        acceptance, threads=8, cpu_shards=8, shards=32, device=2, capq=None, capwords=None):
    inputs, output, cpu_runner, npu_runner, vendors = (p.resolve() for p in (inputs, output, cpu_runner, npu_runner, vendors))
    r.require(not output.exists(), '规模Search结果目录已存在，请换新名字，不覆盖')
    r.require(cpu_runner.is_file() and npu_runner.is_file() and r.integer(device, 0, 2147483647) and
              r.integer(threads, 1, 65536), '实际CPU/NPU程序缺失、卡号或线程数非法')
    print('独立核对正常语料、全量索引和A/B/空查询；这些主机检查不计入Search。', flush=True)
    index, records, snapshot = fixture.validate_inputs(inputs)
    k = records['A']['k']; n = index['num_docs']
    chain.dimensions(n, k, shards)
    r.require(r.integer(cpu_shards, 1, min(threads, n)), 'CPU文章分块应在1..min(线程数,N)')
    capq = max(x['num_queries'] for x in records.values()) if capq is None else capq
    capwords = max(x['query_term_count'] for x in records.values()) if capwords is None else capwords
    r.require(r.integer(capq, 1, 10000) and r.integer(capwords, 0, 16777216) and capq*n <= 2147483647 and
              all(x['num_queries'] <= capq and x['query_term_count'] <= capwords for x in records.values()),
              'CPU/NPU共同声明的查询容量不足或越界')
    installation = chain.validate_install(vendors, score_project, top_project, merge_project, acceptance)
    output.mkdir(parents=True)
    report = {'status':'running', 'index_id':index['index_id'], 'k':k, 'device':device, 'shards':shards,
              'requested_threads':threads, 'cpu_document_shards':cpu_shards,
              'capacity_queries':capq, 'capacity_words':capwords, 'input_sha256':snapshot,
              'installation':installation, 'source_is_real_corpus':False,
              'cpu_runner_sha256':digest(cpu_runner), 'npu_runner_sha256':digest(npu_runner),
              'host_source_sha256':{name:digest(ROOT/name) for name in HOST_SOURCES},
              'tool_source_sha256':{name:digest(ROOT/name) for name in TOOL_SOURCES},
              'scope':'Normal synthetic corpus; five diagnostic Search calls per backend, same logical public boundary; no steady-state benchmark or speedup claim.'}
    r.write_json(output/'report.json', report)
    try:
        raw = output/'raw_queries'; raw.mkdir(); raw_hashes = {}
        for name in records:
            unused, queries = r.validate_queries(inputs/name, index)
            file = raw/(name+'.txt'); npu.write_queries(file, queries); raw_hashes[str(file)] = digest(file)
        lines = []
        for i, name in enumerate(chain.SEQUENCE):
            path = str(raw/(name+'.txt'))
            r.require(not any(c in path for c in '\r\n\t'), '原始词编号路径含控制字符')
            lines.append(f'call{i}_{name}\t{path}')
        plan = output/'requests.tsv'; plan.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
        raw_hashes[str(plan)] = digest(plan)
        command = [str(cpu_runner), str(inputs/'index'), str(plan), str(output/'cpu'), str(threads),
                   str(k), str(cpu_shards), str(capq), str(capwords), 'raw_query_ids_v1']
        report.update(cpu_command=command, raw_query_sha256=raw_hashes)
        r.write_json(output/'report.json', report)
        print('实际CPU Search：原始词编号→常驻索引→FP32算分→CPU Top-K。', flush=True)
        print(json.dumps(command, ensure_ascii=False), flush=True)
        chain.score.execute(command, output/'cpu.log', os.environ.copy())
        cpu_session = cpu.check_session(output/'cpu', index, records, threads, k, cpu_shards, capq, capwords)
        checks = []
        for i, name in enumerate(chain.SEQUENCE):
            print(f'核对CPU call{i}_{name}完整分数及严格排名。', flush=True)
            checks.append({'cpu':cpu.check_cpu_outputs(inputs/'index', inputs/name, output/'cpu'/f'call{i}_{name}', k)})
        report.update(cpu_session=cpu_session, checks=checks, cpu_correctness_passed=True)
        r.write_json(output/'report.json', report)
        unchanged(snapshot, 'CPU运行期间输入改变')
        unchanged(raw_hashes, 'CPU运行期间原始词编号或计划改变')
        workspace = installation['topk_acceptance']['workspace_bytes']
        command = [str(npu_runner), str(inputs/'index'), str(plan), str(output/'device'), str(device), str(vendors),
                   str(k), str(shards), str(capq), str(capwords), str(workspace)]
        report['npu_command'] = command; r.write_json(output/'report.json', report)
        env = os.environ.copy()
        env['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(str((vendors/name).resolve(strict=True)) for name in
            ('bm25_query_score', 'bm25_topk_radix_collect', 'bm25_topk_merge'))
        print('实际NPU Search：索引一次上传，之后五次只上传查询；返回CPU Top-K。', flush=True)
        print(json.dumps(command, ensure_ascii=False), flush=True)
        chain.score.execute(command, output/'device.log', env)
        session = npu.check_session(output/'device', index, records, device, k, shards, capq, capwords, workspace)
        for i, name in enumerate(chain.SEQUENCE):
            label = f'call{i}_{name}'; print(label+'：核对NPU完整分数/候选/严格排名及CPU直接结果。', flush=True)
            actual = output/'device'/label
            checks[i]['npu'] = chain.check_outputs(inputs/'index', inputs/name, actual, k, shards)
            checks[i]['cross'] = cpu.compare_npu(output/'cpu'/label, actual, records[name]['num_queries'], n, k)
        unchanged(snapshot, '运行期间语料/索引/查询/参考改变')
        unchanged(raw_hashes, '运行期间原始查询或计划改变')
        r.require(chain.validate_install(vendors, score_project, top_project, merge_project, acceptance) == installation and
                  digest(cpu_runner) == report['cpu_runner_sha256'] and digest(npu_runner) == report['npu_runner_sha256'] and
                  all(digest(ROOT/name) == value for key in ('host_source_sha256', 'tool_source_sha256')
                      for name, value in report[key].items()), '运行期间安装库/可执行程序/源码改变')
        outputs = {backend+'/'+f'call{i}_{name}/'+file:digest(output/backend/f'call{i}_{name}'/file)
                   for backend, files in (('cpu', cpu.FILES), ('device', chain.FILES))
                   for i, name in enumerate(chain.SEQUENCE) for file in files}
        report.update(status=STATUS, npu_session=session, checks=checks, output_sha256=outputs,
                      cpu_log_sha256=digest(output/'cpu.log'), device_log_sha256=digest(output/'device.log'))
        r.write_json(output/'report.json', report)
        print(f'正常规模验收通过：N={n}，A/B各Q={records["A"]["num_queries"]}，K={k}；两端各五次真实Search及直接结果核对通过。', flush=True)
        print('这里只记录单次完整边界诊断，不计算加速比；后续另做预热/多轮/合理CPU配置的正式性能测试。', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        r.write_json(output/'report.json', report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-runner', type=Path, default=ROOT/'results/cpu_index_search_build/bm25_cpu_index_search_runner')
    p.add_argument('--npu-runner', type=Path, default=ROOT/'results/search_host_build/bm25_search_runner')
    p.add_argument('--vendors', type=Path, default=ROOT/'results/custom_opp/vendors')
    p.add_argument('--score-project', type=Path, default=ROOT/'results/Bm25QueryScore')
    p.add_argument('--topk-project', type=Path, default=ROOT/'results/Bm25TopKRadixCollect')
    p.add_argument('--merge-project', type=Path, default=ROOT/'results/Bm25TopKMerge')
    p.add_argument('--topk-acceptance', type=Path, default=ROOT/'results/radix_collect_topk_validation1')
    p.add_argument('--threads', type=int, default=8); p.add_argument('--cpu-document-shards', type=int, default=8)
    p.add_argument('--shards', type=int, default=32); p.add_argument('--device', type=int, default=2)
    p.add_argument('--max-queries', type=int); p.add_argument('--max-words', type=int)
    args = p.parse_args()
    try:
        run(args.input, args.output, args.cpu_runner, args.npu_runner, args.vendors, args.score_project,
            args.topk_project, args.merge_project, args.topk_acceptance, args.threads, args.cpu_document_shards,
            args.shards, args.device, args.max_queries, args.max_words)
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不用CPU输出替代NPU，不生成成功汇总。', file=sys.stderr); return 1
    return 0


if __name__ == '__main__': sys.exit(main())
