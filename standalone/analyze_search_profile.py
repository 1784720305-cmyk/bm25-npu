"""Read-only msprof analysis for current resident-index three-stage Search, not old prepared-input chains."""
import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import analyze_pipeline_profile as csvtools, profile_search as application
from standalone import resident_index as r

OPERATORS = ('Bm25QueryScore', 'Bm25TopK', 'Bm25TopKMerge')


def stats(values):
    r.require(values, '没有可统计的正式采集样本')
    ordered = sorted(values)
    return {'count':len(values), 'median_us':float(statistics.median(values)),
            'p95_us':float(ordered[math.ceil(.95*len(values))-1]),
            'min_us':float(ordered[0]), 'max_us':float(ordered[-1])}


def phases(session):
    result = [('cold', None), ('before', 'A'), ('before', 'B')]
    for cycle in range(session['warmup_cycles']):
        result.extend(('warmup', 'B' if slot^(cycle&1) else 'A') for slot in range(2))
    result.extend(('measured', sample['case']) for sample in session['samples'])
    return result+[('after', 'A'), ('after', 'B')]


def analyze(root, run):
    root, run = root.resolve(), run.resolve()
    record, session = application.validate_record(run)
    index, records = record['index'], record['records']
    plan = phases(session)
    summary_path = csvtools.find_one(root, 'op_summary*.csv')
    task_path = csvtools.find_one(root, 'task_time*.csv')
    api_path = csvtools.find_one(root, 'api_statistic*.csv')
    summary = csvtools.read_table(summary_path, ('Device_id', 'Task ID', 'Stream ID', 'OP Type', 'Task Type',
        'Task Start Time(us)', 'Task Duration(us)', 'Block Num', 'Input Shapes'))
    r.require(not any(row['OP Type'].startswith('Bm25') and row['OP Type'] not in OPERATORS for row in summary),
              '混入旧贡献/累加流程或其他BM25算子，不套用新Search顺序')
    own = [row for row in summary if row['OP Type'] in OPERATORS]
    r.require(len(own) == 3*len(plan), f'Search设备任务数量不符：应为{3*len(plan)}，实际{len(own)}；不猜测缺失轮次')
    r.require({row['Device_id'].strip() for row in own} == {str(record['device'])} and
              len({row['Stream ID'].strip() for row in own}) == 1 and all(row['Task Type'] == 'AI_VECTOR_CORE' for row in own),
              '采集设备/单Stream/AI_VECTOR_CORE类型与实际程序不符')
    for row in own:
        row['_start'] = csvtools.number(row['Task Start Time(us)'], 'Task Start Time(us)')
        row['_duration'] = csvtools.number(row['Task Duration(us)'], 'Task Duration(us)')
        row['_blocks'] = int(row['Block Num'])
        r.require(row['_duration'] > 0 and row['_blocks'] > 0, '采集任务耗时/工作块非法')
    own.sort(key=lambda row:row['_start'])
    r.require(len({(row['Stream ID'].strip(), row['Task ID'].strip()) for row in own}) == len(own), '设备任务ID重复')
    tasks = csvtools.read_table(task_path, ('Device_id', 'kernel_name', 'kernel_type', 'stream_id',
        'task_id', 'task_time(us)', 'task_start(us)', 'task_stop(us)'))
    by_task = {}
    for row in tasks:
        key = tuple(row[field].strip() for field in ('Device_id', 'stream_id', 'task_id'))
        r.require(key not in by_task, 'task_time设备/Stream/Task ID重复')
        by_task[key] = row
    tolerance = Decimal('0.002')  # Same actual CSV schema: each time printed to 0.001 us.
    for row in own:
        key = tuple(row[field].strip() for field in ('Device_id', 'Stream ID', 'Task ID'))
        r.require(key in by_task, 'op_summary任务未在task_time找到')
        task = by_task[key]
        r.require(task['kernel_type'] == row['Task Type'] and task['kernel_name'].startswith(row['OP Type']+'_'),
                  '任务名称/类型在两张设备表中不一致')
        start = csvtools.number(task['task_start(us)'], 'task_start(us)')
        stop = csvtools.number(task['task_stop(us)'], 'task_stop(us)')
        duration = csvtools.number(task['task_time(us)'], 'task_time(us)')
        r.require(stop >= start and abs(start-row['_start']) <= tolerance and
                  abs(duration-row['_duration']) <= tolerance and abs(stop-start-duration) <= tolerance,
                  'op_summary与task_time时间交叉不符')
    measured = []; selected = []; measured_index = 0
    for pos, (phase, name) in enumerate(plan):
        chain = own[pos*3:(pos+1)*3]
        r.require(tuple(row['OP Type'] for row in chain) == OPERATORS, '三段Search顺序不符')
        q = record['capacity_queries'] if phase == 'cold' else records[name]['num_queries']
        words = 1 if phase == 'cold' else max(1, records[name]['query_term_count'])
        n, v, m, s, k = index['num_docs'], index['vocab_size'], index['posting_count'], record['shards'], records['A']['k']
        expected_shapes = (((v+1,), (m,), (m,), (n,), (v,), (q+1,), (words,)),
                           ((q*s, n//s),), ((q, s, k), (q, s, k)))
        for row, expected in zip(chain, expected_shapes):
            r.require(csvtools.input_shapes(row['Input Shapes']) == expected, '采集输入形状与新Search实际请求不符：'+row['OP Type'])
        gaps = [chain[i+1]['_start']-chain[i]['_start']-chain[i]['_duration'] for i in range(2)]
        r.require(all(value >= 0 for value in gaps), '同Stream三段设备时间重叠或异常')
        if phase != 'measured':
            continue
        sample = session['samples'][measured_index]
        r.require(sample['case'] == name, '设备正式轮次与主机样本错位')
        total = sum((row['_duration'] for row in chain), Decimal(0))
        span = chain[-1]['_start']+chain[-1]['_duration']-chain[0]['_start']
        host = Decimal(str(sample['call_wall_ms']))*1000
        r.require(host >= span, '设备链跨度大于对应完整Search时间，不能直接匹配；需检查采集或轮次')
        measured.append({'sample_index':measured_index, 'case':name, 'cycle':sample['cycle'],
            'task_ids':[row['Task ID'].strip() for row in chain],
            'operator_us':{row['OP Type']:float(row['_duration']) for row in chain},
            'kernel_sum_us':float(total), 'device_chain_span_us':float(span),
            'intra_chain_gap_sum_us':float(sum(gaps, Decimal(0))), 'profiled_complete_search_us':float(host),
            'outside_device_chain_elapsed_us':float(host-span)})
        for row in chain: row['_case'] = name
        selected.extend(chain); measured_index += 1
    r.require(measured_index == len(session['samples']), '正式设备样本数量与主机会话不符')
    cases = {}
    for name in ('A', 'B'):
        rows = [row for row in measured if row['case'] == name]
        cases[name] = {'operators':{op:{**stats([row['operator_us'][op] for row in rows]),
            'block_counts':sorted({task['_blocks'] for task in selected if task['OP Type'] == op and task['_case'] == name})} for op in OPERATORS},
            **{key:stats([row[key] for row in rows]) for key in ('kernel_sum_us', 'device_chain_span_us',
                'intra_chain_gap_sum_us', 'profiled_complete_search_us', 'outside_device_chain_elapsed_us')}}
    api_rows = csvtools.read_table(api_path, ('Device_id', 'Level', 'API Name', 'Time(us)', 'Count', 'Avg(us)', 'Min(us)', 'Max(us)'))
    api = [{'name':row['API Name'], 'level':row['Level'], 'count':int(row['Count']),
            'whole_application_total_us':float(csvtools.number(row['Time(us)'], 'API Time(us)'))} for row in api_rows]
    api.sort(key=lambda row:row['whole_application_total_us'], reverse=True)
    return {'status':'current_search_profile_rows_and_measured_samples_cross_checked',
        'device':record['device'], 'docs':index['num_docs'], 'queries':records['A']['num_queries'], 'k':records['A']['k'],
        'operators':list(OPERATORS), 'all_search_calls':len(plan), 'own_operator_rows':len(own),
        'excluded_cold_calls':1, 'excluded_before_checkpoints':2, 'excluded_warmup_calls':2*session['warmup_cycles'],
        'excluded_after_checkpoints':2, 'measured_search_calls':len(measured),
        'ignored_other_summary_rows':len(summary)-len(own), 'ignored_other_task_rows':len(tasks)-len(own),
        'cases':cases, 'matched_samples':measured, 'whole_application_api_totals_not_steady_state':api,
        'source_sha256':{str(path):application.digest(path) for path in (summary_path, task_path, api_path, run/'report.json')},
        'scope':'Profiling diagnosis only. Outside-device-chain elapsed includes packing/transfers/preparation/submission/sync/framework/profiler effects; not a pure CPU or memcpy measurement. API totals include initialization/warmup/debug/cleanup. No subtraction of medians, cross-run wall/device subtraction, whole-card utilization or speedup claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile-root', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        r.require(not args.output.exists(), '采集分析报告已存在，请换新名字，不覆盖')
        result = analyze(args.profile_root, args.run)
        args.output.parent.mkdir(parents=True, exist_ok=True); r.write_json(args.output, result)
        print(f'新Search采集交叉核对通过：{result["own_operator_rows"]}条三段算子任务；仅统计正式{result["measured_search_calls"]}次A/B Search。')
        for name, item in result['cases'].items():
            print(name+'：')
            for operator, data in item['operators'].items():
                print(f'  {operator}：中位数={data["median_us"]:.6g}us，p95={data["p95_us"]:.6g}us，工作块={data["block_counts"]}')
            print('  同次设备链跨度中位数={:.6g}us；采集下完整Search={:.6g}us；逐次计算的链外耗时中位数={:.6g}us。'.format(
                item['device_chain_span_us']['median_us'], item['profiled_complete_search_us']['median_us'],
                item['outside_device_chain_elapsed_us']['median_us']))
        print('链外时间不能直接称为拷贝或CPU耗时；API表包含冷启动/预热/调试，不用于稳态占比。采集时间不替换普通正式基准。')
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不猜轮次、不产生成功分析。', file=sys.stderr); return 1
    return 0


if __name__ == '__main__': sys.exit(main())
