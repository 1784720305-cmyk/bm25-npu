"""只读分析msprof表，核对轮次后排除首次/预热；不执行NPU、不作加速结论。"""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re


OPERATORS = ('Bm25Contribution', 'Bm25Accumulate', 'Bm25TopK')
OUTPUTS = ('contributions.bin', 'scores.bin', 'top_scores.bin', 'top_ids.bin')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def number(value, field):
    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation as error:
        raise ValueError('不能解析数值：' + field) from error
    require(parsed.is_finite() and parsed >= 0, '非法数值：' + field)
    return parsed


def percentile(values, fraction):
    ordered = sorted(values)
    return float(ordered[max(1, math.ceil(len(ordered) * fraction)) - 1])


def statistics(values):
    require(bool(values), '没有有效样本')
    return {'count': len(values), 'median_us': percentile(values, .5),
            'p95_us': percentile(values, .95), 'min_us': float(min(values)),
            'max_us': float(max(values))}


def read_table(path, fields):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(fields) <= set(reader.fieldnames),
                'CSV字段不匹配，停止，不猜单位：' + str(path))
        rows = list(reader)
    require(bool(rows), 'CSV只有表头：' + str(path))
    require(all(None not in row and all(value is not None for value in row.values())
                for row in rows), 'CSV行与表头不匹配：' + str(path))
    return rows


def find_one(root, pattern):
    paths = sorted(root.rglob(pattern))
    require(len(paths) == 1, f'需要唯一的{pattern}，实际{len(paths)}份；请缩小profile-root范围')
    return paths[0]


def analyze(root, run):
    root, run = root.resolve(), run.resolve()
    metadata_path, timing_path = run / 'run_metadata.json', run / 'npu_timing.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    timing = json.loads(timing_path.read_text(encoding='utf-8'))
    require(metadata['status'] == 'device_repeated_call_completed_file_checks_passed',
            '运行记录未通过原来的首末/重复/文件检查')
    require(metadata.get('topk_variant') == 'striped_local_topk_warp_merge_v1',
            '本分析器要求当前已核对的并行Top-K版本')
    require(timing['status'] == 'device_repeated_completed_independent_correctness_not_yet_checked',
            '计时记录状态不匹配')
    require(timing['mode'] == metadata['mode'] == 'resident',
            '当前只分析普通resident；不混入逐段同步或含传输模式')
    for key in ('device', 'queries', 'docs', 'k', 'warmup', 'repeats'):
        require(type(timing[key]) is int and timing[key] == metadata[key], '运行参数不一致：' + key)
    require(timing['device'] >= 0 and all(timing[key] > 0 for key in
            ('queries', 'docs', 'k', 'warmup', 'repeats')), '运行参数非法')
    require(type(timing['additional_plain_warmup']) is int and timing['additional_plain_warmup'] == 1,
            '普通预热次数与已核对程序不一致')
    require(len(timing['samples']) == timing['repeats'], '正式计时样本数量不一致')
    for name in OUTPUTS:
        require(digest(run / name) == metadata['output_sha256'][name], '输出指纹不一致：' + name)
    value_bytes = (run / 'contributions.bin').stat().st_size
    require(value_bytes > 0 and value_bytes % 4 == 0, '贡献输出尺寸非法')
    values = value_bytes // 4
    expected_counts = {'scores.bin': timing['queries'] * timing['docs'],
                       'top_scores.bin': timing['queries'] * timing['k'],
                       'top_ids.bin': timing['queries'] * timing['k']}
    require(all((run / name).stat().st_size == count * 4 for name, count in expected_counts.items()),
            '运行输出尺寸与元信息不符')

    summary_path = find_one(root, 'op_summary*.csv')
    api_path = find_one(root, 'api_statistic*.csv')
    task_path = find_one(root, 'task_time*.csv')
    summary = read_table(summary_path, ('Device_id', 'Task ID', 'Stream ID', 'Op Name',
        'OP Type', 'Task Type', 'Task Start Time(us)', 'Task Duration(us)', 'Block Num', 'Input Shapes'))
    own = [row for row in summary if row['OP Type'] in OPERATORS]
    skipped = 1 + timing['warmup'] + timing['additional_plain_warmup']
    rounds = skipped + timing['repeats']
    require(len(own) == rounds * 3,
            f'BM25设备记录数量不符：期望{rounds * 3}，实际{len(own)}；不猜正式轮次')
    require({row['Device_id'].strip() for row in own} == {str(timing['device'])}, 'CSV设备号与运行记录不同')
    require(len({row['Stream ID'].strip() for row in own}) == 1, '出现多个Stream，不能按三段顺序分轮')
    require(all(row['Task Type'] == 'AI_VECTOR_CORE' for row in own), '任务类型与当前SIMT实现不符')
    for row in own:
        row['_start'] = number(row['Task Start Time(us)'], 'Task Start Time(us)')
        row['_duration'] = number(row['Task Duration(us)'], 'Task Duration(us)')
        require(row['_duration'] > 0, '设备任务耗时为0，不能当成有效采集')
        row['_blocks'] = int(row['Block Num'])
        require(row['_blocks'] > 0, '工作块数非法')
        shapes = {'Bm25Contribution': [values, values, values],
                  'Bm25Accumulate': [values, timing['queries'] * timing['docs'] + 1],
                  'Bm25TopK': [timing['queries'], timing['docs']]}
        require([int(n) for n in re.findall(r'\d+', row['Input Shapes'])] == shapes[row['OP Type']],
                '算子输入形状与本次运行不同：' + row['OP Type'])
    own.sort(key=lambda row: row['_start'])
    require(len({(row['Stream ID'].strip(), row['Task ID'].strip()) for row in own}) == len(own),
            '重复设备任务ID')

    tasks = read_table(task_path, ('Device_id', 'kernel_name', 'kernel_type', 'stream_id',
                                  'task_id', 'task_time(us)', 'task_start(us)', 'task_stop(us)'))
    by_task = {}
    for row in tasks:
        key = (row['Device_id'].strip(), row['stream_id'].strip(), row['task_id'].strip())
        require(key not in by_task, 'task_time出现重复设备/Stream/任务ID')
        by_task[key] = row
    for row in own:
        key = (row['Device_id'].strip(), row['Stream ID'].strip(), row['Task ID'].strip())
        require(key in by_task, 'op_summary任务未在task_time找到')
        task = by_task[key]
        require(task['kernel_name'].startswith(row['OP Type'] + '_') and
                task['kernel_type'] == row['Task Type'], '设备任务名称或类型交叉不符')
        start = number(task['task_start(us)'], 'task_start(us)')
        stop = number(task['task_stop(us)'], 'task_stop(us)')
        duration = number(task['task_time(us)'], 'task_time(us)')
        tolerance = Decimal('0.002')  # CSV时间以0.001us显示，允许末位舍入差。
        require(stop >= start and abs(start - row['_start']) <= tolerance and
                abs(duration - row['_duration']) <= tolerance and
                abs(stop - start - duration) <= tolerance, '两张表的任务时间不一致')

    batches = []
    for index in range(rounds):
        triplet = own[index * 3:index * 3 + 3]
        require(tuple(row['OP Type'] for row in triplet) == OPERATORS, '三段顺序不一致，不能映射正式轮次')
        gaps = [triplet[i + 1]['_start'] - triplet[i]['_start'] - triplet[i]['_duration'] for i in (0, 1)]
        require(all(gap >= 0 for gap in gaps), '同Stream三段出现重叠或时间异常')
        if index < skipped:
            continue
        durations = {row['OP Type']: float(row['_duration']) for row in triplet}
        batches.append({'measured_index': index - skipped,
            'task_ids': [row['Task ID'] for row in triplet], 'durations_us': durations,
            'kernel_sum_us': float(sum((row['_duration'] for row in triplet), Decimal(0))),
            'intra_chain_gaps_us': [float(gap) for gap in gaps],
            'device_chain_span_us': float(triplet[-1]['_start'] + triplet[-1]['_duration'] - triplet[0]['_start'])})
    selected = own[skipped * 3:]
    operator_stats = {name: {
        **statistics([row['_duration'] for row in selected if row['OP Type'] == name]),
        'block_counts': sorted({row['_blocks'] for row in selected if row['OP Type'] == name})}
        for name in OPERATORS}
    api = read_table(api_path, ('Device_id', 'Level', 'API Name', 'Time(us)', 'Count', 'Avg(us)', 'Min(us)', 'Max(us)'))
    api_items = [{'name': row['API Name'], 'level': row['Level'], 'count': int(row['Count']),
        'total_us': float(number(row['Time(us)'], 'API Time(us)')),
        'average_us': float(number(row['Avg(us)'], 'API Avg(us)'))} for row in api]
    return {'status': 'profile_rows_cross_checked_measured_rounds_selected',
        'scope': 'Profiling diagnostic only, not an unprofiled benchmark or speedup verdict. '
                 'API statistics include initialization/warmup/cleanup; do not mix them into steady-state timing. '
                 'AIV instruction ratios and block counts are not whole-device utilization.',
        'device': timing['device'], 'queries': timing['queries'], 'docs': timing['docs'], 'k': timing['k'],
        'source_own_operator_rows': len(own), 'ignored_other_summary_rows': len(summary) - len(own),
        'skipped_bootstrap_rounds': 1, 'skipped_validation_warmups': timing['warmup'],
        'skipped_plain_warmups': timing['additional_plain_warmup'], 'measured_rounds': timing['repeats'],
        'operator_statistics': operator_stats,
        'kernel_sum': statistics([batch['kernel_sum_us'] for batch in batches]),
        'device_chain_span': statistics([batch['device_chain_span_us'] for batch in batches]),
        'intra_chain_gap_sum': statistics([sum(batch['intra_chain_gaps_us']) for batch in batches]),
        'profiled_host_path_median_us': float(number(timing['median_batch_path_ms'], 'host path')) * 1000,
        'api_whole_capture': api_items, 'samples': batches,
        'association': 'Matched by device, Top-K shape, exact launch counts/order and cross-table task IDs/timestamps; source hashes retained.',
        'source_paths': {'profile_root': str(root), 'run': str(run)},
        'source_sha256': {str(path): digest(path) for path in
                          (summary_path, api_path, task_path, metadata_path, timing_path)},
        'run_output_sha256': metadata['output_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-root', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), '报告已存在，请使用新路径')
    record = analyze(Path(args.profile_root), Path(args.run))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'设备表交叉核对通过：{record["source_own_operator_rows"]}条BM25任务；'
          f'排除首次1轮、检查预热{record["skipped_validation_warmups"]}轮、普通预热1轮；'
          f'只统计正式{record["measured_rounds"]}轮。')
    for name, stats in record['operator_statistics'].items():
        print(f'{name}: 中位数={stats["median_us"]:.6g}us，p95={stats["p95_us"]:.6g}us，工作块={stats["block_counts"]}')
    print(f'逐轮三算子设备耗时之和中位数={record["kernel_sum"]["median_us"]:.6g}us；'
          f'设备链首尾跨度中位数={record["device_chain_span"]["median_us"]:.6g}us；'
          f'同一轮链内间隔之和中位数={record["intra_chain_gap_sum"]["median_us"]:.6g}us。')
    print('API整段采集含初始化、预热和释放；DeviceReset/DevMalloc等不计入稳态计算。')
    print('这是已开启采集的诊断；不将各段中位数相加冒充批次中位数，不由vec_ratio推算整卡利用率。')
    print('报告：', output)


if __name__ == '__main__':
    main()
