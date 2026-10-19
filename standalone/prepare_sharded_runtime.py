"""为四段实验建立独立vendor别名，核对实际工程/头文件/库；不编译或运行设备。"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.prepare_topk_parallel_runtime import prototypes, scratch_bytes
from standalone.validate_topk_merge import digest, dimensions, validate_install, write_json
from standalone import validate_topk_warp as warp
from standalone import validate_topk_radix as radix
from standalone import validate_topk_radix_zero as radix_zero

VARIANT = 'document_sharded_topk_v1'
WARP_VARIANT = 'document_sharded_warp_sort32_topk_v1'
RADIX_VARIANT = 'document_sharded_exact_radix_topk_v1'
RADIX_ZERO_VARIANT = 'document_sharded_exact_radix_zero_aggregate_topk_v1'
RADIX_VARIANTS = (RADIX_VARIANT, RADIX_ZERO_VARIANT)
ALIASES = ('bm25', 'bm25_accumulate', 'bm25_topk', 'bm25_topk_merge')
HEADERS = ('aclnn_bm25_contribution.h', 'aclnn_bm25_accumulate.h', 'aclnn_bm25_top_k.h', 'aclnn_bm25_top_k_merge.h')


def installed(vendors, parallel_project, merge_project, implementation='parallel', warp_validation=None):
    vendors, parallel_project, merge_project = (p.resolve() for p in (vendors, parallel_project, merge_project))
    if implementation not in ('parallel', 'warp', 'radix', 'radix_zero'):
        raise ValueError('未知局部Top-K实现')
    operator = {'parallel': 'topk_parallel', 'warp': 'topk_warp', 'radix': 'topk_radix', 'radix_zero': 'topk_radix_zero'}[implementation]
    project = json.loads((parallel_project / 'bm25_project.json').read_text(encoding='utf-8'))
    if project['operator'] != operator or project['vendor_name'] != 'bm25_' + operator:
        raise ValueError('需要所选局部Top-K实现的实际工程/vendor，不能自动替换版本')
    for name in ('op_host/bm25_top_k.cpp', 'op_host/bm25_top_k_tiling.h', 'op_kernel/bm25_top_k.cpp'):
        if digest(parallel_project / name) != project['source_sha256'][name] or digest(parallel_project / name) != digest(ROOT / 'npu_prototype' / operator / name):
            raise ValueError(f'局部Top-K工程与当前源码不一致：{name}')
    topk_vendor = vendors / ('bm25_' + operator)
    # Radix has its own confirmed actual signature/acceptance gate; no legacy Top-K installation required.
    if implementation not in ('radix', 'radix_zero') and prototypes(vendors / 'bm25_topk/op_api/include/aclnn_bm25_top_k.h') != prototypes(topk_vendor / 'op_api/include/aclnn_bm25_top_k.h'):
        raise ValueError('新旧Top-K实际接口不一致')
    accepted = None
    if implementation == 'warp':
        installation = warp.validate_install(topk_vendor, parallel_project)
        accepted = validate_warp_acceptance(warp_validation, installation)
    elif implementation in ('radix', 'radix_zero'):
        checker = radix_zero if implementation == 'radix_zero' else radix
        installation = checker.validate_install(topk_vendor, parallel_project)
        accepted = validate_radix_acceptance(warp_validation, installation, checker)
    merge = validate_install(vendors / 'bm25_topk_merge', merge_project)
    targets = dict(zip(ALIASES, (vendors / 'bm25', vendors / 'bm25_accumulate', topk_vendor, vendors / 'bm25_topk_merge')))
    libraries = {}
    for (alias, destination), header in zip(targets.items(), HEADERS):
        destination = destination.resolve(strict=True)
        library = destination / 'op_api/lib/libcust_opapi.so'
        libraries[alias] = {'actual_directory': str(destination), 'api_sha256': digest(library),
                            'header_sha256': digest(destination / 'op_api/include' / header)}
    if implementation == 'warp':
        return {'variant': WARP_VARIANT, 'api_libraries': libraries, 'topk_implementation': 'warp',
                'vendors': str(vendors), 'warp_project': str(parallel_project), 'merge_project': str(merge_project),
                'warp_installation': installation, 'warp_validation': accepted, 'merge_installation': merge,
                'scope': 'four installed vendors and prior warp unit acceptance checked; four-stage execution not yet verified'}
    if implementation in ('radix', 'radix_zero'):
        return {'variant': RADIX_ZERO_VARIANT if implementation == 'radix_zero' else RADIX_VARIANT,
                'api_libraries': libraries, 'topk_implementation': 'radix',
                'vendors': str(vendors), 'radix_project': str(parallel_project), 'merge_project': str(merge_project),
                'radix_installation': installation, 'radix_validation': accepted, 'merge_installation': merge,
                'scope': ('four installed vendors and all 23 prior zero-aggregate Radix alternating-input unit cases checked; four-stage execution not yet verified'
                          if implementation == 'radix_zero' else
                          'four installed vendors and all 19 prior Radix alternating-input unit cases checked; four-stage execution not yet verified')}
    return {'variant': VARIANT, 'api_libraries': libraries,
            'vendors': str(vendors), 'parallel_project': str(parallel_project), 'merge_project': str(merge_project),
            'parallel_source_sha256': project['source_sha256'], 'merge_installation': merge,
            'scope': 'four installed vendors checked; source/build/install provenance also needs actual logs; not device execution'}


def validate_warp_acceptance(directory, installation):
    if directory is None:
        raise ValueError('warp版本须提供已通过的独立设备验收目录')
    directory = directory.resolve(strict=True)
    path = directory / 'report.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    if (record.get('status') != 'warp_topk_unit_device_calls_and_full_sort_checks_passed' or
            record.get('variant') != warp.VARIANT or record.get('installation') != installation or
            type(record.get('calls_per_case')) is not int or record['calls_per_case'] < 2 or
            len(record.get('cases', [])) != len(warp.SPECS)):
        raise ValueError('warp独立验收未完成，或与本次实际安装不一致')
    for index, (case, spec) in enumerate(zip(record['cases'], warp.SPECS)):
        q, docs, k = spec
        if case.get('status') != 'passed' or tuple(case.get(key) for key in ('queries', 'docs', 'k')) != spec:
            raise ValueError('warp验收用例缺失、次序或尺寸不符')
        root = directory / f'case{index}_q{q}_d{docs}_k{k}'
        if digest(root / 'input/scores.bin') != case['input_sha256'] or digest(root / 'device.log') != case['device_log_sha256']:
            raise ValueError('warp验收输入或设备日志指纹改变')
        for name in ('top_scores.bin', 'top_ids.bin'):
            if digest(root / 'actual' / name) != case['output_sha256'][name]:
                raise ValueError('warp验收输出指纹改变')
        validation = json.loads((root / 'actual/device_validation.json').read_text(encoding='utf-8'))
        scratch = warp.scratch_bytes(q, docs, k)
        expected = {'status': 'device_calls_completed_full_sort_not_yet_checked', 'queries': q, 'docs': docs,
                    'k': k, 'device': record['device'], 'validated_calls': record['calls_per_case'],
                    'workspace_bytes': record['system_bytes'] + scratch, 'user_scratch_bytes': scratch,
                    'system_bytes': record['system_bytes'], 'every_call_poisoned': True}
        if validation != expected or case.get('validation') != expected:
            raise ValueError('warp设备验收记录与报告不一致')
        with redirect_stdout(io.StringIO()):
            warp.check(root / 'input', root / 'actual/top_scores.bin', root / 'actual/top_ids.bin')
    return {'directory': str(directory), 'report_sha256': digest(path)}


def validate_radix_acceptance(directory, installation, checker=radix):
    if directory is None:
        raise ValueError('radix版本须提供所选版本全部独立设备验收目录，冒烟报告不够')
    directory = directory.resolve(strict=True)
    path = directory / 'report.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    if (record.get('status') != 'radix_all_unit_device_and_full_sort_passed' or
            record.get('suite') != 'all' or record.get('variant') != checker.VARIANT or
            record.get('installation') != installation or type(record.get('calls_per_case')) is not int or
            not 5 <= record['calls_per_case'] <= 100 or type(record.get('device')) is not int or
            not 0 <= record['device'] <= 2147483647 or len(record.get('cases', [])) != len(checker.SPECS)):
        raise ValueError('Radix完整验收未完成，或与本次实际安装不一致')
    checker_path = 'standalone/' + checker.__name__.rsplit('.', 1)[-1] + '.py'
    host_sources = (checker.HOST_SOURCE_PATHS if checker is radix_zero else
                    tuple('npu_prototype/topk_radix_host/' + name for name in checker.HOST_FILES))
    expected_tools = {checker_path, 'standalone/topk_fixture.py'} | set(host_sources)
    if set(record.get('tool_source_sha256', {})) != expected_tools or any(
            digest(ROOT / name) != record['tool_source_sha256'][name] for name in expected_tools):
        raise ValueError('Radix独立验收使用的调用器/检查器源码已改变')
    workspaces = set()
    for index, (case, spec) in enumerate(zip(record['cases'], checker.SPECS)):
        q, docs, k = spec
        if case.get('status') != 'passed' or tuple(case.get(key) for key in ('queries', 'docs', 'k')) != spec:
            raise ValueError('Radix验收用例缺失、次序或尺寸不符')
        root = directory / f'case{index}_q{q}_d{docs}_k{k}'
        command = case.get('command', [])
        if (len(command) != 9 or digest(Path(command[0])) != record['runner_sha256'] or
                command[1:] != [str(root), str(root / 'actual'), str(q), str(docs), str(k),
                                 str(record['device']), installation['api_library'], str(record['calls_per_case'])]):
            raise ValueError('Radix实际调用器指纹或命令与验收报告不符')
        if digest(root / 'device.log') != case['device_log_sha256']:
            raise ValueError('Radix验收设备日志指纹改变')
        validation = json.loads((root / 'actual/device_validation.json').read_text(encoding='utf-8'))
        checker.validate_record(validation, q, docs, k, record['device'], record['calls_per_case'])
        if validation != case.get('validation'):
            raise ValueError('Radix设备验收记录与总报告不同')
        workspaces.add(validation['workspace_bytes'])
        for variant in ('a', 'b'):
            if digest(root / f'input_{variant}/scores.bin') != case['input_sha256'][f'input_{variant}']:
                raise ValueError('Radix验收输入指纹改变')
            for suffix in ('top_scores.bin', 'top_ids.bin'):
                name = variant + '_' + suffix
                if digest(root / 'actual' / name) != case['output_sha256'][name]:
                    raise ValueError('Radix验收输出指纹改变')
            with redirect_stdout(io.StringIO()):
                checker.check(root / f'input_{variant}', root / f'actual/{variant}_top_scores.bin',
                            root / f'actual/{variant}_top_ids.bin')
    if len(workspaces) != 1:
        raise ValueError('当前Radix只有系统GM工作区，但各验收尺寸返回不一致，须先定位')
    return {'directory': str(directory), 'report_sha256': digest(path),
            'workspace_bytes': workspaces.pop()}


def prepare(vendors, parallel_project, merge_project, output, implementation='parallel', warp_validation=None):
    output = output.resolve()
    if output.exists():
        raise ValueError('运行目录已存在，请选择新目录')
    if implementation == 'parallel':
        record = installed(vendors, parallel_project, merge_project)
    else:
        record = installed(vendors, parallel_project, merge_project, implementation, warp_validation)
    runtime = output / 'vendors'
    runtime.mkdir(parents=True)
    for alias in ALIASES:
        (runtime / alias).symlink_to(record['api_libraries'][alias]['actual_directory'], target_is_directory=True)
    write_json(runtime / 'bm25_sharded_runtime.json', record)
    print('独立四段运行目录：' + str(runtime), flush=True)
    print('四个安装库/头文件与工程源码核对通过；尚未运行NPU，旧运行目录未修改。', flush=True)
    return record


def validate_runtime(vendors, info, shards):
    dimensions(info['num_queries'], info['num_docs'], shards, info['k'])
    record = json.loads((vendors / 'bm25_sharded_runtime.json').read_text(encoding='utf-8'))
    if record['variant'] not in (VARIANT, WARP_VARIANT) + RADIX_VARIANTS:
        raise ValueError('四段运行版本不匹配')
    is_warp = record['variant'] == WARP_VARIANT
    is_radix = record['variant'] in RADIX_VARIANTS
    if is_radix:
        current = installed(Path(record['vendors']), Path(record['radix_project']), Path(record['merge_project']),
                            'radix_zero' if record['variant'] == RADIX_ZERO_VARIANT else 'radix',
                            Path(record['radix_validation']['directory']))
    elif is_warp:
        current = installed(Path(record['vendors']), Path(record['warp_project']), Path(record['merge_project']),
                            'warp', Path(record['warp_validation']['directory']))
    else:
        current = installed(Path(record['vendors']), Path(record['parallel_project']), Path(record['merge_project']))
    if current != record:
        raise ValueError('实际工程、安装库或生成头文件已改变，请核对后另建运行目录')
    for alias in ALIASES:
        if str((vendors / alias).resolve(strict=True)) != record['api_libraries'][alias]['actual_directory']:
            raise ValueError(f'实际运行vendor指向改变：{alias}')
    scratch_function = warp.scratch_bytes if is_warp else scratch_bytes
    scratch = 0 if is_radix else scratch_function(info['num_queries'] * shards, info['num_docs'] // shards, info['k'])
    environment = ':'.join(record['api_libraries'][alias]['actual_directory'] for alias in ALIASES)
    return record, scratch, environment


def validate_topk_workspace(record, timing, scratch):
    capacities = timing.get('workspace_capacity_bytes', [])
    if type(scratch) is not int or scratch < 0 or len(capacities) != 4 or any(type(n) is not int or n < 0 for n in capacities):
        raise ValueError('四段实际工作区记录不合法')
    if record['variant'] in RADIX_VARIANTS:
        if (scratch != 0 or timing.get('topk_implementation') != 'radix' or
                type(timing.get('local_topk_user_scratch_bytes')) is not int or
                timing.get('local_topk_user_scratch_bytes') != 0 or
                capacities[2] != record['radix_validation']['workspace_bytes']):
            raise ValueError('Radix实现标识、零GM候选空间或独立验收工作区不匹配')
    else:
        if capacities[2] < capacities[0] + scratch:
            raise ValueError('分块Top-K额外GM工作区不匹配')
        if record['variant'] == WARP_VARIANT and (timing.get('topk_implementation') != 'warp' or
                timing.get('local_topk_user_scratch_bytes') != scratch or capacities[2] != capacities[0] + scratch):
            raise ValueError('Warp实现标识或精确工作区不匹配')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendors', default=str(ROOT / 'results/custom_opp/vendors'))
    parser.add_argument('--parallel-project', default=str(ROOT / 'results/Bm25TopKParallel'))
    parser.add_argument('--topk-implementation', choices=('parallel', 'warp', 'radix', 'radix_zero'), default='parallel')
    parser.add_argument('--warp-project', default=str(ROOT / 'results/Bm25TopKWarp'))
    parser.add_argument('--warp-validation', default=str(ROOT / 'results/topk_warp_validation1'))
    parser.add_argument('--radix-project', default=str(ROOT / 'results/Bm25TopKRadix'))
    parser.add_argument('--radix-validation', default=str(ROOT / 'results/radix_topk_validation1'))
    parser.add_argument('--radix-zero-project', default=str(ROOT / 'results/Bm25TopKRadixZero'))
    parser.add_argument('--radix-zero-validation', default=str(ROOT / 'results/radix_zero_topk_validation1'))
    parser.add_argument('--merge-project', default=str(ROOT / 'results/Bm25TopKMerge'))
    parser.add_argument('--output', default=str(ROOT / 'results/sharded_runtime'))
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise ValueError('请在黄区Linux用实际安装文件建立运行目录')
    project = {'parallel': args.parallel_project, 'warp': args.warp_project, 'radix': args.radix_project, 'radix_zero': args.radix_zero_project}[args.topk_implementation]
    validation = {'parallel': None, 'warp': args.warp_validation, 'radix': args.radix_validation, 'radix_zero': args.radix_zero_validation}[args.topk_implementation]
    prepare(Path(args.vendors), Path(project), Path(args.merge_project), Path(args.output), args.topk_implementation,
            Path(validation) if validation is not None else None)


if __name__ == '__main__':
    main()
