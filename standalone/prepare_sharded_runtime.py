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

VARIANT = 'document_sharded_topk_v1'
WARP_VARIANT = 'document_sharded_warp_sort32_topk_v1'
ALIASES = ('bm25', 'bm25_accumulate', 'bm25_topk', 'bm25_topk_merge')
HEADERS = ('aclnn_bm25_contribution.h', 'aclnn_bm25_accumulate.h', 'aclnn_bm25_top_k.h', 'aclnn_bm25_top_k_merge.h')


def installed(vendors, parallel_project, merge_project, implementation='parallel', warp_validation=None):
    vendors, parallel_project, merge_project = (p.resolve() for p in (vendors, parallel_project, merge_project))
    if implementation not in ('parallel', 'warp'):
        raise ValueError('未知局部Top-K实现')
    operator = 'topk_warp' if implementation == 'warp' else 'topk_parallel'
    project = json.loads((parallel_project / 'bm25_project.json').read_text(encoding='utf-8'))
    if project['operator'] != operator or project['vendor_name'] != 'bm25_' + operator:
        raise ValueError('需要所选局部Top-K实现的实际工程/vendor，不能自动替换版本')
    for name in ('op_host/bm25_top_k.cpp', 'op_host/bm25_top_k_tiling.h', 'op_kernel/bm25_top_k.cpp'):
        if digest(parallel_project / name) != project['source_sha256'][name] or digest(parallel_project / name) != digest(ROOT / 'npu_prototype' / operator / name):
            raise ValueError(f'局部Top-K工程与当前源码不一致：{name}')
    topk_vendor = vendors / ('bm25_' + operator)
    if prototypes(vendors / 'bm25_topk/op_api/include/aclnn_bm25_top_k.h') != prototypes(topk_vendor / 'op_api/include/aclnn_bm25_top_k.h'):
        raise ValueError('新旧Top-K实际接口不一致')
    accepted = None
    if implementation == 'warp':
        installation = warp.validate_install(topk_vendor, parallel_project)
        accepted = validate_warp_acceptance(warp_validation, installation)
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
    if record['variant'] not in (VARIANT, WARP_VARIANT):
        raise ValueError('四段运行版本不匹配')
    is_warp = record['variant'] == WARP_VARIANT
    if is_warp:
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
    scratch = scratch_function(info['num_queries'] * shards, info['num_docs'] // shards, info['k'])
    environment = ':'.join(record['api_libraries'][alias]['actual_directory'] for alias in ALIASES)
    return record, scratch, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendors', default=str(ROOT / 'results/custom_opp/vendors'))
    parser.add_argument('--parallel-project', default=str(ROOT / 'results/Bm25TopKParallel'))
    parser.add_argument('--topk-implementation', choices=('parallel', 'warp'), default='parallel')
    parser.add_argument('--warp-project', default=str(ROOT / 'results/Bm25TopKWarp'))
    parser.add_argument('--warp-validation', default=str(ROOT / 'results/topk_warp_validation1'))
    parser.add_argument('--merge-project', default=str(ROOT / 'results/Bm25TopKMerge'))
    parser.add_argument('--output', default=str(ROOT / 'results/sharded_runtime'))
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise ValueError('请在黄区Linux用实际安装文件建立运行目录')
    project = args.warp_project if args.topk_implementation == 'warp' else args.parallel_project
    prepare(Path(args.vendors), Path(project), Path(args.merge_project), Path(args.output), args.topk_implementation,
            Path(args.warp_validation) if args.topk_implementation == 'warp' else None)


if __name__ == '__main__':
    main()
