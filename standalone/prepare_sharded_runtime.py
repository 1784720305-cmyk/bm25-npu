"""为四段实验建立独立vendor别名，核对实际工程/头文件/库；不编译或运行设备。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.prepare_topk_parallel_runtime import prototypes, scratch_bytes
from standalone.validate_topk_merge import digest, dimensions, validate_install, write_json

VARIANT = 'document_sharded_topk_v1'
ALIASES = ('bm25', 'bm25_accumulate', 'bm25_topk', 'bm25_topk_merge')
HEADERS = ('aclnn_bm25_contribution.h', 'aclnn_bm25_accumulate.h', 'aclnn_bm25_top_k.h', 'aclnn_bm25_top_k_merge.h')


def installed(vendors, parallel_project, merge_project):
    vendors, parallel_project, merge_project = (p.resolve() for p in (vendors, parallel_project, merge_project))
    project = json.loads((parallel_project / 'bm25_project.json').read_text(encoding='utf-8'))
    if project['operator'] != 'topk_parallel' or project['vendor_name'] != 'bm25_topk_parallel':
        raise ValueError('需要实际并行Top-K工程，不能使用旧单线程版本')
    for name in ('op_host/bm25_top_k.cpp', 'op_host/bm25_top_k_tiling.h', 'op_kernel/bm25_top_k.cpp'):
        if digest(parallel_project / name) != project['source_sha256'][name] or digest(parallel_project / name) != digest(ROOT / 'npu_prototype/topk_parallel' / name):
            raise ValueError(f'并行Top-K工程与当前源码不一致：{name}')
    if prototypes(vendors / 'bm25_topk/op_api/include/aclnn_bm25_top_k.h') != prototypes(vendors / 'bm25_topk_parallel/op_api/include/aclnn_bm25_top_k.h'):
        raise ValueError('新旧Top-K实际接口不一致')
    merge = validate_install(vendors / 'bm25_topk_merge', merge_project)
    targets = dict(zip(ALIASES, (vendors / 'bm25', vendors / 'bm25_accumulate', vendors / 'bm25_topk_parallel', vendors / 'bm25_topk_merge')))
    libraries = {}
    for (alias, destination), header in zip(targets.items(), HEADERS):
        destination = destination.resolve(strict=True)
        library = destination / 'op_api/lib/libcust_opapi.so'
        libraries[alias] = {'actual_directory': str(destination), 'api_sha256': digest(library),
                            'header_sha256': digest(destination / 'op_api/include' / header)}
    return {'variant': VARIANT, 'api_libraries': libraries,
            'vendors': str(vendors), 'parallel_project': str(parallel_project), 'merge_project': str(merge_project),
            'parallel_source_sha256': project['source_sha256'], 'merge_installation': merge,
            'scope': 'four installed vendors checked; source/build/install provenance also needs actual logs; not device execution'}


def prepare(vendors, parallel_project, merge_project, output):
    output = output.resolve()
    if output.exists():
        raise ValueError('运行目录已存在，请选择新目录')
    record = installed(vendors, parallel_project, merge_project)
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
    if record['variant'] != VARIANT:
        raise ValueError('四段运行版本不匹配')
    current = installed(Path(record['vendors']), Path(record['parallel_project']), Path(record['merge_project']))
    if current != record:
        raise ValueError('实际工程、安装库或生成头文件已改变，请核对后另建运行目录')
    for alias in ALIASES:
        if str((vendors / alias).resolve(strict=True)) != record['api_libraries'][alias]['actual_directory']:
            raise ValueError(f'实际运行vendor指向改变：{alias}')
    scratch = scratch_bytes(info['num_queries'] * shards, info['num_docs'] // shards, info['k'])
    environment = ':'.join(record['api_libraries'][alias]['actual_directory'] for alias in ALIASES)
    return record, scratch, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendors', default=str(ROOT / 'results/custom_opp/vendors'))
    parser.add_argument('--parallel-project', default=str(ROOT / 'results/Bm25TopKParallel'))
    parser.add_argument('--merge-project', default=str(ROOT / 'results/Bm25TopKMerge'))
    parser.add_argument('--output', default=str(ROOT / 'results/sharded_runtime'))
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise ValueError('请在黄区Linux用实际安装文件建立运行目录')
    prepare(Path(args.vendors), Path(args.parallel_project), Path(args.merge_project), Path(args.output))


if __name__ == '__main__':
    main()
