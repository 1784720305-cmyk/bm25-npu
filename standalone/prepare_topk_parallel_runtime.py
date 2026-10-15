"""核对实际新旧生成头文件和工程指纹，建立独立运行路径；不编译或运行NPU。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'striped_local_topk_warp_merge_v1'


def partition_count(docs):
    partitions = 32
    while partitions < 256 and docs > partitions * 16:
        partitions *= 2
    return partitions


def scratch_bytes(queries, docs, k):
    return queries * partition_count(docs) * (8*k + 4)


def prototypes(path):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', path.read_text(encoding='utf-8'), flags=re.S)
    result = {}
    for name in ('aclnnBm25TopKGetWorkspaceSize', 'aclnnBm25TopK'):
        matches = re.findall(r'\baclnnStatus\s+' + name + r'\s*\((.*?)\)\s*;', text, flags=re.S)
        if len(matches) != 1:
            raise ValueError(f'不能唯一识别实际生成声明：{path}: {name}')
        result[name] = re.sub(r'\s+', '', matches[0])
    return result


def prepare(vendors, project, output):
    vendors, project, output = (p.resolve() for p in (vendors, project, output))
    if output.exists():
        raise ValueError('运行目录已存在，请选择新目录')
    project_info = json.loads((project / 'bm25_project.json').read_text(encoding='utf-8'))
    if project_info['operator'] != 'topk_parallel' or project_info['vendor_name'] != 'bm25_topk_parallel':
        raise ValueError('需要独立bm25_topk_parallel工程/vendor')
    for relative in ('op_host/bm25_top_k.cpp', 'op_host/bm25_top_k_tiling.h', 'op_kernel/bm25_top_k.cpp'):
        digest = hashlib.sha256((project / relative).read_bytes()).hexdigest()
        if digest != project_info['source_sha256'][relative] or digest != hashlib.sha256((ROOT / 'npu_prototype/topk_parallel' / relative).read_bytes()).hexdigest():
            raise ValueError(f'工程不是本次并行源码：{relative}')
    old_header = vendors / 'bm25_topk/op_api/include/aclnn_bm25_top_k.h'
    new_vendor = vendors / 'bm25_topk_parallel'
    new_header = new_vendor / 'op_api/include/aclnn_bm25_top_k.h'
    if prototypes(old_header) != prototypes(new_header):
        raise ValueError('实际新旧头文件声明不同，停止，需核对接口')
    print('实际生成的新旧Top-K两阶段函数声明一致：', json.dumps(prototypes(new_header)), flush=True)
    targets = {'bm25': vendors / 'bm25', 'bm25_accumulate': vendors / 'bm25_accumulate', 'bm25_topk': new_vendor}
    libraries = {}
    for alias, destination in targets.items():
        library = destination / 'op_api/lib/libcust_opapi.so'
        if not library.is_file():
            raise ValueError(f'安装库缺失：{library}')
        libraries[alias] = {'actual_directory': str(destination),
                            'api_sha256': hashlib.sha256(library.read_bytes()).hexdigest()}
    # New runtime directory only. Symlinks reuse the installed two unchanged operators.
    output.mkdir(parents=True)
    runtime_vendors = output / 'vendors'
    runtime_vendors.mkdir()
    for alias, destination in targets.items():
        (runtime_vendors / alias).symlink_to(destination, target_is_directory=True)
    record = {'variant': VARIANT, 'status': 'installed_files_checked_not_device_verified',
              'api_libraries': libraries, 'project_source_sha256': project_info['source_sha256'],
              'generated_topk_header_sha256': hashlib.sha256(new_header.read_bytes()).hexdigest(),
              'ascend_custom_opp_path': ':'.join(str(p) for p in targets.values())}
    (runtime_vendors / 'bm25_runtime.json').write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))
    print('独立vendor目录：', runtime_vendors)
    print('当前终端需明确设置：export ASCEND_CUSTOM_OPP_PATH="' + record['ascend_custom_opp_path'] + '"')
    print('尚未编译新调用器或运行设备；后续调用校验API指纹和额外工作区，失败就停止。')


def validate_runtime(vendors, info):
    record = json.loads((vendors / 'bm25_runtime.json').read_text(encoding='utf-8'))
    if record['variant'] != VARIANT:
        raise ValueError('不认识的并行Top-K运行版本')
    for alias, expected in record['api_libraries'].items():
        actual_dir = (vendors / alias).resolve()
        if str(actual_dir) != expected['actual_directory'] or hashlib.sha256((actual_dir / 'op_api/lib/libcust_opapi.so').read_bytes()).hexdigest() != expected['api_sha256']:
            raise ValueError(f'实际运行API库改变：{alias}')
    header = vendors / 'bm25_topk/op_api/include/aclnn_bm25_top_k.h'
    if hashlib.sha256(header.read_bytes()).hexdigest() != record['generated_topk_header_sha256']:
        raise ValueError('生成头文件指纹改变')
    environment = ':'.join(str((vendors / alias).resolve()) for alias in ('bm25', 'bm25_accumulate', 'bm25_topk'))
    return record, scratch_bytes(info['num_queries'], info['num_docs'], info['k']), environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendors', default=str(ROOT / 'results/custom_opp/vendors'))
    parser.add_argument('--project', default=str(ROOT / 'results/Bm25TopKParallel'))
    parser.add_argument('--output', default=str(ROOT / 'results/topk_parallel_runtime'))
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise ValueError('运行目录在黄区Linux创建，使用实际安装文件')
    prepare(Path(args.vendors), Path(args.project), Path(args.output))


if __name__ == '__main__':
    main()
