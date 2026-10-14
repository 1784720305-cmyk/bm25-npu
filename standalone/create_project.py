"""用设备端msopgen建立独立CANN工程，替换为自研贡献算子；不安装、不运行。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def populate(project, compute_unit, vendor, operator='contribution'):
    # 只接受生成器已建立的预期结构，模板变化时停止而非假定成功。
    if not (project / 'build.sh').is_file():
        raise RuntimeError('生成工程没有build.sh，需检查当前工具模板')
    stem = 'bm25_contribution' if operator == 'contribution' else 'bm25_accumulate'
    pairs = [(f'op_host/{stem}.cpp', f'op_host/{stem}.cpp'),
             (f'op_host/{stem}_tiling.h', f'op_host/{stem}_tiling.h'),
             (f'op_kernel/{stem}.cpp', f'op_kernel/{stem}.cpp')]
    for relative in (f'op_host/{stem}.cpp', f'op_kernel/{stem}.cpp'):
        if not (project / relative).is_file():
            raise RuntimeError(f'生成器没有预期的{relative}，需核对命名规则')
    presets_path = project / 'CMakePresets.json'
    presets = json.loads(presets_path.read_text(encoding='utf-8'))
    changed = 0
    for preset in presets.get('configurePresets', []):
        variables = preset.get('cacheVariables', {})
        if 'ASCEND_COMPUTE_UNIT' in variables:
            old = variables['ASCEND_COMPUTE_UNIT']
            if isinstance(old, dict):
                old['value'] = compute_unit
            else:
                variables['ASCEND_COMPUTE_UNIT'] = compute_unit
            variables['vendor_name'] = vendor
            changed += 1
    if not changed:
        raise RuntimeError('模板未提供ASCEND_COMPUTE_UNIT，需核对生成工程，未替换源码')
    # 工程输出必须新建，备份生成的模板只用于本次比较，不动已有用户工程。
    backup = project / 'bm25_template_backup'
    backup.mkdir()
    for source, destination in pairs:
        target = project / destination
        if target.exists():
            saved = backup / destination
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, saved)
        shutil.copyfile(ROOT / 'npu_prototype' / source, target)
    shutil.copyfile(presets_path, backup / 'CMakePresets.json')
    presets_path.write_bytes((json.dumps(presets, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    metadata = {
        'status': 'project_generated_not_compiled_not_device_verified',
        'compute_unit': compute_unit, 'vendor_name': vendor,
        'operator': operator,
        'source_sha256': {dest: hashlib.sha256((project / dest).read_bytes()).hexdigest()
                          for _, dest in pairs},
    }
    (project / 'bm25_project.json').write_bytes((json.dumps(metadata, indent=2) + '\n').encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compute-resource', required=True,
                        help='当前msopgen支持的ai_core-...目标，先核对工具帮助，不猜型号')
    parser.add_argument('--compute-unit', default='ascend950')
    parser.add_argument('--vendor', help='默认贡献为bm25，累加为bm25_accumulate，避免覆盖已安装包')
    parser.add_argument('--operator', choices=['contribution', 'accumulate'], default='contribution')
    parser.add_argument('--msopgen', default='msopgen')
    parser.add_argument('--output')
    args = parser.parse_args()
    vendor = args.vendor or ('bm25' if args.operator == 'contribution' else 'bm25_accumulate')
    project_name = 'Bm25Contribution' if args.operator == 'contribution' else 'Bm25Accumulate'
    project = Path(args.output or ROOT / 'results' / project_name).resolve()
    if project.exists():
        parser.error('输出已存在，停止避免覆盖；请指定新的工作目录')
    executable = shutil.which(args.msopgen)
    if executable is None:
        parser.error('未找到msopgen，请先加载CANN环境并确认工具位置')
    if not args.compute_resource.startswith('ai_core-'):
        parser.error('compute-resource应为工具支持的ai_core-...目标')
    spec = 'operator.json' if args.operator == 'contribution' else 'accumulate_operator.json'
    command = [executable, 'gen', '-i', str(ROOT / 'standalone' / spec),
               '-f', 'aclnn', '-c', args.compute_resource, '-lan', 'cpp', '-out', str(project)]
    project.parent.mkdir(parents=True, exist_ok=True)
    print('创建独立CANN工程：', command, flush=True)
    subprocess.run(command, check=True, timeout=300)
    populate(project, args.compute_unit, vendor, args.operator)
    print('工程及自研源码已准备：', project)
    print('下一步使用生成的build.sh编译。尚未编译、安装或执行NPU。')


if __name__ == '__main__':
    main()
