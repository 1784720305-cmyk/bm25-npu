"""用设备端msopgen建立独立CANN工程，替换为自研算子；不安装、不运行。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OPERATORS = {
    'contribution': ('Bm25Contribution', 'bm25_contribution', 'operator.json', 'bm25'),
    'accumulate': ('Bm25Accumulate', 'bm25_accumulate', 'accumulate_operator.json', 'bm25_accumulate'),
    'topk': ('Bm25TopK', 'bm25_top_k', 'topk_operator.json', 'bm25_topk'),
    'topk_parallel': ('Bm25TopKParallel', 'bm25_top_k', 'topk_operator.json', 'bm25_topk_parallel'),
    'topk_merge': ('Bm25TopKMerge', 'bm25_top_k_merge', 'topk_merge_operator.json', 'bm25_topk_merge'),
}


def generator_command(msopgen, python=None):
    executable = shutil.which(msopgen)
    if executable is None and python is not None and Path(msopgen).is_file():
        executable = str(Path(msopgen).resolve())
    if executable is None:
        raise ValueError('未找到msopgen，请先加载CANN环境并确认工具位置')
    if python is None:
        return [executable]
    interpreter = shutil.which(python)
    if interpreter is None:
        raise ValueError('指定的msopgen Python不存在或不可执行：' + python)
    # Only a Python text entry can be invoked through an explicit Python interpreter.
    with Path(executable).open('rb') as stream:
        first = stream.readline(512)
    if not first.startswith(b'#!') or b'python' not in first.lower():
        raise ValueError('msopgen入口不是可识别的Python脚本，停止，不改安装文件')
    return [interpreter, str(Path(executable).resolve())]


def populate(project, compute_unit, vendor, operator='contribution'):
    # 只接受生成器已建立的预期结构，模板变化时停止而非假定成功。
    if not (project / 'build.sh').is_file():
        raise RuntimeError('生成工程没有build.sh，需检查当前工具模板')
    if operator not in OPERATORS:
        raise ValueError(f'不支持的算子：{operator}')
    stem = OPERATORS[operator][1]
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
                variables['ASCEND_COMPUTE_UNIT'] = {'type': 'STRING', 'value': compute_unit}
            # CANN preset_parse.py expects cache values as {type, value}, not bare strings.
            old_vendor = variables.get('vendor_name')
            if isinstance(old_vendor, dict):
                old_vendor['value'] = vendor
            else:
                variables['vendor_name'] = {'type': 'STRING', 'value': vendor}
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
        source_root = ROOT / 'npu_prototype'
        if operator in ('topk_parallel', 'topk_merge'):
            source_root = source_root / operator
        shutil.copyfile(source_root / source, target)
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
    parser.add_argument('--vendor', help='默认使用各算子独立vendor，避免覆盖已安装包')
    parser.add_argument('--operator', choices=list(OPERATORS), default='contribution')
    parser.add_argument('--msopgen', default='msopgen')
    parser.add_argument('--msopgen-python', help='显式运行msopgen的Python；先用该Python执行入口gen --help核对兼容性')
    parser.add_argument('--output')
    args = parser.parse_args()
    project_name, _, spec, default_vendor = OPERATORS[args.operator]
    vendor = args.vendor or default_vendor
    if args.operator in ('topk_parallel', 'topk_merge') and vendor != default_vendor:
        parser.error('本实验使用独立vendor ' + default_vendor + '，避免覆盖已有算子')
    project = Path(args.output or ROOT / 'results' / project_name).resolve()
    if project.exists():
        parser.error('输出已存在，停止避免覆盖；请指定新的工作目录')
    try:
        prefix = generator_command(args.msopgen, args.msopgen_python)
    except ValueError as error:
        parser.error(str(error))
    if not args.compute_resource.startswith('ai_core-'):
        parser.error('compute-resource应为工具支持的ai_core-...目标')
    command = prefix + ['gen', '-i', str(ROOT / 'standalone' / spec),
               '-f', 'aclnn', '-c', args.compute_resource, '-lan', 'cpp', '-out', str(project)]
    project.parent.mkdir(parents=True, exist_ok=True)
    print('创建独立CANN工程：', command, flush=True)
    try:
        subprocess.run(command, check=True, timeout=300)
    except FileNotFoundError as error:
        raise RuntimeError('生成器未能启动。入口存在时，也可能是首行指定的解释器不存在；'
                           '先用现有Python执行msopgen入口gen --help，通过后加--msopgen-python显式指定。'
                           '本次未生成可用工程，不继续编译。') from error
    populate(project, args.compute_unit, vendor, args.operator)
    metadata_path = project / 'bm25_project.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata['generator_command'] = command
    metadata_path.write_bytes((json.dumps(metadata, indent=2) + '\n').encode('utf-8'))
    print('工程及自研源码已准备：', project)
    print('下一步使用生成的build.sh编译。尚未编译、安装或执行NPU。')


if __name__ == '__main__':
    main()
