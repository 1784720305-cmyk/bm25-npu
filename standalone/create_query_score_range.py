"""Generate an isolated Ascend950 range-scoring experiment; never overwrite/install/run accepted operators."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.create_project import generator_command
from standalone import resident_index as r

VENDOR = 'bm25_query_score_range'
SOURCES = ('op_host/bm25_query_score.cpp', 'op_host/bm25_query_score_tiling.h', 'op_kernel/bm25_query_score.cpp')


def populate(project):
    r.require((project/'build.sh').is_file() and all((project/path).is_file() for path in
              ('op_host/bm25_query_score.cpp', 'op_kernel/bm25_query_score.cpp')), '实际msopgen模板结构/命名不符')
    preset_path = project/'CMakePresets.json'
    presets = json.loads(preset_path.read_text(encoding='utf-8'))
    changed = 0
    for preset in presets.get('configurePresets', []):
        variables = preset.get('cacheVariables', {})
        if 'ASCEND_COMPUTE_UNIT' not in variables:
            continue
        for key, value in (('ASCEND_COMPUTE_UNIT', 'ascend950'), ('vendor_name', VENDOR)):
            current = variables.get(key)
            if isinstance(current, dict): current['value'] = value
            else: variables[key] = {'type':'STRING', 'value':value}
        changed += 1
    r.require(changed > 0, '实际模板未提供目标平台选项，停止')
    backup = project/'bm25_template_backup'; backup.mkdir()
    for name in SOURCES+('CMakePresets.json',):
        target = project/name
        if target.is_file():
            saved = backup/name; saved.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(target, saved)
    for name in SOURCES:
        shutil.copyfile(ROOT/'npu_prototype/query_score_range'/name, project/name)
    preset_path.write_bytes((json.dumps(presets, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    r.write_json(project/'bm25_project.json', {'status':'project_generated_not_compiled_not_device_verified',
        'operator':'query_score_range', 'compute_unit':'ascend950', 'vendor_name':VENDOR,
        'max_terms_per_query':64, 'formula':'same_FP32_query_time_formula_and_original_term_order',
        'source_sha256':{name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in SOURCES}})


def generate(output, compute_resource, msopgen='msopgen', python=None):
    output = output.resolve()
    r.require(not output.exists(), '实验工程目录已存在，请换新目录；不覆盖原工程')
    r.require(compute_resource == 'ai_core-ascend950', '本独立实验仅使用已核对的ai_core-ascend950目标')
    prefix = generator_command(msopgen, python)
    command = prefix+['gen', '-i', str(ROOT/'standalone/query_score_operator.json'), '-f', 'aclnn',
        '-c', compute_resource, '-lan', 'cpp', '-out', str(output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    print('创建独立评分实验工程：'+json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True, timeout=300)
    populate(output)
    record_path = output/'bm25_project.json'; record = json.loads(record_path.read_text(encoding='utf-8'))
    record['generator_command'] = command; r.write_json(record_path, record)
    print('实验源码已准备：'+str(output), flush=True)
    print('独立vendor='+VENDOR+'；只支持每查询最多64词，超限返回错误结果并须拒绝。未编译/安装/执行NPU。', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compute-resource', required=True)
    p.add_argument('--msopgen', default='msopgen'); p.add_argument('--msopgen-python')
    p.add_argument('--output', type=Path, default=ROOT/'results/Bm25QueryScoreRange')
    args = p.parse_args()
    try: generate(args.output, args.compute_resource, args.msopgen, args.msopgen_python)
    except (Exception, KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不安装、不执行或替代NPU。', file=sys.stderr); return 1
    return 0


if __name__ == '__main__': sys.exit(main())
