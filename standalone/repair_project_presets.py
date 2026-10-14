"""修复已生成工程的CMake cache变量对象格式；备份原文件，不改算子源码。"""
import argparse
import json
from pathlib import Path
import tempfile


def repair(project):
    path = project / 'CMakePresets.json'
    original = path.read_bytes()
    data = json.loads(original.decode('utf-8'))
    changed = []
    for preset in data.get('configurePresets', []):
        variables = preset.get('cacheVariables', {})
        for key, value in list(variables.items()):
            if isinstance(value, dict):
                if 'value' not in value:
                    raise ValueError(f'配置{key}缺少value，停止，需人工核对')
                continue
            if isinstance(value, bool):
                variables[key] = {'type': 'BOOL', 'value': 'ON' if value else 'OFF'}
            elif isinstance(value, str):
                variables[key] = {'type': 'STRING', 'value': value}
            else:
                raise ValueError(f'配置{key}不是支持的字符串/布尔值，停止')
            changed.append(key)
    if not changed:
        print('配置已经使用对象格式，无需修改。')
        return False
    # Validate all changes before writing, keep the original bytes for recovery.
    with tempfile.NamedTemporaryFile(prefix='CMakePresets.before_repair_', suffix='.json',
                                     dir=project, delete=False) as backup:
        backup.write(original)
        backup_path = backup.name
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    print('已修复配置项：', ', '.join(changed))
    print('原文件备份：', backup_path)
    print('算子源码和输入未改；请重新构建并检查日志。')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, help='已有生成工程目录')
    args = parser.parse_args()
    repair(Path(args.project).resolve())


if __name__ == '__main__':
    main()
