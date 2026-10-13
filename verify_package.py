"""核对源码快照的文件SHA256，不依赖第三方包。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'package_manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents:
            raise ValueError('清单路径不在包内')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'文件缺失或指纹不同：{relative}')
    print(f"文件完整性检查通过：{len(manifest['files'])}个文件。")
    print('此检查不代表算子已编译或设备已运行。')


if __name__ == '__main__':
    main()
