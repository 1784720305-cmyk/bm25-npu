"""重建快照清单；文本必须已经是LF，二进制按原始字节计算，不改文件。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BINARY_SUFFIXES = {'.bin', '.npz', '.whl'}


def main():
    path = ROOT / 'package_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    # 沿用明确的发布文件列表，避免把results、缓存或.git误加入清单。
    names = sorted(set(manifest['files']) | {'rebuild_manifest.py'})
    hashes = {}
    for relative in names:
        source = (ROOT / relative).resolve()
        if ROOT not in source.parents or source == path:
            raise ValueError(f'非法清单路径：{relative}')
        data = source.read_bytes()
        if source.suffix not in BINARY_SUFFIXES:
            data.decode('utf-8')
            if b'\r' in data:
                raise ValueError(f'文本尚未统一为LF：{relative}')
        hashes[relative] = hashlib.sha256(data).hexdigest()
    manifest['text_line_endings'] = 'LF'
    manifest['files'] = hashes
    path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(f'已重建{len(hashes)}个文件的原始字节清单；没有修改样本或源码。')


if __name__ == '__main__':
    main()
