"""准备官方离线Numba依赖，或安装至项目私有目录；不改CANN/NumPy，不运行NPU。"""
import argparse
from contextlib import contextmanager
import email
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WHEELS = ROOT/'offline_dependencies/numba_py39_linux'
PINS = {'numba':'0.60.0', 'llvmlite':'0.43.0'}
NETWORK_TIMEOUT = 30


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, record):
    path.write_bytes((json.dumps(record, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))


@contextmanager
def download_response(url, stage):
    print(f'{stage}：{url}（连接/读取超时设置={NETWORK_TIMEOUT}秒，非整次下载总时限）', flush=True)
    try:
        with urllib.request.urlopen(url, timeout=NETWORK_TIMEOUT) as stream:
            yield stream
    except OSError as error:
        raise RuntimeError(f'{stage}失败：{url}；{type(error).__name__}: {error}') from error


def check_wheels():
    record = json.loads((WHEELS/'lock.json').read_text(encoding='utf-8'))
    require(record.get('pins') == PINS and len(record.get('wheels', [])) == 2, '离线依赖版本清单不符')
    require({x['name'] for x in record['wheels']} == set(PINS), '离线包缺失或重复')
    for item in record['wheels']:
        name = item['filename']
        require(Path(name).name == name and '-cp39-cp39-' in name and
                'manylinux2014_x86_64' in name and name.endswith('.whl'), '离线包不是目标CPython3.9/Linux x86_64二进制wheel')
        path = WHEELS/name
        require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], '离线包字节/大小不符：'+name)
        with zipfile.ZipFile(path) as archive:
            require(archive.testzip() is None, 'wheel内部CRC不符')
            names = [p for p in archive.namelist() if p.endswith('.dist-info/METADATA')]
            require(len(names) == 1, 'wheel元数据不唯一')
            info = email.message_from_bytes(archive.read(names[0]))
            require(info['Name'].lower() == item['name'] and info['Version'] == PINS[item['name']], 'wheel名称/版本不符')
    return record


def check_target():
    require(sys.platform == 'linux' and platform.machine() == 'x86_64' and sys.version_info[:2] == (3,9),
            '安装目标须为黄区Linux x86_64/CPython3.9，不安装Windows包或其他Python版本')
    require(metadata.version('numpy') == '2.0.2', '目标NumPy不是已核对的2.0.2，停止，不自动升级/降级')
    libc, version = platform.libc_ver()
    require(libc == 'glibc' and tuple(map(int, version.split('.')[:2])) >= (2,17), '目标glibc不满足manylinux2014要求')
    require('NUMBA_DISABLE_JIT' not in __import__('os').environ, '当前设置了NUMBA_DISABLE_JIT，停止，不能把解释执行当编译路径')


def download():
    require(sys.platform == 'win32', '下载步骤请在外网Windows执行；黄区只用check/install，不联网')
    print('开始准备两个官方Linux/Python3.9离线包；只下载和核对，不安装。', flush=True)
    print(f'保存目录：{WHEELS}', flush=True)
    WHEELS.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, version) in enumerate(PINS.items(), 1):
        endpoint = f'https://pypi.org/pypi/{name}/{version}/json'
        with download_response(endpoint, f'[{index}/{len(PINS)}] 查询{name}=={version}版本信息') as stream:
            info = json.load(stream)
        print(f'[{index}/{len(PINS)}] 版本信息已收到，选择目标平台文件。', flush=True)
        items = [x for x in info['urls'] if '-cp39-cp39-' in x['filename'] and
                 'manylinux2014_x86_64' in x['filename'] and x['filename'].endswith('.whl')]
        require(len(items) == 1, '官方目标wheel不唯一，停止，不猜其他平台')
        item = items[0]
        filename = item['filename']
        require(Path(filename).name == filename and item['url'].startswith('https://files.pythonhosted.org/'), '包地址/名称不符')
        path = WHEELS/filename
        if not path.exists():
            print(f'下载官方包：{filename}，{item["size"]}字节', flush=True)
            partial = path.with_suffix('.whl.part')
            received = 0
            started = last_progress = time.monotonic()
            with download_response(item['url'], '连接文件下载服务') as stream, partial.open('wb') as target:
                while True:
                    block = stream.read(64*1024)
                    if not block:
                        break
                    target.write(block)
                    received += len(block)
                    require(received <= item['size'], '下载字节超过官方大小，停止')
                    now = time.monotonic()
                    if now-last_progress >= 1 or received == item['size']:
                        print(f'下载进度：{received}/{item["size"]}字节 '
                              f'({received/item["size"]:.1%})，已用{now-started:.1f}秒', flush=True)
                        last_progress = now
            print('文件传输结束，核对大小和SHA256。', flush=True)
            require(partial.stat().st_size == item['size'] and digest(partial) == item['digests']['sha256'], '下载校验失败，停止')
            partial.replace(path)
        else:
            print(f'已有文件，核对大小和SHA256：{filename}', flush=True)
        require(path.stat().st_size == item['size'] and digest(path) == item['digests']['sha256'], '已有文件与官方PyPI摘要不同，不覆盖')
        print(f'[{index}/{len(PINS)}] {name}文件SHA256通过。', flush=True)
        records.append({'name':name,'version':version,'filename':filename,'bytes':item['size'],
                        'sha256':item['digests']['sha256'],'url':item['url'],'metadata_url':endpoint})
    lock = {'format':1,'pins':PINS,'target':'CPython3.9 Linux x86_64, NumPy2.0.2, glibc>=2.17',
            'compatibility_source':'https://numba.readthedocs.io/en/0.60.0/user/installing.html#version-support-information',
            'scope':'Official wheel archive/hash checks only; installation/JIT/BM25S target validation still required.', 'wheels':records}
    write_json(WHEELS/'lock.json', lock)
    print('两个文件已取得，检查wheel内部CRC、平台和版本。', flush=True)
    check_wheels()
    # Add exact verified binary bytes to the same LF/raw-byte package list; no Git command.
    manifest_path = ROOT/'package_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for path in [WHEELS/'lock.json']+[WHEELS/x['filename'] for x in records]:
        manifest['files'][path.relative_to(ROOT).as_posix()] = digest(path)
    manifest['optional_offline_dependencies'] = {'numba': {'state':'downloaded_verified_not_installed',
        'pins':PINS,'lock':'offline_dependencies/numba_py39_linux/lock.json'}}
    write_json(manifest_path, manifest)
    print('离线包检查通过，重建并校验项目文件清单。', flush=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'rebuild_manifest.py')],check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'verify_package.py')],check=True)
    print('两包已按PyPI SHA256核对并加入清单。未安装；请将新增offline_dependencies文件随Git提交。', flush=True)


PROBE = '''import sys,json
sys.path.insert(0,sys.argv[1])
import numpy as np,numba,llvmlite
from numba import njit,prange
from pathlib import Path
root=Path(sys.argv[1]).resolve()
assert root in Path(numba.__file__).resolve().parents
assert root in Path(llvmlite.__file__).resolve().parents
assert (numba.__version__,llvmlite.__version__,np.__version__)==('0.60.0','0.43.0','2.0.2')
@njit(parallel=True)
def compute(x):
    y=np.zeros(x.shape[0],dtype=np.float32)
    for i in prange(len(y)):
        for j in range(x.shape[1]):
            y[i]+=x[i,j]
    return y
x=np.arange(128,dtype=np.float32).reshape(8,16)
assert np.array_equal(compute(x),x.sum(axis=1))
assert compute.nopython_signatures
print(json.dumps({'numba':numba.__version__,'llvmlite':llvmlite.__version__,'numpy':np.__version__,'cpu_jit_and_parallel_smoke':'passed','threading_layer':numba.threading_layer()}))
'''


def installed_record(directory):
    check_target()
    lock = check_wheels()
    directory = directory.resolve(strict=True)
    record = json.loads((directory/'installation.json').read_text(encoding='utf-8'))
    require(record.get('status') == 'private_cpu_dependencies_and_jit_smoke_passed' and
            record.get('lock_sha256') == digest(WHEELS/'lock.json') and
            record.get('python_executable') == str(Path(sys.executable).resolve()), '私有依赖安装记录与当前环境不匹配')
    require(record.get('pins') == lock['pins'] and bool(record.get('installed_sha256')), '安装记录版本或文件清单不符')
    for name, expected in record['installed_sha256'].items():
        path = (directory/name).resolve(strict=True)
        require(directory in path.parents and digest(path) == expected, '私有安装文件改变：'+name)
    return record


def install(output):
    check_target()
    lock = check_wheels()
    output = output.resolve()
    require(not output.exists(), '私有依赖目录已存在，请用新目录，不覆盖或改系统环境')
    results = (ROOT/'results').resolve()
    require(results in output.parents, '依赖只能安装到本项目results下的新目录')
    output.mkdir(parents=True)
    record = {'status':'installing','pins':PINS,'lock_sha256':digest(WHEELS/'lock.json'),
              'python_executable':str(Path(sys.executable).resolve()),'scope':'Private CPU reference dependencies only; no CANN/NumPy changes or NPU work.'}
    write_json(output/'installation.json', record)
    try:
        requirements = output/'requirements.lock.txt'
        requirements.write_bytes(('\n'.join(f'{x["name"]} @ {(WHEELS/x["filename"]).resolve().as_uri()} --hash=sha256:{x["sha256"]}' for x in lock['wheels'])+'\n').encode('utf-8'))
        command = [sys.executable,'-m','pip','install','--no-index','--no-deps','--no-compile','--no-cache-dir',
                   '--disable-pip-version-check','--require-hashes','--target',str(output),'-r',str(requirements)]
        print('离线安装至私有目录：'+str(output), flush=True)
        subprocess.run(command,check=True)
        probe = subprocess.run([sys.executable,'-B','-c',PROBE,str(output)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        record['probe_stdout'] = probe.stdout
        record['probe_stderr'] = probe.stderr
        if probe.returncode != 0:
            print(probe.stdout, flush=True)
            print(probe.stderr, file=sys.stderr, flush=True)
            probe.check_returncode()
        record['cpu_probe'] = json.loads(probe.stdout.strip().splitlines()[-1])
        record['installed_sha256'] = {p.relative_to(output).as_posix():digest(p) for p in output.rglob('*')
                                      if p.is_file() and p.name != 'installation.json' and '__pycache__' not in p.parts}
        record['status'] = 'private_cpu_dependencies_and_jit_smoke_passed'
        write_json(output/'installation.json',record)
        print('私有依赖版本、CPU编译和并行小检查通过；尚未运行BM25S或NPU。', flush=True)
    except (Exception,KeyboardInterrupt) as error:
        record.update(status='failed',error=str(error))
        write_json(output/'installation.json',record)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('download')
    sub.add_parser('check')
    p=sub.add_parser('install')
    p.add_argument('--output',default=str(ROOT/'results/bm25s_numba_py39'))
    args=parser.parse_args()
    try:
        if args.command == 'download': download()
        elif args.command == 'check':
            record=check_wheels()
            print('两个官方离线wheel指纹/平台/版本通过，总字节：',sum(x['bytes'] for x in record['wheels']))
        else: install(Path(args.output))
    except (Exception,KeyboardInterrupt) as error:
        print('FAILED：'+str(error)+'；停止，不切换其他版本或改系统依赖。',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
