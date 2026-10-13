"""只读检查当前机器的昇腾开发环境；只把报告写到指定文件，不安装或修改系统。"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def probe(command):
    executable = shutil.which(command[0])
    if executable is None:
        return {"status": "not_found", "command": command}
    try:
        result = subprocess.run(
            [executable, *command[1:]], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20,
        )
        return {"status": "ok" if result.returncode == 0 else "failed",
                "command": command, "executable": executable,
                "returncode": result.returncode,
                "stdout": result.stdout[:20000], "stderr": result.stderr[:5000]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "command": command, "executable": executable}
    except OSError as error:
        return {"status": "failed", "command": command, "error": str(error)}


def version_files(extra_root):
    # 只检查常见安装目录和用户明确指定的目录，不遍历整个文件系统。
    roots = [Path("/usr/local/Ascend"), Path.home() / "Ascend"] if platform.system() == "Linux" else []
    if extra_root:
        roots.append(Path(extra_root).expanduser())
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if os.environ.get(key):
            roots.append(Path(os.environ[key]))
    found = {}
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("version.info", "*/version.info", "*/*/version.info", "*/*/*/version.info",
                        "ascend_toolkit_install.info", "*/ascend_toolkit_install.info",
                        "*/*/ascend_toolkit_install.info", "*/*/*/ascend_toolkit_install.info"):
            for path in root.glob(pattern):
                if str(path) in found:
                    continue
                try:
                    # 安装信息中只保留版本/架构/包名行，不收集安装用户等信息。
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    lines = [line for line in lines if any(k in line.lower() for k in ("version", "arch", "package"))]
                    found[str(path)] = "\n".join(lines)[:4000]
                except OSError as error:
                    found[str(path)] = "无法读取：" + str(error)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/npu_environment.json")
    parser.add_argument("--ascend-root", help="可选：实际昇腾软件安装目录")
    args = parser.parse_args()
    commands = {
        "npu_status": probe(["npu-smi", "info"]),
        "npu_ids": probe(["npu-smi", "info", "-l"]),
        "compiler": probe(["bisheng", "--version"]),
        "compiler_help": probe(["bisheng", "--help"]),
        "atc": probe(["atc", "--version"]),
        "cmake": probe(["cmake", "--version"]),
        "host_compiler": probe(["g++", "--version"]),
    }
    packages = {}
    for name in ("numpy", "bm25s", "numba", "torch", "torch-npu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    report = {
        "system": platform.system(), "machine": platform.machine(),
        "python": platform.python_version(), "python_executable": sys.executable,
        "packages": packages, "commands": commands,
        "ascend_paths": {key: os.environ.get(key) for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME", "ASCEND_VERSION")},
        "version_files": version_files(args.ascend_root),
        "simt_option_visible_in_help": "--enable-simt" in commands["compiler_help"].get("stdout", ""),
        "simt_execution_verified": False,
        "note": "工具可见不等于算子运行成功；须再编译并执行匹配版本的官方 SIMT 示例。",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("系统：", report["system"], report["machine"])
    print("NPU 状态工具：", commands["npu_status"]["status"])
    print("昇腾编译器：", commands["compiler"]["status"])
    print("找到版本信息文件：", len(report["version_files"]))
    print("报告：", output)
    print("注意：这只是环境检查，没有运行 NPU 算子。")
    if commands["npu_status"]["status"] != "ok" or commands["compiler"]["status"] != "ok":
        print("环境尚未就绪：可能不在 NPU 服务器、工具未安装，或当前终端未加载环境。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
