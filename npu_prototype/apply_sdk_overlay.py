"""先核对覆盖文件；只有显式 --apply 才备份并合入指定SDK工作副本。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--overlay", default=str(PACKAGE_ROOT / "results/indexsdk_bm25_overlay"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    sdk, overlay = Path(args.sdk_root).resolve(), Path(args.overlay).resolve()
    manifest = json.loads((overlay / "manifest.json").read_text(encoding="utf-8"))
    base = "feature_retrieval/src/ascendfaiss/ops/ascendc/"
    cmake = base + "op_host/CMakeLists.txt"
    allowed = {cmake, base + "op_host/bm25_contribution.cpp",
               base + "op_host/bm25_contribution_tiling.h", base + "op_kernel/bm25_contribution.cpp"}
    if manifest["cmake_path"] != cmake or set(manifest["files"]) != allowed:
        parser.error("清单不是预期的4个贡献算子文件")
    destinations = {}
    for relative, expected in manifest["files"].items():
        source, target = (overlay / relative).resolve(), (sdk / relative).resolve()
        if overlay not in source.parents or sdk not in target.parents:
            parser.error("路径不在指定工程内部")
        if digest(source) != expected:
            parser.error(f"覆盖文件发生变化，请重新生成：{relative}")
        if relative != cmake and target.exists():
            parser.error(f"目标已有文件，停止避免覆盖：{relative}")
        destinations[relative] = target
    if digest(destinations[cmake]) != manifest["original_cmake_sha256"]:
        parser.error("黄区SDK的原CMake与准备时不同，请在黄区重新生成覆盖材料")
    print("4个文件指纹、目标路径和原CMake检查通过。")
    if not args.apply:
        print("仅检查，SDK没有修改；确认目标工作副本后可加 --apply。")
        return
    backup = Path(tempfile.mkdtemp(prefix="bm25_backup_", dir=sdk))
    shutil.copyfile(destinations[cmake], backup / "CMakeLists.txt.original")
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for relative, target in destinations.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(overlay / relative, target)
    print("接入文件已合入，原CMake备份：", backup)
    print("尚未构建或运行。若中途文件写入失败，按清单检查已写文件并从备份恢复CMake。")


if __name__ == "__main__":
    main()
