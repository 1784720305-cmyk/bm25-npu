"""生成可审阅的IndexSDK接入覆盖文件，绝不直接修改参考仓库。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--output", default=str(PACKAGE_ROOT / "results/indexsdk_bm25_overlay"))
    args = parser.parse_args()
    sdk = Path(args.sdk_root).resolve()
    out = Path(args.output).resolve()
    if out == sdk or sdk in out.parents:
        parser.error("输出目录不能在参考SDK内部")
    if out.exists():
        parser.error("输出目录已存在，请指定新目录，避免覆盖旧工程材料")
    base = Path("feature_retrieval/src/ascendfaiss/ops/ascendc")
    cmake_rel = base / "op_host/CMakeLists.txt"
    original = (sdk / cmake_rel).read_bytes()
    text = original.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    anchor = "set(ops_srcs" + newline
    if text.count(anchor) != 1 or "bm25_contribution.cpp" in text:
        parser.error("SDK源码列表结构变化或已接入BM25，需重新核对")
    text = text.replace(anchor, anchor + "    ${CMAKE_CURRENT_LIST_DIR}/bm25_contribution.cpp" + newline, 1)
    prototype = Path(__file__).resolve().parent
    files = [("op_host/bm25_contribution.cpp", base / "op_host/bm25_contribution.cpp"),
             ("op_host/bm25_contribution_tiling.h", base / "op_host/bm25_contribution_tiling.h"),
             ("op_kernel/bm25_contribution.cpp", base / "op_kernel/bm25_contribution.cpp")]
    for _, rel in files:
        if (sdk / rel).exists():
            parser.error(f"目标已有同名文件：{rel}")
    (out / cmake_rel).parent.mkdir(parents=True)
    (out / cmake_rel).write_bytes(text.encode("utf-8"))
    for source, rel in files:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prototype / source, out / rel)
    manifest = {"status": "prepared_only_not_applied_not_built", "sdk_root": str(sdk),
                "cmake_path": cmake_rel.as_posix(), "original_cmake_sha256": hashlib.sha256(original).hexdigest(),
                "files": {rel.as_posix(): hashlib.sha256((out / rel).read_bytes()).hexdigest()
                          for rel in [cmake_rel] + [rel for _, rel in files]}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("已生成4个接入文件与原始CMake指纹：", out)
    print("参考SDK未修改。黄区使用前需核对原CMake指纹并备份，再合入覆盖文件。")


if __name__ == "__main__":
    main()
