"""Relocate an installation and compile/run a Common or IndexSDK consumer without device work."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = r'''
#include "common/bm25/index_data.h"
#include "common/bm25/fixed_query_packet.h"
#include <vector>
int main()
{
    const ascend::bm25::Index index({0, 1}, {0}, {1}, {1}, {2});
    const bm25plan::PacketLayout packet(1, 2, 1);
    std::vector<unsigned char> memory(packet.inputBytes);
    return index.idf[0] == 2 && packet.Pack({{0, 0}}, memory.data()) == 24 ? 0 : 1;
}
'''
CONSUMER = '''cmake_minimum_required(VERSION 3.17)
project(bm25_installed_consumer LANGUAGES CXX)
find_package(AscendBm25 CONFIG REQUIRED COMPONENTS Common)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE AscendBm25::common)
'''
SDK_PROGRAM = r'''
#include "ascendhost/include/index/AscendIndexBm25.h"
#include "common/bm25/index_data.h"
int main()
{
    faiss::ascend::AscendIndexBm25 model;
    faiss::ascend::Bm25SearchResult result{{9}, {9}};
    if (model.Search({{0}}, result) != ascend::APP_ERR_ILLEGAL_OPERATION) { return 1; }
    if (result.scores[0] != 9 || result.labels[0] != 9) { return 2; }
    const ascend::bm25::Index index({0, 1}, {0}, {1}, {1}, {2});
    const faiss::ascend::Bm25Config missingResources;
    if (model.Init(index, missingResources) != ascend::APP_ERR_INVALID_PARAM) { return 3; }
    if (model.GetLastError().message[0] == '\0') { return 4; }
    if (model.Close() != ascend::APP_ERR_OK || model.GetLastError().message[0] != '\0') { return 5; }
    return 0;
}
'''


def execute(command, log, expected_success=True):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log.write_bytes(result.stdout)
    if (result.returncode == 0) != expected_success:
        raise RuntimeError(f"Unexpected command result {result.returncode}; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--component", choices=("Common", "IndexSDK"), default="Common")
    parser.add_argument("--toolkit-home", type=Path)
    parser.add_argument("--indexsdk-source", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Fresh results directory for the consumer and logs")
    args = parser.parse_args()
    build = args.build_dir.resolve(strict=True)
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    entries = dict(
        line.split("=", 1) for line in cache.splitlines() if "=" in line and not line.startswith(("#", "//"))
    )
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    work = (
        args.output_dir.resolve()
        if args.output_dir
        else Path(tempfile.mkdtemp(prefix="installed_package_", dir=results)).resolve()
    )
    if not work.is_relative_to(results.resolve()):
        raise ValueError("Consumer output must stay within the workspace results directory")
    if args.output_dir:
        if work.exists() or work.is_relative_to(build) or build.is_relative_to(work):
            raise ValueError("Fresh independent consumer output directory required")
        work.mkdir()
    original = work / "original"
    execute(
        [args.cmake, "--install", str(build), "--prefix", str(original), "--component", "Development"],
        work / "install.log",
    )
    relocated = work / "relocated"
    shutil.copytree(original, relocated)
    # Keep both prefixes and logs for review. No deletion or move of build/workspace files.
    for path in relocated.rglob("*"):
        parts = path.relative_to(relocated).parts
        if "mock" in parts or "ut" in parts or "test" in parts:
            raise RuntimeError(f"Test files leaked into installation: {path}")
        if path.suffix == ".cmake":
            text = path.read_text(encoding="utf-8").replace("\\", "/").lower()
            if ROOT.as_posix().lower() in text or build.as_posix().lower() in text:
                raise RuntimeError(f"Workspace path leaked into exported package: {path}")
    source = work / "consumer"
    source.mkdir()
    consumer = CONSUMER
    if args.component == "IndexSDK":
        consumer = consumer.replace("COMPONENTS Common", "COMPONENTS IndexSDK")
        consumer = consumer.replace("AscendBm25::common", "AscendBm25::indexsdk")
    (source / "CMakeLists.txt").write_text(consumer, encoding="utf-8")
    (source / "main.cpp").write_text(SDK_PROGRAM if args.component == "IndexSDK" else PROGRAM, encoding="utf-8")
    consumer_build = work / "consumer_build"
    command = [args.cmake, "-S", str(source), "-B", str(consumer_build), f"-DCMAKE_PREFIX_PATH={relocated}"]
    if args.toolkit_home:
        command.append(f"-DAscendBm25_TOOLKIT_HOME={args.toolkit_home.resolve(strict=True)}")
    if args.indexsdk_source:
        command.append(f"-DAscendBm25_INDEXSDK_SOURCE={args.indexsdk_source.resolve(strict=True)}")
    generator = entries.get("CMAKE_GENERATOR:INTERNAL")
    if generator:
        command += ["-G", generator]
    compiler = entries.get("CMAKE_CXX_COMPILER:FILEPATH", entries.get("CMAKE_CXX_COMPILER:STRING"))
    if compiler:
        command.append(f"-DCMAKE_CXX_COMPILER={compiler}")
    execute(command, work / "configure.log")
    installed_cache = (consumer_build / "CMakeCache.txt").read_text(encoding="utf-8").replace("\\", "/")
    found_paths = [line.split("=", 1)[1] for line in installed_cache.splitlines() if line.startswith("AscendBm25_DIR:")]
    if len(found_paths) != 1 or not Path(found_paths[0]).resolve().is_relative_to(relocated):
        raise RuntimeError("Consumer did not discover the relocated package")
    execute([args.cmake, "--build", str(consumer_build), "--config", "Release"], work / "build.log")
    executable = "consumer.exe" if os.name == "nt" else "consumer"
    candidates = [consumer_build / executable, consumer_build / "Release" / executable]
    binary = next((p for p in candidates if p.is_file()), None)
    if binary is None:
        raise RuntimeError("Consumer executable was not produced")
    execute([str(binary)], work / "consumer.log")
    package_configs = list(relocated.rglob("AscendBm25Config.cmake"))
    if len(package_configs) != 1:
        raise RuntimeError("Exactly one package configuration must be installed")
    package_text = package_configs[0].read_text(encoding="utf-8")
    missing_host_checked = "set(AscendBm25_Host_FOUND OFF)" in package_text
    if missing_host_checked and args.component == "Common":
        (source / "CMakeLists.txt").write_text(
            CONSUMER.replace("COMPONENTS Common", "COMPONENTS Host"), encoding="utf-8"
        )
        execute(command, work / "missing_host.log", expected_success=False)
        if "Component Host is unavailable" not in (work / "missing_host.log").read_text(encoding="utf-8"):
            raise RuntimeError("Missing Host component did not produce the expected diagnostic")
    report = {
        "status": "passed",
        "component": args.component,
        "relocated_consumer_built_and_run": True,
        "workspace_paths_absent_from_exports": True,
        "missing_host_rejected": missing_host_checked,
        "host_or_indexsdk_link_validated": args.component == "IndexSDK",
        "indexsdk_non_device_api_smoke": args.component == "IndexSDK",
        "shared_context_execution_validated": False,
    }
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: relocated installed {args.component} consumer; report: {work / 'report.json'}")


if __name__ == "__main__":
    main()
