"""Run real Linux/CANN/SDK host acceptance, then optional A5 shared-resource and full Search checks."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTOR_STATUS = "actual_simd_selector_guards_and_strict_bit_rankings_passed"
MERGE_STATUS = "actual_global_id_merge_tree_boundaries_guards_and_bits_passed"
SHARED_STATUS = "shared_SDK_context_oracle_retry_lifetime_and_stream_passed"
SEARCH_STATUS = "actual_final_search_six_rounds_checked"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"Expected JSON object: {path}")
    return data


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_elf(path):
    require(path.is_file(), f"Actual binary/library missing: {path}")
    with path.open("rb") as stream:
        require(stream.read(4) == b"\x7fELF", f"Expected real Linux ELF binary/library: {path}")


def validate_device_reports(selector, merge, shared, search, expected_device, expected_runners):
    require(selector.get("status") == SELECTOR_STATUS and selector.get("calls") == 54, "Incomplete selector checks")
    require(merge.get("status") == MERGE_STATUS and merge.get("calls") == 36, "Incomplete merge checks")
    require(
        shared.get("status") == SHARED_STATUS
        and shared.get("device") == expected_device
        and shared.get("init_close_cycles") == 2
        and shared.get("queries") == 8
        and shared.get("k") == 10,
        "Shared SDK context acceptance incomplete",
    )
    rounds = search.get("per_round")
    require(
        search.get("status") == SEARCH_STATUS and isinstance(rounds, list) and len(rounds) == 6,
        "Six complete Search rounds required",
    )
    require(search.get("runner_sha256") == expected_runners, "Full Search used different binaries")
    cpu_decision = search.get("cpu_decision")
    require(isinstance(cpu_decision, dict), "Performance decision missing")
    decision = cpu_decision.get("performance_target_met_at_this_scale")
    require(type(decision) is bool, "Performance decision missing or not boolean")
    cpu_roles = cpu_decision.get("cpu_roles")
    require(
        isinstance(cpu_roles, list)
        and all(isinstance(role, str) and role.startswith("cpu") for role in cpu_roles)
        and len(set(cpu_roles)) == len(cpu_roles)
        and {"cpu", "cpu_directory"}.issubset(cpu_roles),
        "CPU opponents missing or invalid",
    )
    for row in rounds:
        require(isinstance(row, dict), "Incomplete round record")
        for role in ["npu", *cpu_roles]:
            record = row.get(role)
            require(isinstance(record, dict) and isinstance(record.get("summary"), dict), f"Missing round role: {role}")
            for case in ("A", "B"):
                summary = record["summary"].get(case)
                require(isinstance(summary, dict) and summary.get("samples") == 30, "Incomplete measured samples")
                for metric in ("median_batch_ms", "p95_batch_ms"):
                    number = summary.get(metric)
                    require(
                        type(number) in (int, float) and math.isfinite(number) and number > 0,
                        f"Invalid measured timing: {metric}",
                    )
    cases = cpu_decision.get("cases")
    require(isinstance(cases, dict), "Per-case performance decisions missing")
    passed = []
    for case in ("A", "B"):
        cpu = [min(row[role]["summary"][case]["median_batch_ms"] for role in cpu_roles) for row in rounds]
        npu = [row["npu"]["summary"][case]["median_batch_ms"] for row in rounds]
        cp95 = [min(row[role]["summary"][case]["p95_batch_ms"] for role in cpu_roles) for row in rounds]
        np95 = [row["npu"]["summary"][case]["p95_batch_ms"] for row in rounds]
        actual = all(a < b for a, b in zip(npu, cpu)) and statistics.median(np95) < statistics.median(cp95)
        case_result = cases.get(case)
        require(
            isinstance(case_result, dict) and case_result.get("passed") is actual,
            "Inconsistent per-case performance result",
        )
        passed.append(actual)
    require(decision is all(passed), "Performance flag differs from measured rounds")
    return decision


class Run:
    def __init__(self, work, report, environment):
        self.work = work
        self.report = report
        self.environment = environment

    def save(self):
        write_json(self.work / "report.json", self.report)

    def stage(self, name, command, timeout=7200):
        log = self.work / f"{len(self.report['stages']):02d}_{name}.log"
        row = {"name": name, "status": "running", "command": list(map(str, command)), "log": log.name}
        self.report["stages"].append(row)
        self.report["status"] = "running"
        self.report["current_stage"] = name
        self.save()
        print(f"START {name}: {json.dumps(row['command'], ensure_ascii=False)}", flush=True)
        started = time.monotonic()
        try:
            with log.open("wb") as stream:
                result = subprocess.run(
                    row["command"],
                    cwd=ROOT,
                    env=self.environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            row.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - started)
            row["status"] = "passed" if result.returncode == 0 else "failed"
            self.save()
            require(result.returncode == 0, f"Stage {name} failed, exit={result.returncode}; see {log}")
        except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
            row["status"] = "failed"
            self.save()
            raise
        print(f"PASS {name}", flush=True)


def parameters(args, saved):
    values = {}
    for name in ("toolkit_home", "indexsdk_source", "sdk_library", "vendors", "securec_include"):
        value = getattr(args, name) or saved.get(name)
        require(value is not None or name == "securec_include", f"--{name.replace('_', '-')} is required")
        values[name] = str(Path(value).resolve(strict=True)) if value else None
    require(platform.system() == "Linux", "Real acceptance requires Linux; Windows/mock checks cannot pass this gate")
    toolkit = Path(values["toolkit_home"])
    sdk = Path(values["indexsdk_source"])
    require((toolkit / "include/acl/acl.h").is_file(), "CANN acl/acl.h missing")
    require((sdk / "feature_retrieval/src/ascendfaiss/common/ErrorCode.h").is_file(), "IndexSDK source headers missing")
    require_elf(Path(values["sdk_library"]))
    for library in ("libascendcl.so", "libnnopbase.so"):
        require_elf(toolkit / "lib64" / library)
    aliases = Path(values["vendors"])
    for alias, vendor in (
        ("bm25_fused_score_topk", "bm25_fused_score_topk_warp_append_large"),
        ("bm25_topk_merge", "bm25_topk_merge_tree"),
    ):
        actual = (aliases / alias).resolve(strict=True)
        require(actual.name == vendor, f"Final accepted vendor required: {alias}")
        require_elf(actual / "op_api/lib/libcust_opapi.so")
    return values


def environment(values):
    result = os.environ.copy()
    vendors = Path(values["vendors"])
    opp = [str((vendors / name).resolve(strict=True)) for name in ("bm25_fused_score_topk", "bm25_topk_merge")]
    opp += result.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
    result["ASCEND_CUSTOM_OPP_PATH"] = ":".join(dict.fromkeys(p for p in opp if p))
    paths = [str(Path(values["toolkit_home"]) / "lib64"), str(Path(values["sdk_library"]).parent)]
    paths += result.get("LD_LIBRARY_PATH", "").split(":")
    result["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in paths if p))
    return result


def dependency_snapshot(values):
    toolkit = Path(values["toolkit_home"])
    sdk = Path(values["indexsdk_source"]) / "feature_retrieval/src/ascendfaiss"
    meta_header = next(
        (
            path
            for path in (toolkit / "include/aclnn/acl_meta.h", toolkit / "include/aclnnop/aclnn/acl_meta.h")
            if path.is_file()
        ),
        toolkit / "include/aclnn/acl_meta.h",
    )
    paths = {
        "sdk_library": Path(values["sdk_library"]),
        "acl_library": toolkit / "lib64/libascendcl.so",
        "nnop_library": toolkit / "lib64/libnnopbase.so",
        "acl_header": toolkit / "include/acl/acl.h",
        "acl_base_rt_header": toolkit / "include/acl/acl_base_rt.h",
        "acl_rt_header": toolkit / "include/acl/acl_rt.h",
        "acl_meta_header": meta_header,
        "sdk_error_header": sdk / "common/ErrorCode.h",
        "sdk_proxy_header": sdk / "ascenddaemon/AscendResourcesProxy.h",
        "sdk_resources_header": sdk / "ascenddaemon/StandardAscendResources.h",
    }
    for alias, header in (
        ("bm25_fused_score_topk", "aclnn_bm25_fused_score_top_k.h"),
        ("bm25_topk_merge", "aclnn_bm25_top_k_merge.h"),
    ):
        vendor = Path(values["vendors"]) / alias
        paths[f"{alias}_api_header"] = vendor / "op_api/include" / header
        paths[f"{alias}_api_library"] = vendor / "op_api/lib/libcust_opapi.so"
    return {name: {"path": str(path.resolve(strict=True)), "sha256": digest(path)} for name, path in paths.items()}


def snapshot(build):
    names = (
        "bm25_large_search_benchmark",
        "bm25_cpu_search_benchmark",
        "bm25_directory_cpu_benchmark",
        "bm25_fused_simd_selector_check",
        "bm25_sdk_shared_check",
    )
    result = {}
    for name in names:
        path = build / name
        require_elf(path)
        result[name] = digest(path)
    return result


def host_checks(run, args, values):
    build = run.work / "build"
    run.stage("acceptance_contract_tests", [sys.executable, ROOT / "tools/test_acceptance_contract.py"])
    command = [
        args.cmake,
        "-S",
        ROOT,
        "-B",
        build,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBM25_BUILD_NPU=ON",
        "-DBM25_BUILD_BENCHMARKS=ON",
        "-DBUILD_TESTING=ON",
        f"-DTOOLKIT_HOME={values['toolkit_home']}",
        f"-DBM25_VENDOR_DIR={values['vendors']}",
        f"-DBM25_INDEXSDK_SOURCE={values['indexsdk_source']}",
        f"-DBM25_INDEXSDK_LIBRARY={values['sdk_library']}",
    ]
    if values["securec_include"]:
        command.append(f"-DBM25_SECUREC_INCLUDE_DIR={values['securec_include']}")
    run.stage("configure", command)
    run.stage("build", [args.cmake, "--build", build, "--parallel", args.jobs])
    run.stage("unit_tests", [args.ctest, "--test-dir", build, "--output-on-failure"])
    run.stage(
        "production_clang_tidy",
        [
            sys.executable,
            ROOT / "tools/run_clang_tidy.py",
            "--build-dir",
            build,
            "--clang-tidy",
            args.clang_tidy,
            "--include-tests",
        ],
    )
    tidy = read_json(build / "bm25_clang_tidy/report.json")
    require(
        tidy.get("status") == "passed"
        and tidy.get("production_compile_db_coverage_complete") is True
        and tidy.get("allow_test_stubs") is False,
        "Real production static coverage not established",
    )
    for component in ("Common", "IndexSDK"):
        directory = run.work / f"installed_{component.lower()}"
        run.stage(
            f"installed_{component.lower()}",
            [
                sys.executable,
                ROOT / "tools/check_installed_package.py",
                "--build-dir",
                build,
                "--cmake",
                args.cmake,
                "--component",
                component,
                "--toolkit-home",
                values["toolkit_home"],
                "--indexsdk-source",
                values["indexsdk_source"],
                "--output-dir",
                directory,
            ],
        )
        result = read_json(directory / "report.json")
        require(
            result.get("status") == "passed" and result.get("component") == component,
            f"Installed {component} consumer incomplete",
        )
        if component == "IndexSDK":
            require(result.get("host_or_indexsdk_link_validated") is True, "IndexSDK consumer was not linked")
    run.report["binary_sha256"] = snapshot(build)
    run.report["dependency_sha256"] = dependency_snapshot(values)
    run.report["host_checks_passed"] = True
    run.report["status"] = "host_build_install_static_passed_device_pending"
    run.save()


def device_checks(run, args, values):
    require(args.device >= 0 and args.input is not None, "Device phase requires --input and a nonnegative --device")
    inputs = args.input.resolve(strict=True)
    require((inputs / "preparation_report.json").is_file(), "Existing validated large_corpus input directory required")
    build = run.work / "build"
    require(
        snapshot(build) == run.report.get("binary_sha256"), "Host binaries changed; repeat host phase in a new output"
    )
    require(
        dependency_snapshot(values) == run.report.get("dependency_sha256"),
        "CANN/SDK/operator dependency changed since host phase",
    )
    number = 1
    while (run.work / f"device_{number:02d}").exists():
        number += 1
    device = run.work / f"device_{number:02d}"
    device.mkdir()
    run.report.update(
        device_attempt=device.name,
        shared_context_device_passed=False,
        six_round_correctness_passed=False,
        performance_target_met=None,
    )
    run.save()
    run.stage(
        "prepare_small_sdk_input",
        [
            sys.executable,
            ROOT / "standalone/large_corpus.py",
            "--output",
            device / "smoke_inputs",
            "--docs",
            1024,
            "--vocab",
            1024,
            "--seed",
            20261014,
            "--queries-a",
            ROOT / "fixtures/A.txt",
            "--queries-b",
            ROOT / "fixtures/B.txt",
        ],
    )
    run.stage(
        "selector", [build / "bm25_fused_simd_selector_check", values["vendors"], args.device, device / "selector"]
    )
    run.stage(
        "merge",
        [build / "bm25_large_search_benchmark", "--merge-check", values["vendors"], args.device, device / "merge"],
    )
    run.stage(
        "shared_sdk_context",
        [
            build / "bm25_sdk_shared_check",
            device / "smoke_inputs",
            values["vendors"],
            args.device,
            device / "shared_sdk.json",
        ],
    )
    run.stage(
        "six_round_search",
        [
            sys.executable,
            ROOT / "standalone/benchmark_final.py",
            "--input",
            inputs,
            "--output",
            device / "full_search",
            "--vendors",
            values["vendors"],
            "--device",
            args.device,
            "--host-build",
            build,
        ],
    )
    runners = {
        role: run.report["binary_sha256"][name]
        for role, name in (
            ("npu", "bm25_large_search_benchmark"),
            ("cpu", "bm25_cpu_search_benchmark"),
            ("cpu_directory", "bm25_directory_cpu_benchmark"),
        )
    }
    run.report["current_stage"] = "validate_device_reports"
    require(
        snapshot(build) == run.report["binary_sha256"]
        and dependency_snapshot(values) == run.report["dependency_sha256"],
        "Binary/dependency changed during device acceptance",
    )
    met = validate_device_reports(
        read_json(device / "selector/report.json"),
        read_json(device / "merge/report.json"),
        read_json(device / "shared_sdk.json"),
        read_json(device / "full_search/report.json"),
        args.device,
        runners,
    )
    run.report.update(
        shared_context_device_passed=True,
        six_round_correctness_passed=True,
        performance_target_met=met,
        device=args.device,
        input=str(inputs),
    )
    run.report["status"] = (
        "all_requested_checks_passed" if met else "device_correctness_passed_performance_target_not_met"
    )
    run.save()
    return 0 if met else 2


def bundle(run):
    number = 1
    while (run.work / f"acceptance_logs_{number:02d}.zip").exists():
        number += 1
    target = run.work / f"acceptance_logs_{number:02d}.zip"
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.work.rglob("*")):
            if path.suffix not in (".json", ".log") or not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(run.work)
            if any(part in ("original", "relocated", "smoke_inputs") for part in relative.parts):
                continue
            archive.write(path, relative.as_posix())
    print(f"RESULT: {run.report['status']}\nREPORT: {run.work / 'report.json'}\nLOG BUNDLE: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("host", "device", "all"), default="host")
    parser.add_argument("--output", required=True, type=Path)
    for name in ("toolkit-home", "indexsdk-source", "sdk-library", "vendors", "securec-include", "input"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ctest", default="ctest")
    parser.add_argument("--clang-tidy", default="clang-tidy")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    work = args.output.resolve()
    require(
        work.is_relative_to(ROOT / "results") and work != ROOT / "results",
        "Output must be a new directory under project results",
    )
    if args.phase == "device":
        report = read_json(work / "report.json")
        require(report.get("host_checks_passed") is True, "Complete host phase first")
    else:
        require(not work.exists(), "Fresh host output required; existing runs are preserved")
        work.mkdir(parents=True)
        report = {
            "status": "running",
            "host_checks_passed": False,
            "stages": [],
            "source_lock_sha256": digest(ROOT / "source_lock.json"),
            "manifest_sha256": digest(ROOT / "package_manifest.json"),
        }
    run = Run(work, report, os.environ.copy())
    report["phase_requested"] = args.phase
    code = 1
    try:
        require(
            report["source_lock_sha256"] == digest(ROOT / "source_lock.json")
            and report["manifest_sha256"] == digest(ROOT / "package_manifest.json"),
            "Package changed since host phase",
        )
        require(args.jobs > 0, "Positive --jobs required")
        run.stage("verify_package", [sys.executable, ROOT / "verify_package.py"])
        report["current_stage"] = "preflight"
        run.save()
        values = parameters(args, report.get("parameters", {}))
        if args.phase == "device":
            require(values == report["parameters"], "Host dependencies changed; start a new host acceptance output")
        report["parameters"] = values
        run.environment = environment(values)
        report["runtime_paths"] = {key: run.environment[key] for key in ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH")}
        run.save()
        if args.phase in ("host", "all"):
            host_checks(run, args, values)
        code = device_checks(run, args, values) if args.phase in ("device", "all") else 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        report.update(status="failed", failed_stage=report.get("current_stage", "preflight"), error=str(error))
        run.save()
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
    finally:
        bundle(run)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
