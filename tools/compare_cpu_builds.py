"""Compare accepted CPU binaries in alternating pairs or a three-way compiler-policy diagnostic."""

import argparse
import hashlib
import io
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import tarfile
import zipfile
from pathlib import Path

from a5_environment import command_output, cpu_snapshot

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "9a99bbc6719bb7144db3ebca8303393496db32bb"
CHECKPOINTS = ("before_A", "before_B", "after_A", "after_B")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cpu_list(text):
    result = set()
    for part in text.split(","):
        bounds = part.split("-")
        require(1 <= len(bounds) <= 2 and all(v.isdigit() for v in bounds), "Invalid CPU list")
        low, high = int(bounds[0]), int(bounds[-1])
        require(0 <= low <= high < 1048576, "Invalid CPU range")
        result.update(range(low, high + 1))
    return sorted(result)


def build_metadata(directory):
    cache = directory / "CMakeCache.txt"
    selected = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(
                ("CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS", "CMAKE_EXE_LINKER_FLAGS", "CMAKE_BUILD_TYPE:")
            ):
                key, _, value = line.partition("=")
                selected[key.split(":")[0]] = value
    flags = {}
    for name in ("bm25_cpu_search_benchmark", "bm25_directory_cpu_benchmark"):
        matches = list(directory.rglob(f"{name}.dir/flags.make"))
        flags[name] = {str(p.relative_to(directory)): p.read_text(encoding="utf-8", errors="replace") for p in matches}
    database = directory / "compile_commands.json"
    commands = []
    if database.is_file():
        commands = [
            row
            for row in json.loads(database.read_text(encoding="utf-8"))
            if Path(row["file"]).name in ("cpu_benchmark.cpp", "cpu_directory_benchmark.cpp")
        ]
    return {"directory": str(directory), "cache": selected, "flags_make": flags, "compile_commands": commands}


def restore_legacy_cpu_flags(text):
    replacements = {
        "-O3;-Wall;-Wextra;-fstack-protector-all;-fno-strict-aliasing;-Wreturn-type;-fno-common": "-O3;-Wall;-Wextra",
        '    target_compile_definitions(bm25_common INTERFACE "$<BUILD_INTERFACE:_FORTIFY_SOURCE=2>")\n': "",
        '    target_compile_options(bm25_common INTERFACE "$<BUILD_INTERFACE:-fPIC>")\n': "",
        '    target_link_options(bm25_common INTERFACE "$<BUILD_INTERFACE:-Wl,-z,relro;-Wl,-z,now;-Wl,-z,noexecstack>")\n': "",
    }
    for before, after in replacements.items():
        require(text.count(before) == 1, "Source recipe differs from the reviewed experiment: " + before)
        text = text.replace(before, after)
    return text


def prepare_legacy_flags(work, cmake, metadata, environment):
    source = work / "legacy_flags_source"
    source.mkdir()
    archive = subprocess.check_output(
        ["git", "-C", str(ROOT), "archive", SOURCE_COMMIT, "--", "CMakeLists.txt", "cmake", "feature_retrieval"]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            destination = (source / member.name).resolve()
            require(destination.is_relative_to(source.resolve()), "Archive path leaves experiment directory")
            require(member.isdir() or member.isfile(), "Archive links/special files rejected")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as incoming, destination.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    recipe = source / "feature_retrieval/src/ascendfaiss/CMakeLists.txt"
    original = recipe.read_text(encoding="utf-8")
    with recipe.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(restore_legacy_cpu_flags(original))
    build = work / "legacy_flags_build"
    command = [
        cmake,
        "-S",
        str(source),
        "-B",
        str(build),
        "-DBM25_BUILD_NPU=OFF",
        "-DBUILD_TESTING=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    for key in (
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "CMAKE_EXE_LINKER_FLAGS",
        "CMAKE_EXE_LINKER_FLAGS_RELEASE",
    ):
        if key in metadata["cache"]:
            command.append(f"-D{key}={metadata['cache'][key]}")
    for name, args in (
        ("configure", command),
        ("build", [cmake, "--build", str(build), "--target", "bm25_directory_cpu_benchmark", "--parallel", "8"]),
    ):
        with (work / f"legacy_flags_{name}.log").open("wb") as log:
            result = subprocess.run(
                args, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False
            )
        require(result.returncode == 0, f"Legacy-flags experiment {name} failed; see {log.name}")
    effective = build_metadata(build)
    directory_entries = [
        row for row in effective["compile_commands"] if Path(row["file"]).name == "cpu_directory_benchmark.cpp"
    ]
    require(len(directory_entries) == 1, "Experimental compilation record missing")
    rendered = directory_entries[0].get("command", " ".join(directory_entries[0].get("arguments", [])))
    for forbidden in ("-fno-strict-aliasing", "-fstack-protector-all", "-fPIC", "-D_FORTIFY_SOURCE"):
        require(forbidden not in rendered, f"Experimental old policy still contains {forbidden}")
    return build, effective


def checkpoint_hashes(actual):
    result = {}
    for checkpoint in CHECKPOINTS:
        for name in ("top_scores.bin", "top_ids.bin", "scores.bin"):
            path = actual / checkpoint / name
            require(path.is_file() and not path.is_symlink(), f"Checkpoint missing: {path}")
            result[f"{checkpoint}/{name}"] = {"sha256": digest(path), "bytes": path.stat().st_size}
    for case in ("A", "B"):
        for name in ("top_scores.bin", "top_ids.bin", "scores.bin"):
            require(result[f"before_{case}/{name}"] == result[f"after_{case}/{name}"], "In-process checkpoint changed")
    return result


def timing_summary(record, queries, repeats):
    require(
        record.get("queries") == queries and record.get("measured_cycles") == repeats, "Unexpected benchmark workload"
    )
    require(record.get("index_readback_byte_identical") is True, "Resident index verification incomplete")
    result = {}
    for case in ("A", "B"):
        samples = [row["call_wall_ms"] for row in record["samples"] if row["case"] == case]
        require(
            len(samples) == repeats and all(type(v) in (int, float) and 0 < v < float("inf") for v in samples),
            "Incomplete/invalid timing samples",
        )
        result[case] = {
            "samples": len(samples),
            "median_batch_ms": statistics.median(samples),
            "p95_batch_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
        }
    return result


def trial_order(protocol, pairs):
    if protocol == "three-way":
        return ["old_binary", "new_binary", "new_source_old_flags", "new_source_old_flags", "new_binary", "old_binary"]
    require(protocol == "paired", "Unknown comparison protocol")
    require(pairs >= 4 and pairs % 2 == 0, "Use an even number of pairs, at least four, to balance process order")
    return [
        name
        for pair in range(pairs)
        for name in (("old_binary", "new_binary") if pair % 2 == 0 else ("new_binary", "old_binary"))
    ]


def paired_summary(trials):
    require(len(trials) >= 8 and len(trials) % 4 == 0, "Incomplete order-balanced paired comparison")
    expected_order = trial_order("paired", len(trials) // 2)
    pairs = []
    for offset in range(0, len(trials), 2):
        members = trials[offset : offset + 2]
        require([t["variant"] for t in members] == expected_order[offset : offset + 2], "Unexpected pair order")
        by_name = {row["variant"]: row for row in members}
        changes = {}
        for case in ("A", "B"):
            changes[case] = {}
            for metric in ("median_batch_ms", "p95_batch_ms"):
                old, new = (by_name[name]["summary"][case][metric] for name in ("old_binary", "new_binary"))
                require(
                    all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (old, new)),
                    "Invalid paired latency",
                )
                changes[case][metric] = {"old": old, "new": new, "new_vs_old_percent": 100 * (new / old - 1)}
        pairs.append({"pair": offset // 2 + 1, "process_order": [t["variant"] for t in members], "changes": changes})
    summary = {}
    for case in ("A", "B"):
        summary[case] = {}
        for metric in ("median_batch_ms", "p95_batch_ms"):
            values = [pair["changes"][case][metric]["new_vs_old_percent"] for pair in pairs]
            summary[case][metric] = {
                "paired_changes_percent": values,
                "median_paired_change_percent": statistics.median(values),
                "min_paired_change_percent": min(values),
                "max_paired_change_percent": max(values),
                "pairs_new_slower": sum(value > 0 for value in values),
                "pairs": len(values),
            }
    return {"pairs": pairs, "summary": summary, "business_acceptance": "not_decided"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-build", required=True, type=Path)
    parser.add_argument("--after-build", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpus", default="0-31")
    parser.add_argument("--numa-node", default=0, type=int)
    parser.add_argument("--threads", default=32, type=int)
    parser.add_argument("--queries", default=8, type=int)
    parser.add_argument("--max-words", default=24, type=int)
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--repeats", default=30, type=int)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--protocol", choices=("three-way", "paired"), default="three-way")
    parser.add_argument("--pairs", type=int, default=6, help="Paired protocol: even number of old/new process pairs")
    parser.add_argument(
        "--reservation-note", default="", help="Record actual CPU reservation arrangements; does not reserve resources"
    )
    args = parser.parse_args()
    require(platform.system() == "Linux", "This comparison requires the actual Linux host")
    work = args.output.resolve()
    require(work.is_relative_to(ROOT / "results") and work != ROOT / "results", "Output must be under project results")
    require(not work.exists(), "Fresh experiment directory required")
    work.mkdir(parents=True)
    report = {
        "status": "running",
        "diagnostic_only": True,
        "source_commit": SOURCE_COMMIT,
        "scope": "Directory CPU only; not an A5 or business-service acceptance replacement.",
        "protocol": args.protocol,
        "business_acceptance": "not_decided",
        "trials": [],
        "cpus": args.cpus,
        "numa_node": args.numa_node,
    }
    code = 1
    try:
        order = trial_order(args.protocol, args.pairs)
        report["planned_process_order"] = order
        cpus = cpu_list(args.cpus)
        require(
            len(cpus) == args.threads and set(cpus).issubset(os.sched_getaffinity(0)),
            "Need one allowed CPU per requested thread",
        )
        cores = set()
        siblings = set()
        for cpu in cpus:
            base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
            require((base / f"node{args.numa_node}").exists(), "Selected CPUs span a different NUMA node")
            cores.add(
                (
                    (base / "topology/physical_package_id").read_text().strip(),
                    (base / "topology/core_id").read_text().strip(),
                )
            )
            siblings.update(cpu_list((base / "topology/thread_siblings_list").read_text().strip()))
        require(len(cores) == args.threads, "Select distinct physical cores rather than SMT siblings")
        report["resource_reservation"] = {
            "operator_note": args.reservation_note or None,
            "selected_physical_cores_all_logical_cpus": sorted(siblings),
            "exclusive_reservation_verified_by_tool": False,
        }
        require(
            args.repeats >= 10 and args.queries > 0 and args.max_words >= 0 and 1 <= args.k <= 32,
            "Invalid measurement dimensions",
        )
        before, after, inputs = (p.resolve(strict=True) for p in (args.before_build, args.after_build, args.input))
        docs = (inputs / "index/doc_norm.bin").stat().st_size // 4
        checkpoint_budget = docs * args.queries * 4 * 4 * (len(order) if args.keep_checkpoints else 1) + 512 * 1024**2
        require(shutil.disk_usage(work).free >= checkpoint_budget, "Insufficient space for experiment checkpoints")
        binary = "bm25_directory_cpu_benchmark"
        variants = {"old_binary": before / binary, "new_binary": after / binary}
        for path in variants.values():
            require(path.is_file(), f"Actual CPU binary missing: {path}")
        accepted = json.loads((ROOT / "evidence/indexsdk_a5_acceptance_20261104.json").read_text(encoding="utf-8"))
        require(
            digest(variants["new_binary"]) == accepted["binary_sha256"][binary],
            "Select the accepted 9a99bbc after-build for a consistent comparison",
        )
        report["initial_cpu"] = cpu_snapshot()
        report["lscpu"] = command_output(["lscpu"])
        report["numa_hardware"] = command_output(["numactl", "--hardware"])
        environment = os.environ.copy()
        environment.update(
            OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE", OMP_PROC_BIND="close", OMP_PLACES="cores"
        )
        environment.pop("GOMP_CPU_AFFINITY", None)
        prefix = ["numactl", f"--physcpubind={args.cpus}", f"--membind={args.numa_node}"]
        probe = command_output([*prefix, "numactl", "--show"], environment)
        report["binding_probe"] = probe
        require(
            probe["exit_code"] == 0,
            "CPU/NUMA binding unavailable; inspect binding_probe instead of running an unbound comparison",
        )
        report["builds"] = {"old_binary": build_metadata(before), "new_binary": build_metadata(after)}
        if args.protocol == "three-way":
            legacy_build, legacy_metadata = prepare_legacy_flags(
                work, args.cmake, report["builds"]["old_binary"], environment
            )
            variants["new_source_old_flags"] = legacy_build / binary
            report["builds"]["new_source_old_flags"] = legacy_metadata
        report["binary_sha256"] = {name: digest(path) for name, path in variants.items()}
        report["runtime_dependencies"] = {}
        for name, path in variants.items():
            dependencies = command_output(["ldd", str(path)], environment)
            report["runtime_dependencies"][name] = dependencies
            require(
                dependencies["exit_code"] == 0 and "not found" not in dependencies["output"],
                f"CPU runtime dependencies unresolved: {name}",
            )
        report["input"] = {
            "directory": str(inputs),
            "queries": args.queries,
            "k": args.k,
            "max_query_words": args.max_words,
            "A_sha256": digest(inputs / "A.txt"),
            "B_sha256": digest(inputs / "B.txt"),
        }
        baseline = None
        for number, name in enumerate(order, 1):
            target = work / f"trial_{number:02d}_{name}"
            target.mkdir()
            actual = target / "actual"
            command = [
                *prefix,
                str(variants[name]),
                str(inputs / "index"),
                str(inputs / "A.txt"),
                str(inputs / "B.txt"),
                str(actual),
                str(args.threads),
                str(args.k),
                str(args.threads),
                str(args.queries),
                str(args.max_words),
                "5",
                str(args.repeats),
            ]
            row = {"variant": name, "command": command, "before": cpu_snapshot(environment)}
            if args.protocol == "paired":
                row["pair"] = (number + 1) // 2
            report["trials"].append(row)
            save(work / "report.json", report)
            print(f"START {number}/{len(order)} {name}", flush=True)
            with (target / "run.log").open("wb") as log:
                result = subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False
                )
            row.update(exit_code=result.returncode, after=cpu_snapshot(environment))
            require(result.returncode == 0, f"CPU trial failed: {target / 'run.log'}")
            record = json.loads((actual / "benchmark_session.json").read_text(encoding="utf-8"))
            require(record.get("actual_threads") == args.threads, "Actual OpenMP team size differs")
            for key, expected in (
                ("k", args.k),
                ("partitions", args.threads),
                ("capacity_queries", args.queries),
                ("capacity_words", args.max_words),
                ("warmup_cycles", 5),
            ):
                require(record.get(key) == expected, f"Unexpected benchmark setting: {key}")
            row["summary"] = timing_summary(record, args.queries, args.repeats)
            row["checkpoint_sha256"] = checkpoint_hashes(actual)
            if baseline is None:
                baseline = row["checkpoint_sha256"]
            require(row["checkpoint_sha256"] == baseline, "Old/new output bits differ; comparison stopped")
            if not args.keep_checkpoints:
                for relative in row["checkpoint_sha256"]:
                    path = actual / relative
                    require(
                        not path.is_symlink() and path.resolve().is_relative_to(work), "Cleanup path leaves experiment"
                    )
                    path.unlink()
                row["large_checkpoints_removed_after_hash_verification"] = True
            save(work / "report.json", report)
            print(f"PASS {name}: {json.dumps(row['summary'])}", flush=True)
        require(
            all(digest(path) == report["binary_sha256"][name] for name, path in variants.items()),
            "Binary changed during comparison",
        )
        report["summary"] = {
            name: {
                case: statistics.median(
                    row["summary"][case]["median_batch_ms"] for row in report["trials"] if row["variant"] == name
                )
                for case in ("A", "B")
            }
            for name in variants
        }
        report["status"] = "completed_with_bit_identical_outputs"
        report["interpretation"] = (
            "Inspect repeated paired differences and actual resource reservation. A stable slowdown under low interference is a measured version cost, not automatically attributable to load or a specific flag. Business acceptance requires workload and latency/throughput limits."
            if args.protocol == "paired"
            else "Two trials per variant are diagnostic observations. Inspect paired timings and CPU snapshots before attributing a regression to flags or scheduling."
        )
        if args.protocol == "paired":
            report["paired_comparison"] = paired_summary(report["trials"])
            print("PAIRED changes percent: " + json.dumps(report["paired_comparison"]["summary"]), flush=True)
        print("SUMMARY median batch ms: " + json.dumps(report["summary"]), flush=True)
        code = 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report.update(status="failed", error=str(error))
        print(f"FAILED: {error}", flush=True)
    finally:
        save(work / "report.json", report)
        print(f"REPORT: {work / 'report.json'}", flush=True)
        target = work / "cpu_comparison_logs.zip"
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.rglob("*")):
                if (
                    path.suffix in (".json", ".log")
                    and path.is_file()
                    and not path.is_symlink()
                    and "legacy_flags_source" not in path.relative_to(work).parts
                ):
                    archive.write(path, path.relative_to(work).as_posix())
        print(f"LOG BUNDLE: {target}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
