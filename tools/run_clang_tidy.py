"""Check compiled BM25 translation units; reject incomplete/mock coverage unless explicitly allowed."""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "feature_retrieval/src/ascendfaiss"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uses_stubs(entry):
    command = entry.get("command", " ".join(entry.get("arguments", []))).replace("\\", "/").lower()
    return any(marker in command for marker in ("/mock", "_stubs", "/stubs"))


def compiler_openmp_header(entry, cache):
    arguments = entry.get("arguments") or shlex.split(entry["command"], posix=os.name != "nt")
    if not any(value == "-fopenmp" or value.startswith("-fopenmp=") for value in arguments):
        return None
    # CMake can prepend a compiler launcher to its compilation database entries.
    while arguments and Path(arguments[0].strip('"')).stem in ("ccache", "sccache", "distcc"):
        arguments = arguments[1:]
    if not arguments:
        return None
    compiler = arguments[0].strip('"')
    key = (compiler, entry["directory"])
    if key not in cache:
        header = None
        try:
            result = subprocess.run(
                [compiler, "-print-file-name=include/omp.h"],
                cwd=entry["directory"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                candidate = Path(result.stdout.strip())
                if not candidate.is_absolute():
                    candidate = Path(entry["directory"]) / candidate
                if candidate.is_file():
                    header = str(candidate.resolve())
        except (OSError, subprocess.TimeoutExpired):
            pass  # Preserve clang-tidy's actual missing-header diagnostic.
        cache[key] = header
    return cache[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--clang-tidy", default="clang-tidy")
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--allow-test-stubs", action="store_true")
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--extra-arg-before", action="append", default=[])
    args = parser.parse_args()
    build = args.build_dir.resolve(strict=True)
    database = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    selected = {}
    for entry in database:
        file = Path(entry["file"])
        if not file.is_absolute():
            file = Path(entry["directory"]) / file
        file = file.resolve()
        if not file.is_relative_to(SOURCE) or file.suffix != ".cpp":
            continue
        relative = file.relative_to(SOURCE).as_posix()
        if relative.startswith("ops/") or (not args.include_tests and not relative.startswith("ascendhost/")):
            continue
        if uses_stubs(entry) and not args.allow_test_stubs:
            continue
        key = file.as_posix()
        # Prefer the real host build over duplicate UT compilations of the same source.
        if key not in selected or (uses_stubs(selected[key]) and not uses_stubs(entry)):
            selected[key] = {**entry, "file": str(file)}
    required = {
        str((SOURCE / "ascendhost/src/bm25" / name).resolve()).replace("\\", "/")
        for name in ("runtime.cpp", "fused_runtime.cpp", "planned_search.cpp", "large_search.cpp")
    }
    if any("AscendIndexBm25.cpp" in row["file"] for row in database):
        required.add((SOURCE / "ascendhost/src/bm25/AscendIndexBm25.cpp").as_posix())
    missing = sorted(required - selected.keys())
    if not selected or (missing and not args.allow_test_stubs):
        raise ValueError(f"Production compilation coverage missing: {missing}. Use a real CANN build database.")
    output = build / "bm25_clang_tidy"
    output.mkdir(exist_ok=True)
    (output / "compile_commands.json").write_text(json.dumps(list(selected.values()), indent=2), encoding="utf-8")
    config = ROOT / ".clang-tidy"
    verified = subprocess.run(
        [args.clang_tidy, "--verify-config", f"--config-file={config}"], capture_output=True, text=True, check=False
    )
    (output / "config.log").write_text(verified.stdout + verified.stderr, encoding="utf-8")
    if verified.returncode or "unknown check" in (verified.stdout + verified.stderr).lower():
        raise RuntimeError(f"clang-tidy configuration rejected; see {output / 'config.log'}")
    version = subprocess.run([args.clang_tidy, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    checks = []
    compiler_headers = {}
    header_filter = (
        "^" + re.escape(SOURCE.as_posix()).replace("/", r"[\\/]") + r"[\\/](common[\\/]bm25|ascendhost)[\\/]"
    )
    for number, (file, entry) in enumerate(sorted(selected.items())):
        command = [args.clang_tidy, file, f"-p={output}", f"--config-file={config}", "--quiet"]
        command.append(f"--header-filter={header_filter}")
        extra_arguments = list(args.extra_arg)
        omp_header = compiler_openmp_header(entry, compiler_headers)
        if omp_header:
            # Search after Clang's builtin and standard headers; GCC's stddef.h
            # must not replace Clang's own resource headers.
            extra_arguments.append(f"-idirafter{Path(omp_header).parent}")
        command += [f"--extra-arg={value}" for value in extra_arguments]
        command += [f"--extra-arg-before={value}" for value in args.extra_arg_before]
        if args.fix:
            command.append("--fix")
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
        )
        log = output / f"{number:03d}.log"
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        checks.append(
            {
                "file": file,
                "sha256": digest(Path(file)),
                "uses_test_stubs": uses_stubs(entry),
                "compiler_openmp_header": omp_header,
                "extra_arguments": extra_arguments,
                "exit_code": result.returncode,
                "log": str(log),
            }
        )
        print(f"{Path(file).name}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
    passed = all(item["exit_code"] == 0 for item in checks)
    report = {
        "status": "passed" if passed else "failed",
        "tool": version,
        "config_sha256": digest(config),
        "fix_requested": args.fix,
        "allow_test_stubs": args.allow_test_stubs,
        "missing_production_files": missing,
        "checks": checks,
        "production_compile_db_coverage_complete": passed and not missing and not args.allow_test_stubs,
        "hardware_validation": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output / 'report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
