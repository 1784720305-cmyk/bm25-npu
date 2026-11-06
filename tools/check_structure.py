"""Check BM25 source layout and the portable/runtime/test dependency boundaries."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "feature_retrieval/src/ascendfaiss"


def main():
    checked = 0
    for path in SOURCE.rglob("*"):
        if path.suffix not in (".h", ".cpp"):
            continue
        relative = path.relative_to(SOURCE).as_posix()
        text = path.read_text(encoding="utf-8")
        if relative.startswith(("common/", "ascendhost/")) and re.search(
            r"\b(?:std::ifstream|std::ofstream|LoadQueries|LoadRequests)\b", text
        ):
            raise ValueError(f"Benchmark file IO in implementation layer: {relative}")
        for include in re.findall(r'^#include "([^"]+)"', text, re.MULTILINE):
            if ".." in Path(include).parts:
                raise ValueError(f"Parent-relative include: {relative}: {include}")
            if include.startswith(("common/", "ascendhost/", "test/", "ut/", "ops/")):
                # The optional SDK boundary consumes the upstream header directly.
                if (
                    relative
                    in {
                        "ascendhost/include/impl/Bm25ErrorAdapter.h",
                        "ascendhost/include/index/AscendIndexBm25.h",
                        "ascendhost/src/bm25/AscendIndexBm25.cpp",
                        "ut/cases/bm25/sdk_error_adapter_tests.cpp",
                        "ut/cases/bm25/sdk_facade_tests.cpp",
                        "test/ascendhost/bm25/sdk_shared_check.cpp",
                        "test/ascendhost/bm25/sdk_benchmark.cpp",
                    }
                    and include == "common/ErrorCode.h"
                ):
                    continue
                if (
                    relative
                    in {
                        "test/ascendhost/bm25/sdk_shared_check.cpp",
                        "test/ascendhost/bm25/sdk_benchmark.cpp",
                    }
                    and include == "common/utils/SocUtils.h"
                ):
                    continue
                if not (SOURCE / include).is_file():
                    raise ValueError(f"Missing local header: {relative}: {include}")
                if relative.startswith("common/") and not include.startswith("common/"):
                    raise ValueError(f"Common depends on another layer: {relative}: {include}")
                if relative.startswith("ascendhost/") and include.startswith(("test/", "ut/")):
                    raise ValueError(f"Runtime depends on tests: {relative}: {include}")
        if path.suffix == ".h" and not re.search(r"^#ifndef \w+\n#define \w+", text, re.MULTILINE):
            raise ValueError(f"Missing header guard: {relative}")
        if relative.startswith(("common/", "ascendhost/")) and re.search(r"\bint main\s*\(", text):
            raise ValueError(f"Executable in implementation layer: {relative}")
        checked += 1
    for directory in ("npu_prototype", "cpu_reference", "common", "operators"):
        if any(p.suffix in (".h", ".cpp") for p in (ROOT / directory).rglob("*")):
            raise ValueError(f"Legacy source directory: {directory}")
    print(f"PASS: {checked} C++ files; layout, header guards and layer boundaries")


if __name__ == "__main__":
    main()
