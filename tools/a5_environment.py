"""Read runtime prerequisites and CPU placement without changing the machine configuration."""

import json
import os
import platform
import subprocess
import time
from pathlib import Path

CPU_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_WAIT_POLICY",
    "GOMP_CPU_AFFINITY",
    "GOMP_SPINCOUNT",
    "OPENBLAS_NUM_THREADS",
)
RUNTIME_ENVIRONMENT = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_CUSTOM_OPP_PATH",
    "LD_LIBRARY_PATH",
    "MX_INDEX_USE_ONLINEOP",
    "MX_INDEX_MODELPATH",
)


def read_optional(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def command_output(command, environment=None):
    try:
        result = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        return {
            "command": list(map(str, command)),
            "exit_code": result.returncode,
            "output": (result.stdout + result.stderr).strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": list(map(str, command)), "exit_code": None, "output": str(error)}


def cpu_snapshot(environment=None):
    environment = os.environ if environment is None else environment
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    frequencies = {}
    for cpu in affinity or []:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
        governor = read_optional(base / "scaling_governor")
        frequency = read_optional(base / "scaling_cur_freq")
        if governor is not None or frequency is not None:
            frequencies[str(cpu)] = {"governor": governor, "frequency_khz": frequency}
    return {
        "snapshot_scope": "Observer process and system state; child placement is specified separately in its command.",
        "wall_time_seconds": time.time(),
        "system": platform.platform(),
        "allowed_cpus": affinity,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "openmp_environment": {name: environment.get(name) for name in CPU_ENVIRONMENT},
        "cpu_pressure": read_optional("/proc/pressure/cpu"),
        "memory_info": read_optional("/proc/meminfo"),
        "cgroup_membership": read_optional("/proc/self/cgroup"),
        "cgroup_v2_cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
        "cgroup_v2_cpu_stat": read_optional("/sys/fs/cgroup/cpu.stat"),
        "cpu_frequencies": frequencies,
    }


def runtime_preflight(values, environment):
    errors = []
    paths = {name: environment.get(name) for name in RUNTIME_ENVIRONMENT}
    toolkit = Path(values["toolkit_home"]).resolve()
    securec = Path(values.get("securec_include") or toolkit / "include") / "securec.h"
    if not securec.is_file():
        errors.append("securec.h missing; set --securec-include to its actual directory")
    for name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH"):
        value = paths[name]
        if not value or not Path(value).is_dir():
            errors.append(f"{name} must name an existing directory; source the selected CANN set_env.sh first")
    if paths["ASCEND_HOME_PATH"] and Path(paths["ASCEND_HOME_PATH"]).is_dir():
        home = Path(paths["ASCEND_HOME_PATH"]).resolve()
        if not (toolkit.is_relative_to(home) or home.is_relative_to(toolkit)):
            errors.append("ASCEND_HOME_PATH and --toolkit-home select different installations")
    if paths["MX_INDEX_USE_ONLINEOP"] not in (None, "0", "1"):
        errors.append("MX_INDEX_USE_ONLINEOP must be unset, 0 or 1")
    if paths["MX_INDEX_USE_ONLINEOP"] == "1":
        sdk_mode = "online"
    else:
        sdk_mode = "offline"
        model = paths["MX_INDEX_MODELPATH"]
        if not model or not Path(model).is_dir() or not any(Path(model).rglob("*.om")):
            errors.append(
                "SDK offline models are not configured; this shared-resource acceptance can use MX_INDEX_USE_ONLINEOP=1"
            )
    libraries = [Path(values["sdk_library"])]
    vendors = Path(values["vendors"])
    packages = []
    for alias in ("bm25_fused_score_topk", "bm25_topk_merge"):
        vendor = (vendors / alias).resolve()
        configurations = list(vendor.rglob("binary_info_config.json"))
        tiling = list(vendor.glob("op_impl/ai_core/tbe/op_tiling/lib/linux/*/*.so"))
        if not configurations:
            errors.append(f"{alias}: binary_info_config.json missing")
        if not tiling:
            errors.append(f"{alias}: actual host tiling library missing")
        for config in configurations:
            try:
                data = json.loads(config.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not data:
                    errors.append(f"Empty/invalid operator configuration: {config}")
            except (OSError, ValueError) as error:
                errors.append(f"Cannot read operator configuration {config}: {error}")
        packages.append(
            {
                "alias": alias,
                "directory": str(vendor),
                "configurations": list(map(str, configurations)),
                "tiling_libraries": list(map(str, tiling)),
            }
        )
        libraries += [vendor / "op_api/lib/libcust_opapi.so", *tiling]
    dependencies = []
    if platform.system() == "Linux":
        for library in dict.fromkeys(libraries):
            result = command_output(["ldd", str(library)], environment)
            dependencies.append(result)
            if result["exit_code"] != 0 or "not found" in result["output"]:
                errors.append(f"Runtime library dependencies unresolved: {library}")
    else:
        errors.append("Runtime preflight requires the real Linux CANN environment")
    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "runtime_environment": paths,
        "sdk_operator_mode": sdk_mode,
        "offline_model_execution_validated": False,
        "operator_packages": packages,
        "library_dependencies": dependencies,
        "cpu": cpu_snapshot(environment),
        "device_execution_performed": False,
    }
