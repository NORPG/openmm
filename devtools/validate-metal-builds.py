#!/usr/bin/env python3
"""Build and GPU-test the offline/runtime x shared/static Metal matrix.

Requires Python 3.8+, macOS on Apple Silicon, a visible Metal GPU, CMake, Ninja
(by default), and Xcode's Metal Toolchain for the offline cases.  No third-party
Python modules are needed.  A skipped GPU test is a validation failure.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


SOURCE = Path(__file__).resolve().parents[1]
CASES = ("offline-shared", "offline-static", "runtime-shared", "runtime-static")
TESTS = ("TestMetalComputeContext", "TestMetalPlatform", "TestMetalRuntime",
         "TestMetalVerticalSlice", "TestMetalNonbondedForce")
DISABLED = ("C_AND_FORTRAN_WRAPPERS", "PYTHON_WRAPPERS", "CPU_LIB", "CUDA_LIB",
            "OPENCL_LIB", "HIP_LIB", "COMMON", "AMOEBA_PLUGIN", "RPMD_PLUGIN",
            "DRUDE_PLUGIN", "PME_PLUGIN", "REFERENCE_TESTS",
            "SERIALIZATION_TESTS", "EXAMPLES")


def run(command, directory, log):
    print("  {} (log: {})".format(shlex.join(command), log), flush=True)
    with log.open("w") as output:
        result = subprocess.run(command, cwd=directory, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-35:])
        raise RuntimeError("Command exited with {}:\n{}".format(result.returncode, tail))


def read_cache(directory):
    result = {}
    path = directory / "CMakeCache.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            match = re.match(r"([^#/:][^:]*):[^=]+=(.*)", line)
            if match:
                result[match[1]] = match[2]
    return result


def check_results(directory, expected, mode):
    """Check actual CTest records, including legacy tests returning 0 on skip."""
    log = (directory / "test.log").read_text(errors="replace")
    if "Test skipped:" in log:
        raise RuntimeError("A Metal test skipped GPU execution; see test.log")
    marker = "Fixed-point oracle kernels: " + (
        "offline metallib" if mode == "offline" else "runtime compilation")
    markers = re.findall(r"Fixed-point oracle kernels: [^\r\n]+", log)
    if markers != [marker]:
        raise RuntimeError("Expected oracle path {!r}, observed {!r}".format(marker, markers))
    tag = (directory / "Testing" / "TAG").read_text().splitlines()[0]
    records = ET.parse(directory / "Testing" / tag / "Test.xml").getroot().findall("./Testing/Test")
    results = [{"name": test.findtext("Name"), "status": test.get("Status")}
               for test in records]
    if sorted(test["name"] for test in results) != sorted(expected):
        raise RuntimeError("CTest did not record exactly the five expected Metal tests: {!r}".format(results))
    if any(test["status"] != "passed" for test in results):
        raise RuntimeError("Not all GPU tests passed: {!r}".format(results))
    return results


def validate_case(name, args, result):
    mode, linkage = name.split("-")
    directory = args.build_root / name
    directory.mkdir(parents=True, exist_ok=True)
    result["build_directory"] = str(directory)
    result["phase"] = "configure"
    cache = read_cache(directory)
    if cache and Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != SOURCE:
        raise RuntimeError("Refusing to reconfigure a build belonging to another source tree")
    options = {
        "BUILD_TESTING": "ON",
        "OPENMM_BUILD_METAL_LIB": "ON",
        "OPENMM_BUILD_METAL_TESTS": "ON",
        "OPENMM_METAL_KERNEL_COMPILATION": "ON" if mode == "offline" else "OFF",
        "OPENMM_BUILD_SHARED_LIB": "ON" if linkage == "shared" else "OFF",
        "OPENMM_BUILD_STATIC_LIB": "ON" if linkage == "static" else "OFF",
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_OSX_ARCHITECTURES": "arm64",
        "CMAKE_OSX_DEPLOYMENT_TARGET": "13.0",
    }
    options.update(("OPENMM_BUILD_" + name, "OFF") for name in DISABLED)
    run(["cmake", "-S", str(SOURCE), "-B", str(directory), "-G", args.generator]
        + ["-D{}={}".format(key, value) for key, value in options.items()],
        SOURCE, directory / "configure.log")
    cache = read_cache(directory)
    for key, value in options.items():
        if cache.get(key) != value:
            raise RuntimeError("Unexpected {}={!r}; expected {!r}".format(key, cache.get(key), value))
    result["configuration"] = options

    expected = [name + ("Static" if linkage == "static" else "") for name in TESTS]
    pattern = "^(" + "|".join(expected) + ")$"
    result["phase"] = "test-discovery"
    run(["ctest", "-C", "Release", "--show-only=json-v1", "-R", pattern], directory,
        directory / "test-discovery.json")
    discovered = json.loads((directory / "test-discovery.json").read_text())
    if sorted(test["name"] for test in discovered["tests"]) != sorted(expected):
        raise RuntimeError("The configured build is missing expected Metal tests")

    result["phase"] = "build"
    run(["cmake", "--build", str(directory), "--config", "Release", "--parallel", str(args.jobs),
         "--target"] + expected, SOURCE, directory / "build.log")

    result["phase"] = "test"
    # Dashboard XML is available at the project's CMake 3.17 minimum, unlike
    # newer JUnit options.  Verbose output exposes legacy success-on-skip tests.
    run(["ctest", "-C", "Release", "-T", "Test", "-R", pattern, "-V",
         "--output-on-failure", "--timeout", "180", "--parallel", "1"], directory,
        directory / "test.log")
    result["tests"] = check_results(directory, expected, mode)
    result["oracle_compilation"] = mode
    result["phase"] = "complete"


def git_output(*arguments):
    try:
        return subprocess.check_output(["git"] + list(arguments), cwd=SOURCE,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=SOURCE / "build" / "metal-matrix",
                        help="parent of four dedicated build directories (default: build/metal-matrix)")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--generator", default="Ninja")
    parser.add_argument("--case", choices=CASES, action="append", dest="cases",
                        help="run only this case; repeat to select several (default: all four)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("This GPU validation requires native Apple Silicon macOS")
    args.build_root = args.build_root.resolve()
    args.build_root.mkdir(parents=True, exist_ok=True)
    cases = list(dict.fromkeys(args.cases or CASES))
    status = git_output("status", "--porcelain")
    report = {"source": str(SOURCE), "revision": git_output("rev-parse", "HEAD"),
              "dirty": None if status is None else bool(status),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "host": platform.platform(), "cases": [], "status": "running",
              "full_matrix": set(cases) == set(CASES)}
    report_path = args.build_root / "validation-results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for name in cases:
        print("\n=== {} ===".format(name), flush=True)
        started = time.monotonic()
        result = {"case": name, "status": "running"}
        report["cases"].append(result)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        try:
            validate_case(name, args, result)
            result["status"] = "passed"
        except (OSError, ValueError, KeyError, RuntimeError, ET.ParseError) as error:
            result["status"] = "failed"
            result["error"] = str(error)
            print("  FAILED ({}): {}".format(result.get("phase", "setup"), error), flush=True)
        result["seconds"] = round(time.monotonic() - started, 2)
        print("  {}: {} ({:.2f}s)".format(name, result["status"], result["seconds"]), flush=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    passed = all(result["status"] == "passed" for result in report["cases"])
    report["status"] = "passed" if passed else "failed"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("\n{}: {} selected cases. Report: {}".format(report["status"].upper(), len(cases), report_path))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
