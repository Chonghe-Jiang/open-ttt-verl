# Copyright 2026 Chonghe Jiang and/or contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


STDCPP_INCLUDE_BUNDLE = """#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
"""


@dataclass
class CppRunResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


@dataclass
class CppSandboxResult:
    binary_path: Path | None
    work_dir: Path
    compile_stdout: str
    compile_stderr: str
    error: str | None


def extract_cpp_code(response: str) -> str | None:
    patterns = [
        r"```(?:cpp|c\+\+|cc|cxx)\s+([\s\S]*?)\s*```",
        r"```\s+([\s\S]*?#include[\s\S]*?)\s*```",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()
    open_block = list(re.finditer(r"```(?:cpp|c\+\+|cc|cxx)\s+([\s\S]*)\Z", response, re.IGNORECASE))
    if open_block:
        code = open_block[-1].group(1).strip()
        return code or None
    stripped = response.strip()
    if "#include" in stripped and "int main" in stripped:
        return stripped
    if stripped.startswith("using namespace std") and "int main" in stripped:
        return "#include <bits/stdc++.h>\n" + stripped
    return None


def compile_cpp_code(code: str, *, timeout_s: int, work_dir: str | Path | None = None) -> CppSandboxResult:
    parent = Path(work_dir) if work_dir else None
    tmp = tempfile.TemporaryDirectory(dir=str(parent) if parent else None)
    tmp_path = Path(tmp.name)
    source_path = tmp_path / "solution.cpp"
    binary_path = tmp_path / "solution"
    portable_code = code.replace("#include <bits/stdc++.h>", STDCPP_INCLUDE_BUNDLE)
    source_path.write_text(portable_code)
    try:
        proc = subprocess.run(
            ["g++", "-O2", "-pipe", "-std=gnu++17", str(source_path), "-o", str(binary_path)],
            cwd=tmp_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(1, int(timeout_s)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        tmp.cleanup()
        return CppSandboxResult(
            binary_path=None,
            work_dir=tmp_path,
            compile_stdout=exc.stdout or "",
            compile_stderr=exc.stderr or "",
            error=f"Compile timed out after {timeout_s}s",
        )

    if proc.returncode != 0:
        tmp.cleanup()
        return CppSandboxResult(
            binary_path=None,
            work_dir=tmp_path,
            compile_stdout=proc.stdout,
            compile_stderr=proc.stderr,
            error=f"Compile failed with exit code {proc.returncode}",
        )

    result = CppSandboxResult(
        binary_path=binary_path,
        work_dir=tmp_path,
        compile_stdout=proc.stdout,
        compile_stderr=proc.stderr,
        error=None,
    )
    result._tmp = tmp  # type: ignore[attr-defined]
    return result


def run_cpp_binary(binary_path: Path, input_text: str, *, timeout_s: int) -> CppRunResult:
    try:
        proc = subprocess.run(
            [str(binary_path)],
            input=input_text,
            cwd=binary_path.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(1, int(timeout_s)),
            check=False,
        )
        return CppRunResult(stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
    except subprocess.TimeoutExpired as exc:
        return CppRunResult(
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            returncode=-1,
            timed_out=True,
        )


def cleanup_cpp_sandbox(result: CppSandboxResult) -> None:
    tmp = getattr(result, "_tmp", None)
    if tmp is not None:
        tmp.cleanup()
