#!/usr/bin/env python3
"""Print and validate the runtime fingerprint for TTT GPT-OSS runs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


DEFAULT_DISTRIBUTIONS = [
    ("torch", "torch"),
    ("vllm", "vllm"),
    ("transformers", "transformers"),
    ("flash-attn", "flash_attn"),
    ("triton", "triton"),
    ("flashinfer-python", None),
    ("ray", "ray"),
    ("peft", "peft"),
    ("accelerate", "accelerate"),
    ("tensordict", "tensordict"),
    ("torchdata", "torchdata"),
    ("datasets", "datasets"),
    ("hydra-core", "hydra"),
    ("sentencepiece", "sentencepiece"),
    ("tiktoken", "tiktoken"),
    ("protobuf", "google.protobuf"),
]


def _run(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def _dist_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _import_info(module_name: str | None) -> dict[str, Any]:
    if not module_name:
        return {"module": None, "importable": None}
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "module": module_name,
            "importable": False,
            "error": repr(exc),
        }
    return {
        "module": module_name,
        "importable": True,
        "import_version": getattr(module, "__version__", None),
        "file": getattr(module, "__file__", None),
    }


def _torch_gpu_info(torch_module: Any | None) -> dict[str, Any]:
    if torch_module is None:
        return {"cuda_available": None, "device_count": 0, "devices": []}
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception:
        cuda_available = False
    devices = []
    if cuda_available:
        try:
            count = torch_module.cuda.device_count()
            for idx in range(count):
                devices.append(
                    {
                        "index": idx,
                        "name": torch_module.cuda.get_device_name(idx),
                        "capability": list(torch_module.cuda.get_device_capability(idx)),
                    }
                )
        except Exception as exc:
            devices.append({"error": repr(exc)})
    return {
        "cuda_available": cuda_available,
        "device_count": len(devices),
        "devices": devices,
    }


def _hub_kernel_paths() -> list[str]:
    candidates = []
    for root in [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        str(Path.home() / ".cache" / "huggingface"),
    ]:
        if not root:
            continue
        hub_root = Path(root).expanduser()
        if (hub_root / "hub").exists():
            hub_root = hub_root / "hub"
        if not hub_root.exists():
            continue
        try:
            for path in hub_root.glob("models--kernels-community--vllm-flash-attn3*"):
                candidates.append(str(path))
        except OSError:
            continue
    return sorted(set(candidates))


def collect_fingerprint() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    torch_module = None
    for distribution, module_name in DEFAULT_DISTRIBUTIONS:
        info: dict[str, Any] = {
            "distribution": distribution,
            "dist_version": _dist_version(distribution),
        }
        import_info = _import_info(module_name)
        info.update(import_info)
        if distribution == "torch" and import_info.get("importable"):
            torch_module = importlib.import_module("torch")
            info["torch_cuda"] = getattr(torch_module.version, "cuda", None)
        packages[distribution] = info

    nvidia_smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "USE_HUB_KERNELS": os.environ.get("USE_HUB_KERNELS"),
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
        },
        "tools": {
            "nvidia_smi": nvidia_smi,
            "nvcc": _run([shutil.which("nvcc") or "nvcc", "--version"]) if shutil.which("nvcc") else None,
        },
        "gpu": _torch_gpu_info(torch_module),
        "packages": packages,
        "hf_hub_kernels": {
            "vllm_flash_attn3_paths": _hub_kernel_paths(),
        },
    }


def _load_lock(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _locked_packages(lock: dict[str, Any]) -> dict[str, Any]:
    packages = lock.get("packages", {})
    return {name: spec for name, spec in packages.items() if spec.get("required", True)}


def emit_pip_requirements(lock: dict[str, Any], exclude: set[str]) -> str:
    lines = []
    for name, spec in _locked_packages(lock).items():
        if name in exclude:
            continue
        version = spec.get("dist_version") or spec.get("version")
        if not version:
            continue
        lines.append(f"{name}=={version}")
    return "\n".join(lines) + ("\n" if lines else "")


def package_version_from_lock(lock: dict[str, Any], package: str) -> str:
    spec = lock.get("packages", {}).get(package)
    version = (spec.get("version") or spec.get("dist_version")) if spec else None
    if not version:
        raise SystemExit(f"Package {package!r} is not present in the lock file")
    return str(version)


def compare_fingerprint(
    actual: dict[str, Any],
    lock: dict[str, Any],
    *,
    strict_python: bool,
    strict_gpu: bool,
    forbid_vllm_flash_attn3: bool,
) -> list[str]:
    errors: list[str] = []

    expected_env = lock.get("env", {})
    for key, expected in expected_env.items():
        if expected is None:
            continue
        actual_value = actual.get("env", {}).get(key)
        if actual_value != expected:
            errors.append(f"env.{key}: expected {expected!r}, got {actual_value!r}")

    if strict_python:
        expected_python = lock.get("python", {}).get("version")
        actual_python = actual.get("python", {}).get("version")
        if expected_python and actual_python != expected_python:
            errors.append(f"python.version: expected {expected_python!r}, got {actual_python!r}")

    for name, expected in _locked_packages(lock).items():
        actual_pkg = actual.get("packages", {}).get(name, {})
        for field in ("dist_version", "import_version", "importable", "torch_cuda"):
            if field not in expected:
                continue
            if actual_pkg.get(field) != expected[field]:
                errors.append(f"packages.{name}.{field}: expected {expected[field]!r}, got {actual_pkg.get(field)!r}")

    if strict_gpu:
        expected_devices = lock.get("gpu", {}).get("devices", [])
        actual_devices = actual.get("gpu", {}).get("devices", [])
        if expected_devices and actual_devices:
            expected_cap = expected_devices[0].get("capability")
            actual_cap = actual_devices[0].get("capability")
            if expected_cap and actual_cap != expected_cap:
                errors.append(f"gpu.capability: expected {expected_cap!r}, got {actual_cap!r}")

    paths = actual.get("hf_hub_kernels", {}).get("vllm_flash_attn3_paths", [])
    if forbid_vllm_flash_attn3 and paths:
        errors.append("HF cache contains kernels-community/vllm-flash-attn3: " + ", ".join(paths))
    return errors


def print_human(fingerprint: dict[str, Any]) -> None:
    print(f"python={fingerprint['python']['version']}")
    print(f"USE_HUB_KERNELS={fingerprint['env'].get('USE_HUB_KERNELS')}")
    for name, info in fingerprint["packages"].items():
        print(
            f"{name}={info.get('dist_version')} import_version={info.get('import_version')} "
            f"importable={info.get('importable')} "
            f"module={info.get('module')} file={info.get('file')}"
        )
    print(f"gpu_devices={fingerprint['gpu'].get('devices')}")
    print(f"vllm_flash_attn3_paths={fingerprint['hf_hub_kernels'].get('vllm_flash_attn3_paths')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the current fingerprint as JSON.")
    parser.add_argument("--compare-lock", help="Compare the current environment against a lock JSON file.")
    parser.add_argument("--lock", help="Read a lock JSON file for helper operations.")
    parser.add_argument("--emit-pip-requirements", action="store_true", help="Emit pinned pip requirements from --lock.")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude a package from emitted requirements.")
    parser.add_argument("--package-version", help="Print one package version from --lock.")
    parser.add_argument("--strict-python", action="store_true", help="Require exact Python version match.")
    parser.add_argument("--strict-gpu", action="store_true", help="Require GPU capability match.")
    parser.add_argument(
        "--forbid-vllm-flash-attn3",
        action="store_true",
        help="Fail if kernels-community/vllm-flash-attn3 is present in the HF cache.",
    )
    args = parser.parse_args()

    if args.emit_pip_requirements:
        if not args.lock:
            raise SystemExit("--emit-pip-requirements requires --lock")
        print(emit_pip_requirements(_load_lock(args.lock), set(args.exclude)), end="")
        return

    if args.package_version:
        if not args.lock:
            raise SystemExit("--package-version requires --lock")
        print(package_version_from_lock(_load_lock(args.lock), args.package_version))
        return

    fingerprint = collect_fingerprint()

    if args.compare_lock:
        lock = _load_lock(args.compare_lock)
        errors = compare_fingerprint(
            fingerprint,
            lock,
            strict_python=args.strict_python,
            strict_gpu=args.strict_gpu,
            forbid_vllm_flash_attn3=args.forbid_vllm_flash_attn3,
        )
        if errors:
            print("Runtime fingerprint mismatch:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            raise SystemExit(1)
        print("Runtime fingerprint matches lock.")
        return

    if args.json:
        print(json.dumps(fingerprint, indent=2, sort_keys=True))
    else:
        print_human(fingerprint)


if __name__ == "__main__":
    main()
