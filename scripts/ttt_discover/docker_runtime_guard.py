#!/usr/bin/env python3
"""Guard the Docker runtime used by B200 GPT-OSS TTT runs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _find_vllm_flash_attn3_cache() -> list[str]:
    roots = [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        str(Path.home() / ".cache" / "huggingface"),
    ]
    paths: list[str] = []
    for root in roots:
        if not root:
            continue
        hub_root = Path(root).expanduser()
        if (hub_root / "hub").exists():
            hub_root = hub_root / "hub"
        if not hub_root.exists():
            continue
        paths.extend(str(path) for path in hub_root.glob("models--kernels-community--vllm-flash-attn3*"))
    return sorted(set(paths))


def _load_lock(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def _require_equal(errors: list[str], label: str, actual: str | None, expected: str | None) -> None:
    if expected and actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def _require_prefix(errors: list[str], label: str, actual: str | None, expected_prefix: str | None) -> None:
    if expected_prefix and (actual is None or not actual.startswith(expected_prefix)):
        errors.append(f"{label}: expected prefix {expected_prefix!r}, got {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-flash-attn", action="store_true")
    parser.add_argument("--require-use-hub-kernels-zero", action="store_true")
    parser.add_argument("--forbid-vllm-flash-attn3", action="store_true")
    parser.add_argument("--lock", default=os.environ.get("RUNTIME_LOCK"), help="Runtime lock JSON file.")
    args = parser.parse_args()

    lock = _load_lock(args.lock)
    package_lock = lock.get("packages", {})
    env_lock = lock.get("env", {})

    expected_torch_prefix = os.environ.get(
        "EXPECT_TORCH_PREFIX", package_lock.get("torch", {}).get("version_prefix", "2.10.0")
    )
    expected_torch_cuda = os.environ.get(
        "EXPECT_TORCH_CUDA", package_lock.get("torch", {}).get("torch_cuda", "12.9")
    )
    expected_vllm = os.environ.get("EXPECT_VLLM_VERSION", package_lock.get("vllm", {}).get("version", "0.17.0"))
    expected_flash_attn = os.environ.get(
        "EXPECT_FLASH_ATTN_VERSION", package_lock.get("flash-attn", {}).get("version", "2.8.3")
    )
    expected_use_hub_kernels = env_lock.get("USE_HUB_KERNELS", "0")

    errors: list[str] = []
    torch = importlib.import_module("torch")
    vllm = importlib.import_module("vllm")
    flash_attn = importlib.import_module("flash_attn")

    print(f"torch={torch.__version__}")
    print(f"torch_dist={_dist_version('torch')}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"vllm={getattr(vllm, '__version__', None)}")
    print(f"vllm_dist={_dist_version('vllm')}")
    print(f"flash_attn={getattr(flash_attn, '__version__', None)}")
    print(f"flash_attn_dist={_dist_version('flash-attn')}")
    print(f"USE_HUB_KERNELS={os.environ.get('USE_HUB_KERNELS')}")
    print(f"HF_HOME={os.environ.get('HF_HOME')}")
    print(f"RUNTIME_LOCK={args.lock}")

    _require_prefix(errors, "torch.__version__", torch.__version__, expected_torch_prefix)
    _require_equal(errors, "torch.version.cuda", torch.version.cuda, expected_torch_cuda)
    _require_equal(errors, "vllm.__version__", getattr(vllm, "__version__", None), expected_vllm)
    _require_equal(errors, "flash_attn.__version__", getattr(flash_attn, "__version__", None), expected_flash_attn)

    if args.require_cuda:
        cuda_available = bool(torch.cuda.is_available())
        print(f"cuda_available={cuda_available}")
        print(f"cuda_device_count={torch.cuda.device_count() if cuda_available else 0}")
        if not cuda_available:
            errors.append("CUDA is not available inside the container")

    if args.require_use_hub_kernels_zero and os.environ.get("USE_HUB_KERNELS") != expected_use_hub_kernels:
        errors.append(
            f"USE_HUB_KERNELS must be set to {expected_use_hub_kernels!r} to avoid the HF vllm-flash-attn3 GPT-OSS path"
        )

    if args.check_flash_attn:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input  # noqa: F401
        from flash_attn.ops.triton.rotary import apply_rotary  # noqa: F401

        print("import flash_attn.bert_padding: ok")
        print("import flash_attn.ops.triton.rotary: ok")

    flash_attn3_paths = _find_vllm_flash_attn3_cache()
    print(f"vllm_flash_attn3_cache_paths={flash_attn3_paths}")
    if args.forbid_vllm_flash_attn3 and flash_attn3_paths and os.environ.get("ALLOW_VLLM_FLASH_ATTN3_CACHE") != "1":
        errors.append(
            "HF cache contains kernels-community/vllm-flash-attn3; remove it or set a clean HF_HOME before B200 GPT-OSS TTT"
        )

    if errors:
        print("Docker runtime guard failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)

    print("Docker runtime guard passed.")


if __name__ == "__main__":
    main()
