from scripts.ttt_discover.env_fingerprint import compare_fingerprint, emit_pip_requirements, package_version_from_lock


def test_emit_pip_requirements_uses_locked_dist_versions():
    lock = {
        "packages": {
            "torch": {"dist_version": "2.11.0"},
            "flash-attn": {"dist_version": "2.8.3"},
            "optional-missing": {"required": False, "dist_version": "1.0.0"},
        }
    }

    requirements = emit_pip_requirements(lock, {"flash-attn"})

    assert requirements == "torch==2.11.0\n"


def test_package_version_from_lock_accepts_dist_version():
    lock = {"packages": {"flash-attn": {"dist_version": "2.8.3"}}}

    assert package_version_from_lock(lock, "flash-attn") == "2.8.3"


def test_compare_fingerprint_reports_package_mismatch():
    actual = {
        "env": {"USE_HUB_KERNELS": "0"},
        "packages": {"vllm": {"dist_version": "0.17.0", "importable": True}},
        "hf_hub_kernels": {"vllm_flash_attn3_paths": []},
    }
    lock = {
        "env": {"USE_HUB_KERNELS": "0"},
        "packages": {"vllm": {"dist_version": "0.21.0", "importable": True}},
    }

    errors = compare_fingerprint(
        actual,
        lock,
        strict_python=False,
        strict_gpu=False,
        forbid_vllm_flash_attn3=True,
    )

    assert errors == ["packages.vllm.dist_version: expected '0.21.0', got '0.17.0'"]


def test_compare_fingerprint_forbids_vllm_flash_attn3_cache():
    actual = {
        "env": {"USE_HUB_KERNELS": "0"},
        "packages": {},
        "hf_hub_kernels": {"vllm_flash_attn3_paths": ["/hf_cache/hub/models--kernels-community--vllm-flash-attn3"]},
    }
    lock = {"env": {"USE_HUB_KERNELS": "0"}, "packages": {}}

    errors = compare_fingerprint(
        actual,
        lock,
        strict_python=False,
        strict_gpu=False,
        forbid_vllm_flash_attn3=True,
    )

    assert "kernels-community/vllm-flash-attn3" in errors[0]
