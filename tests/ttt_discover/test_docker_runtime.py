import json
from pathlib import Path

from scripts.ttt_discover import docker_runtime_guard


def test_docker_runtime_guard_detects_vllm_flash_attn3_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "hub" / "models--kernels-community--vllm-flash-attn3"
    cache_path.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    assert docker_runtime_guard._find_vllm_flash_attn3_cache() == [str(cache_path)]


def test_ttt_vllm_docker_runtime_forces_b200_flash_attention_path():
    dockerfile = Path("docker/Dockerfile.ttt-vllm").read_text()
    build_script = Path("scripts/ttt_discover/docker_build_ttt_vllm.sh").read_text()
    run_script = Path("scripts/ttt_discover/docker_run_ttt_vllm.sh").read_text()
    lock = json.loads(Path("docker/b200_vllm017_env.lock.json").read_text())

    assert lock["packages"]["torch"]["version_prefix"] == "2.10.0"
    assert lock["packages"]["torch"]["torch_cuda"] == "12.9"
    assert lock["packages"]["vllm"]["version"] == "0.17.0"
    assert lock["packages"]["flash-attn"]["version"] == "2.8.3"
    assert lock["env"]["USE_HUB_KERNELS"] == "0"
    assert "USE_HUB_KERNELS=0" in dockerfile
    assert "EXPECT_TORCH_PREFIX=2.10.0" in dockerfile
    assert "EXPECT_TORCH_CUDA=12.9" in dockerfile
    assert "EXPECT_VLLM_VERSION=0.17.0" in dockerfile
    assert "EXPECT_FLASH_ATTN_VERSION=2.8.3" in dockerfile
    assert "RUNTIME_LOCK=docker/b200_vllm017_env.lock.json" in dockerfile
    assert 'RUNTIME_LOCK="${RUNTIME_LOCK:-docker/b200_vllm017_env.lock.json}"' in build_script
    assert '--build-arg "RUNTIME_LOCK=${RUNTIME_LOCK}"' in build_script
    assert "--forbid-vllm-flash-attn3" in dockerfile
    assert 'USE_HUB_KERNELS="${USE_HUB_KERNELS:-0}"' in run_script
    assert 'RUNTIME_LOCK="${RUNTIME_LOCK:-docker/b200_vllm017_env.lock.json}"' in run_script
    assert '-e "USE_HUB_KERNELS=${USE_HUB_KERNELS}"' in run_script
    assert '-e "RUNTIME_LOCK=${RUNTIME_LOCK}"' in run_script
    assert "preflight-forward" in run_script
    assert 'attn_impl = os.environ.get("ATTN_IMPL") or "flash_attention_2"' in run_script
    assert "attn_implementation=attn_impl" in run_script
    assert "scripts/ttt_discover/docker_runtime_guard.py" in run_script
    assert run_script.index("scripts/ttt_discover/docker_runtime_guard.py") < run_script.index(
        "scripts/ttt_discover/run_erdos_gptoss_bf16_4gpu_b200.sh"
    )


def test_ttt_vllm_docker_runtime_exposes_qwen3_8b_official_entrypoint():
    run_script = Path("scripts/ttt_discover/docker_run_ttt_vllm.sh").read_text()
    readme = Path("README.md").read_text()

    assert "run-qwen8b" in run_script
    assert 'MODE}" == "run-qwen8b"' in run_script
    assert 'CONFIG="verl_ttt_discover/config/erdos_4gpu_b200_qwen3_8b_official.yaml"' in run_script
    assert "scripts/ttt_discover/run_erdos_qwen3_8b_4gpu_b200.sh" in run_script
    assert "erdos_4gpu_b200_qwen3_8b_official.yaml" in readme
    assert "run-qwen8b" in readme
