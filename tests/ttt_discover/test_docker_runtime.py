from pathlib import Path


def test_ttt_vllm_docker_runtime_exposes_qwen3_8b_official_entrypoint():
    run_script = Path("scripts/ttt_discover/docker_run_ttt_vllm.sh").read_text()
    readme = Path("README.md").read_text()

    assert "run-qwen8b" in run_script
    assert 'MODE}" == "run-qwen8b"' in run_script
    assert 'CONFIG="verl_ttt_discover/config/erdos_4gpu_b200_qwen3_8b_official.yaml"' in run_script
    assert "scripts/ttt_discover/run_erdos_qwen3_8b_4gpu_b200.sh" in run_script
    assert "erdos_4gpu_b200_qwen3_8b_official.yaml" in readme
    assert "run-qwen8b" in readme


def test_modal_qwen3_8b_runtime_uses_h200_g4_n16_config_and_persistent_volumes():
    modal_app = Path("scripts/ttt_discover/modal_qwen3_8b_erdos.py").read_text()
    modal_wrapper = Path("scripts/ttt_discover/modal_run_qwen3_8b_erdos.sh").read_text()
    readme = Path("README.md").read_text()

    assert 'APP_NAME = "polynomino-qwen3-8b-erdos"' in modal_app
    assert 'DEFAULT_CONFIG = "verl_ttt_discover/config/erdos_2gpu_h200_qwen3_8b_g4_n16.yaml"' in modal_app
    assert 'DEFAULT_BASE_IMAGE = "verlai/verl:vllm017.latest"' in modal_app
    assert 'DEFAULT_GPUS = "0,1"' in modal_app
    assert "def _resolve_repo_root()" in modal_app
    assert 'Path(os.environ.get("MODAL_REPO_ROOT", REMOTE_ROOT))' in modal_app
    assert "_load_modal_credentials_from_dotenv()" in modal_app
    assert '"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_SECRET_KEY"' in modal_app
    assert '"PYTORCH_ALLOC_CONF": "expandable_segments:True"' in modal_app
    assert "flash_attn_version=" in modal_app
    assert "attn_implementation='flash_attention_2'" in modal_app
    assert 'gpu=_env("MODAL_GPU", "H200:2")' in modal_app
    assert 'modal.Volume.from_name("polynomino-qwen3-8b-hf-cache", create_if_missing=True)' in modal_app
    assert 'modal.Volume.from_name("polynomino-qwen3-8b-outputs", create_if_missing=True)' in modal_app
    assert "scripts/ttt_discover/run_erdos_qwen3_8b_2gpu_h200.sh" in modal_app
    assert "MODAL_SECRET_KEY" in modal_wrapper
    assert "MODAL_TOKEN_SECRET" in modal_wrapper
    assert "modal_run_qwen3_8b_erdos.sh prepare" in readme
    assert "GPU request `H200:2`" in readme
