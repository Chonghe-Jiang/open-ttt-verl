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
