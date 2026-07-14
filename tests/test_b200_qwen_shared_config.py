from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "guidance_ttt/config"
SMOKE_CONFIG = CONFIG_DIR / "polyomino_b200_4gpu_qwen3_8b_exec_shared_batch8_group32_code_delta_8192_smoke.yaml"
FALLBACK_CONFIG = (
    CONFIG_DIR / "polyomino_b200_4gpu_qwen3_8b_exec_shared_batch8_group32_code_delta_8192_fallback_smoke.yaml"
)
LONG_CONFIG = CONFIG_DIR / "polyomino_b200_4gpu_qwen3_8b_exec_shared_batch8_group32_code_delta_8192_50step.yaml"


def test_qwen_shared_smoke_keeps_four_training_gpus_and_8x32_shape():
    config = yaml.safe_load(SMOKE_CONFIG.read_text())

    assert config["run"]["model_path"] == "models/Qwen3-8B"
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["run"]["num_steps"] == 1
    assert config["run"]["ppo_mini_batch_size"] == 8
    assert config["run"]["ppo_micro_batch_size_per_gpu"] == 2
    assert config["run"]["gpu_memory_utilization"] == 0.55
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 32
    assert config["ttt"]["prompt_mode"] == "code_delta"


def test_qwen_shared_execution_uses_native_thinking_sampling():
    execution = yaml.safe_load(SMOKE_CONFIG.read_text())["llm"]["execution"]

    assert execution["model"] == "Qwen/Qwen3-8B"
    assert execution["prompt_style"] == "qwen_native_thinking"
    assert execution["chat_template_kwargs"] == {"enable_thinking": True}
    assert execution["temperature"] == 0.6
    assert execution["top_p"] == 0.95
    assert execution["top_k"] == 20
    assert execution["min_p"] == 0.0
    assert execution["max_tokens"] == 16384
    assert execution["concurrency"] == 4


def test_qwen_shared_fallback_preserves_batch_and_reduces_memory_pressure():
    config = yaml.safe_load(FALLBACK_CONFIG.read_text())

    assert config["run"]["gpu_memory_utilization"] == 0.50
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 32
    assert config["llm"]["execution"]["concurrency"] == 2
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.50" in config["verl_overrides"]


def test_qwen_shared_long_run_uses_native_context_and_saves_every_ten_steps():
    config = yaml.safe_load(LONG_CONFIG.read_text())

    assert config["run"]["num_steps"] == 50
    assert config["run"]["total_epochs"] == 50
    assert config["run"]["save_freq"] == 10
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 32
    assert config["llm"]["execution"]["max_tokens"] is None


def test_qwen_shared_slurm_layout_and_reasoning_server_flags():
    smoke = (ROOT / "scripts/slurm_polyomino_b200_4gpu_qwen_exec_shared_group32_smoke.sbatch").read_text()
    long_run = (ROOT / "scripts/slurm_polyomino_b200_4gpu_qwen_exec_shared_group32_50step.sbatch").read_text()
    runner = (ROOT / "scripts/run_polyomino_b200_4gpu_qwen_exec_shared.sh").read_text()

    for script in (smoke, long_run):
        assert "#SBATCH --gres=gpu:b200:4" in script
        assert "export TRAINING_GPUS=0,1,2,3" in script
        assert "export EXECUTION_GPU=3" in script
    assert "COLOCATION_PROFILE" in smoke
    assert "EXECUTION_GPU_MEMORY_UTILIZATION=0.18" in smoke
    assert "--reasoning-parser qwen3" in runner
    assert "--enable-reasoning" not in runner
    assert "--served-model-name Qwen/Qwen3-8B" in runner
    assert "models/gpt-oss-120b/config.json" not in runner
    assert "nvidia-smi-qwen-shared-monitor" in runner
    assert "validate_qwen_shared_smoke.py" in runner
