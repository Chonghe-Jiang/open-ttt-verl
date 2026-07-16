from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "guidance_ttt/config"
CONFIGS = {
    "code_delta": CONFIG_DIR
    / "polyomino_b200_2gpu_qwen3_8b_qwen3_coder_next_fp8_exec_batch8_group8_code_delta_entropic_best_child_500step.yaml",
    "summary_only": CONFIG_DIR
    / "polyomino_b200_2gpu_qwen3_8b_qwen3_coder_next_fp8_exec_batch8_group8_summary_only_entropic_best_child_500step.yaml",
}


def test_coder_next_variants_preserve_the_qwen3_8b_training_method():
    for mode, path in CONFIGS.items():
        config = yaml.safe_load(path.read_text())

        assert config["run"]["model_path"] == "models/Qwen3-8B"
        assert config["run"]["num_steps"] == 500
        assert config["run"]["total_epochs"] == 500
        assert config["run"]["n_gpus_per_node"] == 1
        assert config["run"]["adv_estimator"] == "entropic_adaptive_beta"
        assert config["run"]["save_freq"] == 1
        assert config["ttt"]["groups_per_batch"] == 8
        assert config["ttt"]["group_size"] == 8
        assert config["ttt"]["prompt_mode"] == mode
        assert config["ttt"]["puct_q_mode"] == "best_child"
        assert "trainer.resume_mode=auto" in config["verl_overrides"]
        assert "trainer.max_actor_ckpt_to_keep=1" in config["verl_overrides"]


def test_coder_next_code_delta_reserves_more_context_for_growing_source_code():
    run = yaml.safe_load(CONFIGS["code_delta"].read_text())["run"]

    assert run["max_prompt_length"] == 12288
    assert run["max_response_length"] == 4096
    assert run["max_prompt_length"] + run["max_response_length"] == 16384
    assert run["truncation"] == "middle"


def test_coder_next_execution_is_fp8_and_non_thinking():
    for path in CONFIGS.values():
        execution = yaml.safe_load(path.read_text())["llm"]["execution"]

        assert execution["model"] == "Qwen/Qwen3-Coder-Next-FP8"
        assert execution["prompt_style"] == "qwen_no_thinking"
        assert execution["chat_template_kwargs"] == {"enable_thinking": False}
        expected_temperature = 0.0 if path == CONFIGS["summary_only"] else 0.6
        assert execution["temperature"] == expected_temperature
        assert execution["top_p"] == 0.95
        assert execution["top_k"] == 20
        assert execution["min_p"] == 0.0
        assert execution["max_tokens"] is None
        assert execution["concurrency"] == 16


def test_coder_next_stage_uses_one_train_and_one_execution_b200():
    stage = (
        ROOT / "scripts/slurm_polyomino_b200_2gpu_qwen3_coder_next_fp8_exec_group8_stage.sbatch"
    ).read_text()
    runner = (ROOT / "scripts/run_polyomino_b200_long.sh").read_text()

    assert "#SBATCH --gres=gpu:b200:2" in stage
    assert "TRAINING_GPUS=0" in stage
    assert "EXECUTION_GPU=1" in stage
    assert 'EXECUTION_MODEL_DIR="models/Qwen3-Coder-Next-FP8"' in stage
    assert 'EXECUTION_MODEL_NAME="Qwen/Qwen3-Coder-Next-FP8"' in stage
    assert 'EXECUTION_VLLM_DTYPE="auto"' in stage
    assert 'EXECUTION_VLLM_MAX_MODEL_LEN="32768"' in stage
    assert 'EXECUTION_VLLM_REASONING_PARSER=""' in stage
    assert "EXECUTION_EXTRA_CONTAINER_PYTHONPATH" in stage
    assert "APPTAINER_EXECUTION_BASE" in runner


def test_coder_next_setup_and_dependency_gated_two_day_chain():
    setup = (ROOT / "scripts/slurm_setup_qwen3_coder_next_fp8_b200.sbatch").read_text()
    submitter = (
        ROOT / "scripts/submit_qwen3_coder_next_fp8_exec_b200_2gpu_group8_two_day.sh"
    ).read_text()

    assert "Qwen/Qwen3-Coder-Next-FP8" in setup
    assert 'quantization.get("quant_method") == "fp8"' in setup
    assert "Qwen3NextForCausalLM" in setup
    assert "for mode in code_delta summary_only" in submitter
    assert "--partition=b200-devel --time=02:00:00" in submitter
    assert 'dependency="afterok:${smoke}"' in submitter
    assert 'dependency="afterok:${day1}"' in submitter
