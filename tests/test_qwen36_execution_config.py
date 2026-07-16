from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "guidance_ttt/config"
CONFIGS = {
    "code_delta": CONFIG_DIR
    / "polyomino_b200_2gpu_qwen3_8b_qwen36_35b_exec_batch8_group8_code_delta_entropic_best_child_500step.yaml",
    "summary_only": CONFIG_DIR
    / "polyomino_b200_2gpu_qwen3_8b_qwen36_35b_exec_batch8_group8_summary_only_entropic_best_child_500step.yaml",
}


def test_qwen36_execution_variants_preserve_the_running_qwen3_8b_method():
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


def test_qwen36_execution_disables_thinking_and_uses_two_final_blocks():
    for path in CONFIGS.values():
        execution = yaml.safe_load(path.read_text())["llm"]["execution"]

        assert execution["model"] == "Qwen/Qwen3.6-35B-A3B"
        assert execution["prompt_style"] == "qwen_no_thinking"
        assert execution["chat_template_kwargs"] == {"enable_thinking": False}
        assert execution["temperature"] == 0.6
        assert execution["top_p"] == 0.95
        assert execution["top_k"] == 20
        assert execution["min_p"] == 0.0
        assert execution["max_tokens"] is None
        assert execution["concurrency"] == 16


def test_qwen36_execution_stage_keeps_training_and_serving_runtimes_isolated():
    stage = (ROOT / "scripts/slurm_polyomino_b200_2gpu_qwen36_35b_exec_group8_stage.sbatch").read_text()
    runner = (ROOT / "scripts/run_polyomino_b200_long.sh").read_text()

    assert "#SBATCH --gres=gpu:b200:2" in stage
    assert "TRAINING_GPUS=0" in stage
    assert "EXECUTION_GPU=1" in stage
    assert 'EXECUTION_MODEL_DIR="models/Qwen3.6-35B-A3B"' in stage
    assert 'EXECUTION_MODEL_NAME="Qwen/Qwen3.6-35B-A3B"' in stage
    assert 'EXECUTION_VLLM_LANGUAGE_MODEL_ONLY="1"' in stage
    assert 'EXECUTION_VLLM_REASONING_PARSER=""' in stage
    assert "EXECUTION_EXTRA_CONTAINER_PYTHONPATH" in stage
    assert "APPTAINER_EXECUTION_BASE" in runner
    assert "EXECUTION_VLLM_ARGS" in runner


def test_qwen36_setup_and_submitter_are_prepared_but_not_implicitly_run():
    setup = (ROOT / "scripts/slurm_setup_qwen36_35b_exec_b200.sbatch").read_text()
    submitter = (ROOT / "scripts/submit_qwen36_35b_exec_b200_2gpu_group8_two_day.sh").read_text()

    assert "Qwen/Qwen3.6-35B-A3B" in setup
    assert "'vllm==0.19.0'" in setup
    assert "'mistral-common==1.11.5'" in setup
    assert 'mistral_common-*.dist-info' in setup
    assert "ReasoningEffort" in setup
    assert "MistralCommonBackend" in setup
    assert "is_mistral_common_available()" in setup
    assert ".qwen36-exec-complete" in setup
    assert ".mistral-common-1.11.5-complete" in submitter
    assert "for mode in code_delta summary_only" in submitter
    assert "--partition=b200-devel --time=02:00:00" in submitter
    assert 'dependency="afterok:${smoke}"' in submitter
    assert 'dependency="afterok:${day1}"' in submitter
