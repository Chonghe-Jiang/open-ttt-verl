from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "guidance_ttt/config/polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_code_delta_prompt8192_entropic_best_child_500step.yaml"
)


def test_qwen36_27b_code_delta_extends_prompt_and_context_without_changing_method():
    config = yaml.safe_load(CONFIG.read_text())
    run = config["run"]
    ttt = config["ttt"]
    execution = config["llm"]["execution"]
    overrides = config["verl_overrides"]

    assert run["model_path"] == "models/Qwen3.6-27B"
    assert run["n_gpus_per_node"] == 4
    assert run["max_prompt_length"] == 8192
    assert run["max_response_length"] == 8192
    assert run["adv_estimator"] == "entropic_adaptive_beta"
    assert ttt["prompt_mode"] == "code_delta"
    assert ttt["groups_per_batch"] == 8
    assert ttt["group_size"] == 8
    assert ttt["puct_q_mode"] == "best_child"
    assert execution["model"] == "openai/gpt-oss-120b"
    assert execution["temperature"] == 0.0
    assert "actor_rollout_ref.rollout.max_model_len=16384" in overrides
    assert (
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096"
        in overrides
    )


def test_qwen36_27b_code_delta_has_a_five_gpu_smoke_gated_chain():
    stage = (
        ROOT
        / "scripts/slurm_polyomino_b200_5gpu_qwen36_27b_guidance_code_delta_prompt8192_group8_stage.sbatch"
    ).read_text()
    submitter = (
        ROOT
        / "scripts/submit_qwen36_27b_guidance_code_delta_prompt8192_b200_5gpu_group8_two_day.sh"
    ).read_text()

    assert "#SBATCH --gres=gpu:b200:5" in stage
    assert "TRAINING_GPUS=0,1,2,3" in stage
    assert "EXECUTION_GPU=4" in stage
    assert "EXPECTED_PROMPT_MODE=code_delta" in stage
    assert 'dependency="afterok:${smoke}"' in submitter
    assert 'dependency="afterok:${day1}"' in submitter
    assert 'group_size="${GROUP_SIZE:-16}"' in submitter
    assert 'GROUP_SIZE_OVERRIDE="${group_size}"' in submitter
