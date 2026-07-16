from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "guidance_ttt/config/polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_summary_only_entropic_best_child_500step.yaml"
)
BLENDED_CONFIG = (
    ROOT
    / "guidance_ttt/config/polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_summary_only_entropic_blended_500step.yaml"
)


def test_qwen36_27b_guidance_preserves_the_summary_only_method():
    config = yaml.safe_load(CONFIG.read_text())
    run = config["run"]
    ttt = config["ttt"]
    execution = config["llm"]["execution"]

    assert run["model_path"] == "models/Qwen3.6-27B"
    assert run["n_gpus_per_node"] == 4
    assert run["num_steps"] == 500
    assert run["adv_estimator"] == "entropic_adaptive_beta"
    assert run["temperature"] == 0.9
    assert run["save_freq"] == 1
    assert ttt["prompt_mode"] == "summary_only"
    assert ttt["groups_per_batch"] == 8
    assert ttt["group_size"] == 8
    assert ttt["puct_q_mode"] == "best_child"
    assert execution["model"] == "openai/gpt-oss-120b"
    assert execution["temperature"] == 0.0


def test_qwen36_27b_guidance_uses_language_only_lora_and_bounded_checkpoints():
    overrides = yaml.safe_load(CONFIG.read_text())["verl_overrides"]

    assert "actor_rollout_ref.model.exclude_modules='.*visual.*'" in overrides
    assert "actor_rollout_ref.actor.freeze_vision_tower=True" in overrides
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True" in overrides
    assert "actor_rollout_ref.rollout.free_cache_engine=True" in overrides
    assert "actor_rollout_ref.rollout.enforce_eager=True" in overrides
    assert (
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096"
        in overrides
    )
    assert "trainer.resume_mode=auto" in overrides
    assert "trainer.max_actor_ckpt_to_keep=1" in overrides


def test_qwen36_27b_five_gpu_stage_and_smoke_gated_two_day_chain():
    stage = (
        ROOT / "scripts/slurm_polyomino_b200_5gpu_qwen36_27b_guidance_group8_stage.sbatch"
    ).read_text()
    submitter = (
        ROOT / "scripts/submit_qwen36_27b_guidance_b200_5gpu_group8_two_day.sh"
    ).read_text()
    setup = (ROOT / "scripts/slurm_setup_qwen36_27b_guidance_b200.sbatch").read_text()

    assert "#SBATCH --gres=gpu:b200:5" in stage
    assert "TRAINING_GPUS=0,1,2,3" in stage
    assert "EXECUTION_GPU=4" in stage
    assert "EXPECTED_PROMPT_MODE=summary_only" in stage
    assert "Qwen/Qwen3.6-27B" in setup
    assert "vllm==0.19.0" in setup
    assert "vllm-0.19.0-fla-triton-allocator.patch" in setup
    assert 'dependency="afterok:${smoke}"' in submitter
    assert 'dependency="afterok:${day1}"' in submitter
    assert 'group_size="${GROUP_SIZE:-16}"' in submitter
    assert 'GROUP_SIZE_OVERRIDE="${group_size}"' in submitter
    assert "for mode in" not in submitter


def test_qwen36_27b_blended_ablation_changes_only_the_puct_q_rule():
    baseline = yaml.safe_load(CONFIG.read_text())
    blended = yaml.safe_load(BLENDED_CONFIG.read_text())

    assert blended["ttt"]["puct_q_mode"] == "blended"
    assert baseline["ttt"]["puct_q_mode"] == "best_child"

    baseline["ttt"].pop("puct_q_mode")
    blended["ttt"].pop("puct_q_mode")
    baseline["run"].pop("output_dir")
    baseline["run"].pop("experiment_name")
    blended["run"].pop("output_dir")
    blended["run"].pop("experiment_name")
    assert blended == baseline


def test_qwen36_27b_blended_submitter_uses_group16_and_smoke_gate():
    stage = (
        ROOT / "scripts/slurm_polyomino_b200_5gpu_qwen36_27b_guidance_group8_stage.sbatch"
    ).read_text()
    submitter = (
        ROOT
        / "scripts/submit_qwen36_27b_guidance_blended_b200_5gpu_group16_two_day.sh"
    ).read_text()

    assert 'CONFIG_OVERRIDE:-${DEFAULT_CONFIG}' in stage
    assert 'PUCT_Q_MODE_OVERRIDE:-best_child' in stage
    assert "EXPECTED_PUCT_Q_MODE" in stage
    assert "GROUP_SIZE_OVERRIDE=16" not in submitter
    assert "group_size=16" in submitter
    assert "PUCT_Q_MODE_OVERRIDE=blended" in submitter
    assert 'dependency="afterok:${smoke}"' in submitter
    assert 'dependency="afterok:${day1}"' in submitter
