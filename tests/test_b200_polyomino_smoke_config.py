from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "guidance_ttt/config/polyomino_b200_3gpu_gpt_oss_120b_batch8_group16_code_delta_8192_smoke.yaml"
)
CONFIG_5GPU_PATH = (
    ROOT
    / "guidance_ttt/config/polyomino_b200_5gpu_gpt_oss_120b_batch8_group32_code_delta_8192_smoke.yaml"
)
CONFIG_5GPU_50STEP_PATH = (
    ROOT
    / "guidance_ttt/config/polyomino_b200_5gpu_gpt_oss_120b_batch8_group32_code_delta_8192_50step.yaml"
)
CONFIG_5GPU_GROUP64_PATH = (
    ROOT
    / "guidance_ttt/config/polyomino_b200_5gpu_gpt_oss_120b_batch8_group64_code_delta_8192_smoke.yaml"
)
CONFIG_5GPU_GROUP64_50STEP_PATH = (
    ROOT
    / "guidance_ttt/config/polyomino_b200_5gpu_gpt_oss_120b_batch8_group64_code_delta_8192_50step.yaml"
)
CONFIG_QWEN14B_GROUP32_50STEP_PATH = (
    ROOT
    / "guidance_ttt/config/polyomino_b200_5gpu_qwen3_14b_gpt_oss_120b_batch8_group32_code_delta_8192_50step.yaml"
)


def test_active_training_configs_release_rollout_cache_between_phases():
    config_dir = ROOT / "guidance_ttt/config"
    active_configs = sorted(config_dir.glob("*.yaml"))

    assert active_configs
    for path in active_configs:
        assert "actor_rollout_ref.rollout.free_cache_engine=False" not in path.read_text(), path.name


def test_b200_smoke_preserves_full_8x16_one_step_shape():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["run"]["num_steps"] == 1
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["model_path"] == "models/Qwen3-8B"
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 16
    assert config["ttt"]["prompt_mode"] == "code_delta"
    assert config["llm"]["execution"]["model"] == "openai/gpt-oss-120b"
    assert config["llm"]["execution"]["concurrency"] == 16


def test_b200_smoke_keeps_outputs_models_and_caches_in_checkout():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["run"]["output_dir"].startswith("outputs/")
    assert config["run"]["hf_cache_dir"] == ".hf_cache"
    assert config["run"]["ray_temp_dir"] == ".ray_tmp"
    assert config["run"]["tmp_dir"] == ".tmp"
    assert config["run"]["triton_cache_dir"] == ".triton_cache"
    assert config["task"]["frontiercs"]["base_dir"] == "reference/Frontier-CS"


def test_b200_slurm_job_requests_three_batch_gpus_and_runs_acceptance():
    script = (ROOT / "scripts/slurm_polyomino_b200_3gpu_smoke.sbatch").read_text()

    assert "#SBATCH --partition=b200-batch" in script
    assert "#SBATCH --gres=gpu:b200:3" in script
    assert "run_polyomino_b200_3gpu_smoke.sh preflight" in script
    assert "run_polyomino_b200_3gpu_smoke.sh run" in script


def test_b200_setup_uses_memory_bounded_downloads_and_excludes_extra_weights():
    script = (ROOT / "scripts/setup_polyomino_b200.sh").read_text()

    assert '"Qwen/Qwen3-8B:Qwen3-8B"' in script
    assert '"openai/gpt-oss-120b:gpt-oss-120b"' in script
    assert "--max-workers 1" in script
    assert "--exclude 'metal/*' 'original/*'" in script
    assert ".download-complete" in script


def test_b200_5gpu_smoke_uses_four_training_gpus_and_full_8x32_shape():
    config = yaml.safe_load(CONFIG_5GPU_PATH.read_text())

    assert config["run"]["num_steps"] == 1
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["run"]["ppo_mini_batch_size"] == 8
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 32
    assert config["ttt"]["prompt_mode"] == "code_delta"
    assert config["llm"]["execution"]["temperature"] == 0.0
    assert config["llm"]["execution"]["concurrency"] == 16


def test_b200_5gpu_slurm_job_maps_four_training_cards_and_one_execution_card():
    script = (ROOT / "scripts/slurm_polyomino_b200_5gpu_group32_smoke.sbatch").read_text()

    assert "#SBATCH --partition=b200-batch" in script
    assert "#SBATCH --gres=gpu:b200:5" in script
    assert "export EXPECTED_GPUS=5" in script
    assert "export TRAINING_GPUS=0,1,2,3" in script
    assert "export EXECUTION_GPU=4" in script
    assert "export EXPECTED_GROUP_SIZE=32" in script


def test_b200_qwen14b_configs_keep_5gpu_shape_and_define_oom_fallback():
    primary = yaml.safe_load(
        (
            ROOT
            / "guidance_ttt/config/polyomino_b200_5gpu_qwen3_14b_gpt_oss_120b_batch8_group32_code_delta_8192_smoke.yaml"
        ).read_text()
    )
    fallback = yaml.safe_load(
        (
            ROOT
            / "guidance_ttt/config/polyomino_b200_5gpu_qwen3_14b_gpt_oss_120b_batch8_group24_code_delta_8192_smoke.yaml"
        ).read_text()
    )

    for config in (primary, fallback):
        assert config["run"]["model_path"] == "models/Qwen3-14B"
        assert config["run"]["n_gpus_per_node"] == 4
        assert config["ttt"]["groups_per_batch"] == 8
        assert config["llm"]["execution"]["model"] == "openai/gpt-oss-120b"
        assert config["llm"]["execution"]["temperature"] == 0.0
    assert primary["ttt"]["group_size"] == 32
    assert fallback["ttt"]["group_size"] == 24


def test_b200_5gpu_50step_keeps_smoke_shape_and_saves_every_ten_steps():
    config = yaml.safe_load(CONFIG_5GPU_50STEP_PATH.read_text())
    script = (ROOT / "scripts/slurm_polyomino_b200_5gpu_group32_50step.sbatch").read_text()

    assert config["run"]["model_path"] == "models/Qwen3-8B"
    assert config["run"]["num_steps"] == 50
    assert config["run"]["total_epochs"] == 50
    assert config["run"]["save_freq"] == 10
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 32
    assert "#SBATCH --gres=gpu:b200:5" in script
    assert "#SBATCH --time=1-00:00:00" in script
    assert "export EXPECTED_GROUPS=400" in script


def test_b200_5gpu_group64_smoke_doubles_rollouts_without_raising_concurrency():
    config = yaml.safe_load(CONFIG_5GPU_GROUP64_PATH.read_text())
    script = (ROOT / "scripts/slurm_polyomino_b200_5gpu_group64_smoke.sbatch").read_text()

    assert config["run"]["model_path"] == "models/Qwen3-8B"
    assert config["run"]["num_steps"] == 1
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["run"]["ppo_micro_batch_size_per_gpu"] == 2
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 64
    assert config["llm"]["execution"]["concurrency"] == 16
    assert "#SBATCH --gres=gpu:b200:5" in script
    assert "export EXPECTED_GROUP_SIZE=64" in script


def test_b200_long_configs_run_50_steps_and_save_every_ten():
    qwen8b = yaml.safe_load(CONFIG_5GPU_GROUP64_50STEP_PATH.read_text())
    qwen14b = yaml.safe_load(CONFIG_QWEN14B_GROUP32_50STEP_PATH.read_text())

    for config in (qwen8b, qwen14b):
        assert config["run"]["num_steps"] == 50
        assert config["run"]["total_epochs"] == 50
        assert config["run"]["save_freq"] == 10
        assert config["run"]["n_gpus_per_node"] == 4
        assert config["ttt"]["groups_per_batch"] == 8
    assert qwen8b["run"]["model_path"] == "models/Qwen3-8B"
    assert qwen8b["ttt"]["group_size"] == 64
    assert qwen14b["run"]["model_path"] == "models/Qwen3-14B"
    assert qwen14b["ttt"]["group_size"] == 32


def test_b200_group64_long_job_splits_at_checkpoint_20_for_walltime():
    script = (ROOT / "scripts/slurm_polyomino_b200_5gpu_group64_50step_stage.sbatch").read_text()

    assert "#SBATCH --time=1-00:00:00" in script
    assert "export NUM_STEPS_OVERRIDE=20" in script
    assert "export EXPECTED_GROUPS=160" in script
    assert "export EXPECTED_GROUPS=400" in script
    assert "SHARED_OUTPUT_DIR" in script
    assert "run_polyomino_b200_long.sh run" in script

    runner = (ROOT / "scripts/run_polyomino_b200_long.sh").read_text()
    assert "NUM_STEPS_OVERRIDE" in runner
    assert "TOTAL_EPOCHS_OVERRIDE" in runner


def test_b200_prompt_refinement_long_configs_use_summary_only_semantics():
    paths = (
        ROOT
        / "guidance_ttt/config/polyomino_b200_5gpu_gpt_oss_120b_batch8_group64_prompt_refinement_50step.yaml",
        ROOT
        / "guidance_ttt/config/polyomino_b200_5gpu_qwen3_14b_gpt_oss_120b_batch8_group32_prompt_refinement_50step.yaml",
    )
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert config["run"]["num_steps"] == 50
        assert config["run"]["save_freq"] == 10
        assert config["run"]["max_prompt_length"] == 4096
        assert config["run"]["truncation"] == "middle"
        assert config["run"]["ppo_mini_batch_size"] == 4
        assert config["ttt"]["prompt_mode"] == "summary_only"
        assert "actor_rollout_ref.rollout.max_model_len=12288" in config["verl_overrides"]


def test_b200_prompt_refinement_jobs_pass_summary_mode_to_validator():
    paths = (
        ROOT / "scripts/slurm_polyomino_b200_5gpu_group64_prompt_refinement_50step_stage.sbatch",
        ROOT / "scripts/slurm_polyomino_b200_5gpu_qwen3_14b_group32_prompt_refinement_50step.sbatch",
    )
    for path in paths:
        script = path.read_text()
        assert "#SBATCH --gres=gpu:b200:5" in script
        assert "export EXPECTED_PROMPT_MODE=summary_only" in script
        assert "run_polyomino_b200_long.sh run" in script


def test_b200_paper_aligned_configs_use_entropic_advantages_and_best_child_q():
    paths = (
        ROOT
        / "guidance_ttt/config/polyomino_b200_5gpu_gpt_oss_120b_batch8_group64_code_delta_8192_entropic_best_child_50step.yaml",
        ROOT
        / "guidance_ttt/config/polyomino_b200_5gpu_gpt_oss_120b_batch8_group64_prompt_refinement_entropic_best_child_50step.yaml",
        ROOT
        / "guidance_ttt/config/polyomino_b200_5gpu_qwen3_14b_gpt_oss_120b_batch8_group32_code_delta_8192_entropic_best_child_50step.yaml",
        ROOT
        / "guidance_ttt/config/polyomino_b200_5gpu_qwen3_14b_gpt_oss_120b_batch8_group32_prompt_refinement_entropic_best_child_50step.yaml",
    )

    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert config["run"]["num_steps"] == 50
        assert config["run"]["save_freq"] == 10
        assert config["run"]["adv_estimator"] == "entropic_adaptive_beta"
        assert config["ttt"]["puct_q_mode"] == "best_child"
        assert config["llm"]["execution"]["max_tokens"] is None


def test_b200_paper_aligned_launcher_validates_without_host_python_packages():
    script = (ROOT / "scripts/slurm_polyomino_b200_5gpu_entropic_best_child_50step.sbatch").read_text()

    assert "grep -Eq" in script
    assert "adv_estimator:" in script
    assert "puct_q_mode:" in script
    assert "python -" not in script
    assert "run_polyomino_b200_long.sh preflight" in script
    assert "run_polyomino_b200_long.sh run" in script


def test_qwen35_9b_three_gpu_two_day_pipeline_is_resumable_and_bounded():
    config_path = (
        ROOT
        / "guidance_ttt/config/"
        "polyomino_b200_3gpu_qwen35_9b_gpt_oss_120b_256samples_code_delta_entropic_best_child_50step.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    stage = (ROOT / "scripts/slurm_polyomino_b200_3gpu_qwen35_9b_256samples_stage.sbatch").read_text()
    submitter = (ROOT / "scripts/submit_qwen35_9b_b200_two_day.sh").read_text()

    assert config["run"]["model_path"] == "models/Qwen3.5-9B"
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["save_freq"] == 1
    assert config["run"]["adv_estimator"] == "entropic_adaptive_beta"
    assert config["ttt"]["groups_per_batch"] * config["ttt"]["group_size"] == 256
    assert config["ttt"]["puct_q_mode"] == "best_child"
    assert config["llm"]["execution"]["max_tokens"] is None
    assert "trainer.resume_mode=auto" in config["verl_overrides"]
    assert "trainer.max_actor_ckpt_to_keep=1" in config["verl_overrides"]
    assert "trainer.max_critic_ckpt_to_keep=1" in config["verl_overrides"]
    assert "actor_rollout_ref.model.exclude_modules='.*visual.*'" in config["verl_overrides"]
    assert "actor_rollout_ref.actor.freeze_vision_tower=True" in config["verl_overrides"]
    assert (
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True"
        in config["verl_overrides"]
    )
    assert "actor_rollout_ref.rollout.enforce_eager=False" in config["verl_overrides"]

    assert "#SBATCH --gres=gpu:b200:3" in stage
    assert "TRAINING_GPUS=0,1" in stage
    assert "EXECUTION_GPU=2" in stage
    assert "NUM_STEPS_OVERRIDE=1" in stage
    assert "NUM_STEPS_OVERRIDE=20" in stage
    assert "SAVE_FREQ_OVERRIDE=-1" in stage
    assert ".vllm-0.18.0-complete" in stage
    assert "timeout --foreground --signal=TERM --kill-after=5m 23h" in stage
    assert "for setting in 8x32 16x16" in submitter
    assert submitter.count('dependency="afterok:') == 3
    assert 'smoke_dependency="${previous_smoke}"' in submitter


def test_qwen35_setup_pins_vllm_with_language_model_only_support():
    setup = (ROOT / "scripts/slurm_setup_qwen35_9b_b200.sbatch").read_text()
    patch = (ROOT / "scripts/vllm-0.18.0-qwen35-packed-lora.patch").read_text()

    assert "'vllm==0.18.0'" in setup
    assert 'vllm.__version__ == "0.18.0"' in setup
    assert '"language_model_only" in inspect.signature(EngineArgs).parameters' in setup
    assert "qwen35-packed-lora-complete" in setup
    assert "def expand_packed_lora" in patch
    assert "len(lora_b) != self.n_slices" in patch
    assert "n_slices != len(replacements)" in patch


def test_qwen35_9b_two_gpu_group8_smokes_preserve_full_method_and_lora():
    config_dir = ROOT / "guidance_ttt/config"
    paths = {
        "code_delta": config_dir
        / "polyomino_b200_2gpu_qwen35_9b_gpt_oss_120b_batch8_group8_code_delta_entropic_best_child_500step.yaml",
        "summary_only": config_dir
        / "polyomino_b200_2gpu_qwen35_9b_gpt_oss_120b_batch8_group8_summary_only_entropic_best_child_500step.yaml",
    }
    for mode, path in paths.items():
        config = yaml.safe_load(path.read_text())
        assert config["run"]["model_path"] == "models/Qwen3.5-9B"
        assert config["run"]["num_steps"] == 500
        assert config["run"]["total_epochs"] == 500
        assert config["run"]["n_gpus_per_node"] == 1
        assert config["run"]["adv_estimator"] == "entropic_adaptive_beta"
        assert config["run"]["save_freq"] == 1
        assert config["ttt"]["groups_per_batch"] == 8
        assert config["ttt"]["group_size"] == 8
        assert config["ttt"]["prompt_mode"] == mode
        assert config["ttt"]["puct_q_mode"] == "best_child"
        assert config["llm"]["execution"]["max_tokens"] is None
        assert "actor_rollout_ref.model.lora_rank=32" in config["verl_overrides"]
        assert "actor_rollout_ref.model.exclude_modules='.*visual.*'" in config["verl_overrides"]
        assert (
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True"
            in config["verl_overrides"]
        )
        assert "actor_rollout_ref.rollout.enforce_eager=False" in config["verl_overrides"]
        if mode == "code_delta":
            assert config["run"]["max_prompt_length"] == 12288
            assert config["run"]["max_response_length"] == 4096
            assert config["run"]["truncation"] == "middle"

    stage = (ROOT / "scripts/slurm_polyomino_b200_2gpu_qwen35_9b_group8_stage.sbatch").read_text()
    formal = (ROOT / "scripts/submit_qwen35_9b_b200_2gpu_group8_formal.sh").read_text()
    assert "#SBATCH --gres=gpu:b200:2" in stage
    assert "TRAINING_GPUS=0" in stage
    assert "EXECUTION_GPU=1" in stage
    assert "NUM_STEPS_OVERRIDE=1" in stage
    assert "SAVE_FREQ_OVERRIDE=-1" in stage
    assert ".qwen35-packed-lora-complete" in stage
    assert "timeout --foreground --signal=TERM --kill-after=5m 23h" in stage
    assert "SUCCESSFUL_SMOKE_JOB_ID" in formal
    assert formal.count('dependency="afterok:') == 2


def test_qwen3_8b_two_gpu_group8_two_day_pipelines_match_paper_method():
    config_dir = ROOT / "guidance_ttt/config"
    paths = {
        "code_delta": config_dir
        / "polyomino_b200_2gpu_qwen3_8b_gpt_oss_120b_batch8_group8_code_delta_entropic_best_child_500step.yaml",
        "summary_only": config_dir
        / "polyomino_b200_2gpu_qwen3_8b_gpt_oss_120b_batch8_group8_summary_only_entropic_best_child_500step.yaml",
    }
    for mode, path in paths.items():
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
        assert config["llm"]["execution"]["model"] == "openai/gpt-oss-120b"
        assert config["llm"]["execution"]["max_tokens"] is None
        assert "trainer.resume_mode=auto" in config["verl_overrides"]
        assert "trainer.max_actor_ckpt_to_keep=1" in config["verl_overrides"]
        assert "trainer.max_critic_ckpt_to_keep=1" in config["verl_overrides"]
        if mode == "code_delta":
            assert config["run"]["max_prompt_length"] == 12288
            assert config["run"]["max_response_length"] == 4096
            assert config["run"]["truncation"] == "middle"

    main = (ROOT / "guidance_ttt/main_erdos.py").read_text()
    assert "+data.apply_chat_template_kwargs.enable_thinking=True" in main
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=ttt_reinforce_is" in main

    stage = (ROOT / "scripts/slurm_polyomino_b200_2gpu_qwen3_8b_group8_stage.sbatch").read_text()
    submitter = (ROOT / "scripts/submit_qwen3_8b_b200_2gpu_group8_two_day.sh").read_text()
    assert "#SBATCH --gres=gpu:b200:2" in stage
    assert "TRAINING_GPUS=0" in stage
    assert "EXECUTION_GPU=1" in stage
    assert "NUM_STEPS_OVERRIDE=25" not in stage
    assert stage.count("EXPECTED_GROUPS=4000") == 2
    assert "timeout --foreground --signal=TERM --kill-after=5m 23h" in stage
    assert "for mode in code_delta summary_only" in submitter
    assert submitter.count('dependency="afterok:') == 2
