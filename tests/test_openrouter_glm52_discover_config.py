from pathlib import Path

import yaml

from guidance_ttt.main_erdos import build_verl_overrides, prepare_run


CONFIG_PATH = Path(
    "guidance_ttt/config/"
    "polyomino_b200_1gpu_qwen3_8b_openrouter_glm52_batch8_group16_"
    "summary_only_discover_1day_latest_ckpt.yaml"
)
SCRIPT_PATH = Path(
    "scripts/"
    "slurm_polyomino_b200_1gpu_qwen3_8b_openrouter_glm52_discover_1day_latest_ckpt.sbatch"
)


def test_openrouter_glm52_long_run_uses_exact_discover_and_latest_checkpoint(tmp_path):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["run"]["output_dir"] = str(tmp_path / "discover")
    prepared = prepare_run(config)
    overrides = build_verl_overrides(config, prepared, [])
    library = yaml.safe_load(prepared["library_path"].read_text())

    assert config["run"]["num_steps"] == config["run"]["total_epochs"] == 500
    assert config["run"]["save_freq"] == 1
    assert config["ttt"]["discover_compat"] is True
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 16
    assert config["ttt"]["puct_q_mode"] == "best_child"
    assert library["config"]["discover_compat"] is True
    assert len([node for node in library["nodes"].values() if node["parent_id"] is None]) == 8

    execution = config["llm"]["execution"]
    assert execution["model"] == "z-ai/glm-5.2"
    assert execution["max_tokens"] is None
    assert execution["concurrency"] == 16
    assert execution["reasoning"] == {"effort": "high", "exclude": False}

    assert "algorithm.rollout_correction.rollout_is=null" in overrides
    assert "+algorithm.discover_kl_coef=0.1" in overrides
    assert "+algorithm.remove_constant_reward_groups=True" in overrides
    assert "actor_rollout_ref.actor.use_kl_loss=False" in overrides
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=8" in overrides
    assert "actor_rollout_ref.actor.optim.betas=[0.9,0.95]" in overrides
    assert "actor_rollout_ref.actor.optim.weight_decay=0.0" in overrides
    assert "actor_rollout_ref.rollout.temperature=1.0" in overrides
    assert "actor_rollout_ref.rollout.top_p=1.0" in overrides
    assert "trainer.max_actor_ckpt_to_keep=1" in overrides
    assert "trainer.max_critic_ckpt_to_keep=1" in overrides


def test_openrouter_glm52_long_run_launcher_is_one_day_and_prunes_checkpoints():
    source = SCRIPT_PATH.read_text()

    assert "#SBATCH --time=1-00:00:00" in source
    assert "timeout --foreground --signal=TERM --kill-after=5m 23h" in source
    assert "discover_compat:" in source
    assert "save_freq:" in source
    assert "trainer.max_actor_ckpt_to_keep=1" in source
    assert "prune_to_latest_checkpoint" in source
