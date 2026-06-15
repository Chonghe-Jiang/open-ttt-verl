from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from guidance_ttt.data import write_slot_parquet
from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.tasks.erdos import create_root_node


class GuidanceTTTTaskRunner:
    def __init__(self):
        from verl.trainer.main_ppo import TaskRunner

        self._runner = TaskRunner()

    def run(self, config):
        import guidance_ttt.verl_ext  # noqa: F401

        return self._runner.run(config)


def load_recipe_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def split_overrides(overrides: list[str]) -> tuple[list[str], list[str]]:
    recipe_overrides, verl_overrides = [], []
    for override in overrides:
        if override.startswith(("run.", "ttt.", "llm.")):
            recipe_overrides.append(override)
        else:
            verl_overrides.append(override)
    return recipe_overrides, verl_overrides


def apply_recipe_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    if not overrides:
        return config
    from omegaconf import OmegaConf

    merged = OmegaConf.merge(OmegaConf.create(config), OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(merged, resolve=True)


def prepare_run(config: dict[str, Any]) -> dict[str, Path]:
    run_cfg = config["run"]
    ttt_cfg = config["ttt"]
    output_dir = Path(run_cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    library_path = output_dir / "library.json"
    if not library_path.exists():
        root_nodes = [create_root_node() for _ in range(int(run_cfg.get("num_initial_states", 1)))]
        GuidanceLibrary(
            library_path,
            initial_nodes=root_nodes,
            rollout_n=int(ttt_cfg["group_size"]),
            puct_c=float(ttt_cfg.get("puct_c", 1.0)),
        )

    slot_parquet = output_dir / "ttt_slots.parquet"
    write_slot_parquet(slot_parquet, num_slots=int(ttt_cfg["groups_per_batch"]), library_path=str(library_path))

    agent_loop_config = output_dir / "agent_loop.yaml"
    agent_loop_config.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "guidance_execution_erdos",
                    "_target_": "guidance_ttt.agent_loop.GuidanceExecutionAgentLoop",
                    "execution_llm": config.get("llm", {}).get("execution", {"provider": "mock"}),
                    "summarizer_llm": config.get("llm", {}).get("summarizer", {"provider": "mock"}),
                    "eval_timeout_s": int(ttt_cfg.get("eval_timeout", 60)),
                    "verifier_timeout_s": int(ttt_cfg.get("eval_timeout", 60)),
                }
            ],
            sort_keys=False,
        )
    )
    return {
        "output_dir": output_dir,
        "library_path": library_path,
        "slot_parquet": slot_parquet,
        "agent_loop_config": agent_loop_config,
    }


def build_verl_overrides(config: dict[str, Any], prepared: dict[str, Path], extra_overrides: list[str]) -> list[str]:
    run_cfg = config["run"]
    ttt_cfg = config["ttt"]
    zmq_key = f"{prepared['output_dir']}:{os.getpid()}"
    zmq_suffix = f"guidance-ttt-{hashlib.sha1(zmq_key.encode()).hexdigest()[:12]}"
    overrides = [
        "algorithm.adv_estimator=entropic_adaptive_beta",
        "algorithm.use_kl_in_reward=False",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        f"data.train_files={prepared['slot_parquet']}",
        f"data.val_files={prepared['slot_parquet']}",
        f"data.train_batch_size={int(ttt_cfg['groups_per_batch'])}",
        f"data.max_prompt_length={int(run_cfg.get('max_prompt_length', 8192))}",
        f"data.max_response_length={int(run_cfg.get('max_response_length', 2048))}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={run_cfg['model_path']}",
        f"actor_rollout_ref.actor.optim.lr={float(run_cfg.get('learning_rate', 1e-5))}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={int(run_cfg.get('ppo_mini_batch_size', ttt_cfg['groups_per_batch']))}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={int(run_cfg.get('ppo_micro_batch_size_per_gpu', 1))}",
        "actor_rollout_ref.actor.policy_loss.loss_mode=ttt_reinforce_is",
        f"actor_rollout_ref.actor.use_kl_loss={bool(run_cfg.get('use_kl_loss', True))}",
        f"actor_rollout_ref.actor.kl_loss_coef={float(run_cfg.get('kl_loss_coef', 0.05))}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.rollout.name={run_cfg.get('rollout_engine', 'vllm')}",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.n={int(ttt_cfg['group_size'])}",
        f"actor_rollout_ref.rollout.temperature={float(run_cfg.get('temperature', 1.0))}",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={int(run_cfg.get('tensor_model_parallel_size', 1))}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={float(run_cfg.get('gpu_memory_utilization', 0.5))}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=guidance_execution_erdos",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={prepared['agent_loop_config']}",
        "actor_rollout_ref.model.external_lib=guidance_ttt.verl_ext",
        f"trainer.project_name={run_cfg.get('project_name', 'guidance_ttt')}",
        f"trainer.experiment_name={run_cfg.get('experiment_name', prepared['output_dir'].name)}",
        f"trainer.default_local_dir={prepared['output_dir'] / 'checkpoints'}",
        f"trainer.n_gpus_per_node={int(run_cfg.get('n_gpus_per_node', 1))}",
        f"trainer.nnodes={int(run_cfg.get('nnodes', 1))}",
        f"trainer.total_epochs={int(run_cfg.get('total_epochs', 1))}",
        f"trainer.total_training_steps={int(run_cfg.get('num_steps', 1))}",
        f"trainer.save_freq={int(run_cfg.get('save_freq', -1))}",
        f"trainer.test_freq={int(run_cfg.get('test_freq', -1))}",
        f"trainer.val_before_train={bool(run_cfg.get('val_before_train', False))}",
        "trainer.logger=['console']",
        f"+ray_kwargs.ray_init.address={run_cfg.get('ray_address', 'local')}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_ZMQ_SUFFIX={zmq_suffix}",
    ]
    overrides.extend(config.get("verl_overrides", []))
    overrides.extend(extra_overrides)
    return overrides


def _default_verl_config_dir() -> Path:
    reference = Path("/home/chonghej/scratch/chonghej/Guidance-ttt/reference/open-ttt-verl/verl/trainer/config")
    if reference.exists():
        return reference
    return Path.cwd() / "verl" / "trainer" / "config"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Guidance + Execution TTT on Erdos with verl.")
    parser.add_argument("--config", default="guidance_ttt/config/erdos_smoke.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_recipe_config(args.config)
    recipe_overrides, verl_overrides = split_overrides(args.overrides)
    config = apply_recipe_overrides(config, recipe_overrides)
    prepared = prepare_run(config)
    overrides = build_verl_overrides(config, prepared, verl_overrides)

    print(f"Prepared library: {prepared['library_path']}")
    print(f"Prepared slots: {prepared['slot_parquet']}")
    print(f"Prepared agent loop config: {prepared['agent_loop_config']}")
    if args.prepare_only or config["run"].get("prepare_only", False):
        print("Prepare-only mode; not launching verl trainer.")
        return

    import guidance_ttt.verl_ext  # noqa: F401
    import ray
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo

    config_dir = str(Path(config["run"].get("verl_config_dir", _default_verl_config_dir())).resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        verl_config = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(verl_config)
    task_runner_class = ray.remote(num_cpus=1)(GuidanceTTTTaskRunner)
    run_ppo(verl_config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
