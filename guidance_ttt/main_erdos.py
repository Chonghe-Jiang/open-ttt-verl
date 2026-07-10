from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from guidance_ttt.data import write_slot_parquet
from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.tasks import get_task_spec


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
        if override.startswith(("run.", "ttt.", "llm.", "task.")):
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
    task_cfg = dict(config.get("task") or {"id": "erdos_min_overlap"})
    task_spec = get_task_spec(str(task_cfg.get("id", "erdos_min_overlap")))
    output_dir = Path(run_cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    library_path = output_dir / "library.json"
    if not library_path.exists():
        bootstrap_cfg = dict(ttt_cfg.get("bootstrap") or {})
        seed_library_path = bootstrap_cfg.get("seed_library_path")
        if seed_library_path:
            source_path = Path(str(seed_library_path)).expanduser()
            if not source_path.is_absolute():
                source_path = Path.cwd() / source_path
            temp_library_path = library_path.with_name(f".{library_path.name}.{uuid4().hex}.tmp")
            try:
                shutil.copyfile(source_path, temp_library_path)
                temp_library_path.replace(library_path)
            finally:
                if temp_library_path.exists():
                    temp_library_path.unlink()
        else:
            root_nodes = [task_spec.create_root_node() for _ in range(int(run_cfg.get("num_initial_states", 1)))]
            GuidanceLibrary(
                library_path,
                initial_nodes=root_nodes,
                rollout_n=int(ttt_cfg["group_size"]),
                puct_c=float(ttt_cfg.get("puct_c", 1.0)),
                max_buffer_size=int(ttt_cfg.get("max_buffer_size", 1000)),
                topk_children=int(ttt_cfg.get("topk_children", 2)),
            )

    slot_parquet = output_dir / "ttt_slots.parquet"
    write_slot_parquet(
        slot_parquet,
        num_slots=int(ttt_cfg["groups_per_batch"]),
        library_path=str(library_path),
        task=task_spec.task_id,
        task_config=task_cfg,
        rollout_n=int(ttt_cfg["group_size"]),
        puct_c=float(ttt_cfg.get("puct_c", 1.0)),
        max_buffer_size=int(ttt_cfg.get("max_buffer_size", 1000)),
        topk_children=int(ttt_cfg.get("topk_children", 2)),
    )

    agent_loop_name = str(ttt_cfg.get("agent_loop", run_cfg.get("agent_loop", "guidance_execution_task")))
    if agent_loop_name == "polyomino_discover_task":
        agent_loop_entry = {
            "name": "polyomino_discover_task",
            "_target_": "guidance_ttt.agent_loop.PolyominoDiscoverAgentLoop",
            "task": task_cfg,
            "eval_timeout_s": int(ttt_cfg.get("eval_timeout", 60)),
            "verifier_timeout_s": int(ttt_cfg.get("eval_timeout", 60)),
        }
    else:
        agent_loop_entry = {
            "name": agent_loop_name,
            "_target_": "guidance_ttt.agent_loop.GuidanceExecutionAgentLoop",
            "task": task_cfg,
            "execution_llm": config.get("llm", {}).get("execution", {"provider": "mock"}),
            "eval_timeout_s": int(ttt_cfg.get("eval_timeout", 60)),
            "verifier_timeout_s": int(ttt_cfg.get("eval_timeout", 60)),
        }
    agent_loop_config = output_dir / "agent_loop.yaml"
    agent_loop_config.write_text(
        yaml.safe_dump(
            [agent_loop_entry],
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
    ray_temp_dir = Path(run_cfg.get("ray_temp_dir", Path.cwd() / ".ray_tmp")).expanduser().resolve()
    ray_temp_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = Path(run_cfg.get("hf_cache_dir", Path.cwd() / ".hf_cache")).expanduser().resolve()
    (hf_cache_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (hf_cache_dir / "hub").mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(run_cfg.get("tmp_dir", Path.cwd() / ".tmp")).expanduser().resolve()
    triton_cache_dir = Path(run_cfg.get("triton_cache_dir", Path.cwd() / ".triton_cache")).expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    triton_cache_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"algorithm.adv_estimator={run_cfg.get('adv_estimator', 'grpo')}",
        "algorithm.use_kl_in_reward=False",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        f"data.train_files={prepared['slot_parquet']}",
        f"data.val_files={prepared['slot_parquet']}",
        f"data.train_batch_size={int(ttt_cfg['groups_per_batch'])}",
        f"data.max_prompt_length={int(run_cfg.get('max_prompt_length', 8192))}",
        f"data.max_response_length={int(run_cfg.get('max_response_length', 2048))}",
        f"data.filter_overlong_prompts={bool(run_cfg.get('filter_overlong_prompts', True))}",
        f"data.truncation={run_cfg.get('truncation', 'error')}",
        "+data.apply_chat_template_kwargs.enable_thinking=True",
        f"actor_rollout_ref.model.path={run_cfg['model_path']}",
        "actor_rollout_ref.model.use_remove_padding=False",
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
        f"actor_rollout_ref.rollout.agent.default_agent_loop={ttt_cfg.get('agent_loop', run_cfg.get('agent_loop', 'guidance_execution_task'))}",
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
        "+ray_kwargs.ray_init.include_dashboard=False",
        f"+ray_kwargs.ray_init._temp_dir={ray_temp_dir}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME={hf_cache_dir}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_CACHE={hf_cache_dir / 'datasets'}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.HUGGINGFACE_HUB_CACHE={hf_cache_dir / 'hub'}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_CACHE={hf_cache_dir / 'hub'}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR={tmp_dir}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.TRITON_CACHE_DIR={triton_cache_dir}",
        "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO='0'",
        "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_NO_USAGE_STATS='1'",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_ZMQ_SUFFIX={zmq_suffix}",
    ]
    if "adam_beta1" in run_cfg or "adam_beta2" in run_cfg:
        overrides.append(
            f"actor_rollout_ref.actor.optim.betas=[{float(run_cfg.get('adam_beta1', 0.9))},{float(run_cfg.get('adam_beta2', 0.999))}]"
        )
    if "adam_eps" in run_cfg:
        overrides.append(f"actor_rollout_ref.actor.optim.override_optimizer_config={{eps:{float(run_cfg['adam_eps'])}}}")
    for env_name in ("CC", "CXX", "CUDAHOSTCXX"):
        env_value = os.environ.get(env_name)
        if env_value:
            overrides.append(f"+ray_kwargs.ray_init.runtime_env.env_vars.{env_name}={env_value}")
    overrides.extend(config.get("verl_overrides", []))
    overrides.extend(extra_overrides)
    return overrides


def validate_bootstrap_requirement(config: dict[str, Any], prepared: dict[str, Path]) -> None:
    bootstrap_cfg = ((config.get("ttt") or {}).get("bootstrap") or {})
    if not bootstrap_cfg.get("required", False):
        return
    library = GuidanceLibrary(prepared["library_path"])
    snapshot = library.snapshot()
    entries = snapshot.get("entries") or {}
    root_nodes = [
        node
        for node in (snapshot.get("nodes") or {}).values()
        if isinstance(node, dict) and node.get("parent_id") is None
    ]
    has_root_entry = any(str(node.get("entry_id") or "") in entries for node in root_nodes)
    if not has_root_entry:
        raise RuntimeError(
            "Bootstrap history is required but no root-attached library entry exists. "
            "Run this recipe once with --bootstrap-only before launching training."
        )


def _default_verl_config_dir() -> Path:
    return Path.cwd() / "verl" / "trainer" / "config"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Guidance + Execution TTT with verl.")
    parser.add_argument("--config", default="guidance_ttt/config/backup/erdos_smoke.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--bootstrap-only", action="store_true")
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
    if args.bootstrap_only:
        bootstrap_cfg = ((config.get("ttt") or {}).get("bootstrap") or {})
        if not bootstrap_cfg.get("enabled", False):
            raise RuntimeError("Bootstrap requested but ttt.bootstrap.enabled is not true in the recipe config.")
        from guidance_ttt.bootstrap import bootstrap_library_entries

        result = bootstrap_library_entries(
            prepared["library_path"],
            task_config=dict(config.get("task") or {"id": "erdos_min_overlap"}),
            execution_llm_config=dict((config.get("llm") or {}).get("execution") or {"provider": "mock"}),
            verifier_timeout_s=int((config.get("ttt") or {}).get("eval_timeout", 60)),
            max_attempts=int(bootstrap_cfg.get("max_attempts", 2)),
            overwrite_existing=bool(bootstrap_cfg.get("overwrite_existing", False)),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.prepare_only or config["run"].get("prepare_only", False):
        print("Prepare-only mode; not launching verl trainer.")
        return
    validate_bootstrap_requirement(config, prepared)

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
