import json
from pathlib import Path

import pandas as pd
import yaml

import pytest

from guidance_ttt.main_erdos import (
    _default_verl_config_dir,
    build_verl_overrides,
    load_recipe_config,
    prepare_run,
    validate_bootstrap_requirement,
)
from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.state import LibraryEntry


def test_prepare_run_writes_library_slots_and_agent_loop_config(tmp_path):
    config = {
        "run": {
            "output_dir": str(tmp_path / "outputs" / "erdos_smoke"),
            "model_path": "Qwen/Qwen3-8B",
            "num_initial_states": 1,
        },
        "ttt": {
            "groups_per_batch": 2,
            "group_size": 3,
            "puct_c": 1.5,
            "max_buffer_size": 123,
            "topk_children": 4,
            "eval_timeout": 5,
        },
        "llm": {
            "execution": {"provider": "mock", "model": "mock-exec"},
        },
    }

    prepared = prepare_run(config)

    assert prepared["library_path"].exists()
    assert prepared["slot_parquet"].exists()
    assert prepared["agent_loop_config"].exists()
    library = yaml.safe_load(Path(prepared["library_path"]).read_text())
    assert library["config"] == {
        "rollout_n": 3,
        "puct_c": 1.5,
        "max_buffer_size": 123,
        "topk_children": 4,
    }
    slots = pd.read_parquet(prepared["slot_parquet"]).to_dict("records")
    assert slots[0]["extra_info"]["task"] == "erdos_min_overlap"
    assert slots[0]["extra_info"]["task_config"] == {"id": "erdos_min_overlap"}
    assert slots[0]["extra_info"]["rollout_n"] == 3
    assert slots[0]["extra_info"]["group_size"] == 3
    assert slots[0]["extra_info"]["puct_c"] == 1.5
    assert slots[0]["extra_info"]["max_buffer_size"] == 123
    assert slots[0]["extra_info"]["topk_children"] == 4
    data = yaml.safe_load(Path(prepared["agent_loop_config"]).read_text())
    assert data[0]["name"] == "guidance_execution_task"
    assert data[0]["task"]["id"] == "erdos_min_overlap"
    assert data[0]["verifier_timeout_s"] == 5
    assert "execution_llm" in data[0]
    assert "summarizer_llm" not in data[0]


def test_prepare_run_supports_polyomino_task_config(tmp_path):
    config = {
        "run": {
            "output_dir": str(tmp_path / "outputs" / "polyomino"),
            "model_path": "Qwen/Qwen3-8B",
            "num_initial_states": 2,
        },
        "task": {
            "id": "polyomino_packing",
            "frontiercs": {
                "problem_id": "0",
                "n_cases": 70,
                "time_limit_s": 2,
                "memory_mb": 256,
                "total_timeout_s": 340,
            },
        },
        "ttt": {
            "groups_per_batch": 2,
            "group_size": 3,
            "puct_c": 1.5,
            "max_buffer_size": 123,
            "topk_children": 4,
            "eval_timeout": 340,
        },
        "llm": {
            "execution": {"provider": "mock", "model": "mock-exec"},
        },
    }

    prepared = prepare_run(config)

    library = yaml.safe_load(Path(prepared["library_path"]).read_text())
    root_nodes = list(library["nodes"].values())
    root_problem_ids = {node["problem_id"] for node in root_nodes}
    assert root_problem_ids == {"polyomino_packing"}
    assert len(root_nodes) == 2
    slots = pd.read_parquet(prepared["slot_parquet"]).to_dict("records")
    assert slots[0]["extra_info"]["task"] == "polyomino_packing"
    assert slots[0]["extra_info"]["task_config"] == config["task"]
    data = yaml.safe_load(Path(prepared["agent_loop_config"]).read_text())
    assert data[0]["name"] == "guidance_execution_task"
    assert data[0]["task"] == config["task"]
    assert data[0]["verifier_timeout_s"] == 340


def test_prepare_run_can_initialize_library_from_seed_path(tmp_path):
    seed_library_path = tmp_path / "seed_library.json"
    seed_library_path.write_text(
        json.dumps(
            {
                "nodes": {
                    "seed-root": {
                        "id": "seed-root",
                        "problem_id": "polyomino_packing",
                        "timestep": 0,
                        "entry_id": "seed-entry",
                        "value": 74.55953630142857,
                        "raw_score": 74.55953630142857,
                        "visits": 0,
                        "parent_id": None,
                        "children": [],
                        "metadata": {"bootstrap": True, "verifier_status": "valid"},
                    }
                },
                "entries": {
                    "seed-entry": {
                        "id": "seed-entry",
                        "parent_id": "seed-root",
                        "problem_id": "polyomino_packing",
                        "timestep": 0,
                        "guidance": "Bootstrap execution without guidance.",
                        "execution_thinking": "thinking",
                        "solution": "int main(){return 0;}",
                        "verifier_reward": 74.55953630142857,
                        "verifier_raw_score": 74.55953630142857,
                        "verifier_status": "valid",
                        "verifier_message": "accepted",
                        "summary": "Seed summary",
                        "reusable_idea": "Seed idea",
                        "failure_mode": None,
                        "metadata": {
                            "bootstrap": True,
                            "static_seed": True,
                            "raw_model_summary": "Seed raw summary",
                            "execution_text": "<execution_thinking>x</execution_thinking><solution>y</solution><summary>z</summary>",
                        },
                    }
                },
                "groups": {},
                "best_node_id": "seed-root",
            }
        )
    )
    config = {
        "run": {
            "output_dir": str(tmp_path / "outputs" / "seeded_polyomino"),
            "model_path": "Qwen/Qwen3-8B",
            "num_initial_states": 1,
        },
        "task": {"id": "polyomino_packing"},
        "ttt": {
            "groups_per_batch": 1,
            "group_size": 2,
            "puct_c": 1.0,
            "eval_timeout": 5,
            "bootstrap": {"required": True, "seed_library_path": str(seed_library_path)},
        },
        "llm": {"execution": {"provider": "mock"}},
    }

    prepared = prepare_run(config)

    library = yaml.safe_load(Path(prepared["library_path"]).read_text())
    assert library["nodes"]["seed-root"]["entry_id"] == "seed-entry"
    assert library["entries"]["seed-entry"]["metadata"]["static_seed"] is True
    validate_bootstrap_requirement(config, prepared)


def test_prepare_run_copies_seed_library_via_atomic_replace(tmp_path, monkeypatch):
    seed_library_path = tmp_path / "seed_library.json"
    seed_library_path.write_text(json.dumps({"nodes": {}, "entries": {}, "groups": {}, "best_node_id": None}))
    config = {
        "run": {
            "output_dir": str(tmp_path / "outputs" / "atomic_seeded_polyomino"),
            "model_path": "Qwen/Qwen3-8B",
            "num_initial_states": 1,
        },
        "task": {"id": "polyomino_packing"},
        "ttt": {
            "groups_per_batch": 1,
            "group_size": 2,
            "puct_c": 1.0,
            "eval_timeout": 5,
            "bootstrap": {"seed_library_path": str(seed_library_path)},
        },
        "llm": {"execution": {"provider": "mock"}},
    }
    original_replace = Path.replace
    replace_targets: list[Path] = []

    def tracked_replace(self: Path, target):
        replace_targets.append(Path(target))
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", tracked_replace)

    prepared = prepare_run(config)

    assert Path(prepared["library_path"]) in replace_targets
    assert json.loads(Path(prepared["library_path"]).read_text())["nodes"] == {}


def test_prepare_run_does_not_overwrite_existing_library_with_seed_path(tmp_path):
    seed_library_path = tmp_path / "seed_library.json"
    seed_library_path.write_text(json.dumps({"nodes": {}, "entries": {}, "groups": {}, "best_node_id": None}))
    config = {
        "run": {
            "output_dir": str(tmp_path / "outputs" / "existing_polyomino"),
            "model_path": "Qwen/Qwen3-8B",
            "num_initial_states": 1,
        },
        "task": {"id": "polyomino_packing"},
        "ttt": {
            "groups_per_batch": 1,
            "group_size": 2,
            "puct_c": 1.0,
            "eval_timeout": 5,
            "bootstrap": {"seed_library_path": str(seed_library_path)},
        },
        "llm": {"execution": {"provider": "mock"}},
    }
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True)
    existing_library = output_dir / "library.json"
    existing_library.write_text(
        json.dumps(
            {
                "nodes": {
                    "existing-root": {
                        "id": "existing-root",
                        "problem_id": "polyomino_packing",
                        "timestep": 0,
                        "entry_id": None,
                        "value": 0.0,
                        "raw_score": None,
                        "visits": 0,
                        "parent_id": None,
                        "children": [],
                        "metadata": {"existing": True},
                    }
                },
                "entries": {},
                "groups": {},
                "best_node_id": "existing-root",
            }
        )
    )

    prepared = prepare_run(config)

    library = yaml.safe_load(Path(prepared["library_path"]).read_text())
    assert "existing-root" in library["nodes"]
    assert "seed-root" not in library["nodes"]


def test_default_verl_config_dir_uses_local_verl_tree():
    expected = Path.cwd() / "verl" / "trainer" / "config"

    assert _default_verl_config_dir() == expected


def test_verl_overrides_enable_qwen_thinking_template_for_guidance_actor(tmp_path):
    config = {
        "run": {
            "output_dir": str(tmp_path / "outputs" / "erdos_smoke"),
            "model_path": "Qwen/Qwen3-8B",
            "num_initial_states": 1,
        },
        "ttt": {
            "groups_per_batch": 1,
            "group_size": 1,
            "puct_c": 1.0,
            "eval_timeout": 5,
        },
        "llm": {"execution": {"provider": "mock"}},
    }
    prepared = prepare_run(config)

    overrides = build_verl_overrides(config, prepared, [])

    assert "+data.apply_chat_template_kwargs.enable_thinking=True" in overrides
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=guidance_execution_task" in overrides


def test_verl_overrides_allow_non_failing_prompt_truncation(tmp_path):
    config = {
        "run": {
            "output_dir": str(tmp_path / "outputs" / "polyomino"),
            "model_path": "Qwen/Qwen3-8B",
            "num_initial_states": 1,
            "max_prompt_length": 4096,
            "max_response_length": 8192,
            "filter_overlong_prompts": False,
            "truncation": "middle",
        },
        "ttt": {
            "groups_per_batch": 1,
            "group_size": 1,
            "puct_c": 1.0,
            "eval_timeout": 5,
        },
        "llm": {"execution": {"provider": "mock"}},
    }
    prepared = prepare_run(config)

    overrides = build_verl_overrides(config, prepared, [])

    assert "data.max_prompt_length=4096" in overrides
    assert "data.max_response_length=8192" in overrides
    assert "data.filter_overlong_prompts=False" in overrides
    assert "data.truncation=middle" in overrides


def test_all_smoke_configs_enable_qwen_thinking_template_for_guidance_actor(tmp_path):
    config_dir = Path("guidance_ttt/config")
    config_dirs = [config_dir, config_dir / "backup"]
    smoke_configs = []
    for current_dir in config_dirs:
        smoke_configs.extend(sorted(current_dir.glob("*smoke*.yaml")))
        smoke_configs.extend(sorted(current_dir.glob("*single_summary*.yaml")))
    assert smoke_configs

    for config_path in smoke_configs:
        config = yaml.safe_load(config_path.read_text())
        config["run"]["output_dir"] = str(tmp_path / config_path.stem)
        config["run"]["hf_cache_dir"] = str(tmp_path / config_path.stem / "hf_cache")
        config["run"]["tmp_dir"] = str(tmp_path / config_path.stem / "tmp")
        config["run"]["triton_cache_dir"] = str(tmp_path / config_path.stem / "triton")
        config["run"]["ray_temp_dir"] = str(tmp_path / config_path.stem / "ray")
        prepared = prepare_run(config)

        overrides = build_verl_overrides(config, prepared, [])

        assert "+data.apply_chat_template_kwargs.enable_thinking=True" in overrides, config_path.name
        assert all("enable_thinking=False" not in override for override in overrides), config_path.name


def test_recipe_verl_overrides_are_appended(tmp_path):
    config = {
        "run": {
            "output_dir": str(tmp_path / "outputs" / "erdos_smoke"),
            "model_path": "Qwen/Qwen3-8B",
            "num_initial_states": 1,
        },
        "ttt": {
            "groups_per_batch": 1,
            "group_size": 1,
            "puct_c": 1.0,
            "eval_timeout": 5,
        },
        "llm": {"execution": {"provider": "mock"}},
        "verl_overrides": ["actor_rollout_ref.rollout.load_format=auto"],
    }
    prepared = prepare_run(config)

    overrides = build_verl_overrides(config, prepared, [])

    assert "actor_rollout_ref.rollout.load_format=auto" in overrides


def test_bootstrap_required_fails_when_library_has_only_roots(tmp_path):
    config = {
        "run": {
            "output_dir": str(tmp_path / "outputs" / "erdos_bootstrap_required"),
            "model_path": "Qwen/Qwen3-8B",
            "num_initial_states": 1,
        },
        "ttt": {
            "groups_per_batch": 1,
            "group_size": 1,
            "puct_c": 1.0,
            "eval_timeout": 5,
            "bootstrap": {"enabled": True, "required": True},
        },
        "llm": {"execution": {"provider": "mock"}},
    }
    prepared = prepare_run(config)

    with pytest.raises(RuntimeError, match="--bootstrap-only"):
        validate_bootstrap_requirement(config, prepared)

    library = GuidanceLibrary(prepared["library_path"])
    root_id = next(node_id for node_id, node in library.snapshot()["nodes"].items() if node["parent_id"] is None)
    library.attach_entry_to_root(
        root_id,
        LibraryEntry(
            id="bootstrap-entry",
            parent_id=root_id,
            problem_id="erdos_min_overlap",
            timestep=0,
            guidance="Bootstrap execution without guidance.",
            execution_thinking="thinking",
            solution="def run(seed=42, budget_s=1, **kwargs):\n    return ([0.5, 0.5], 0.5, 2)",
            verifier_reward=2.0,
            verifier_raw_score=0.5,
            verifier_status="valid",
            verifier_message="ok",
            summary="Bootstrap summary",
            reusable_idea="Bootstrap idea",
            failure_mode=None,
            metadata={"bootstrap": True, "raw_model_summary": "Bootstrap raw summary"},
        ),
    )

    validate_bootstrap_requirement(config, prepared)


def test_gpt_oss_recipes_initialize_fsdp_models_in_bfloat16():
    for recipe in [
        "erdos_gpt_oss_20b_5step.yaml",
        "erdos_gpt_oss_20b_8gpu_50step.yaml",
    ]:
        config = yaml.safe_load((Path("guidance_ttt/config/backup") / recipe).read_text())
        overrides = config["verl_overrides"]

        assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" in overrides
        assert "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16" in overrides


def test_polyomino_h200_qwen_discover_recipe_is_direct_policy_mode(tmp_path):
    config_path = Path("guidance_ttt/config/polyomino_h200_4gpu_qwen3_8b_discover.yaml")
    config_text = config_path.read_text()
    config = load_recipe_config(config_path)
    config["run"]["output_dir"] = str(tmp_path / "outputs" / "polyomino_discover")

    prepared = prepare_run(config)
    overrides = build_verl_overrides(config, prepared, [])
    agent_loop = yaml.safe_load(Path(prepared["agent_loop_config"]).read_text())[0]

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["agent_loop"] == "polyomino_discover_task"
    assert config["run"]["adv_estimator"] == "entropic_adaptive_beta"
    assert config["run"]["temperature"] == 1.0
    assert config["run"]["kl_loss_coef"] == 0.1
    assert config["run"]["save_freq"] == 2
    assert config["run"]["adam_beta1"] == 0.9
    assert config["run"]["adam_beta2"] == 0.95
    assert config["run"]["adam_eps"] == 1.0e-8
    assert config["ttt"]["groups_per_batch"] * config["ttt"]["group_size"] == 256
    assert config["ttt"]["bootstrap"] == {"enabled": False, "required": False}
    assert config["task"]["frontiercs"]["total_timeout_s"] == 1000
    assert "rollouts/step: 256" in config_text
    assert "group size 64 / 512 rollouts per step" in config_text
    assert "target KL ln(2)" in config_text

    lowered = config_text.lower()
    assert "modal" not in lowered
    assert "openrouter" not in lowered
    assert "gpt-oss" not in lowered

    assert agent_loop["name"] == "polyomino_discover_task"
    assert agent_loop["_target_"] == "guidance_ttt.agent_loop.PolyominoDiscoverAgentLoop"
    assert "execution_llm" not in agent_loop
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=polyomino_discover_task" in overrides
    assert "algorithm.adv_estimator=entropic_adaptive_beta" in overrides
    assert "actor_rollout_ref.rollout.n=32" in overrides
    assert "actor_rollout_ref.rollout.temperature=1.0" in overrides
    assert "actor_rollout_ref.actor.kl_loss_coef=0.1" in overrides
    assert "actor_rollout_ref.actor.optim.lr=4e-05" in overrides
    assert "actor_rollout_ref.actor.optim.betas=[0.9,0.95]" in overrides
    assert "actor_rollout_ref.actor.optim.override_optimizer_config={eps:1e-08}" in overrides


def test_entropic_adaptive_beta_registration_name_is_available():
    from guidance_ttt.verl_ext import compute_entropic_adaptive_beta, compute_ttt_reinforce_is, register_ttt_algorithms

    registered_adv = {}
    registered_loss = {}

    def register_adv(name):
        def decorator(fn):
            registered_adv[name] = fn
            return fn

        return decorator

    def register_loss(name):
        def decorator(fn):
            registered_loss[name] = fn
            return fn

        return decorator

    register_ttt_algorithms(register_adv_est=register_adv, register_policy_loss=register_loss)

    assert registered_adv["entropic_adaptive_beta"] is compute_entropic_adaptive_beta
    assert registered_loss["ttt_reinforce_is"] is compute_ttt_reinforce_is


def test_polyomino_qwen_discover_staged_submission_scripts_are_ordered():
    stage_runner = Path("scripts/run_local_polyomino_h200_qwen3_8b_discover_stage.sh").read_text()
    local_runner = Path("scripts/run_local_polyomino_h200_qwen3_8b_discover.sh").read_text()
    launcher = Path("scripts/launch_polyomino_h200_qwen3_8b_discover_staged.sh").read_text()
    slurm = Path("scripts/slurm_polyomino_h200_qwen3_8b_discover.sh").read_text()

    assert "tiny_smoke)" in stage_runner
    assert '"run.num_steps=1"' in stage_runner
    assert '"run.total_epochs=1"' in stage_runner
    assert '"run.n_gpus_per_node=1"' in stage_runner
    assert '"run.tensor_model_parallel_size=1"' in stage_runner
    assert 'EXPECTED_ROLLOUT_N="${EXPECTED_ROLLOUT_N:-1}"' in stage_runner
    assert 'EXPECTED_TENSOR_MODEL_PARALLEL_SIZE="${EXPECTED_TENSOR_MODEL_PARALLEL_SIZE:-1}"' in stage_runner
    assert '"ttt.groups_per_batch=1"' in stage_runner
    assert '"ttt.group_size=1"' in stage_runner
    assert '"actor_rollout_ref.rollout.tensor_model_parallel_size=1"' in stage_runner
    assert "one_step)" in stage_runner
    assert 'EXPECTED_ROLLOUT_N="${EXPECTED_ROLLOUT_N:-32}"' in stage_runner
    assert "polyomino_qwen3_8b_direct_discover_h200_4gpu_one_step_full_batch" in stage_runner
    one_step_block = stage_runner.split("one_step)", 1)[1].split("full)", 1)[0]
    assert "ttt.groups_per_batch" not in one_step_block
    assert "ttt.group_size" not in one_step_block

    assert "submit_stage tiny_smoke" in launcher
    assert "submit_stage one_step" in launcher
    assert "submit_stage full" in launcher
    assert 'TINY_GRES="${TINY_GRES:-gpu:h200:1}"' in launcher
    assert 'ONE_STEP_GRES="${ONE_STEP_GRES:-gpu:h200:4}"' in launcher
    assert 'FULL_GRES="${FULL_GRES:-gpu:h200:4}"' in launcher
    assert '--gres="$gres"' in launcher
    assert '--dependency="afterok:$dependency"' in launcher
    assert "one_step_job" in launcher

    assert "#SBATCH --gres=gpu:h200:4" in slurm
    assert "scripts/run_local_polyomino_h200_qwen3_8b_discover_stage.sh" in slurm
    assert "APPTAINER_IMAGE" in slurm
    assert "CONTAINER_PYTHON_BIN" in slurm
    assert "apptainer exec --nv" in local_runner
    assert '"${APPTAINER_EXEC[@]}" "$CONTAINER_PYTHON_BIN" -m guidance_ttt.main_erdos' in local_runner
