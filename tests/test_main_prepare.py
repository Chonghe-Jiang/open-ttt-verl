from pathlib import Path

import pandas as pd
import yaml

from guidance_ttt.main_erdos import _default_verl_config_dir, build_verl_overrides, prepare_run


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


def test_default_verl_config_dir_uses_local_verl_tree():
    expected = Path.cwd() / "verl" / "trainer" / "config"

    assert _default_verl_config_dir() == expected


def test_verl_overrides_disable_qwen_thinking_template_for_guidance_actor(tmp_path):
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

    assert "+data.apply_chat_template_kwargs.enable_thinking=False" in overrides
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=guidance_execution_task" in overrides


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


def test_gpt_oss_recipes_initialize_fsdp_models_in_bfloat16():
    for recipe in [
        "erdos_gpt_oss_20b_5step.yaml",
        "erdos_gpt_oss_20b_8gpu_50step.yaml",
    ]:
        config = yaml.safe_load((Path("guidance_ttt/config") / recipe).read_text())
        overrides = config["verl_overrides"]

        assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" in overrides
        assert "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16" in overrides
