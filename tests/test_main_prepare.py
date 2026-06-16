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
    assert library["config"] == {"rollout_n": 3, "puct_c": 1.5}
    slots = pd.read_parquet(prepared["slot_parquet"]).to_dict("records")
    assert slots[0]["extra_info"]["rollout_n"] == 3
    assert slots[0]["extra_info"]["group_size"] == 3
    assert slots[0]["extra_info"]["puct_c"] == 1.5
    data = yaml.safe_load(Path(prepared["agent_loop_config"]).read_text())
    assert data[0]["name"] == "guidance_execution_erdos"
    assert data[0]["verifier_timeout_s"] == 5
    assert "execution_llm" in data[0]
    assert "summarizer_llm" not in data[0]


def test_default_verl_config_dir_uses_local_verl_tree():
    expected = Path.cwd() / "verl" / "trainer" / "config"

    assert _default_verl_config_dir() == expected


def test_verl_overrides_enable_qwen_thinking_template(tmp_path):
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
