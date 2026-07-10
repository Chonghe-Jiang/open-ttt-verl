# Copyright 2026 Chonghe Jiang and/or contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");

import yaml

from verl_ttt_discover.cpp_sandbox import extract_cpp_code
from verl_ttt_discover.discover_compat import last_codeblock_postprocess
from verl_ttt_discover.main_polyomino import _build_verl_overrides, _prepare_run
from verl_ttt_discover.polyomino_env import (
    BASELINE_CPP_CODE,
    PolyominoEnv,
    SAMPLE_INPUT,
    REWARD_NORMALIZER,
    evaluate_polyomino_cpp,
    evaluate_polyomino_cpp_official,
    parse_polyomino_input,
    verify_polyomino_output,
)


def test_extract_cpp_code_uses_last_cpp_block():
    response = """Draft:
```cpp
int main(){return 1;}
```
Final:
```c++
#include <bits/stdc++.h>
int main(){return 0;}
```"""

    code = extract_cpp_code(response)

    assert code is not None
    assert "return 0" in code


def test_extract_cpp_code_accepts_open_final_cpp_block():
    response = "```cpp\n#include <bits/stdc++.h>\nint main(){return 0;}\n"

    code = extract_cpp_code(response)

    assert code is not None
    assert "int main" in code


def test_polyomino_checker_accepts_statement_sample_output():
    output = """3 3
2 0 0 0
2 2 2 0
0 0 0 0
"""

    ratio = verify_polyomino_output(parse_polyomino_input(SAMPLE_INPUT), output)

    assert ratio == 8 / 9


def test_baseline_cpp_compiles_and_scores_positive():
    score, message = evaluate_polyomino_cpp(BASELINE_CPP_CODE, timeout_s=8)

    assert score > 0
    assert "case1=ratio" in message
    assert score / REWARD_NORMALIZER > 0


def test_baseline_cpp_scores_positive_with_official_checker(monkeypatch, tmp_path):
    monkeypatch.setenv("FRONTIER_CS_POLYOMINO_CASE_LIMIT", "1")
    monkeypatch.setenv("FRONTIER_CS_POLYOMINO_CONCURRENCY", "1")
    monkeypatch.setenv("FRONTIER_CS_CACHE_DIR", str(tmp_path / "frontier_cs"))

    score, message = evaluate_polyomino_cpp_official(
        BASELINE_CPP_CODE,
        timeout_s=8,
        cache_dir=str(tmp_path / "frontier_cs"),
        concurrency=1,
        case_limit=1,
    )

    assert score > 0
    assert "official_cases=1" in message


def test_discover_parser_uses_final_cpp_block_without_separators():
    response = """Plan first.
```cpp
int main(){return 1;}
```
```c++
int main(){return 0;}
```"""

    code = last_codeblock_postprocess(response, codeblock_seps=["cpp", "c++"], keep_separators=False)

    assert code == "int main(){return 0;}"


def test_polyomino_env_accepts_final_cpp_block_and_creates_child_state(monkeypatch, tmp_path):
    monkeypatch.setenv("FRONTIER_CS_POLYOMINO_CASE_LIMIT", "1")
    monkeypatch.setenv("FRONTIER_CS_POLYOMINO_CONCURRENCY", "1")
    monkeypatch.setenv("FRONTIER_CS_CACHE_DIR", str(tmp_path / "frontier_cs"))
    state = PolyominoEnv.create_initial_state("frontier_cs_polyomino")
    env = PolyominoEnv(initial_state=state, problem_type="frontier_cs_polyomino", eval_timeout=8)

    step = env.step_response(f"```cpp\n{BASELINE_CPP_CODE}\n```", step_idx=3)

    assert step.reward > 0
    assert step.verify_result.correctness == 1.0
    assert step.next_state is not None
    assert step.next_state.timestep == 3
    assert step.next_state.raw_score == step.verify_result.raw_score


def test_polyomino_h200_config_preserves_requested_shape():
    config = yaml.safe_load(open("verl_ttt_discover/config/polyomino_2gpu_h200_qwen3_8b_g4_n16.yaml"))

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["ttt"]["groups_per_batch"] == 4
    assert config["ttt"]["group_size"] == 16
    assert config["ttt"]["use_chat_template"] is True
    assert config["ttt"]["cpp_min_tokens"] == 0
    assert config["ttt"]["faithful_discover"] is True
    assert config["ttt"]["update_puct_per_rollout"] is True
    assert config["ttt"]["official_concurrency"] == 16
    assert config["ttt"]["invalid_reward"] == 0.0
    assert "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2" in config["verl_overrides"]
    assert "actor_rollout_ref.rollout.load_format=auto" in config["verl_overrides"]


def test_polyomino_prepare_writes_polyomino_agent_loop_and_slots(monkeypatch, tmp_path):
    monkeypatch.setenv("FRONTIER_CS_POLYOMINO_CASE_LIMIT", "1")
    monkeypatch.setenv("FRONTIER_CS_POLYOMINO_CONCURRENCY", "1")
    monkeypatch.setenv("FRONTIER_CS_CACHE_DIR", str(tmp_path / "frontier_cs"))
    config = {
        "run": {"output_dir": str(tmp_path), "model_path": "dummy-model", "num_initial_states": 1},
        "ttt": {"groups_per_batch": 1, "group_size": 2, "eval_timeout": 8, "phase1_max_tokens": 26000},
    }

    prepared = _prepare_run(config)
    overrides = _build_verl_overrides(config, prepared, [])

    agent_loop_config = prepared["agent_loop_config"].read_text()
    assert "ttt_discover_polyomino" in agent_loop_config
    assert "use_chat_template: True" in agent_loop_config
    assert "cpp_min_tokens: 0" in agent_loop_config
    assert "faithful_discover: True" in agent_loop_config
    assert "update_puct_per_rollout: True" in agent_loop_config
    assert "official_concurrency: 16" in agent_loop_config
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=ttt_discover_polyomino" in overrides
    assert "+data.apply_chat_template_kwargs.enable_thinking=True" in overrides
