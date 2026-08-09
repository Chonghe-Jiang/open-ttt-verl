from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

from guidance_ttt.main_erdos import prepare_run
from guidance_ttt.tasks.trimul import TRIMUL_BASELINE_SOLUTION

CODE_DELTA_CONFIG_PATH = Path(
    "guidance_ttt/config/trimul_modal_h200_1gpu_evolvent_glm52_group2_1step.yaml"
)
SUMMARY_ONLY_CONFIG_PATH = Path(
    "guidance_ttt/config/trimul_modal_h200_1gpu_evolvent_glm52_group2_summary_only_1step.yaml"
)
SCRATCH_SEED_CONFIG_PATH = Path("guidance_ttt/config/trimul_modal_glm52_scratch_seed.yaml")
FROZEN_SCRATCH_SEED_PATH = Path(
    "guidance_ttt/seeds/trimul/glm52_scratch_bootstrap_library.json"
)
SCRIPT_PATH = Path("scripts/modal_trimul_h200_smoke.py")


def test_trimul_modal_config_is_one_step_code_delta_smoke():
    config = yaml.safe_load(CODE_DELTA_CONFIG_PATH.read_text())

    assert config["run"]["num_steps"] == 1
    assert config["run"]["n_gpus_per_node"] == 1
    assert config["run"]["max_prompt_length"] == 16384
    assert config["run"]["max_response_length"] == 8192
    assert config["run"]["filter_overlong_prompts"] is False
    assert config["ttt"]["prompt_mode"] == "code_delta"
    assert config["ttt"]["groups_per_batch"] == 1
    assert config["ttt"]["group_size"] == 2
    assert config["task"]["id"] == "trimul"
    assert config["task"]["verifier"]["provider"] == "modal_http"
    assert config["task"]["verifier"]["api_key_env"] == "TRIMUL_JUDGE_TOKEN"
    assert config["task"]["verifier"]["concurrency"] == 2
    assert config["task"]["verifier"]["max_retries"] == 2
    assert config["task"]["verifier"]["reward_scale"] == 1500.0
    assert config["llm"]["execution"]["model"] == "glm-5.2"
    assert config["llm"]["execution"]["enable_thinking"] is False
    assert config["llm"]["execution"]["temperature"] == 0.0
    assert config["llm"]["execution"]["max_tokens"] == 16384
    assert config["llm"]["execution"]["api_key_env"] == "EVOLVENT_API_KEY"
    assert config["llm"]["execution"]["concurrency"] == 2
    assert config["ttt"]["bootstrap"]["source"] == "scratch"
    assert config["ttt"]["bootstrap"]["seed_library_path"] == str(FROZEN_SCRATCH_SEED_PATH)


def test_trimul_summary_only_config_is_a_matched_one_step_smoke():
    code_delta = yaml.safe_load(CODE_DELTA_CONFIG_PATH.read_text())
    summary_only = yaml.safe_load(SUMMARY_ONLY_CONFIG_PATH.read_text())

    assert summary_only["ttt"]["prompt_mode"] == "summary_only"
    assert "summary_only" in summary_only["run"]["output_dir"]
    assert "summary_only" in summary_only["run"]["experiment_name"]

    for section in ("task", "llm", "verl_overrides"):
        assert summary_only[section] == code_delta[section]
    assert {**summary_only["ttt"], "prompt_mode": "code_delta"} == code_delta["ttt"]
    assert {
        **summary_only["run"],
        "output_dir": code_delta["run"]["output_dir"],
        "experiment_name": code_delta["run"]["experiment_name"],
    } == code_delta["run"]


def test_trimul_scratch_seed_config_uses_no_prior_library_or_task_baseline():
    config = yaml.safe_load(SCRATCH_SEED_CONFIG_PATH.read_text())

    assert config["task"]["id"] == "trimul"
    assert config["ttt"]["prompt_mode"] == "summary_only"
    assert config["ttt"]["bootstrap"]["source"] == "scratch"
    assert "seed_library_path" not in config["ttt"]["bootstrap"]
    assert config["ttt"]["bootstrap"]["max_attempts"] == 1
    assert config["llm"]["execution"]["model"] == "glm-5.2"
    assert config["llm"]["execution"]["temperature"] == 0.0
    assert config["task"]["verifier"]["provider"] == "modal_http"


def test_trimul_smoke_prepare_copies_the_frozen_seed(tmp_path):
    config = yaml.safe_load(CODE_DELTA_CONFIG_PATH.read_text())
    config["run"]["output_dir"] = str(tmp_path / "run")

    paths = prepare_run(config)
    prepared = json.loads(paths["library_path"].read_text())
    entries = list(prepared["entries"].values())

    assert len(entries) == 1
    assert entries[0]["metadata"]["bootstrap_source"] == "scratch"
    assert entries[0]["metadata"]["execution_model"] == "glm-5.2"
    assert hashlib.sha256(entries[0]["solution"].encode()).hexdigest() == (
        "49485cfe6f1e0f5d2ea40df2bead246b59ab8774cff3a79ab03cba4cf08995a9"
    )


def test_trimul_modal_script_pins_h100_eval_and_contains_no_credentials():
    spec = importlib.util.spec_from_file_location("modal_trimul_smoke", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = SCRIPT_PATH.read_text()
    assert module.TRAIN_GPU_CONFIG == "H200:1"
    assert module.EVALUATION_GPU_CONFIG == "H100!"
    assert set(module.REMOTE_CONFIG_PATHS) == {"code_delta", "summary_only"}
    assert "summary_only" in module.REMOTE_CONFIG_PATHS["summary_only"]
    assert module.SCRATCH_SEED_CONFIG_PATH.endswith("trimul_modal_glm52_scratch_seed.yaml")
    assert module.SCRATCH_SEED_OUTPUT_DIR.endswith("trimul_modal_glm52_scratch_seed")
    assert module.FROZEN_SCRATCH_SEED_RELATIVE_PATH == str(FROZEN_SCRATCH_SEED_PATH)
    assert "torch>=2.7.0,<2.8.0" in source
    assert "triton==3.3.1" in source
    assert "run_official_trimul_evaluation" in source
    assert "max_containers=2" in source
    assert re.search(r"\bsk-[A-Za-z0-9_-]{20,}\b", source) is None
    assert re.search(r"\bak-[A-Za-z0-9_-]{20,}\b", source) is None
    assert re.search(r"\bas-[A-Za-z0-9_-]{20,}\b", source) is None


def test_frozen_trimul_scratch_seed_is_valid_and_has_no_prior_candidate_context():
    payload = json.loads(FROZEN_SCRATCH_SEED_PATH.read_text())
    entries = list(payload["entries"].values())
    nodes = list(payload["nodes"].values())

    assert len(entries) == 1
    assert len(nodes) == 1
    assert payload["groups"] == {}
    assert payload["puct_T"] == 0
    entry = entries[0]
    node = nodes[0]
    metadata = entry["metadata"]
    prompt_user = metadata["execution_prompt"]["user"]
    artifacts = metadata["verification_artifacts"]
    solution = entry["solution"]

    assert node["parent_id"] is None
    assert node["entry_id"] == entry["id"]
    assert entry["problem_id"] == "trimul"
    assert entry["timestep"] == 0
    assert entry["verifier_status"] == "valid"
    assert entry["verifier_raw_score"] == pytest.approx(10177.396849081848)
    assert entry["verifier_reward"] == pytest.approx(1500.0 / entry["verifier_raw_score"])
    assert metadata["bootstrap"] is True
    assert metadata["bootstrap_source"] == "scratch"
    assert metadata["execution_model"] == "glm-5.2"
    assert metadata["raw_model_summary"].strip()
    assert "There is no parent candidate, prior solution" in prompt_user
    assert "<parent_code>" not in prompt_user
    assert "<baseline_candidate>" not in prompt_user
    assert "<baseline_code>" not in prompt_user
    assert TRIMUL_BASELINE_SOLUTION not in prompt_user
    assert solution != TRIMUL_BASELINE_SOLUTION
    assert artifacts["test_count"] == 18
    assert artifacts["test"]["passed"] is True
    assert artifacts["benchmark_count"] == 7
    assert artifacts["leaderboard"]["passed"] is True
    assert hashlib.sha256(solution.encode()).hexdigest() == (
        "49485cfe6f1e0f5d2ea40df2bead246b59ab8774cff3a79ab03cba4cf08995a9"
    )


def test_trimul_scratch_seed_audit_rejects_prior_context(tmp_path):
    spec = importlib.util.spec_from_file_location("modal_trimul_scratch_audit", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entry = {
        "id": "scratch-seed",
        "timestep": 0,
        "solution": "import triton\n@triton.jit\ndef kernel():\n    pass\n",
        "verifier_status": "valid",
        "verifier_raw_score": 2000.0,
        "verifier_reward": 0.75,
        "metadata": {
            "bootstrap_source": "scratch",
            "bootstrap_attempts": 1,
            "execution_model": "glm-5.2",
            "raw_model_summary": "A standalone scratch implementation.",
            "execution_prompt": {
                "user": (
                    "There is no parent candidate, prior solution, library summary, guidance, verifier score, "
                    "or search history."
                )
            },
        },
    }
    (tmp_path / "library.json").write_text(json.dumps({"entries": {"scratch-seed": entry}}))

    audit = module._scratch_seed_audit(output_dir=str(tmp_path))
    assert audit["bootstrap_source"] == "scratch"
    assert audit["prompt_has_parent"] is False
    assert audit["prompt_has_baseline"] is False

    entry["metadata"]["execution_prompt"]["user"] += "\n<baseline_candidate>old</baseline_candidate>"
    (tmp_path / "library.json").write_text(json.dumps({"entries": {"scratch-seed": entry}}))
    with pytest.raises(RuntimeError, match="baseline candidate"):
        module._scratch_seed_audit(output_dir=str(tmp_path))


def test_trimul_judge_subprocess_cannot_read_runtime_secrets(monkeypatch):
    spec = importlib.util.spec_from_file_location("modal_trimul_env", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setenv("EVOLVENT_API_KEY", "execution-secret")
    monkeypatch.setenv(module.AUTH_TOKEN_ENV, "judge-secret")
    monkeypatch.setenv(module.LEGACY_AUTH_TOKEN_ENV, "legacy-judge-secret")
    monkeypatch.setenv("SAFE_TEST_VALUE", "retained")

    env = module._judge_subprocess_env()

    assert "EVOLVENT_API_KEY" not in env
    assert module.AUTH_TOKEN_ENV not in env
    assert module.LEGACY_AUTH_TOKEN_ENV not in env
    assert env["SAFE_TEST_VALUE"] == "retained"
    assert env["NO_PROXY"] == "*"


def test_trimul_smoke_summary_distinguishes_global_and_child_best(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("modal_trimul_summary", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = {
        "entries": {
            "root": {
                "id": "root",
                "timestep": 0,
                "verifier_status": "valid",
                "verifier_raw_score": 100.0,
                "metadata": {},
            },
            "valid-child": {
                "id": "valid-child",
                "timestep": 1,
                "verifier_status": "valid",
                "verifier_raw_score": 120.0,
                "solution": "code",
                "metadata": {
                    "raw_model_summary": "delta",
                    "execution_response_metadata": {"finish_reason": "stop"},
                },
            },
            "invalid-child": {
                "id": "invalid-child",
                "timestep": 1,
                "verifier_status": "invalid",
                "verifier_raw_score": None,
                "solution": "code",
                "metadata": {
                    "raw_model_summary": "delta",
                    "execution_response_metadata": {"finish_reason": "stop"},
                },
            },
        },
        "groups": {"group": {"finalized": True}},
        "puct_T": 1,
        "best_node_id": "root",
    }
    (tmp_path / "library.json").write_text(json.dumps(payload))
    monkeypatch.setattr(module, "REMOTE_OUTPUT_DIR", str(tmp_path))

    summary = module._smoke_summary(training_completed=True)

    assert summary["best_runtime_us"] == 100.0
    assert summary["best_child_runtime_us"] == 120.0


def test_trimul_smoke_validator_checks_summary_only_prompt_transport():
    spec = importlib.util.spec_from_file_location("modal_trimul_validator", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = {
        "training_completed": True,
        "root_entries": 1,
        "valid_roots": 1,
        "child_entries": 2,
        "parsed_children": 2,
        "valid_children": 1,
        "groups": 1,
        "finalized_groups": 1,
        "puct_T": 1,
        "finish_reasons": {"stop": 2},
        "prompt_modes": {"summary_only": 3},
        "root_prompt_modes": {"summary_only": 1},
        "child_prompt_modes": {"summary_only": 2},
        "root_bootstrap_sources": {"scratch": 1},
        "summary_only_guidance_contexts": 2,
        "code_delta_guidance_contexts": 0,
        "execution_parent_contexts": 2,
    }

    module._validate_smoke(summary, expected_prompt_mode="summary_only")
    summary["summary_only_guidance_contexts"] = 1
    with pytest.raises(RuntimeError, match="summary_only guidance contexts=1"):
        module._validate_smoke(summary, expected_prompt_mode="summary_only")

    summary.update(
        {
            "prompt_modes": {"summary_only": 1, "code_delta": 2},
            "child_prompt_modes": {"code_delta": 2},
            "summary_only_guidance_contexts": 0,
            "code_delta_guidance_contexts": 2,
        }
    )
    module._validate_smoke(summary, expected_prompt_mode="code_delta")
