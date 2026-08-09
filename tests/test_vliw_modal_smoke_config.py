from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import yaml

CONFIG_PATH = Path(
    "guidance_ttt/config/vliw_kernel_modal_h200_2gpu_glm52_batch8_group16_1step.yaml"
)
SCRIPT_PATH = Path("scripts/modal_vliw_kernel_h200_smoke.py")


def test_vliw_modal_recipe_matches_8x16_one_step_shape():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["run"]["num_steps"] == 1
    assert config["run"]["total_epochs"] == 1
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 8192
    assert config["ttt"]["prompt_mode"] == "code_delta"
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 16
    assert config["task"]["id"] == "vliw_kernel_optimization"
    assert config["task"]["edgebench"]["provider"] == "modal_http"
    assert config["task"]["edgebench"]["concurrency"] == 16
    assert config["task"]["edgebench"]["runner_timeout_s"] == 60
    assert config["task"]["edgebench"]["reward_mode"] == "inverse_cycles"
    assert config["llm"]["execution"]["model"] == "glm-5.2"
    assert config["llm"]["execution"]["api_key_env"] == "EVOLVENT_API_KEY"
    assert config["llm"]["execution"]["enable_thinking"] is False
    assert config["llm"]["execution"]["concurrency"] == 16
    assert config["llm"]["execution"]["max_tokens"] == 8192
    assert config["ttt"]["bootstrap"]["max_attempts"] == 8


def test_vliw_modal_script_uses_two_h200s_and_contains_no_literal_credentials():
    spec = importlib.util.spec_from_file_location("modal_vliw_smoke", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = SCRIPT_PATH.read_text()
    assert module.GPU_CONFIG == "H200:2"
    assert module.JUDGE_IMAGE.endswith(":5cdef0021634")
    assert module.SECRET_NAME == "guidance-ttt-vliw-secrets"
    assert '"pyarrow>=19.0.0,<21.0.0"' in source
    assert '"fsspec[http]==2023.9.2"' in source
    assert "runtime_versions" in source
    assert "slot_dataset_rows" in source
    assert "def bootstrap_vliw_seed" in source
    assert 'get("execution_response_metadata")' in source
    assert re.search(r"\bsk-[A-Za-z0-9_-]{20,}\b", source) is None
    assert re.search(r"\bak-[A-Za-z0-9_-]{20,}\b", source) is None
    assert re.search(r"\bas-[A-Za-z0-9_-]{20,}\b", source) is None


def test_vliw_judge_subprocess_cannot_read_runtime_secrets(monkeypatch):
    spec = importlib.util.spec_from_file_location("modal_vliw_smoke_env", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setenv("EVOLVENT_API_KEY", "execution-secret")
    monkeypatch.setenv("VLIW_JUDGE_TOKEN", "judge-secret")
    monkeypatch.setenv("SAFE_TEST_VALUE", "retained")

    env = module._judge_subprocess_env()

    assert "EVOLVENT_API_KEY" not in env
    assert "VLIW_JUDGE_TOKEN" not in env
    assert env["SAFE_TEST_VALUE"] == "retained"
    assert env["NO_PROXY"] == "*"
    assert env["no_proxy"] == "*"


def test_vliw_modal_upload_ignores_runtime_and_secret_paths():
    spec = importlib.util.spec_from_file_location("modal_vliw_smoke_ignore", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._repo_ignore(Path(".runtime/live/heartbeat")) is True
    assert module._repo_ignore(Path(".secrets/evolvent_api_key")) is True
    assert module._repo_ignore(module.REPO_ROOT / "results/run/library.json") is True
    assert module._repo_ignore(Path("guidance_ttt/agent_loop.py")) is False


def test_vliw_modal_summary_reads_nested_execution_finish_reason(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("modal_vliw_smoke_summary", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = {
        "entries": {
            "root": {
                "id": "root",
                "timestep": 0,
                "verifier_status": "valid",
                "verifier_reward": 1.0,
            },
            "child": {
                "id": "child",
                "timestep": 1,
                "solution": "class KernelBuilder: pass",
                "verifier_status": "valid",
                "verifier_reward": 1.0,
                "metadata": {
                    "execution_model": "glm-5.2",
                    "execution_response_metadata": {"finish_reason": "length"},
                    "raw_model_summary": "Truncated test summary.",
                },
            },
        },
        "groups": {},
        "puct_T": 0,
        "best_node_id": "root",
    }
    (tmp_path / "library.json").write_text(json.dumps(payload))
    monkeypatch.setattr(module, "REMOTE_OUTPUT_DIR", str(tmp_path))

    summary = module._smoke_summary(training_completed=False)

    assert summary["finish_reasons"] == {"length": 1}
    assert summary["complete_valid_children"] == 1
