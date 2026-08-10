from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


LIBRARY_VALIDATOR = Path("scripts/validate_trimul_evolvent_cache_smoke.py")
TRAINING_VALIDATOR = Path("scripts/validate_trimul_training_update.py")
CONFIG_DIR = Path("guidance_ttt/config")
GROUP16_FORMAL_CONFIG = CONFIG_DIR / (
    "trimul_b200_1gpu_qwen3_8b_evolvent_glm52_thinking_cache_"
    "batch8_group16_30step.yaml"
)
GROUP32_FORMAL_CONFIG = CONFIG_DIR / (
    "trimul_b200_1gpu_qwen3_8b_evolvent_glm52_thinking_cache_"
    "batch8_group32_30step.yaml"
)
GROUP32_SMOKE_CONFIG = CONFIG_DIR / (
    "trimul_b200_1gpu_qwen3_8b_evolvent_glm52_thinking_cache_"
    "batch8_group32_1step.yaml"
)


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_group32_formal_changes_only_group_scale_and_run_identifiers():
    group16 = _load_config(GROUP16_FORMAL_CONFIG)
    group32 = _load_config(GROUP32_FORMAL_CONFIG)

    assert group16["ttt"]["group_size"] == 16
    assert group32["ttt"]["group_size"] == 32

    normalized = copy.deepcopy(group32)
    normalized["ttt"]["group_size"] = 16
    normalized["run"]["output_dir"] = group16["run"]["output_dir"]
    normalized["run"]["experiment_name"] = group16["run"]["experiment_name"]
    assert normalized == group16


def test_group32_smoke_preserves_formal_algorithm_settings():
    formal = _load_config(GROUP32_FORMAL_CONFIG)
    smoke = _load_config(GROUP32_SMOKE_CONFIG)

    normalized = copy.deepcopy(smoke)
    normalized["run"]["output_dir"] = formal["run"]["output_dir"]
    normalized["run"]["experiment_name"] = formal["run"]["experiment_name"]
    normalized["run"]["num_steps"] = formal["run"]["num_steps"]
    normalized["run"]["total_epochs"] = formal["run"]["total_epochs"]
    normalized["run"]["save_freq"] = formal["run"]["save_freq"]
    assert normalized == formal


def test_large_smoke_validator_accepts_eight_complete_groups(tmp_path):
    entries = {}
    for root_index in range(8):
        root_id = f"root-{root_index}"
        entries[root_id] = {
            "id": root_id,
            "timestep": 0,
            "verifier_status": "valid",
            "verifier_raw_score": 10000.0,
        }
    groups = {}
    for group_index in range(8):
        child_ids = []
        for child_index in range(16):
            index = group_index * 16 + child_index
            entry_id = f"entry-{index}"
            node_id = f"node-{index}"
            if index == 0:
                child_ids.append(node_id)
            role = "primer_owner" if index == 0 else "follower"
            entries[entry_id] = {
                "id": entry_id,
                "timestep": 1,
                "execution_thinking": "reasoning",
                "solution": "def custom_kernel(data):\n    return data\n",
                "verifier_status": "valid" if index == 0 else "invalid",
                "verifier_raw_score": 9000.0 if index == 0 else None,
                "metadata": {
                    "raw_model_summary": "summary",
                    "execution_response_metadata": {
                        "api_streaming": True,
                        "cache_warm_role": role,
                        "cache_prime_succeeded": True,
                        "finish_reason": "stop",
                    },
                    "execution_response_usage": {
                        "prompt_tokens": 4300,
                        "prompt_tokens_details": {"cached_tokens": 4224},
                    },
                },
            }
        groups[f"group-{group_index}"] = {
            "children": child_ids,
            "entry_ids": [f"entry-{group_index * 16 + i}" for i in range(16)],
            "finalized": True,
            "submitted": 16,
        }
    payload = {
        "entries": entries,
        "groups": groups,
        "puct_T": 128,
        "config": {"discover_compat": True},
    }
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "library.json").write_text(json.dumps(payload))

    completed = subprocess.run(
        [
            sys.executable,
            str(LIBRARY_VALIDATOR),
            str(output_dir),
            "--expected-children",
            "128",
            "--expected-groups",
            "8",
            "--expected-group-size",
            "16",
            "--minimum-cache-hits",
            "16",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["children"] == 128
    assert summary["roots"] == 8
    assert summary["groups"] == 8
    assert summary["puct_T"] == 128
    assert summary["follower_cache_hit_children"] == 127
    assert summary["valid_children"] == 1
    assert summary["root_runtime_us"] == 10000.0
    assert summary["best_child_runtime_us"] == 9000.0


def test_training_update_validator_requires_a_real_optimizer_step(tmp_path):
    trainer_log = tmp_path / "trainer.log"
    trainer_log.write_text(
        "step:1 - actor/pg_loss:0.25 - actor/kl_loss:0.0 - "
        "actor/grad_norm:1.5 - actor/lr:4e-05 - training/global_step:1 - "
        "training/rollout_probs_diff_valid:1 - policy/empty_optimization_mask:0.0 - "
        "response/aborted_ratio:0.0 - timing_s/update_actor:7.5 - "
        "perf/total_num_tokens:250000\n"
    )

    completed = subprocess.run(
        [sys.executable, str(TRAINING_VALIDATOR), str(trainer_log)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["training_global_step"] == 1
    assert summary["actor_grad_norm"] == 1.5
    assert summary["actor_update_s"] == 7.5
