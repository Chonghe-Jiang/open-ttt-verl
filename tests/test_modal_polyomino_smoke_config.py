import importlib.util
import json
import os
from pathlib import Path

import yaml


def _load_modal_smoke_module():
    path = Path("scripts/modal_polyomino_h200_smoke.py")
    spec = importlib.util.spec_from_file_location("modal_polyomino_h200_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modal_polyomino_h200_smoke_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/polyomino_modal_h200_4gpu_smoke.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["num_steps"] == 1
    assert config["run"]["total_epochs"] == 1
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["run"]["tensor_model_parallel_size"] == 4
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 32
    assert config["task"]["id"] == "polyomino_packing"
    assert config["task"]["frontiercs"]["base_dir"] == "/opt/Frontier-CS"
    assert config["task"]["frontiercs"]["judge_url"] == "http://127.0.0.1:8081"

    execution = config["llm"]["execution"]
    assert execution["provider"] == "local_vllm"
    assert execution["model"] == "openai/gpt-oss-20b"
    assert execution["tensor_model_parallel_size"] == 4
    assert execution["enforce_eager"] is True
    assert execution["max_tokens"] is None
    assert execution["phase1_max_tokens"] is None
    assert execution["max_num_seqs"] == 8
    assert execution["max_batch_size"] == 8
    assert execution["batch_wait_ms"] == 50


def test_polyomino_50step_execution_concurrency_is_capped_at_8():
    config_path = Path("guidance_ttt/config/polyomino_gpt_oss_20b_8gpu_50step.yaml")
    config = yaml.safe_load(config_path.read_text())

    execution = config["llm"]["execution"]
    assert execution["max_num_seqs"] == 8
    assert execution["max_batch_size"] == 8
    assert execution["batch_wait_ms"] == 50


def test_modal_polyomino_h200_smoke_script_targets_requested_config():
    module = _load_modal_smoke_module()

    assert module.GPU_CONFIG == "H200:4"
    assert module.REMOTE_CONFIG_PATH.endswith("polyomino_modal_h200_4gpu_smoke.yaml")
    assert module.training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_CONFIG_PATH,
    ]


def test_modal_polyomino_history_smoke_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/polyomino_modal_h200_2gpu_history_smoke.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["num_steps"] == 1
    assert config["run"]["total_epochs"] == 1
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["tensor_model_parallel_size"] == 2
    assert config["ttt"]["groups_per_batch"] == 2
    assert config["ttt"]["group_size"] == 4
    assert config["run"]["ppo_mini_batch_size"] == 2
    assert config["task"]["id"] == "polyomino_packing"

    execution = config["llm"]["execution"]
    assert execution["provider"] == "local_vllm"
    assert execution["model"] == "openai/gpt-oss-20b"
    assert execution["tensor_model_parallel_size"] == 2
    assert execution["max_tokens"] is None
    assert execution["phase1_max_tokens"] is None
    assert execution["max_num_seqs"] == 4
    assert execution["max_batch_size"] == 4


def test_modal_polyomino_history_smoke_script_targets_requested_config():
    module = _load_modal_smoke_module()

    assert module.HISTORY_GPU_CONFIG == "H200:2"
    assert module.REMOTE_HISTORY_CONFIG_PATH.endswith("polyomino_modal_h200_2gpu_history_smoke.yaml")
    assert module.REMOTE_HISTORY_OUTPUT_DIR.endswith("polyomino_modal_h200_2gpu_history_smoke")
    assert module.history_training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_HISTORY_CONFIG_PATH,
    ]


def test_modal_polyomino_single_summary_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/polyomino_modal_h200_2gpu_single_summary.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["num_steps"] == 1
    assert config["run"]["total_epochs"] == 1
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 24576
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["tensor_model_parallel_size"] == 2
    assert config["ttt"]["groups_per_batch"] == 1
    assert config["ttt"]["group_size"] == 2
    assert config["ttt"]["groups_per_batch"] * config["ttt"]["group_size"] == 2
    assert config["ttt"]["bootstrap"] == {
        "enabled": True,
        "required": True,
        "max_attempts": 2,
        "overwrite_existing": False,
    }
    assert config["task"]["id"] == "polyomino_packing"

    execution = config["llm"]["execution"]
    assert execution["provider"] == "local_vllm"
    assert execution["model"] == "openai/gpt-oss-20b"
    assert execution["tensor_model_parallel_size"] == 2
    assert execution["max_tokens"] is None
    assert execution["phase1_max_tokens"] is None
    assert execution["max_num_seqs"] == 1
    assert execution["max_batch_size"] == 1

    overrides = set(config["verl_overrides"])
    assert "actor_rollout_ref.rollout.max_model_len=32768" in overrides
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=32768" in overrides
    assert "actor_rollout_ref.rollout.enforce_eager=True" in overrides
    assert "actor_rollout_ref.actor.use_torch_compile=False" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False" in overrides
    assert "actor_rollout_ref.ref.use_torch_compile=False" in overrides
    assert "actor_rollout_ref.ref.fsdp_config.use_torch_compile=False" in overrides
    assert "+ray_kwargs.ray_init.runtime_env.env_vars.TORCHDYNAMO_DISABLE='1'" in overrides


def test_modal_polyomino_single_summary_script_targets_requested_config():
    module = _load_modal_smoke_module()

    assert module.REMOTE_SINGLE_SUMMARY_CONFIG_PATH.endswith("polyomino_modal_h200_2gpu_single_summary.yaml")
    assert module.REMOTE_SINGLE_SUMMARY_OUTPUT_DIR.endswith("polyomino_modal_h200_2gpu_single_summary")
    assert module.single_summary_bootstrap_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_SINGLE_SUMMARY_CONFIG_PATH,
        "--bootstrap-only",
    ]
    assert module.single_summary_training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_SINGLE_SUMMARY_CONFIG_PATH,
    ]


def test_modal_polyomino_qwen_exec_single_summary_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/polyomino_modal_h200_2gpu_qwen_exec_single_summary.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["num_steps"] == 1
    assert config["run"]["total_epochs"] == 1
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["tensor_model_parallel_size"] == 2
    assert config["run"]["experiment_name"] == "polyomino_qwen8b_actor_qwen8b_exec_modal_h200_2gpu_single_summary"
    assert config["run"]["output_dir"].endswith("polyomino_modal_h200_2gpu_qwen_exec_single_summary")
    assert config["ttt"]["groups_per_batch"] == 1
    assert config["ttt"]["group_size"] == 2
    assert config["ttt"]["bootstrap"]["seed_library_path"] == (
        "guidance_ttt/seeds/polyomino_packing/openrouter_gpt55_bootstrap_library.json"
    )
    assert config["task"]["id"] == "polyomino_packing"

    execution = config["llm"]["execution"]
    assert execution["provider"] == "local_vllm"
    assert execution["model"] == "Qwen/Qwen3-8B"
    assert "reasoning_effort" not in execution
    assert execution["chat_template_kwargs"] == {"enable_thinking": True}
    assert execution["tensor_model_parallel_size"] == 2
    assert execution["max_tokens"] is None
    assert execution["phase1_max_tokens"] is None
    assert execution["max_num_seqs"] == 1
    assert execution["max_batch_size"] == 1
    assert execution["max_model_len"] == 32768


def test_modal_polyomino_qwen_exec_single_summary_script_targets_requested_config():
    module = _load_modal_smoke_module()

    assert module.REMOTE_QWEN_EXEC_SINGLE_SUMMARY_CONFIG_PATH.endswith(
        "polyomino_modal_h200_2gpu_qwen_exec_single_summary.yaml"
    )
    assert module.REMOTE_QWEN_EXEC_SINGLE_SUMMARY_OUTPUT_DIR.endswith(
        "polyomino_modal_h200_2gpu_qwen_exec_single_summary"
    )
    assert module.qwen_exec_single_summary_bootstrap_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_QWEN_EXEC_SINGLE_SUMMARY_CONFIG_PATH,
        "--bootstrap-only",
    ]
    assert module.qwen_exec_single_summary_training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_QWEN_EXEC_SINGLE_SUMMARY_CONFIG_PATH,
    ]


def test_modal_polyomino_openrouter_gpt55_single_summary_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/polyomino_modal_h200_2gpu_openrouter_gpt55_single_summary.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["num_steps"] == 1
    assert config["run"]["total_epochs"] == 1
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 24576
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["tensor_model_parallel_size"] == 2
    assert config["run"]["experiment_name"] == "polyomino_qwen8b_actor_openrouter_gpt55_exec_modal_h200_2gpu_single_summary"
    assert config["run"]["output_dir"].endswith("polyomino_modal_h200_2gpu_openrouter_gpt55_single_summary")
    assert config["ttt"]["groups_per_batch"] == 1
    assert config["ttt"]["group_size"] == 2
    assert config["task"]["id"] == "polyomino_packing"

    execution = config["llm"]["execution"]
    assert execution == {
        "provider": "openai_compatible",
        "model": "openai/gpt-5.5",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "temperature": 0.35,
        "max_tokens": None,
        "phase1_max_tokens": None,
    }


def test_modal_polyomino_openrouter_gpt55_single_summary_script_targets_requested_config_and_secret():
    module = _load_modal_smoke_module()
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()

    assert module.REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_CONFIG_PATH.endswith(
        "polyomino_modal_h200_2gpu_openrouter_gpt55_single_summary.yaml"
    )
    assert module.REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR.endswith(
        "polyomino_modal_h200_2gpu_openrouter_gpt55_single_summary"
    )
    assert module.openrouter_gpt55_single_summary_bootstrap_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_CONFIG_PATH,
        "--bootstrap-only",
    ]
    assert module.openrouter_gpt55_single_summary_training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_CONFIG_PATH,
    ]
    source = script[script.index("def train_openrouter_gpt55_seeded_single_summary_smoke") :]
    source = source[: source.index("@app.function", 1) if "@app.function" in source[1:] else len(source)]
    assert "_reset_output_dir(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)" in source
    assert "openrouter_gpt55_single_summary_training_command()" in source
    assert "openrouter_gpt55_single_summary_bootstrap_command()" not in source
    assert "train_single_summary_openrouter_gpt55_seeded" in script
    assert 'modal.Secret.from_name("openrouter-api-key")' in script
    assert "secrets=[openrouter_secret]" in script


def test_modal_train_image_uses_spawn_for_local_vllm_workers():
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()

    assert '"VLLM_WORKER_MULTIPROC_METHOD": "spawn"' in script


def test_modal_polyomino_single_summary_launcher_bootstraps_before_training():
    launcher = Path("scripts/run_modal_polyomino_single_summary.sh").read_text()

    bootstrap_index = launcher.index("--action bootstrap_single_summary")
    train_index = launcher.index("--action train_single_summary")
    assert bootstrap_index < train_index
    assert "modal run --detach scripts/modal_polyomino_h200_smoke.py --action train_single_summary" in launcher


def test_modal_polyomino_single_summary_training_does_not_auto_prune():
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()
    start = script.index("def train_single_summary_smoke")
    end = script.index("@app.function", start + 1)
    source = script[start:end]

    assert "prune_summary = _prune_to_single_summary" not in source
    assert "_dump_prompt_answer_artifacts" in source


def test_modal_polyomino_prompt_answer_dump_includes_all_entries(tmp_path):
    module = _load_modal_smoke_module()
    output_dir = tmp_path / "single"
    output_dir.mkdir()
    library_path = output_dir / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "nodes": {},
                "entries": {
                    "entry-bootstrap": {
                        "id": "entry-bootstrap",
                        "timestep": 0,
                        "guidance": "Bootstrap execution without guidance.",
                        "summary": "bootstrap canonical summary",
                        "verifier_reward": 10.0,
                        "verifier_raw_score": 10.0,
                        "verifier_status": "valid",
                        "metadata": {
                            "raw_model_summary": "bootstrap raw summary",
                            "execution_prompt": {"system": "bootstrap exec system", "user": "bootstrap exec user"},
                            "execution_text": "<execution_thinking>bootstrap</execution_thinking>",
                            "verification_artifacts": {"cases": 1},
                        },
                    },
                    "entry-guided": {
                        "id": "entry-guided",
                        "timestep": 1,
                        "guidance": "<think>diagnosis</think><guidance>guided change</guidance>",
                        "summary": "guided canonical summary",
                        "verifier_reward": 20.0,
                        "verifier_raw_score": 20.0,
                        "verifier_status": "invalid",
                        "metadata": {
                            "guidance_format_ok": True,
                            "raw_guidance_text": "<think>raw diagnosis</think><guidance>raw guided change</guidance>",
                            "raw_guidance_with_specials": "<|assistant|><think>raw diagnosis</think><guidance>raw guided change</guidance>",
                            "raw_model_summary": "guided raw summary",
                            "guidance_prompt": {"system": "guidance system", "user": "guidance user"},
                            "execution_prompt": {"system": "execution system", "user": "execution user"},
                            "execution_text": "<execution_thinking>guided</execution_thinking>",
                            "verification_artifacts": {"error": "bad packing"},
                        },
                    },
                },
                "best_node_id": None,
            }
        )
    )

    dump_summary = module._dump_prompt_answer_artifacts(str(output_dir))
    markdown = Path(dump_summary["markdown_path"]).read_text()
    data = json.loads(Path(dump_summary["json_path"]).read_text())

    assert dump_summary["entry_count"] == 2
    assert {entry["entry_id"] for entry in data["entries"]} == {"entry-bootstrap", "entry-guided"}
    assert "entry-bootstrap" in markdown
    assert "entry-guided" in markdown
    guided_entry = next(entry for entry in data["entries"] if entry["entry_id"] == "entry-guided")
    assert guided_entry["raw_guidance_text"] == "<think>raw diagnosis</think><guidance>raw guided change</guidance>"
    assert guided_entry["raw_guidance_with_specials"] == "<|assistant|><think>raw diagnosis</think><guidance>raw guided change</guidance>"
    assert "### Raw Guidance Answer" in markdown
    assert "<think>raw diagnosis</think><guidance>raw guided change</guidance>" in markdown
    assert "### Raw Guidance Answer With Specials" in markdown
    assert "<|assistant|><think>raw diagnosis</think><guidance>raw guided change</guidance>" in markdown
    assert "guidance system" in markdown
    assert "execution user" in markdown
    assert "<execution_thinking>guided</execution_thinking>" in markdown


def test_modal_polyomino_train_smoke_resets_output_dir(tmp_path):
    module = _load_modal_smoke_module()
    output_dir = tmp_path / "polyomino_modal_h200_4gpu_smoke"
    stale_file = output_dir / "library.json"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("{}")

    module._reset_output_dir(str(output_dir))

    assert output_dir.exists()
    assert not stale_file.exists()


def test_modal_polyomino_single_summary_prunes_to_one_guided_entry(tmp_path):
    module = _load_modal_smoke_module()
    output_dir = tmp_path / "single"
    output_dir.mkdir()
    library_path = output_dir / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "nodes": {
                    "root": {
                        "id": "root",
                        "problem_id": "0",
                        "timestep": 0,
                        "entry_id": None,
                        "value": 0.0,
                        "raw_score": None,
                        "visits": 0,
                        "parent_id": None,
                        "children": ["node-a", "node-b"],
                        "metadata": {},
                    },
                    "node-a": {
                        "id": "node-a",
                        "problem_id": "0",
                        "timestep": 1,
                        "entry_id": "entry-a",
                        "value": 10.0,
                        "raw_score": 10.0,
                        "visits": 0,
                        "parent_id": "root",
                        "children": [],
                        "metadata": {"verifier_status": "valid"},
                    },
                    "node-b": {
                        "id": "node-b",
                        "problem_id": "0",
                        "timestep": 1,
                        "entry_id": "entry-b",
                        "value": 1.0,
                        "raw_score": 1.0,
                        "visits": 0,
                        "parent_id": "root",
                        "children": [],
                        "metadata": {"verifier_status": "valid"},
                    },
                },
                "entries": {
                    "entry-a": {
                        "id": "entry-a",
                        "guidance": "fallback guidance",
                        "summary": "fallback summary",
                        "verifier_reward": 10.0,
                        "verifier_raw_score": 10.0,
                        "verifier_status": "valid",
                        "metadata": {"guidance_format_ok": False, "execution_text": "code"},
                    },
                    "entry-b": {
                        "id": "entry-b",
                        "guidance": "<mutation>guided</mutation>",
                        "summary": "guided summary",
                        "verifier_reward": 1.0,
                        "verifier_raw_score": 1.0,
                        "verifier_status": "valid",
                        "metadata": {"guidance_format_ok": True, "execution_text": "code"},
                    },
                },
                "groups": {"group": {"children": ["node-a", "node-b"], "submitted": 2, "finalized": True}},
                "best_node_id": "node-a",
                "config": {"rollout_n": 2},
                "rollout_n": 2,
            }
        )
    )

    prune_summary = module._prune_to_single_summary(str(output_dir))
    data = json.loads(library_path.read_text())
    history_summary = module._summarize_history_extraction(str(output_dir))

    assert prune_summary["entry_count"] == 1
    assert set(data["entries"]) == {"entry-b"}
    assert data["best_node_id"] == "node-b"
    assert data["nodes"]["root"]["children"] == ["node-b"]
    assert data["config"]["rollout_n"] == 1
    assert history_summary["entry_count"] == 1
    assert history_summary["guidance_ok_count"] == 1
    assert history_summary["canonical_summary_ok_count"] == 1
    assert history_summary["execution_text_ok_count"] == 1
    module._validate_single_summary_extraction(history_summary)


def test_modal_dotenv_key_loader_handles_workspace(tmp_path):
    module = _load_modal_smoke_module()
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# ignored",
                "OTHER=value",
                "WORKSPACE='modal-workspace'",
            ]
        )
    )

    assert module._load_dotenv_key(env_path, "WORKSPACE") == "modal-workspace"
    assert module._load_dotenv_key(env_path, "MISSING") is None


def test_modal_dotenv_key_loader_handles_export_prefix(tmp_path):
    module = _load_modal_smoke_module()
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "export WORKSPACE=ac-target",
                "export MODAL_TOKEN_ID=ak-target",
            ]
        )
    )

    assert module._load_dotenv_key(env_path, "WORKSPACE") == "ac-target"
    assert module._load_dotenv_keys(env_path, {"MODAL_TOKEN_ID"}) == {"MODAL_TOKEN_ID": "ak-target"}


def test_modal_dotenv_key_loader_handles_modal_token_fields(tmp_path):
    module = _load_modal_smoke_module()
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "WORKSPACE=ac-target",
                "MODAL_TOKEN_ID=ak-target",
                "MODAL_TOKEN_SECRET='as-target'",
            ]
        )
    )

    assert module._load_dotenv_keys(env_path, {"WORKSPACE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}) == {
        "WORKSPACE": "ac-target",
        "MODAL_TOKEN_ID": "ak-target",
        "MODAL_TOKEN_SECRET": "as-target",
    }


def test_modal_auth_uses_dotenv_tokens(tmp_path, monkeypatch):
    module = _load_modal_smoke_module()
    env_path = tmp_path / ".env"
    env_path.write_text("WORKSPACE=ac-target\nMODAL_TOKEN_ID=ak-target\nMODAL_TOKEN_SECRET=as-target\n")
    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    assert module._configure_modal_auth(env_path) == {
        "WORKSPACE": "ac-target",
        "MODAL_TOKEN_ID": "ak-target",
        "MODAL_TOKEN_SECRET": "as-target",
    }
    module._validate_modal_token_env()


def test_modal_auth_uses_modal_secret_key_alias(tmp_path, monkeypatch):
    module = _load_modal_smoke_module()
    env_path = tmp_path / ".env"
    env_path.write_text("WORKSPACE=ac-target\nMODAL_TOKEN_ID=ak-target\nMODAL_SECRET_KEY=as-target\n")
    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("MODAL_SECRET_KEY", raising=False)

    assert module._configure_modal_auth(env_path) == {
        "WORKSPACE": "ac-target",
        "MODAL_TOKEN_ID": "ak-target",
        "MODAL_TOKEN_SECRET": "as-target",
    }
    module._validate_modal_token_env()


def test_modal_auth_uses_modal_secret_key_environment_alias(tmp_path, monkeypatch):
    module = _load_modal_smoke_module()
    env_path = tmp_path / ".env"
    env_path.write_text("WORKSPACE=ac-target\nMODAL_TOKEN_ID=ak-target\n")
    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-env")
    monkeypatch.setenv("MODAL_SECRET_KEY", "as-env")

    assert module._configure_modal_auth(env_path) == {
        "WORKSPACE": "ac-target",
        "MODAL_TOKEN_ID": "ak-env",
        "MODAL_TOKEN_SECRET": "as-env",
    }
    module._validate_modal_token_env()


def test_modal_auth_uses_dotenv_token_id_with_matching_local_secret(tmp_path, monkeypatch):
    module = _load_modal_smoke_module()
    env_path = tmp_path / ".env"
    env_path.write_text("WORKSPACE=ac-target\nMODAL_TOKEN_ID=ak-target\n")
    (tmp_path / ".modal.toml").write_text(
        "[target]\n"
        "token_id = 'ak-target'\n"
        "token_secret = 'as-target'\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    assert module._configure_modal_auth(env_path) == {
        "WORKSPACE": "ac-target",
        "MODAL_TOKEN_ID": "ak-target",
        "MODAL_TOKEN_SECRET": "as-target",
    }
    module._validate_modal_token_env()


def test_modal_token_secret_for_id_returns_matching_secret(tmp_path):
    module = _load_modal_smoke_module()
    config_path = tmp_path / ".modal.toml"
    config_path.write_text(
        "[first]\n"
        "token_id = 'ak-first'\n"
        "token_secret = 'as-first'\n"
        "[second]\n"
        "token_id = 'ak-second'\n"
        "token_secret = 'as-second'\n"
    )

    assert module._modal_token_secret_for_id("ak-second", config_path) == "as-second"
    assert module._modal_token_secret_for_id("ak-missing", config_path) is None


def test_modal_auth_rejects_partial_dotenv_token(tmp_path, monkeypatch):
    module = _load_modal_smoke_module()
    env_path = tmp_path / ".env"
    env_path.write_text("WORKSPACE=ac-target\nMODAL_TOKEN_ID=ak-target\n")
    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    module._configure_modal_auth(env_path)
    try:
        module._validate_modal_token_env()
    except RuntimeError as exc:
        assert "MODAL_TOKEN_SECRET" in str(exc)
    else:
        raise AssertionError("partial Modal token env should be rejected")


def test_modal_profile_uses_workspace_from_dotenv(tmp_path, monkeypatch):
    module = _load_modal_smoke_module()
    env_path = tmp_path / ".env"
    env_path.write_text("WORKSPACE=modal-workspace\n")
    (tmp_path / ".modal.toml").write_text(
        "[modal-workspace]\n"
        "token_id = 'ak-test'\n"
        "token_secret = 'as-test'\n"
    )
    modal_config = importlib.import_module("modal.config")
    monkeypatch.setattr(modal_config, "_profile", "default")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.delenv("MODAL_PROFILE", raising=False)

    assert module._configure_modal_profile(env_path) == "modal-workspace"
    assert os.environ["MODAL_PROFILE"] == "modal-workspace"
    assert modal_config._profile == "modal-workspace"


def test_modal_profile_ignores_workspace_that_is_not_a_profile(tmp_path, monkeypatch):
    module = _load_modal_smoke_module()
    env_path = tmp_path / ".env"
    env_path.write_text("WORKSPACE=ac-not-a-profile\n")
    (tmp_path / ".modal.toml").write_text(
        "[modal-workspace]\n"
        "active = true\n"
        "token_id = 'ak-test'\n"
        "token_secret = 'as-test'\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.delenv("MODAL_PROFILE", raising=False)

    assert module._configure_modal_profile(env_path) == "ac-not-a-profile"
    assert "MODAL_PROFILE" not in os.environ


def test_modal_token_workspace_parser_accepts_workspace_id():
    module = _load_modal_smoke_module()

    assert module._parse_modal_token_workspace(
        "Token: ak-test\nWorkspace: target-workspace (ac-target)\nUser: user (us-test)\n"
    ) == ("target-workspace", "ac-target")


def test_modal_workspace_validation_accepts_name_or_id(monkeypatch):
    module = _load_modal_smoke_module()
    monkeypatch.setattr(module, "_current_modal_token_workspace", lambda: ("target-workspace", "ac-target"))

    module._validate_modal_workspace("target-workspace")
    module._validate_modal_workspace("ac-target")
