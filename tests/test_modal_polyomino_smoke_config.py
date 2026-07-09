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
    config_path = Path("guidance_ttt/config/backup/polyomino_modal_h200_4gpu_smoke.yaml")
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
    config_path = Path("guidance_ttt/config/backup/polyomino_gpt_oss_20b_8gpu_50step.yaml")
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
    config_path = Path("guidance_ttt/config/backup/polyomino_modal_h200_2gpu_history_smoke.yaml")
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
    config_path = Path("guidance_ttt/config/backup/polyomino_modal_h200_2gpu_single_summary.yaml")
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
    config_path = Path("guidance_ttt/config/backup/polyomino_modal_h200_2gpu_qwen_exec_single_summary.yaml")
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
    config_path = Path("guidance_ttt/config/backup/polyomino_modal_h200_2gpu_openrouter_gpt55_single_summary.yaml")
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


def test_modal_polyomino_openrouter_gpt55_4gpu_full_batch_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/backup/polyomino_modal_h200_4gpu_openrouter_gpt55_full_batch.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["num_steps"] == 1
    assert config["run"]["total_epochs"] == 1
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 24576
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["run"]["tensor_model_parallel_size"] == 4
    assert config["run"]["ppo_mini_batch_size"] == 8
    assert config["run"]["output_dir"].endswith("polyomino_modal_h200_4gpu_openrouter_gpt55_full_batch")
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 32
    assert config["ttt"]["bootstrap"]["seed_library_path"] == (
        "guidance_ttt/seeds/polyomino_packing/openrouter_gpt55_bootstrap_library.json"
    )
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

    overrides = set(config["verl_overrides"])
    assert "actor_rollout_ref.rollout.max_model_len=32768" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=8" in overrides
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=32768" in overrides


def test_modal_polyomino_openrouter_gpt55_4gpu_full_batch_script_targets_requested_config_and_secret():
    module = _load_modal_smoke_module()
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()

    assert module.REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_CONFIG_PATH.endswith(
        "polyomino_modal_h200_4gpu_openrouter_gpt55_full_batch.yaml"
    )
    assert module.REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_OUTPUT_DIR.endswith(
        "polyomino_modal_h200_4gpu_openrouter_gpt55_full_batch"
    )
    assert module.openrouter_gpt55_4gpu_full_batch_training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_CONFIG_PATH,
    ]
    function_start = script.rindex("@app.function", 0, script.index("def train_openrouter_gpt55_4gpu_full_batch_smoke"))
    source = script[function_start:]
    source = source[: source.index("@app.function", 1) if "@app.function" in source[1:] else len(source)]
    assert "gpu=GPU_CONFIG" in source
    assert "_reset_output_dir(REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_OUTPUT_DIR)" in source
    assert "openrouter_gpt55_4gpu_full_batch_training_command()" in source
    assert "train_openrouter_gpt55_4gpu_full_batch" in script
    assert "secrets=[openrouter_secret]" in source


def test_modal_polyomino_evolvent_gpt55_4gpu_short_response_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/backup/polyomino_modal_h200_4gpu_evolvent_gpt55_short_response.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["num_steps"] == 1
    assert config["run"]["total_epochs"] == 1
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 4096
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["run"]["tensor_model_parallel_size"] == 4
    assert config["run"]["ppo_mini_batch_size"] == 8
    assert config["run"]["ppo_micro_batch_size_per_gpu"] == 2
    assert config["run"]["output_dir"].endswith("polyomino_modal_h200_4gpu_evolvent_gpt55_short_response")
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 32
    assert config["ttt"]["bootstrap"]["seed_library_path"] == (
        "guidance_ttt/seeds/polyomino_packing/openrouter_gpt55_bootstrap_library.json"
    )
    assert config["task"]["id"] == "polyomino_packing"

    execution = config["llm"]["execution"]
    assert execution == {
        "provider": "openai_compatible",
        "model": "sub2api-gpt-5.4",
        "base_url": "http://llmapi.evolventapi.com/v1",
        "api_key_env": "API_KEY",
        "temperature": 0.35,
        "max_tokens": None,
        "phase1_max_tokens": None,
    }

    overrides = set(config["verl_overrides"])
    assert "actor_rollout_ref.actor.use_remove_padding=True" in overrides
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2" in overrides
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=12288" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=8" in overrides
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=12288" in overrides


def test_modal_polyomino_evolvent_gpt55_4gpu_short_response_script_targets_requested_config_and_secret():
    module = _load_modal_smoke_module()
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()

    assert module.REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_CONFIG_PATH.endswith(
        "polyomino_modal_h200_4gpu_evolvent_gpt55_short_response.yaml"
    )
    assert module.REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_OUTPUT_DIR.endswith(
        "polyomino_modal_h200_4gpu_evolvent_gpt55_short_response"
    )
    assert module.evolvent_gpt55_4gpu_short_response_training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_CONFIG_PATH,
    ]
    function_start = script.rindex("@app.function", 0, script.index("def train_evolvent_gpt55_4gpu_short_response_smoke"))
    source = script[function_start:]
    source = source[: source.index("@app.function", 1) if "@app.function" in source[1:] else len(source)]
    assert "gpu=GPU_CONFIG" in source
    assert "_reset_output_dir(REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_OUTPUT_DIR)" in source
    assert "evolvent_gpt55_4gpu_short_response_training_command()" in source
    assert "train_evolvent_gpt55_4gpu_short_response" in script
    assert 'modal.Secret.from_name("evolvent-api-key")' in script
    assert "secrets=[evolvent_secret]" in source


def test_modal_polyomino_gpt_oss_120b_5gpu_group64_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/backup/polyomino_modal_h200_5gpu_gpt_oss_120b_group64.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["num_steps"] == 1
    assert config["run"]["total_epochs"] == 1
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 4096
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["run"]["tensor_model_parallel_size"] == 4
    assert config["run"]["ppo_mini_batch_size"] == 8
    assert config["run"]["ppo_micro_batch_size_per_gpu"] == 1
    assert config["run"]["output_dir"].endswith("polyomino_modal_h200_5gpu_gpt_oss_120b_group64")
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 64
    assert config["ttt"]["bootstrap"]["seed_library_path"] == (
        "guidance_ttt/seeds/polyomino_packing/openrouter_gpt55_bootstrap_library.json"
    )
    assert config["task"]["id"] == "polyomino_packing"

    execution = config["llm"]["execution"]
    assert execution == {
        "provider": "openai_compatible",
        "model": "openai/gpt-oss-120b",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "local-vllm",
        "temperature": 0.35,
        "max_tokens": None,
        "phase1_max_tokens": None,
        "timeout_s": 1200,
        "concurrency": 8,
    }

    overrides = set(config["verl_overrides"])
    assert "actor_rollout_ref.actor.use_remove_padding=True" in overrides
    assert "actor_rollout_ref.rollout.agent.num_workers=1" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=12288" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=64" in overrides
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=12288" in overrides
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" in overrides
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" in overrides


def test_modal_polyomino_gpt_oss_120b_5gpu_group64_script_targets_requested_config():
    module = _load_modal_smoke_module()
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()

    assert module.GPT_OSS_120B_GPU_CONFIG == "H200:5"
    assert module.REMOTE_GPT_OSS_120B_5GPU_GROUP64_CONFIG_PATH.endswith(
        "polyomino_modal_h200_5gpu_gpt_oss_120b_group64.yaml"
    )
    assert module.REMOTE_GPT_OSS_120B_5GPU_GROUP64_OUTPUT_DIR.endswith(
        "polyomino_modal_h200_5gpu_gpt_oss_120b_group64"
    )
    assert module.gpt_oss_120b_5gpu_group64_training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_GPT_OSS_120B_5GPU_GROUP64_CONFIG_PATH,
    ]
    function_start = script.rindex("@app.function", 0, script.index("def train_gpt_oss_120b_5gpu_group64_smoke"))
    source = script[function_start:]
    source = source[: source.index("@app.function", 1) if "@app.function" in source[1:] else len(source)]
    assert "gpu=GPT_OSS_120B_GPU_CONFIG" in source
    assert "_reset_output_dir(REMOTE_GPT_OSS_120B_5GPU_GROUP64_OUTPUT_DIR)" in source
    assert "_start_gpt_oss_120b_server()" in source
    assert '"CUDA_VISIBLE_DEVICES": "0,1,2,3"' in source
    assert "gpt_oss_120b_5gpu_group64_training_command()" in source
    assert "train_gpt_oss_120b_5gpu_group64" in script
    assert 'def _start_gpt_oss_120b_server(*, cuda_visible_devices: str = "4")' in script
    assert '_start_gpt_oss_120b_server()' in source
    assert '"--max-num-seqs"' in script
    assert '"8"' in script


def test_modal_polyomino_gpt_oss_120b_3gpu_group16_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/backup/polyomino_modal_h200_3gpu_gpt_oss_120b_group16.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["num_steps"] == 1
    assert config["run"]["total_epochs"] == 1
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 4096
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["tensor_model_parallel_size"] == 2
    assert config["run"]["gpu_memory_utilization"] == 0.5
    assert config["run"]["ppo_mini_batch_size"] == 4
    assert config["run"]["ppo_micro_batch_size_per_gpu"] == 2
    assert config["run"]["output_dir"].endswith("polyomino_modal_h200_3gpu_gpt_oss_120b_group16")
    assert config["ttt"]["groups_per_batch"] == 4
    assert config["ttt"]["group_size"] == 16
    assert config["ttt"]["bootstrap"]["seed_library_path"] == (
        "guidance_ttt/seeds/polyomino_packing/openrouter_gpt55_bootstrap_library.json"
    )
    assert config["task"]["id"] == "polyomino_packing"

    execution = config["llm"]["execution"]
    assert execution == {
        "provider": "openai_compatible",
        "model": "openai/gpt-oss-120b",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "local-vllm",
        "temperature": 0.35,
        "max_tokens": None,
        "phase1_max_tokens": None,
        "timeout_s": 1200,
        "concurrency": 8,
    }

    overrides = set(config["verl_overrides"])
    assert "actor_rollout_ref.actor.use_remove_padding=True" in overrides
    assert "actor_rollout_ref.rollout.agent.num_workers=1" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=12288" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=16" in overrides
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=12288" in overrides
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2" in overrides
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2" in overrides


def test_modal_polyomino_gpt_oss_120b_3gpu_group16_script_targets_requested_config():
    module = _load_modal_smoke_module()
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()

    assert module.GPT_OSS_120B_3GPU_CONFIG == "H200:3"
    assert module.REMOTE_GPT_OSS_120B_3GPU_GROUP16_CONFIG_PATH.endswith(
        "polyomino_modal_h200_3gpu_gpt_oss_120b_group16.yaml"
    )
    assert module.REMOTE_GPT_OSS_120B_3GPU_GROUP16_OUTPUT_DIR.endswith(
        "polyomino_modal_h200_3gpu_gpt_oss_120b_group16"
    )
    assert module.gpt_oss_120b_3gpu_group16_training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_GPT_OSS_120B_3GPU_GROUP16_CONFIG_PATH,
    ]
    function_start = script.rindex("@app.function", 0, script.index("def train_gpt_oss_120b_3gpu_group16_smoke"))
    source = script[function_start:]
    source = source[: source.index("@app.function", 1) if "@app.function" in source[1:] else len(source)]
    assert "gpu=GPT_OSS_120B_3GPU_CONFIG" in source
    assert "_reset_output_dir(REMOTE_GPT_OSS_120B_3GPU_GROUP16_OUTPUT_DIR)" in source
    assert '_start_gpt_oss_120b_server(cuda_visible_devices="2")' in source
    assert '"CUDA_VISIBLE_DEVICES": "0,1"' in source
    assert "gpt_oss_120b_3gpu_group16_training_command()" in source
    assert "train_gpt_oss_120b_3gpu_group16" in script


def test_modal_polyomino_gpt_oss_120b_3gpu_group16_h200_tuned_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["num_steps"] == 50
    assert config["run"]["total_epochs"] == 50
    assert config["run"]["save_freq"] == 5
    assert config["run"]["max_prompt_length"] == 4096
    assert config["run"]["max_response_length"] == 8192
    assert config["run"]["filter_overlong_prompts"] is False
    assert config["run"]["truncation"] == "middle"
    assert config["run"]["learning_rate"] == 4.0e-5
    assert config["run"]["kl_loss_coef"] == 0.05
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["tensor_model_parallel_size"] == 1
    assert config["run"]["gpu_memory_utilization"] == 0.7
    assert config["run"]["ppo_mini_batch_size"] == 4
    assert config["run"]["ppo_micro_batch_size_per_gpu"] == 2
    assert config["run"]["output_dir"].endswith("polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned_50step")
    assert config["ttt"]["groups_per_batch"] == 4
    assert config["ttt"]["group_size"] == 16
    assert config["ttt"]["bootstrap"]["seed_library_path"] == (
        "guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json"
    )
    assert config["task"]["id"] == "polyomino_packing"

    execution = config["llm"]["execution"]
    assert execution == {
        "provider": "openai_compatible",
        "model": "openai/gpt-oss-120b",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "local-vllm",
        "temperature": 0.35,
        "max_tokens": None,
        "phase1_max_tokens": None,
        "timeout_s": 1200,
        "concurrency": 8,
    }

    overrides = set(config["verl_overrides"])
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in overrides
    assert "actor_rollout_ref.rollout.load_format=safetensors" in overrides
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" in overrides
    assert "actor_rollout_ref.rollout.free_cache_engine=False" in overrides
    assert "actor_rollout_ref.rollout.layered_summon=False" in overrides
    assert "actor_rollout_ref.rollout.enforce_eager=False" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=False" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False" in overrides
    assert "actor_rollout_ref.ref.fsdp_config.param_offload=False" in overrides
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=12288" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=16" in overrides
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=32768" in overrides
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2" in overrides
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2" in overrides


def test_modal_polyomino_gpt_oss_120b_bootstrap_seed_config_creates_fresh_library():
    config_path = Path("guidance_ttt/config/polyomino_modal_h200_gpt_oss_120b_bootstrap_seed.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["output_dir"].endswith("polyomino_modal_h200_gpt_oss_120b_bootstrap_seed")
    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["task"]["id"] == "polyomino_packing"
    assert config["ttt"]["groups_per_batch"] == 1
    assert config["ttt"]["group_size"] == 1
    assert config["ttt"]["bootstrap"] == {
        "enabled": True,
        "required": False,
        "max_attempts": 4,
        "overwrite_existing": True,
    }

    execution = config["llm"]["execution"]
    assert execution["provider"] == "openai_compatible"
    assert execution["model"] == "openai/gpt-oss-120b"
    assert execution["base_url"] == "http://127.0.0.1:8000/v1"
    assert execution["api_key"] == "local-vllm"
    assert execution["temperature"] == 0.0
    assert execution["max_tokens"] is None
    assert execution["timeout_s"] == 1200


def test_polyomino_gpt_oss_120b_seed_library_is_valid_and_model_matched():
    seed_path = Path("guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json")
    data = json.loads(seed_path.read_text())
    entries = list(data["entries"].values())

    assert len(entries) == 1
    entry = entries[0]
    assert entry["verifier_status"] == "valid"
    assert entry["verifier_raw_score"] is not None
    assert entry["verifier_reward"] == entry["verifier_raw_score"]
    assert entry["metadata"]["execution_model"] == "openai/gpt-oss-120b"
    assert entry["metadata"]["execution_provider"] == "openai_compatible"
    assert entry["metadata"]["raw_model_summary"]
    assert entry["metadata"]["execution_text"].strip().endswith("</summary>")


def test_modal_polyomino_gpt_oss_120b_3gpu_group16_h200_tuned_script_targets_requested_config():
    module = _load_modal_smoke_module()
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()

    assert module.GPT_OSS_120B_3GPU_CONFIG == "H200:3"
    assert module.REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_CONFIG_PATH.endswith(
        "polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned.yaml"
    )
    assert module.REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_OUTPUT_DIR.endswith(
        "polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned_50step"
    )
    assert module.gpt_oss_120b_3gpu_group16_h200_tuned_training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_CONFIG_PATH,
    ]


def test_modal_polyomino_gpt_oss_120b_3gpu_batch8_group8_temp09_config_matches_requested_shape():
    config_path = Path("guidance_ttt/config/polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group8_temp09.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["num_steps"] == 20
    assert config["run"]["total_epochs"] == 20
    assert config["run"]["save_freq"] == 5
    assert config["run"]["temperature"] == 0.9
    assert config["run"]["output_dir"].endswith("polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group8_temp09_20step")
    assert config["run"]["max_prompt_length"] == 4096
    assert config["run"]["max_response_length"] == 8192
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["tensor_model_parallel_size"] == 1
    assert config["run"]["ppo_mini_batch_size"] == 4
    assert config["run"]["ppo_micro_batch_size_per_gpu"] == 2
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 8
    assert config["ttt"]["groups_per_batch"] * config["ttt"]["group_size"] == 64
    assert config["ttt"]["bootstrap"]["seed_library_path"] == (
        "guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json"
    )
    assert config["task"]["id"] == "polyomino_packing"

    execution = config["llm"]["execution"]
    assert execution == {
        "provider": "openai_compatible",
        "model": "openai/gpt-oss-120b",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "local-vllm",
        "temperature": 0.0,
        "max_tokens": None,
        "phase1_max_tokens": None,
        "timeout_s": 1200,
        "concurrency": 8,
    }

    overrides = set(config["verl_overrides"])
    assert "actor_rollout_ref.rollout.top_p=0.95" in overrides
    assert "actor_rollout_ref.rollout.val_kwargs.do_sample=False" in overrides
    assert "actor_rollout_ref.rollout.val_kwargs.temperature=0" in overrides
    assert "actor_rollout_ref.rollout.val_kwargs.top_p=1.0" in overrides
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=16" in overrides
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=32768" in overrides


def test_modal_polyomino_gpt_oss_120b_3gpu_batch8_group8_temp09_script_targets_requested_config():
    module = _load_modal_smoke_module()
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()

    assert module.GPT_OSS_120B_3GPU_CONFIG == "H200:3"
    assert module.REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_CONFIG_PATH.endswith(
        "polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group8_temp09.yaml"
    )
    assert module.REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_OUTPUT_DIR.endswith(
        "polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group8_temp09_20step"
    )
    assert module.gpt_oss_120b_3gpu_batch8_group8_temp09_training_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_CONFIG_PATH,
    ]
    function_start = script.rindex(
        "@app.function",
        0,
        script.index("def train_gpt_oss_120b_3gpu_batch8_group8_temp09_smoke"),
    )
    source = script[function_start:]
    source = source[: source.index("@app.function", 1) if "@app.function" in source[1:] else len(source)]
    assert "gpu=GPT_OSS_120B_3GPU_CONFIG" in source
    assert "_reset_output_dir(REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_OUTPUT_DIR)" in source
    assert '_start_gpt_oss_120b_server(cuda_visible_devices="2")' in source
    assert '"CUDA_VISIBLE_DEVICES": "0,1"' in source
    assert "gpt_oss_120b_3gpu_batch8_group8_temp09_training_command()" in source
    assert "train_gpt_oss_120b_3gpu_batch8_group8_temp09" in script


def test_modal_polyomino_gpt_oss_120b_bootstrap_seed_script_targets_single_h200():
    module = _load_modal_smoke_module()
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()

    assert module.GPT_OSS_120B_SINGLE_GPU_CONFIG == "H200:1"
    assert module.REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_CONFIG_PATH.endswith(
        "polyomino_modal_h200_gpt_oss_120b_bootstrap_seed.yaml"
    )
    assert module.REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_OUTPUT_DIR.endswith(
        "polyomino_modal_h200_gpt_oss_120b_bootstrap_seed"
    )
    assert module.gpt_oss_120b_bootstrap_seed_command() == [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        module.REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_CONFIG_PATH,
        "--bootstrap-only",
    ]

    function_start = script.rindex("@app.function", 0, script.index("def bootstrap_gpt_oss_120b_seed_smoke"))
    source = script[function_start:]
    source = source[: source.index("@app.function", 1) if "@app.function" in source[1:] else len(source)]
    assert "gpu=GPT_OSS_120B_SINGLE_GPU_CONFIG" in source
    assert "_reset_output_dir(REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_OUTPUT_DIR)" in source
    assert '_start_gpt_oss_120b_server(cuda_visible_devices="0")' in source
    assert "gpt_oss_120b_bootstrap_seed_command()" in source
    assert "bootstrap_gpt_oss_120b_seed" in script
    function_start = script.rindex(
        "@app.function",
        0,
        script.index("def train_gpt_oss_120b_3gpu_group16_h200_tuned_smoke"),
    )
    source = script[function_start:]
    source = source[: source.index("@app.function", 1) if "@app.function" in source[1:] else len(source)]
    assert "gpu=GPT_OSS_120B_3GPU_CONFIG" in source
    assert "_reset_output_dir(REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_OUTPUT_DIR)" in source
    assert '_start_gpt_oss_120b_server(cuda_visible_devices="2")' in source
    assert '"CUDA_VISIBLE_DEVICES": "0,1"' in source
    assert "gpt_oss_120b_3gpu_group16_h200_tuned_training_command()" in source
    assert "train_gpt_oss_120b_3gpu_group16_h200_tuned" in script


def test_modal_train_image_installs_and_checks_flash_attn_for_remove_padding():
    script = Path("scripts/modal_polyomino_h200_smoke.py").read_text()

    assert "FLASH_ATTN_TORCH29_CU12_WHEEL" in script
    assert "flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" in script
    assert ".uv_pip_install(FLASH_ATTN_TORCH29_CU12_WHEEL)" in script
    assert "import flash_attn; print('flash_attn'" in script
    assert "def check_training_packages_smoke()" in script
    assert "elif action == \"check_training_packages\"" in script

    for function_name in [
        "check_training_packages_smoke",
        "train_gpt_oss_120b_5gpu_group64_smoke",
        "train_gpt_oss_120b_3gpu_group16_smoke",
        "train_gpt_oss_120b_3gpu_group16_h200_tuned_smoke",
    ]:
        function_start = script.index(f"def {function_name}")
        source = script[function_start:]
        next_function = source.find("\ndef ", 1)
        source = source[:next_function] if next_function != -1 else source
        assert "_assert_training_packages_available()" in source


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
