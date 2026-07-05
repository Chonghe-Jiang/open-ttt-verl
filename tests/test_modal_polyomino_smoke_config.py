from pathlib import Path
import importlib.util

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


def test_modal_polyomino_train_smoke_resets_output_dir(tmp_path):
    module = _load_modal_smoke_module()
    output_dir = tmp_path / "polyomino_modal_h200_4gpu_smoke"
    stale_file = output_dir / "library.json"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("{}")

    module._reset_output_dir(str(output_dir))

    assert output_dir.exists()
    assert not stale_file.exists()
