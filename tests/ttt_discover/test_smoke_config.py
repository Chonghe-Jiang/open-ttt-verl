# Copyright 2026 Chonghe Jiang and/or contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import yaml

from verl_ttt_discover.agent_loop import phase1_generation_budget
from verl_ttt_discover.main_erdos import _apply_recipe_overrides, _build_verl_overrides, _prepare_run, _split_overrides


def test_gpu_smoke_config_is_minimal_single_gpu_vllm_run():
    config_path = Path("verl_ttt_discover/config/erdos_smoke_gpu.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["prepare_only"] is False
    assert config["run"]["rollout_engine"] == "vllm"
    assert config["run"]["n_gpus_per_node"] == 1
    assert config["run"]["num_steps"] == 1
    assert config["run"]["ppo_mini_batch_size"] == 1
    assert config["run"]["use_remove_padding"] is False
    assert config["ttt"]["groups_per_batch"] == 1
    assert config["ttt"]["group_size"] == 2
    assert "actor_rollout_ref.model.lora_rank=32" in config["verl_overrides"]
    assert "actor_rollout_ref.model.external_lib=verl_ttt_discover.verl_ext" in config["verl_overrides"]
    assert "+actor_rollout_ref.model.override_config.attn_implementation=eager" in config["verl_overrides"]
    assert "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072" in config["verl_overrides"]
    assert "actor_rollout_ref.rollout.agent.num_workers=1" in config["verl_overrides"]
    assert "actor_rollout_ref.rollout.enforce_eager=False" in config["verl_overrides"]
    assert "actor_rollout_ref.rollout.max_model_len=9216" in config["verl_overrides"]


def test_verl_overrides_enable_rollout_importance_weights(tmp_path):
    config = {
        "run": {"model_path": "dummy-model"},
        "ttt": {"groups_per_batch": 1, "group_size": 2},
    }
    prepared = {
        "output_dir": tmp_path,
        "slot_parquet": tmp_path / "slots.parquet",
        "agent_loop_config": tmp_path / "agent_loop.yaml",
    }

    overrides = _build_verl_overrides(config, prepared, [])

    assert "algorithm.rollout_correction.rollout_is=token" in overrides
    assert "algorithm.rollout_correction.rollout_is_threshold=2.0" in overrides


def test_cli_overrides_split_recipe_fields_from_verl_fields():
    recipe_overrides, verl_overrides = _split_overrides(
        [
            "run.num_steps=1",
            "ttt.group_size=8",
            "actor_rollout_ref.rollout.max_num_seqs=64",
        ]
    )

    assert recipe_overrides == ["run.num_steps=1", "ttt.group_size=8"]
    assert verl_overrides == ["actor_rollout_ref.rollout.max_num_seqs=64"]


def test_recipe_overrides_update_preparation_config_before_verl_mapping(tmp_path):
    config = {
        "run": {"model_path": "dummy-model", "num_steps": 50, "save_freq": 5},
        "ttt": {"groups_per_batch": 8, "group_size": 64},
    }
    prepared = {
        "output_dir": tmp_path,
        "slot_parquet": tmp_path / "slots.parquet",
        "agent_loop_config": tmp_path / "agent_loop.yaml",
    }

    config = _apply_recipe_overrides(config, ["run.num_steps=1", "run.save_freq=-1", "ttt.group_size=8"])
    overrides = _build_verl_overrides(config, prepared, [])

    assert "trainer.total_training_steps=1" in overrides
    assert "trainer.save_freq=-1" in overrides
    assert "actor_rollout_ref.rollout.n=8" in overrides


def test_phase1_generation_budget_matches_official_prompt_plus_thinking_budget():
    assert phase1_generation_budget(prompt_len=2000, response_len=26000, phase1_max_tokens=26000) == 24000
    assert phase1_generation_budget(prompt_len=8192, response_len=26000, phase1_max_tokens=26000) == 17808
    assert phase1_generation_budget(prompt_len=30000, response_len=26000, phase1_max_tokens=26000) == 0
    assert phase1_generation_budget(prompt_len=2000, response_len=8192, phase1_max_tokens=None) == 8192


def test_prepare_run_writes_phase1_budget_to_agent_loop_config(tmp_path):
    config = {
        "run": {"output_dir": str(tmp_path), "num_initial_states": 1},
        "ttt": {
            "groups_per_batch": 1,
            "group_size": 2,
            "eval_timeout": 20,
            "phase1_max_tokens": 26000,
        },
    }

    prepared = _prepare_run(config)

    assert "phase1_max_tokens: 26000" in prepared["agent_loop_config"].read_text()


def test_verl_overrides_start_isolated_local_ray_by_default(tmp_path):
    config = {
        "run": {"model_path": "dummy-model"},
        "ttt": {"groups_per_batch": 1, "group_size": 2},
    }
    prepared = {
        "output_dir": tmp_path,
        "slot_parquet": tmp_path / "slots.parquet",
        "agent_loop_config": tmp_path / "agent_loop.yaml",
    }

    overrides = _build_verl_overrides(config, prepared, [])

    assert "+ray_kwargs.ray_init.address=local" in overrides
    suffix_override = next(
        item
        for item in overrides
        if item.startswith("+ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_ZMQ_SUFFIX=")
    )
    suffix = suffix_override.rsplit("=", 1)[1]
    assert suffix.startswith("ttt-")
    assert len(suffix) <= 16


def test_verl_overrides_allow_explicit_ray_address(tmp_path):
    config = {
        "run": {"model_path": "dummy-model", "ray_address": "auto"},
        "ttt": {"groups_per_batch": 1, "group_size": 2},
    }
    prepared = {
        "output_dir": tmp_path,
        "slot_parquet": tmp_path / "slots.parquet",
        "agent_loop_config": tmp_path / "agent_loop.yaml",
    }

    overrides = _build_verl_overrides(config, prepared, [])

    assert "+ray_kwargs.ray_init.address=auto" in overrides
    assert "+ray_kwargs.ray_init.address=local" not in overrides


def _load_config(name: str) -> dict:
    return yaml.safe_load(Path("verl_ttt_discover/config", name).read_text())


def test_public_configs_do_not_pin_chonghej_local_model_snapshots():
    for config_path in Path("verl_ttt_discover/config").glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text())
        model_path = config["run"]["model_path"]
        assert "chonghej" not in model_path, f"{config_path} pins a private local path"


def _assert_two_gpu_flash_config(config: dict) -> None:
    overrides = config["verl_overrides"]

    assert config["run"]["rollout_engine"] == "vllm"
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["tensor_model_parallel_size"] == 2
    assert config["run"]["use_remove_padding"] is True
    assert "actor_rollout_ref.model.external_lib=verl_ttt_discover.verl_ext" in overrides
    assert "trainer.use_legacy_worker_impl=disable" in overrides
    assert "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False" in overrides
    assert "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16" in overrides
    assert "actor_rollout_ref.rollout.enforce_eager=False" in overrides
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend=naive" in overrides
    assert "actor_rollout_ref.rollout.free_cache_engine=False" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=16384" in overrides


def test_two_gpu_smoke_flash_config_uses_qwen_and_small_batch():
    config = _load_config("erdos_2gpu_smoke_flash.yaml")

    _assert_two_gpu_flash_config(config)
    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["output_dir"] == "outputs/ttt_erdos/2gpu_smoke_flash"
    assert config["run"]["num_steps"] == 1
    assert config["run"]["ppo_mini_batch_size"] == 1
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 1024
    assert config["ttt"]["groups_per_batch"] == 1
    assert config["ttt"]["group_size"] == 2
    assert config["ttt"]["eval_timeout"] == 20
    assert "actor_rollout_ref.rollout.agent.num_workers=2" in config["verl_overrides"]


def test_two_gpu_16k_probe_flash_config_uses_qwen_and_16k_context():
    config = _load_config("erdos_2gpu_16k_probe_flash.yaml")

    _assert_two_gpu_flash_config(config)
    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["output_dir"] == "outputs/ttt_erdos/2gpu_16k_probe_flash"
    assert config["run"]["num_steps"] == 1
    assert config["run"]["ppo_mini_batch_size"] == 2
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 8192
    assert config["ttt"]["groups_per_batch"] == 2
    assert config["ttt"]["group_size"] == 4
    assert config["ttt"]["eval_timeout"] == 60
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=32768" in config["verl_overrides"]
    assert "actor_rollout_ref.rollout.agent.num_workers=8" in config["verl_overrides"]


def test_two_gpu_scale_flash_config_uses_gpt_oss_20b_and_larger_batch():
    config = _load_config("erdos_2gpu_scale_gptoss20b_flash.yaml")

    _assert_two_gpu_flash_config(config)
    assert config["run"]["model_path"] == "unsloth/gpt-oss-20b-BF16"
    assert config["run"]["output_dir"] == "outputs/ttt_erdos/2gpu_scale_gptoss20b_flash_g4_n16"
    assert config["run"]["num_steps"] == 10
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 8192
    assert config["run"]["ppo_mini_batch_size"] == 4
    assert config["ttt"]["groups_per_batch"] == 4
    assert config["ttt"]["group_size"] == 16
    assert config["ttt"]["eval_timeout"] == 300
    assert "actor_rollout_ref.rollout.agent.num_workers=16" in config["verl_overrides"]


def test_two_gpu_gptoss_bf16_smoke_config_is_blackwell_compatible_minimal_run():
    config = _load_config("erdos_2gpu_smoke_gptoss20b_bf16_flash.yaml")
    overrides = config["verl_overrides"]

    assert config["run"]["model_path"] == "unsloth/gpt-oss-20b-BF16"
    assert config["run"]["n_gpus_per_node"] == 2
    assert config["run"]["tensor_model_parallel_size"] == 2
    assert config["run"]["num_steps"] == 1
    assert config["run"]["ppo_mini_batch_size"] == 2
    assert config["run"]["max_prompt_length"] == 1920
    assert config["run"]["max_response_length"] == 128
    assert config["run"]["save_freq"] == -1
    assert config["ttt"]["groups_per_batch"] == 2
    assert config["ttt"]["group_size"] == 1
    assert "+actor_rollout_ref.model.override_config.attn_implementation=eager" in overrides
    assert "actor_rollout_ref.rollout.load_format=auto" in overrides
    assert "actor_rollout_ref.rollout.layered_summon=True" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=2048" in overrides
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=2048" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=1" in overrides


def test_four_gpu_b200_gptoss_bf16_official_config_matches_ttt_erdos_defaults():
    config = _load_config("erdos_4gpu_b200_gptoss20b_bf16_official.yaml")
    overrides = config["verl_overrides"]

    assert config["run"]["model_path"] == "unsloth/gpt-oss-20b-BF16"
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["run"]["tensor_model_parallel_size"] == 4
    assert config["run"]["num_initial_states"] == 8
    assert config["run"]["num_steps"] == 50
    assert config["run"]["learning_rate"] == 4.0e-5
    assert config["run"]["kl_loss_coef"] == 0.1
    assert config["run"]["ppo_mini_batch_size"] == 8
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 26000
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 64
    assert config["ttt"]["phase1_max_tokens"] == 26000
    assert config["ttt"]["eval_timeout"] == 1000
    assert "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2" in overrides
    assert "actor_rollout_ref.model.lora_rank=32" in overrides
    assert "actor_rollout_ref.model.lora_alpha=32" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" in overrides
    assert "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16" in overrides
    assert "actor_rollout_ref.rollout.load_format=auto" in overrides
    assert "actor_rollout_ref.rollout.layered_summon=True" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=32768" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=512" in overrides
    assert "actor_rollout_ref.rollout.agent.num_workers=64" in overrides


def test_four_gpu_b200_qwen3_8b_official_config_matches_ttt_erdos_defaults():
    config = _load_config("erdos_4gpu_b200_qwen3_8b_official.yaml")
    overrides = config["verl_overrides"]

    assert config["run"]["model_path"] == "Qwen/Qwen3-8B"
    assert config["run"]["output_dir"] == "outputs/ttt_erdos/4gpu_b200_qwen3_8b_official_g8_n64"
    assert config["run"]["experiment_name"] == "erdos_4gpu_b200_qwen3_8b_official_g8_n64"
    assert config["run"]["n_gpus_per_node"] == 4
    assert config["run"]["tensor_model_parallel_size"] == 4
    assert config["run"]["num_initial_states"] == 8
    assert config["run"]["num_steps"] == 50
    assert config["run"]["learning_rate"] == 4.0e-5
    assert config["run"]["ppo_mini_batch_size"] == 8
    assert config["run"]["max_prompt_length"] == 8192
    assert config["run"]["max_response_length"] == 26000
    assert config["ttt"]["groups_per_batch"] == 8
    assert config["ttt"]["group_size"] == 64
    assert config["ttt"]["phase1_max_tokens"] == 26000
    assert config["ttt"]["eval_timeout"] == 1000
    assert "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2" in overrides
    assert "actor_rollout_ref.model.lora_rank=32" in overrides
    assert "actor_rollout_ref.model.lora_alpha=32" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" in overrides
    assert "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16" in overrides
    assert "actor_rollout_ref.rollout.max_model_len=32768" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=512" in overrides
    assert "actor_rollout_ref.rollout.agent.num_workers=64" in overrides
