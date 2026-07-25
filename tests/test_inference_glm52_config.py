from pathlib import Path

import yaml


CONFIG_PATH = Path("guidance_ttt/config/polyomino_cpu_openrouter_glm52_inference_puct.yaml")


def test_glm52_inference_config_is_cpu_only_openrouter_sampling():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    execution = config["llm"]["execution"]
    search = config["search"]

    assert config["run"]["num_steps"] == 50
    assert config["task"]["id"] == "polyomino_packing"
    assert search["discover_compat"] is True
    assert search["groups_per_batch"] == 8
    assert search["group_size"] == 16
    assert search["generation_concurrency"] == 128
    assert search["evaluation_concurrency"] == 128
    assert search["puct_c"] == 1.0
    assert search["puct_q_mode"] == "best_child"
    assert search["max_buffer_size"] == 1000
    assert search["topk_children"] == 2
    assert execution["provider"] == "openai_compatible"
    assert execution["model"] == "z-ai/glm-5.2"
    assert execution["base_url"] == "https://openrouter.ai/api/v1"
    assert execution["api_key_env"] == "OR_KEY"
    assert execution["reasoning"] == {"effort": "high", "exclude": False}
    assert execution["temperature"] == 1.0
    assert execution["max_tokens"] == 32768
    assert execution["concurrency"] == 128
    assert "guidance" not in config["llm"]
