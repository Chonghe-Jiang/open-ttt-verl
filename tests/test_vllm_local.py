import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from openevolve.config import Config

from scripts.check_vllm_server import check_vllm_server


def test_qwen3_8b_vllm_config_loads_for_local_endpoint():
    config = Config.from_yaml("configs/erdos_qwen3_8b_vllm.yaml")

    assert config.llm.models[0].name == "Qwen/Qwen3-8B"
    assert config.llm.models[0].api_base == "http://127.0.0.1:8000/v1"
    assert config.llm.models[0].api_key == "EMPTY"
    assert config.evaluator.parallel_evaluations == 1


def test_check_vllm_server_accepts_available_model():
    payload = json.dumps({"data": [{"id": "Qwen/Qwen3-8B"}]}).encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return payload

    with patch("scripts.check_vllm_server.urllib.request.urlopen", return_value=Response()):
        result = check_vllm_server("http://127.0.0.1:8000/v1", "Qwen/Qwen3-8B", timeout_s=1)

    assert result.ok is True
    assert result.models == ["Qwen/Qwen3-8B"]


def test_check_vllm_server_reports_missing_server():
    with patch("scripts.check_vllm_server.urllib.request.urlopen", side_effect=URLError("refused")):
        result = check_vllm_server("http://127.0.0.1:8000/v1", "Qwen/Qwen3-8B", timeout_s=1)

    assert result.ok is False
    assert "Could not reach vLLM server" in result.message


def test_run_script_documents_local_qwen_option():
    script = Path("scripts/run_openevolve.sh").read_text()

    assert "--local-qwen-vllm" in script
    assert "scripts/check_vllm_server.py" in script
    assert "--api-base" in script
    assert "--primary-model" in script
