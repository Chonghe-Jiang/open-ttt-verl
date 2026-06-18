import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openevolve"))

from openevolve.config import Config
from openevolve.llm.openai import OpenAILLM


class TestContextWindowGuard(unittest.TestCase):
    def _model_cfg(self):
        config = Config.from_dict(
            {
                "llm": {
                    "api_base": "http://127.0.0.1:18004/v1",
                    "api_key": "EMPTY",
                    "max_tokens": 90,
                    "max_context_tokens": 100,
                    "context_guard_buffer": 5,
                    "models": [{"name": "Qwen/Qwen3-8B", "weight": 1.0}],
                }
            }
        )
        config.llm.update_model_params({"system_message": "system"})
        return config.llm.models[0]

    def test_config_propagates_context_guard_to_model(self):
        model_cfg = self._model_cfg()

        self.assertEqual(model_cfg.max_context_tokens, 100)
        self.assertEqual(model_cfg.context_guard_buffer, 5)

    def test_context_guard_reduces_max_tokens_before_api_call(self):
        model_cfg = self._model_cfg()

        with patch("openai.OpenAI"):
            llm = OpenAILLM(model_cfg)

        async def fake_call_api(params):
            return params

        llm._call_api = fake_call_api
        llm._estimate_message_tokens = Mock(return_value=80)

        params = asyncio.run(
            llm.generate_with_context(
                system_message="system",
                messages=[{"role": "user", "content": "prompt"}],
            )
        )

        self.assertEqual(params["max_tokens"], 15)


if __name__ == "__main__":
    unittest.main()
