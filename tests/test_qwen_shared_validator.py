import json

import pytest

from scripts.validate_qwen_shared_smoke import validate


def _write_library(tmp_path, entries):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "library.json").write_text(json.dumps({"entries": entries}))
    return output_dir


def test_qwen_shared_validator_accepts_95_percent_parse_and_reasoning(tmp_path):
    entries = {}
    for index in range(20):
        complete = index < 19
        entries[str(index)] = {
            "timestep": 1,
            "solution": "int main(){}" if complete else "",
            "summary": "delta" if complete else "",
            "execution_thinking": "native reasoning" if complete else "",
            "verifier_status": "valid" if complete else "parse_error",
            "metadata": {"execution_model": "Qwen/Qwen3-8B"},
        }
    output_dir = _write_library(tmp_path, entries)

    result = validate(output_dir, expected_children=20)

    assert result["parse_fraction"] == 0.95
    assert result["native_reasoning_fraction"] == 0.95
    assert result["execution_errors"] == 0


def test_qwen_shared_validator_rejects_transport_execution_errors(tmp_path):
    output_dir = _write_library(
        tmp_path,
        {
            "child": {
                "timestep": 1,
                "solution": "",
                "summary": "",
                "execution_thinking": "",
                "verifier_status": "execution_error",
                "metadata": {"execution_model": "Qwen/Qwen3-8B"},
            }
        },
    )

    with pytest.raises(RuntimeError, match="execution_error entries=1"):
        validate(
            output_dir,
            expected_children=1,
            min_parse_fraction=0.0,
            min_reasoning_fraction=0.0,
        )
