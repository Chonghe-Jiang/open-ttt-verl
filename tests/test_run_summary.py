from pathlib import Path

import yaml

from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.run_summary import summarize_run, write_run_summary
from guidance_ttt.state import LibraryEntry
from guidance_ttt.tasks.erdos import create_root_node


def _entry(parent_id: str, *, timestep: int, raw_score: float | None, status: str = "valid") -> LibraryEntry:
    return LibraryEntry(
        id=f"entry-{timestep}-{raw_score}",
        parent_id=parent_id,
        problem_id="erdos",
        timestep=timestep,
        guidance=f"guidance {timestep}",
        execution_thinking=f"thinking {timestep}",
        solution="def run(seed=42, budget_s=1, **kwargs):\n    return ([0.5, 0.5], 0.5, 2)",
        verifier_reward=0.0 if raw_score is None else 1.0 / raw_score,
        verifier_raw_score=raw_score,
        verifier_status=status,
        verifier_message="ok" if status == "valid" else "failed",
        summary=f"summary {timestep}",
        reusable_idea=f"idea {timestep}",
        failure_mode=None if status == "valid" else status,
        metadata={
            "guidance_prompt": {"system": "guidance system", "user": f"guidance prompt {timestep}"},
            "execution_prompt": {"system": "execution system", "user": f"execution prompt {timestep}"},
            "execution_text": f"execution output {timestep}",
            "execution_model": "models/gpt-oss-20b",
            "execution_provider": "local",
        },
    )


def test_summarize_run_groups_entries_and_reports_min_c5_score(tmp_path):
    output_dir = tmp_path / "run"
    library = GuidanceLibrary(output_dir / "library.json", initial_nodes=[create_root_node()], rollout_n=2)
    root_id = next(node_id for node_id, node in library.snapshot()["nodes"].items() if node["parent_id"] is None)

    library.acquire_group("1:a")
    library.submit_child("1:a", _entry(root_id, timestep=1, raw_score=0.4))
    library.acquire_group("1:b")
    library.submit_child("1:b", _entry(root_id, timestep=1, raw_score=0.3))
    library.acquire_group("2:a")
    library.submit_child("2:a", _entry(root_id, timestep=2, raw_score=None, status="execution_error"))

    config_path = output_dir / "recipe.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"model_path": "models/gpt-oss-20b", "num_steps": 5},
                "llm": {"execution": {"provider": "local", "model": "models/gpt-oss-20b"}},
            }
        )
    )

    summary = summarize_run(output_dir, config_path=config_path)

    assert summary["params"]["recipe_config"]["run"]["num_steps"] == 5
    assert summary["params"]["recipe_config"]["llm"]["execution"]["provider"] == "local"
    assert summary["steps"][0]["timestep"] == 1
    assert summary["steps"][0]["min_c5_score"] == 0.3
    assert summary["steps"][1]["timestep"] == 2
    assert summary["steps"][1]["min_c5_score"] is None
    assert summary["steps"][0]["entries"][0]["prompts"]["execution"]["user"] == "execution prompt 1"
    assert summary["steps"][0]["entries"][0]["execution_output"] == "execution output 1"


def test_write_run_summary_writes_json_and_markdown(tmp_path):
    output_dir = tmp_path / "run"
    GuidanceLibrary(output_dir / "library.json", initial_nodes=[create_root_node()], rollout_n=1)

    paths = write_run_summary(output_dir)

    assert Path(paths["json"]).exists()
    markdown = Path(paths["markdown"]).read_text()
    assert "# Guidance TTT Run Summary" in markdown
    assert "No completed execution entries" in markdown


def test_write_run_summary_handles_mixed_int_and_string_config_keys(tmp_path):
    output_dir = tmp_path / "run"
    GuidanceLibrary(output_dir / "library.json", initial_nodes=[create_root_node()], rollout_n=1)
    config_path = output_dir / "recipe.yaml"
    mixed_key_config = {
        "llm": {
            "execution": {
                "model_kwargs": {
                    "max_memory": {
                        0: "90GiB",
                        1: "90GiB",
                        "cpu": "700GiB",
                    }
                }
            }
        }
    }
    config_path.write_text(yaml.safe_dump(mixed_key_config))
    (output_dir / "agent_loop.yaml").write_text(yaml.safe_dump([{"execution_llm": mixed_key_config["llm"]["execution"]}]))

    paths = write_run_summary(output_dir, config_path=config_path)

    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
