from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "archive_guidance_history.py"


def _write_run(
    outputs: Path,
    name: str,
    *,
    step: int | None,
    entries: bool = True,
) -> None:
    run_dir = outputs / "guidance_ttt" / name
    run_dir.mkdir(parents=True)
    payload: dict[str, object] = {"entries": {}}
    if entries:
        metadata = {} if step is None else {"step": step}
        payload["entries"] = {"node-1": {"metadata": metadata}}
    (run_dir / "library.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "agent_loop.yaml").write_text("rollout_n: 16\n", encoding="utf-8")


def _inventory(tmp_path: Path) -> dict[str, object]:
    outputs = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(tmp_path),
            "--outputs",
            str(outputs),
            "--staging",
            str(tmp_path / "staging"),
            "--inventory-only",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _build(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    outputs = tmp_path / "outputs"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(tmp_path),
            "--outputs",
            str(outputs),
            "--staging",
            str(tmp_path / "staging"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def test_inventory_applies_approved_selection_policy(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    _write_run(outputs, "ordinary_ten_steps", step=10)
    _write_run(outputs, "ordinary_nine_steps", step=9)
    _write_run(outputs, "formal_discover_1day", step=3)
    _write_run(outputs, "formal_two_day_empty", step=None, entries=False)
    _write_run(outputs, "long_smoke", step=50)
    _write_run(outputs, "discover_prepare", step=50)
    _write_run(outputs, "discover_preflight", step=50)
    _write_run(outputs, "discover_setup", step=50)

    inventory = _inventory(tmp_path)
    decisions = {run["name"]: run for run in inventory["runs"]}

    assert decisions["ordinary_ten_steps"]["included"] is True
    assert decisions["ordinary_ten_steps"]["reason"] == "completed_at_least_10_steps"
    assert decisions["ordinary_nine_steps"]["included"] is False
    assert decisions["formal_discover_1day"]["included"] is True
    assert decisions["formal_discover_1day"]["reason"] == "formal_run_with_history"
    assert decisions["formal_two_day_empty"]["included"] is False
    assert decisions["long_smoke"]["included"] is False
    assert decisions["discover_prepare"]["included"] is False
    assert decisions["discover_preflight"]["included"] is False
    assert decisions["discover_setup"]["included"] is False


def test_inventory_detects_steps_across_small_scan_chunks(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_dir = outputs / "guidance_ttt" / "chunk_boundary_history"
    run_dir.mkdir(parents=True)
    (run_dir / "library.json").write_text(
        '{"entries":{"n":{"padding":"xxxxxxxxxxxxxxx","metadata":{"global_step":12}}}}',
        encoding="utf-8",
    )

    inventory = _inventory(tmp_path)
    decision = inventory["runs"][0]

    assert decision["max_step"] == 12
    assert decision["included"] is True


def test_inventory_detects_legacy_timestep_fields(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_dir = outputs / "guidance_ttt" / "legacy_history"
    run_dir.mkdir(parents=True)
    (run_dir / "library.json").write_text(
        (
            '{"entries":{"n":{"metadata":{"timestep":11,'
            '"visible_timestep_exclusive":12}}}}'
        ),
        encoding="utf-8",
    )

    inventory = _inventory(tmp_path)
    decision = inventory["runs"][0]

    assert decision["max_step"] == 12
    assert decision["included"] is True


def test_inventory_assigns_stable_experiment_families(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    _write_run(outputs, "qwen3_8b_openrouter_glm52_discover_1day", step=2)
    _write_run(outputs, "qwen36_27b_gpt_oss_120b_two_day", step=12)
    _write_run(outputs, "qwen3_coder_next_50step", step=11)
    _write_run(outputs, "entropic_best_child_14b", step=10)

    inventory = _inventory(tmp_path)
    families = {run["name"]: run["family"] for run in inventory["runs"]}

    assert families["qwen3_8b_openrouter_glm52_discover_1day"] == "openrouter_glm52"
    assert families["qwen36_27b_gpt_oss_120b_two_day"] == "gpt_oss_120b"
    assert families["qwen3_coder_next_50step"] == "qwen3_coder_next"
    assert families["entropic_best_child_14b"] == "entropic_best_child"


def test_build_stages_history_metadata_and_manifest_without_checkpoints(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_name = "qwen3_8b_openrouter_glm52_discover_1day"
    _write_run(outputs, run_name, step=12)
    run_dir = outputs / "guidance_ttt" / run_name
    (run_dir / "library.before_resume.json").write_text(
        '{"entries":{"node-2":{"metadata":{"step":10}}}}',
        encoding="utf-8",
    )
    (run_dir / "ttt_slots.parquet").write_bytes(b"PAR1 metadata")
    (run_dir / "run_summary.json").write_text('{"best_score": 85.24}\n')
    (run_dir / "library.json.lock").write_text("")
    (run_dir / ".library.json.deadbeef.tmp").write_text("temporary")
    checkpoints = run_dir / "checkpoints" / "global_step_12" / "actor"
    checkpoints.mkdir(parents=True)
    (checkpoints / "model.pt").write_bytes(b"model weights")

    result = _build(tmp_path)

    assert result.returncode == 0, result.stderr
    staging = tmp_path / "staging"
    archived = staging / "experiments" / "openrouter_glm52" / run_name
    assert (archived / "history" / "library.json").is_file()
    assert (archived / "history" / "library.before_resume.json").is_file()
    assert (archived / "config" / "agent_loop.yaml").is_file()
    assert (archived / "metadata" / "ttt_slots.parquet").is_file()
    assert (archived / "metadata" / "run_summary.json").is_file()
    assert not any(path.name == "model.pt" for path in staging.rglob("*"))
    assert not any(path.name.endswith(".lock") for path in staging.rglob("*"))
    assert not any(path.name.endswith(".tmp") for path in staging.rglob("*"))

    manifest = json.loads((staging / "manifest.json").read_text())
    manifest_by_path = {entry["path"]: entry for entry in manifest["files"]}
    library_path = (
        f"experiments/openrouter_glm52/{run_name}/history/library.json"
    )
    library_bytes = (archived / "history" / "library.json").read_bytes()
    assert manifest_by_path[library_path]["size"] == len(library_bytes)
    assert manifest_by_path[library_path]["sha256"] == hashlib.sha256(
        library_bytes
    ).hexdigest()
    assert (staging / "README.md").is_file()
    assert (staging / "reports" / "runs.jsonl").is_file()
    assert (staging / "reports" / "overview.md").is_file()


def test_build_excludes_runs_rejected_by_selection(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    _write_run(outputs, "long_smoke", step=50)
    _write_run(outputs, "ordinary_nine_steps", step=9)
    _write_run(outputs, "formal_two_day", step=3)

    result = _build(tmp_path)

    assert result.returncode == 0, result.stderr
    staged_names = {
        path.name
        for path in (tmp_path / "staging" / "experiments").glob("*/*")
    }
    assert staged_names == {"formal_two_day"}


def test_build_rejects_credentials_in_staged_text(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    _write_run(outputs, "formal_discover_1day", step=3)
    config = (
        outputs
        / "guidance_ttt"
        / "formal_discover_1day"
        / "agent_loop.yaml"
    )
    config.write_text(
        "api_key: hf_abcdefghijklmnopqrstuvwxyz123456\n",
        encoding="utf-8",
    )

    result = _build(tmp_path)

    assert result.returncode != 0
    assert "credential" in result.stderr.lower()


def test_build_associates_job_logs_and_archives_reproduction_sources(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_name = "qwen3_8b_gpt_oss_120b_discover_1day_123456"
    _write_run(outputs, run_name, step=12)

    slurm = outputs / "slurm"
    slurm.mkdir()
    (slurm / "guidance-run-123456.out").write_text("completed\n")
    (slurm / "guidance-run-123456.err").write_text("")
    (slurm / "gojudge-123456.log").write_text("judge ready\n")
    (slurm / "unrelated-654321.out").write_text("unrelated\n")
    (slurm / "old-smoke-123456.log").write_text("smoke\n")

    configs = tmp_path / "guidance_ttt" / "config"
    configs.mkdir(parents=True)
    (configs / "production_recipe.yaml").write_text("trainer:\n  steps: 50\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "slurm_guidance_production.sbatch").write_text("#!/bin/bash\n")
    (scripts / "unrelated_helper.txt").write_text("not archived\n")

    result = _build(tmp_path)

    assert result.returncode == 0, result.stderr
    staging = tmp_path / "staging"
    run_logs = staging / "experiments" / "gpt_oss_120b" / run_name / "logs"
    assert (run_logs / "guidance-run-123456.out").is_file()
    assert (run_logs / "guidance-run-123456.err").is_file()
    assert (run_logs / "gojudge-123456.log").is_file()
    assert not (run_logs / "unrelated-654321.out").exists()
    assert not (run_logs / "old-smoke-123456.log").exists()
    assert (
        staging / "source" / "configs" / "production_recipe.yaml"
    ).is_file()
    assert (
        staging / "source" / "scripts" / "slurm_guidance_production.sbatch"
    ).is_file()
    assert not (
        staging / "source" / "scripts" / "unrelated_helper.txt"
    ).exists()

    runs = [
        json.loads(line)
        for line in (staging / "reports" / "runs.jsonl").read_text().splitlines()
    ]
    decision = next(run for run in runs if run["name"] == run_name)
    assert decision["job_ids"] == ["123456"]
