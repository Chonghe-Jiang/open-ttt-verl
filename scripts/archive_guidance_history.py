#!/usr/bin/env python3
"""Inventory and stage meaningful Guidance-TTT experiment history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


STEP_RE = re.compile(
    rb'"(?:global_step|trainer_step|training_step|step|timestep|'
    rb'visible_timestep_exclusive)"\s*:\s*(\d+)',
    re.IGNORECASE,
)
ENTRIES_RE = re.compile(rb'"entries"\s*:\s*\{\s*"[^"]+"', re.IGNORECASE)
EXCLUDED_NAME_RE = re.compile(
    r"(?:^|[_-])(?:smoke|prepare|preflight|setup)(?:[_-]|$)",
    re.IGNORECASE,
)
FORMAL_TAGS = ("1day", "two_day", "50step", "discover")
JOB_ID_RE = re.compile(r"(?:^|[_-])(\d{6})(?:$|[_-])")
SOURCE_SCRIPT_RE = re.compile(
    r"guidance|discover|openrouter|glm|qwen|prepare|validate",
    re.IGNORECASE,
)
SMOKE_NAME_RE = re.compile(r"(?:^|[_-])smoke(?:[-_.]|$)", re.IGNORECASE)
TEXT_SUFFIXES = {
    ".err",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".out",
    ".py",
    ".sbatch",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
CREDENTIAL_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}", re.IGNORECASE),
    re.compile(
        rb"authorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._-]{16,}",
        re.IGNORECASE,
    ),
    re.compile(
        rb"(?:^|[\s\"'])"
        rb"(?:api_key|token)\s*[:=]\s*[\"']?"
        rb"(?!null\b|none\b|env\b|\$|\{)[A-Za-z0-9._-]{20,}",
        re.IGNORECASE,
    ),
)


class CredentialError(RuntimeError):
    pass


def scan_history(path: Path, chunk_size: int = 1024 * 1024) -> tuple[int | None, bool]:
    """Return the maximum recorded step and whether entries exist."""
    max_step: int | None = None
    has_entries = False
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            data = overlap + chunk
            for match in STEP_RE.finditer(data):
                step = int(match.group(1))
                max_step = step if max_step is None else max(max_step, step)
            if not has_entries and ENTRIES_RE.search(data):
                has_entries = True
            overlap = data[-256:]
    return max_step, has_entries


def experiment_family(name: str) -> str:
    lowered = name.lower()
    if "openrouter" in lowered or "glm52" in lowered:
        return "openrouter_glm52"
    if "gpt_oss_120b" in lowered or "gpt-oss-120b" in lowered:
        return "gpt_oss_120b"
    if "coder_next" in lowered or "coder-next" in lowered:
        return "qwen3_coder_next"
    if "entropic_best_child" in lowered:
        return "entropic_best_child"
    if "qwen36" in lowered or "qwen3.6" in lowered:
        return "qwen36"
    if "evolvent" in lowered or "gpt54" in lowered:
        return "evolvent_gpt54"
    return "other"


def classify_run(run_dir: Path) -> dict[str, Any]:
    history_files = sorted(run_dir.glob("library*.json"))
    max_step: int | None = None
    has_entries = False
    history_bytes = 0
    for path in history_files:
        history_bytes += path.stat().st_size
        file_step, file_has_entries = scan_history(path)
        if file_step is not None:
            max_step = file_step if max_step is None else max(max_step, file_step)
        has_entries = has_entries or file_has_entries

    name = run_dir.name
    formal_tags = [tag for tag in FORMAL_TAGS if tag in name.lower()]
    if EXCLUDED_NAME_RE.search(name):
        included = False
        reason = "excluded_run_name"
    elif max_step is not None and max_step >= 10:
        included = True
        reason = "completed_at_least_10_steps"
    elif formal_tags and has_entries:
        included = True
        reason = "formal_run_with_history"
    else:
        included = False
        reason = "below_step_threshold"

    return {
        "name": name,
        "source_path": str(run_dir),
        "family": experiment_family(name),
        "included": included,
        "reason": reason,
        "max_step": max_step,
        "formal_tags": formal_tags,
        "job_ids": sorted(set(JOB_ID_RE.findall(name))),
        "history_file_count": len(history_files),
        "history_bytes": history_bytes,
        "substantive_history": has_entries,
    }


def inventory(outputs: Path) -> dict[str, Any]:
    runs_root = outputs / "guidance_ttt"
    runs = []
    if runs_root.is_dir():
        runs = [
            classify_run(path)
            for path in sorted(runs_root.iterdir(), key=lambda item: item.name)
            if path.is_dir()
        ]
    return {
        "selection_policy": {
            "minimum_step": 10,
            "formal_tags": list(FORMAL_TAGS),
            "excluded_name_tokens": ["smoke", "prepare", "preflight", "setup"],
        },
        "runs": runs,
    }


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _scan_credentials(path: Path, chunk_size: int = 1024 * 1024) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            data = overlap + chunk
            if any(pattern.search(data) for pattern in CREDENTIAL_PATTERNS):
                raise CredentialError(f"credential-like content detected in {path}")
            overlap = data[-512:]


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entry(
    *,
    source: Path,
    destination: Path,
    staging: Path,
) -> dict[str, Any]:
    _scan_credentials(destination)
    return {
        "path": destination.relative_to(staging).as_posix(),
        "source_path": str(source),
        "size": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _write_dataset_documents(staging: Path, result: dict[str, Any]) -> None:
    included = [run for run in result["runs"] if run["included"]]
    excluded = [run for run in result["runs"] if not run["included"]]
    (staging / "reports").mkdir(parents=True, exist_ok=True)
    with (staging / "reports" / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for run in result["runs"]:
            handle.write(json.dumps(run, sort_keys=True) + "\n")
    (staging / "reports" / "overview.md").write_text(
        "\n".join(
            [
                "# Guidance-TTT History Overview",
                "",
                f"- Included runs: {len(included)}",
                f"- Excluded runs: {len(excluded)}",
                f"- Included history bytes: {sum(run['history_bytes'] for run in included)}",
                "",
                "See `runs.jsonl` for every selection decision.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (staging / "README.md").write_text(
        """---
pretty_name: Guidance-TTT Experiment History
tags:
- test-time-training
- guidance
- experiment-tracking
---

# Guidance-TTT Experiment History

This private dataset preserves meaningful Guidance-TTT experiment histories.
Runs are included when they completed at least 10 steps or are substantive
formal 1day, two-day, 50-step, or Discover runs. Smoke, setup, prepare,
preflight, checkpoint, cache, lock, temporary, model, and credential files are
excluded.

See `reports/overview.md`, `reports/runs.jsonl`, and `manifest.json`.
""",
        encoding="utf-8",
    )


def build_archive(
    *,
    repo_root: Path,
    outputs: Path,
    staging: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    staging = staging.resolve()
    if staging in {Path("/"), repo_root}:
        raise ValueError(f"unsafe staging path: {staging}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    result = inventory(outputs)
    payload_sources: list[tuple[Path, Path]] = []
    slurm_root = outputs / "slurm"
    slurm_files = (
        [path for path in sorted(slurm_root.iterdir()) if path.is_file()]
        if slurm_root.is_dir()
        else []
    )
    for run in result["runs"]:
        if not run["included"]:
            continue
        source_dir = Path(run["source_path"])
        run_root = staging / "experiments" / run["family"] / run["name"]
        for source in sorted(source_dir.glob("library*.json")):
            payload_sources.append((source, run_root / "history" / source.name))
        for name in ("agent_loop.yaml", "agent_loop.yml"):
            source = source_dir / name
            if source.is_file():
                payload_sources.append((source, run_root / "config" / source.name))
        for name in ("ttt_slots.parquet", "run_summary.json"):
            source = source_dir / name
            if source.is_file():
                payload_sources.append((source, run_root / "metadata" / source.name))
        for source in slurm_files:
            if EXCLUDED_NAME_RE.search(source.name):
                continue
            if any(job_id in source.name for job_id in run["job_ids"]):
                payload_sources.append((source, run_root / "logs" / source.name))

    config_root = repo_root / "guidance_ttt" / "config"
    if config_root.is_dir():
        for pattern in ("*.yaml", "*.yml"):
            for source in sorted(config_root.glob(pattern)):
                if SMOKE_NAME_RE.search(source.name):
                    continue
                payload_sources.append(
                    (source, staging / "source" / "configs" / source.name)
                )
    scripts_root = repo_root / "scripts"
    if scripts_root.is_dir():
        for source in sorted(scripts_root.iterdir()):
            if (
                source.is_file()
                and source.suffix.lower() in {".py", ".sbatch", ".sh"}
                and SOURCE_SCRIPT_RE.search(source.name)
                and not SMOKE_NAME_RE.search(source.name)
            ):
                payload_sources.append(
                    (source, staging / "source" / "scripts" / source.name)
                )

    manifest_files = []
    for source, destination in payload_sources:
        _copy_or_link(source, destination)
        manifest_files.append(
            _manifest_entry(
                source=source,
                destination=destination,
                staging=staging,
            )
        )

    _write_dataset_documents(staging, result)
    for destination in (
        staging / "README.md",
        staging / "reports" / "overview.md",
        staging / "reports" / "runs.jsonl",
    ):
        manifest_files.append(
            _manifest_entry(
                source=destination,
                destination=destination,
                staging=staging,
            )
        )
    manifest = {
        "selection_policy": result["selection_policy"],
        "run_count": sum(run["included"] for run in result["runs"]),
        "file_count": len(manifest_files),
        "total_bytes": sum(item["size"] for item in manifest_files),
        "files": sorted(manifest_files, key=lambda item: item["path"]),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"inventory": result, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = inventory(args.outputs)
    if args.inventory_only:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    try:
        archive = build_archive(
            repo_root=args.repo_root,
            outputs=args.outputs,
            staging=args.staging,
        )
    except CredentialError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    summary = {
        "included_runs": archive["manifest"]["run_count"],
        "files": archive["manifest"]["file_count"],
        "total_bytes": archive["manifest"]["total_bytes"],
        "staging": str(args.staging.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
