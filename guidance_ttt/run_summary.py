from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def summarize_run(output_dir: str | Path, *, config_path: str | Path | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    library_path = output_dir / "library.json"
    if not library_path.exists():
        raise FileNotFoundError(f"Run library not found: {library_path}")

    library = json.loads(library_path.read_text())
    params = {
        "output_dir": str(output_dir),
        "library_path": str(library_path),
        "library_config": library.get("config", {}),
        "recipe_config": _load_yaml(config_path) if config_path else {},
        "agent_loop_config": _load_yaml(output_dir / "agent_loop.yaml") if (output_dir / "agent_loop.yaml").exists() else {},
    }

    entries_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in library.get("entries", {}).values():
        entries_by_step[int(entry.get("timestep", 0))].append(_summarize_entry(entry))

    steps = []
    for timestep in sorted(entries_by_step):
        entries = sorted(entries_by_step[timestep], key=lambda item: item["entry_id"])
        raw_scores = [entry["c5_score"] for entry in entries if entry["c5_score"] is not None]
        steps.append(
            {
                "timestep": timestep,
                "min_c5_score": min(raw_scores) if raw_scores else None,
                "entries": entries,
            }
        )

    return {
        "params": params,
        "steps": steps,
        "best": _summarize_best(library),
    }


def write_run_summary(
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    json_name: str = "run_summary.json",
    markdown_name: str = "run_summary.md",
) -> dict[str, str]:
    output_dir = Path(output_dir).expanduser().resolve()
    summary = summarize_run(output_dir, config_path=config_path)
    json_path = output_dir / json_name
    markdown_path = output_dir / markdown_name
    json_path.write_text(json.dumps(_stringify_mapping_keys(summary), indent=2, sort_keys=True))
    markdown_path.write_text(_to_markdown(summary))
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _summarize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(entry.get("metadata") or {})
    return {
        "entry_id": entry.get("id"),
        "parent_id": entry.get("parent_id"),
        "status": entry.get("verifier_status"),
        "message": entry.get("verifier_message"),
        "reward": entry.get("verifier_reward"),
        "c5_score": entry.get("verifier_raw_score"),
        "guidance": entry.get("guidance", ""),
        "execution_thinking": entry.get("execution_thinking", ""),
        "summary": entry.get("summary", ""),
        "solution": entry.get("solution", ""),
        "execution_provider": metadata.get("execution_provider"),
        "execution_model": metadata.get("execution_model"),
        "prompts": {
            "guidance": metadata.get("guidance_prompt", {}),
            "execution": metadata.get("execution_prompt", {}),
        },
        "execution_output": metadata.get("execution_text", ""),
    }


def _summarize_best(library: dict[str, Any]) -> dict[str, Any] | None:
    best_node_id = library.get("best_node_id")
    best_node = (library.get("nodes") or {}).get(best_node_id)
    if not best_node:
        return None
    entry = (library.get("entries") or {}).get(best_node.get("entry_id"))
    return {
        "node_id": best_node_id,
        "value": best_node.get("value"),
        "c5_score": best_node.get("raw_score"),
        "entry_id": best_node.get("entry_id"),
        "status": entry.get("verifier_status") if entry else best_node.get("metadata", {}).get("verifier_status"),
    }


def _load_yaml(path: str | Path | None) -> Any:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _stringify_mapping_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Guidance TTT Run Summary", ""]
    params = summary["params"]
    lines.extend(
        [
            f"- Output dir: `{params['output_dir']}`",
            f"- Library: `{params['library_path']}`",
            f"- Library config: `{json.dumps(_stringify_mapping_keys(params.get('library_config', {})), sort_keys=True)}`",
            "",
            "## Per-Step C5 Scores",
            "",
        ]
    )
    if not summary["steps"]:
        lines.append("No completed execution entries were found.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Step | Min C5 score | Entries |")
    lines.append("| --- | ---: | ---: |")
    for step in summary["steps"]:
        min_score = "n/a" if step["min_c5_score"] is None else f"{step['min_c5_score']:.12g}"
        lines.append(f"| {step['timestep']} | {min_score} | {len(step['entries'])} |")
    lines.append("")
    lines.append("## Prompts And Outputs")
    lines.append("")
    for step in summary["steps"]:
        lines.append(f"### Step {step['timestep']}")
        for entry in step["entries"]:
            lines.extend(
                [
                    "",
                    f"#### Entry `{entry['entry_id']}`",
                    "",
                    f"- Status: `{entry['status']}`",
                    f"- C5 score: `{entry['c5_score']}`",
                    f"- Reward: `{entry['reward']}`",
                    f"- Execution model: `{entry['execution_model']}`",
                    "",
                    "**Guidance Prompt**",
                    "",
                    "```text",
                    str(entry["prompts"].get("guidance", {}).get("user", "")),
                    "```",
                    "",
                    "**Execution Prompt**",
                    "",
                    "```text",
                    str(entry["prompts"].get("execution", {}).get("user", "")),
                    "```",
                    "",
                    "**Execution Output**",
                    "",
                    "```text",
                    str(entry["execution_output"]),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a Guidance TTT run.")
    parser.add_argument("output_dir")
    parser.add_argument("--config", dest="config_path")
    args = parser.parse_args()
    paths = write_run_summary(args.output_dir, config_path=args.config_path)
    print(f"Wrote JSON summary: {paths['json']}")
    print(f"Wrote Markdown summary: {paths['markdown']}")


if __name__ == "__main__":
    main()
