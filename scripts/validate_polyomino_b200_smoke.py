from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(
    output_dir: Path,
    expected_groups: int = 8,
    expected_group_size: int = 16,
    expected_prompt_mode: str = "code_delta",
) -> dict[str, int | str]:
    library_path = output_dir / "library.json"
    if not library_path.is_file():
        raise RuntimeError(f"missing smoke library: {library_path}")
    payload = json.loads(library_path.read_text())
    config = payload.get("config") or {}
    entries = payload.get("entries") or {}
    groups = payload.get("groups") or {}
    child_entries = [
        entry
        for entry in entries.values()
        if isinstance(entry, dict) and int(entry.get("timestep") or 0) > 0
    ]
    finalized_groups = [group for group in groups.values() if isinstance(group, dict) and group.get("finalized")]

    errors: list[str] = []
    expected_children = expected_groups * expected_group_size
    if int(config.get("rollout_n", 0)) != expected_group_size:
        errors.append(f"rollout_n={config.get('rollout_n')!r}, expected {expected_group_size}")
    if len(groups) != expected_groups or len(finalized_groups) != expected_groups:
        errors.append(
            f"groups/finalized={len(groups)}/{len(finalized_groups)}, "
            f"expected {expected_groups}/{expected_groups}"
        )
    if len(child_entries) != expected_children:
        errors.append(f"child entries={len(child_entries)}, expected {expected_children}")
    for group_id, group in groups.items():
        if not isinstance(group, dict):
            errors.append(f"group {group_id!r} is malformed")
            continue
        if (
            int(group.get("submitted", 0)) != expected_group_size
            or len(group.get("children") or []) != expected_group_size
        ):
            errors.append(
                f"group {group_id!r} does not contain {expected_group_size} completed rollouts"
            )
    prompt_mode_count = 0
    valid_count = 0
    length_stop_count = 0
    for entry in child_entries:
        metadata = entry.get("metadata") or {}
        prompt_mode_count += metadata.get("prompt_mode") == expected_prompt_mode
        valid_count += entry.get("verifier_status") == "valid"
        length_stop_count += str(metadata.get("guidance_stop_reason") or "").lower() == "length"
    if prompt_mode_count != expected_children:
        errors.append(
            f"{expected_prompt_mode} entries={prompt_mode_count}, expected {expected_children}"
        )
    if valid_count == 0:
        errors.append("no rollout passed FrontierCS verification")
    if length_stop_count:
        errors.append(f"{length_stop_count} guidance rollouts stopped at the token limit")
    if int(payload.get("puct_T", 0)) != expected_groups:
        errors.append(f"puct_T={payload.get('puct_T')!r}, expected {expected_groups}")
    if errors:
        raise RuntimeError(
            f"B200 {expected_groups}x{expected_group_size} smoke acceptance failed: "
            + "; ".join(errors)
        )
    return {
        "output_dir": str(output_dir),
        "groups": len(groups),
        "children": len(child_entries),
        "valid_children": valid_count,
        "puct_T": int(payload["puct_T"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-groups", type=int, default=8)
    parser.add_argument("--expected-group-size", type=int, default=16)
    parser.add_argument("--expected-prompt-mode", default="code_delta")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.output_dir,
                args.expected_groups,
                args.expected_group_size,
                args.expected_prompt_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
