from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(
    output_dir: Path,
    *,
    expected_children: int = 256,
    min_parse_fraction: float = 0.95,
    min_reasoning_fraction: float = 0.95,
) -> dict[str, int | float | str]:
    library_path = output_dir / "library.json"
    if not library_path.is_file():
        raise RuntimeError(f"missing Qwen shared smoke library: {library_path}")
    payload = json.loads(library_path.read_text())
    entries = payload.get("entries") or {}
    children = [
        entry
        for entry in entries.values()
        if isinstance(entry, dict) and int(entry.get("timestep") or 0) > 0
    ]

    model_count = 0
    parsed_count = 0
    reasoning_count = 0
    execution_error_count = 0
    for entry in children:
        metadata = entry.get("metadata") or {}
        model_count += metadata.get("execution_model") == "Qwen/Qwen3-8B"
        parsed_count += bool(str(entry.get("solution") or "").strip()) and bool(
            str(entry.get("summary") or "").strip()
        )
        reasoning_count += bool(str(entry.get("execution_thinking") or "").strip())
        execution_error_count += entry.get("verifier_status") == "execution_error"

    total = len(children)
    parse_fraction = parsed_count / total if total else 0.0
    reasoning_fraction = reasoning_count / total if total else 0.0
    errors: list[str] = []
    if total != expected_children:
        errors.append(f"child entries={total}, expected {expected_children}")
    if model_count != total:
        errors.append(f"Qwen execution model entries={model_count}, expected {total}")
    if execution_error_count:
        errors.append(f"execution_error entries={execution_error_count}, expected 0")
    if parse_fraction < min_parse_fraction:
        errors.append(
            f"parsed solution+summary fraction={parse_fraction:.3f}, expected >= {min_parse_fraction:.3f}"
        )
    if reasoning_fraction < min_reasoning_fraction:
        errors.append(
            f"native reasoning fraction={reasoning_fraction:.3f}, expected >= {min_reasoning_fraction:.3f}"
        )
    if errors:
        raise RuntimeError("Qwen shared smoke acceptance failed: " + "; ".join(errors))
    return {
        "output_dir": str(output_dir),
        "children": total,
        "parsed_children": parsed_count,
        "parse_fraction": parse_fraction,
        "native_reasoning_children": reasoning_count,
        "native_reasoning_fraction": reasoning_fraction,
        "execution_errors": execution_error_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-children", type=int, default=256)
    parser.add_argument("--min-parse-fraction", type=float, default=0.95)
    parser.add_argument("--min-reasoning-fraction", type=float, default=0.95)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.output_dir,
                expected_children=args.expected_children,
                min_parse_fraction=args.min_parse_fraction,
                min_reasoning_fraction=args.min_reasoning_fraction,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
