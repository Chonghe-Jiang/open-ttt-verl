from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_slot_records(num_slots: int, archive_path: str, task: str = "erdos_min_overlap") -> list[dict[str, Any]]:
    return [
        {
            "data_source": "ttt_erdos",
            "prompt": [{"role": "user", "content": ""}],
            "reward_model": {"ground_truth": ""},
            "extra_info": {
                "task": task,
                "slot_id": f"slot_{slot_idx}",
                "archive_path": archive_path,
            },
        }
        for slot_idx in range(int(num_slots))
    ]


def write_slot_parquet(path: str | Path, *, num_slots: int, archive_path: str) -> Path:
    """Write verl-compatible static slot data.

    Requires pandas/pyarrow at runtime. The JSON fallback is useful for quick inspection but
    verl training should consume the parquet path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = build_slot_records(num_slots=num_slots, archive_path=archive_path)
    try:
        import pandas as pd

        pd.DataFrame(records).to_parquet(output_path)
    except ModuleNotFoundError as exc:
        fallback_path = output_path.with_suffix(".jsonl")
        fallback_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        raise RuntimeError(
            f"Writing parquet requires pandas and pyarrow. Wrote inspectable fallback to {fallback_path}."
        ) from exc
    return output_path
