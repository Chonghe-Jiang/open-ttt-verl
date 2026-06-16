from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_slot_records(
    num_slots: int,
    library_path: str,
    task: str = "erdos_min_overlap",
    *,
    rollout_n: int = 1,
    puct_c: float = 1.0,
) -> list[dict[str, Any]]:
    return [
        {
            "data_source": "guidance_ttt",
            "prompt": [{"role": "user", "content": ""}],
            "reward_model": {"ground_truth": ""},
            "extra_info": {
                "task": task,
                "slot_id": f"slot_{slot_idx}",
                "uid": f"slot_{slot_idx}",
                "library_path": library_path,
                "archive_path": library_path,
                "rollout_n": int(rollout_n),
                "group_size": int(rollout_n),
                "puct_c": float(puct_c),
            },
        }
        for slot_idx in range(int(num_slots))
    ]


def write_slot_parquet(
    path: str | Path,
    *,
    num_slots: int,
    library_path: str,
    rollout_n: int = 1,
    puct_c: float = 1.0,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = build_slot_records(
        num_slots=num_slots,
        library_path=library_path,
        rollout_n=rollout_n,
        puct_c=puct_c,
    )
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
