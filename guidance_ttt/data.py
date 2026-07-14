from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guidance_ttt.puct import PUCT_Q_BLEND, normalize_puct_q_mode


def _slot_task_config(task: str, task_config: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": task}
    if task_config is not None:
        payload.update(json.loads(json.dumps(task_config)))
    payload["id"] = str(payload.get("id") or task)
    return payload


def build_slot_records(
    num_slots: int,
    library_path: str,
    task: str = "erdos_min_overlap",
    *,
    task_config: dict[str, Any] | None = None,
    rollout_n: int = 1,
    puct_c: float = 1.0,
    puct_q_mode: str = PUCT_Q_BLEND,
    max_buffer_size: int = 1000,
    topk_children: int = 2,
) -> list[dict[str, Any]]:
    task_config_payload = _slot_task_config(task, task_config)
    normalized_puct_q_mode = normalize_puct_q_mode(puct_q_mode)
    records = [
        {
            "data_source": "guidance_ttt",
            "prompt": [{"role": "user", "content": ""}],
            "reward_model": {"ground_truth": ""},
            "extra_info": {
                "task": task,
                "task_config": json.loads(json.dumps(task_config_payload)),
                "slot_id": f"slot_{slot_idx}",
                "uid": f"slot_{slot_idx}",
                "library_path": library_path,
                "archive_path": library_path,
                "rollout_n": int(rollout_n),
                "group_size": int(rollout_n),
                "puct_c": float(puct_c),
                "max_buffer_size": int(max_buffer_size),
                "topk_children": int(topk_children),
            },
        }
        for slot_idx in range(int(num_slots))
    ]
    if normalized_puct_q_mode != PUCT_Q_BLEND:
        for record in records:
            record["extra_info"]["puct_q_mode"] = normalized_puct_q_mode
    return records


def write_slot_parquet(
    path: str | Path,
    *,
    num_slots: int,
    library_path: str,
    task: str = "erdos_min_overlap",
    task_config: dict[str, Any] | None = None,
    rollout_n: int = 1,
    puct_c: float = 1.0,
    puct_q_mode: str = PUCT_Q_BLEND,
    max_buffer_size: int = 1000,
    topk_children: int = 2,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = build_slot_records(
        num_slots=num_slots,
        library_path=library_path,
        task=task,
        task_config=task_config,
        rollout_n=rollout_n,
        puct_c=puct_c,
        puct_q_mode=puct_q_mode,
        max_buffer_size=max_buffer_size,
        topk_children=topk_children,
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
