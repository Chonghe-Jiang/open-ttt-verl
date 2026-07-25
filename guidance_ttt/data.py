from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    puct_q_mode: str = "blended",
    max_buffer_size: int = 1000,
    topk_children: int = 2,
    discover_compat: bool = False,
    groups_per_batch: int = 1,
    score_direction: str = "max",
) -> list[dict[str, Any]]:
    task_config_payload = _slot_task_config(task, task_config)
    return [
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
                "puct_q_mode": str(puct_q_mode),
                "max_buffer_size": int(max_buffer_size),
                "topk_children": int(topk_children),
                "discover_compat": bool(discover_compat),
                "groups_per_batch": int(groups_per_batch),
                "score_direction": str(score_direction),
            },
        }
        for slot_idx in range(int(num_slots))
    ]


def write_slot_parquet(
    path: str | Path,
    *,
    num_slots: int,
    library_path: str,
    task: str = "erdos_min_overlap",
    task_config: dict[str, Any] | None = None,
    rollout_n: int = 1,
    puct_c: float = 1.0,
    puct_q_mode: str = "blended",
    max_buffer_size: int = 1000,
    topk_children: int = 2,
    discover_compat: bool = False,
    groups_per_batch: int = 1,
    score_direction: str = "max",
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
        discover_compat=discover_compat,
        groups_per_batch=groups_per_batch,
        score_direction=score_direction,
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
