# Copyright 2026 Chonghe Jiang and/or contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
