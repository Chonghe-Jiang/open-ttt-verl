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

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass
class DiscoveryState:
    timestep: int
    value: float
    raw_score: float | None
    code: str
    construction: list[Any] | None
    id: str = field(default_factory=lambda: str(uuid4()))
    parent_values: list[float] = field(default_factory=list)
    parents: list[dict[str, Any]] = field(default_factory=list)
    observation: str = ""

    def __post_init__(self) -> None:
        self.construction = _jsonable(self.construction)
        self.value = float(self.value) if self.value is not None else None
        self.raw_score = float(self.raw_score) if self.raw_score is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "raw_score": self.raw_score,
            "code": self.code,
            "construction": _jsonable(self.construction),
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveryState":
        return cls(
            id=data["id"],
            timestep=int(data["timestep"]),
            value=data.get("value"),
            raw_score=data.get("raw_score"),
            code=data.get("code", ""),
            construction=data.get("construction"),
            parent_values=list(data.get("parent_values", [])),
            parents=list(data.get("parents", [])),
            observation=data.get("observation", ""),
        )

    def to_prompt(self, target: float, metric_name: str = "value", maximize: bool = True, language: str = "") -> str:
        value_ctx = f"You are iteratively optimizing {metric_name}."
        improvement_direction = "higher" if maximize else "lower"

        if self.code and self.code.strip():
            value_ctx += "\nHere is the last code we ran:\n"
            if language:
                value_ctx += f"```{language}\n{self.code}\n```"
            else:
                value_ctx += self.code
        else:
            value_ctx += "\nNo previous code available."

        if self.parent_values and self.value is not None and self.construction:
            before_value = self.parent_values[0] if maximize else -self.parent_values[0]
            after_value = self.value if maximize else -self.value
            current_gap = target - after_value if maximize else after_value - target
            value_ctx += (
                f"\nHere is the {metric_name} before and after running the code above "
                f"({improvement_direction} is better): {before_value:.6f} -> {after_value:.6f}"
            )
            value_ctx += f"\nTarget: {target}. Current gap: {current_gap:.6f}. Further improvements will also be generously rewarded."
        elif self.value is not None:
            after_value = self.value if maximize else -self.value
            current_gap = target - after_value if maximize else after_value - target
            value_ctx += f"\nCurrent {metric_name} ({improvement_direction} is better): {after_value:.6f}"
            value_ctx += f"\nTarget: {target}. Current gap: {current_gap:.6f}. Further improvements will also be generously rewarded."
        else:
            value_ctx += f"\nTarget {metric_name}: {target}"

        if self.observation and self.observation.strip():
            stdout = self.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        return value_ctx
