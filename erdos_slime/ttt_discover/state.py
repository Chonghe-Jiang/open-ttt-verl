from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
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
