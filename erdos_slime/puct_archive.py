from __future__ import annotations

from pathlib import Path

from erdos_slime.ttt_discover.archive import PUCTArchive
from erdos_slime.ttt_discover.erdos_env import create_random_initial_state
from erdos_slime.ttt_discover.state import DiscoveryState


def create_initial_archive(
    path: str | Path,
    *,
    num_states: int = 4,
    rollout_n: int = 8,
    puct_c: float = 1.0,
    topk_children: int = 2,
    max_buffer_size: int = 1000,
) -> PUCTArchive:
    archive_path = Path(path)
    initial_states = None
    if not archive_path.exists():
        initial_states = [create_random_initial_state(seed=i) for i in range(num_states)]
    return PUCTArchive(
        archive_path,
        initial_states=initial_states,
        rollout_n=rollout_n,
        puct_c=puct_c,
        topk_children=topk_children,
        max_buffer_size=max_buffer_size,
    )


__all__ = ["DiscoveryState", "PUCTArchive", "create_initial_archive"]
