"""Slime reward model for the Erdos TTT discovery task."""
from __future__ import annotations

import asyncio
import os
import sys

_PROJECT_ROOT = os.environ.get("ERDOS_PROJECT_ROOT", "/root/workspace/erdos")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _compute(sample) -> float:
    from erdos_slime.erdos_env import score_erdos_result
    from erdos_slime.erdos_sandbox import evaluate_python_code, extract_python_code
    from erdos_slime.puct_archive import DiscoveryState, PUCTArchive

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    code = extract_python_code(sample.response)
    group_uid = metadata.get("erdos_group_uid")
    archive = None

    if metadata.get("erdos_archive_path"):
        archive = PUCTArchive(
            metadata["erdos_archive_path"],
            rollout_n=int(os.environ.get("ERDOS_ROLLOUT_N", "8")),
            puct_c=float(os.environ.get("ERDOS_PUCT_C", "1.0")),
            topk_children=int(os.environ.get("ERDOS_TOPK_CHILDREN", "2")),
        )

    if not code:
        if archive is not None and group_uid:
            archive.submit_child(group_uid, None)
        metadata.update({"erdos_valid": False, "erdos_error": "No python code block found"})
        return 0.0

    state_data = metadata.get("erdos_state")
    if not state_data:
        metadata.update({"erdos_valid": False, "erdos_error": "Missing Erdos parent state"})
        return 0.0

    state = DiscoveryState.from_dict(state_data)
    budget_s = int(metadata.get("erdos_budget_s") or os.environ.get("ERDOS_BUDGET_S", "60"))
    sandbox_result = evaluate_python_code(code, state=state, timeout_s=budget_s)
    if sandbox_result.error is not None:
        if archive is not None and group_uid:
            archive.submit_child(group_uid, None)
        metadata.update({"erdos_valid": False, "erdos_error": sandbox_result.error})
        return 0.0

    try:
        timestep = metadata.get("erdos_global_step", 0)
        scored = score_erdos_result(
            sandbox_result.output,
            code=code,
            timestep=int(timestep) if str(timestep).isdigit() else 0,
            stdout=sandbox_result.stdout,
        )
    except Exception as exc:
        if archive is not None and group_uid:
            archive.submit_child(group_uid, None)
        metadata.update({"erdos_valid": False, "erdos_error": str(exc)})
        return 0.0

    if archive is not None and group_uid:
        archive.submit_child(group_uid, scored.state)
    metadata.update(
        {
            "erdos_valid": True,
            "erdos_raw_score": scored.raw_score,
            "erdos_message": scored.message,
        }
    )
    return float(scored.reward)


async def reward(args, sample):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _compute, sample)


__all__ = ["reward"]
