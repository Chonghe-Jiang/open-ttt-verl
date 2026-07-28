#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from guidance_ttt.library import GuidanceLibrary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align a Guidance-TTT library with the latest complete VERL checkpoint."
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_root = output_dir / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    library_path = output_dir / "library.json"
    if not tracker.is_file():
        raise FileNotFoundError(f"Missing checkpoint tracker: {tracker}")
    if not library_path.is_file():
        raise FileNotFoundError(f"Missing persisted Guidance-TTT library: {library_path}")

    latest_step = int(tracker.read_text().strip())
    latest_checkpoint = checkpoint_root / f"global_step_{latest_step}"
    if not (latest_checkpoint / "actor").is_dir():
        raise FileNotFoundError(f"Missing actor checkpoint: {latest_checkpoint / 'actor'}")
    if not (latest_checkpoint / "data.pt").is_file():
        raise FileNotFoundError(f"Missing dataloader checkpoint: {latest_checkpoint / 'data.pt'}")

    library = GuidanceLibrary(library_path)
    rollback = library.rollback_incomplete_steps_after(latest_step)
    snapshot = library.snapshot()
    remaining_future_groups = [
        group_uid
        for group_uid, group in (snapshot.get("groups") or {}).items()
        if int(group.get("visible_timestep_exclusive") or group_uid.split(":", 1)[0]) > latest_step
    ]
    if remaining_future_groups:
        raise RuntimeError(
            f"Library still contains {len(remaining_future_groups)} groups after checkpoint step {latest_step}"
        )

    print(
        f"Resume state ready at global_step_{latest_step}: "
        f"rolled_back_groups={rollback['groups']} "
        f"rolled_back_submissions={rollback['submitted']} "
        f"rolled_back_entries={rollback['entries']} "
        f"rolled_back_nodes={rollback['nodes']}"
    )


if __name__ == "__main__":
    main()
