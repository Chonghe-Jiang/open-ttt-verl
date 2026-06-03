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

import threading

from verl_ttt_discover.archive import PUCTArchive
from verl_ttt_discover.state import DiscoveryState


def test_acquire_group_binds_same_uid_to_one_state(tmp_path):
    states = [
        DiscoveryState(timestep=-1, value=1.0, raw_score=1.0, code="", construction=[0.5, 0.5], id="state-a"),
        DiscoveryState(timestep=-1, value=0.5, raw_score=2.0, code="", construction=[0.4, 0.6], id="state-b"),
    ]
    archive = PUCTArchive(tmp_path / "archive.json", initial_states=states, rollout_n=3)

    first = archive.acquire_group("7:uid-0")
    second = archive.acquire_group("7:uid-0")

    assert first.id == second.id
    assert archive.snapshot()["groups"]["7:uid-0"]["state_id"] == first.id


def test_concurrent_acquire_group_is_atomic(tmp_path):
    states = [
        DiscoveryState(timestep=-1, value=float(i), raw_score=float(10 - i), code="", construction=[0.5], id=f"s{i}")
        for i in range(5)
    ]
    archive = PUCTArchive(tmp_path / "archive.json", initial_states=states, rollout_n=8)
    barrier = threading.Barrier(8)
    picked_ids = []

    def acquire_once():
        barrier.wait()
        picked_ids.append(archive.acquire_group("3:uid-0").id)

    threads = [threading.Thread(target=acquire_once) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(picked_ids)) == 1


def test_submit_child_finalizes_group_once_and_updates_puct(tmp_path):
    parent = DiscoveryState(timestep=-1, value=1.0, raw_score=1.0, code="", construction=[0.5], id="parent")
    archive = PUCTArchive(tmp_path / "archive.json", initial_states=[parent], rollout_n=3, topk_children=2)
    archive.acquire_group("1:uid-0")

    assert not archive.submit_child(
        "1:uid-0",
        DiscoveryState(timestep=1, value=1.2, raw_score=0.8, code="a", construction=[0.6], id="child-a"),
    )
    assert not archive.submit_child(
        "1:uid-0",
        DiscoveryState(timestep=1, value=0.8, raw_score=1.2, code="b", construction=[0.4], id="child-b"),
    )
    assert archive.submit_child(
        "1:uid-0",
        DiscoveryState(timestep=1, value=2.0, raw_score=0.5, code="c", construction=[0.7], id="child-c"),
    )
    assert not archive.submit_child(
        "1:uid-0",
        DiscoveryState(timestep=1, value=3.0, raw_score=0.3, code="late", construction=[0.8], id="child-late"),
    )

    snapshot = archive.snapshot()
    assert snapshot["puct_T"] == 1
    assert snapshot["puct_n"]["parent"] == 1
    assert snapshot["puct_m"]["parent"] == 2.0
    assert snapshot["groups"]["1:uid-0"]["finalized"] is True
    assert snapshot["best_state_id"] == "child-c"
    assert [state["id"] for state in snapshot["states"] if state["id"].startswith("child")] == ["child-c", "child-a"]


def test_multiple_archive_instances_do_not_drop_submit_counts(tmp_path):
    parent = DiscoveryState(timestep=-1, value=1.0, raw_score=1.0, code="", construction=[0.5], id="parent")
    path = tmp_path / "archive.json"
    PUCTArchive(path, initial_states=[parent], rollout_n=2)

    first_process_view = PUCTArchive(path, rollout_n=2)
    second_process_view = PUCTArchive(path, rollout_n=2)

    assert first_process_view.acquire_group("1:uid-0").id == "parent"
    assert second_process_view.acquire_group("1:uid-0").id == "parent"
    assert not first_process_view.submit_child("1:uid-0", None)
    assert second_process_view.submit_child("1:uid-0", None)

    snapshot = PUCTArchive(path, rollout_n=2).snapshot()
    assert snapshot["groups"]["1:uid-0"]["submitted"] == 2
    assert snapshot["groups"]["1:uid-0"]["finalized"] is True
    assert snapshot["puct_T"] == 1


def test_duplicate_construction_is_not_added_to_archive(tmp_path):
    parent = DiscoveryState(timestep=-1, value=1.0, raw_score=1.0, code="", construction=[0.5], id="parent")
    archive = PUCTArchive(tmp_path / "archive.json", initial_states=[parent], rollout_n=1)
    archive.acquire_group("1:uid-0")

    assert archive.submit_child(
        "1:uid-0",
        DiscoveryState(timestep=1, value=2.0, raw_score=0.5, code="duplicate", construction=[0.5], id="duplicate"),
    )

    snapshot = archive.snapshot()
    assert [state["id"] for state in snapshot["states"]] == ["parent"]
    assert snapshot["puct_m"]["parent"] == 2.0


def test_lineage_blocking_avoids_same_step_family(tmp_path):
    parent = DiscoveryState(timestep=-1, value=3.0, raw_score=0.3, code="", construction=[0.3], id="parent")
    child = DiscoveryState(
        timestep=0,
        value=2.0,
        raw_score=0.5,
        code="child",
        construction=[0.2],
        id="child",
        parents=[{"id": "parent", "timestep": -1}],
    )
    other = DiscoveryState(timestep=-1, value=1.0, raw_score=1.0, code="", construction=[0.1], id="other")
    archive = PUCTArchive(tmp_path / "archive.json", initial_states=[parent, child, other], rollout_n=2)

    assert archive.acquire_group("4:uid-0").id == "parent"
    assert archive.acquire_group("4:uid-1").id == "other"


def test_archive_writes_step_snapshot_and_puct_stats(tmp_path):
    parent = DiscoveryState(timestep=-1, value=1.0, raw_score=1.0, code="", construction=[0.5], id="parent")
    archive = PUCTArchive(tmp_path / "archive.json", initial_states=[parent], rollout_n=1)
    archive.acquire_group("12:uid-0")

    assert archive.submit_child(
        "12:uid-0",
        DiscoveryState(timestep=12, value=1.2, raw_score=0.8, code="child", construction=[0.6], id="child"),
    )

    assert (tmp_path / "archive_snapshots" / "step_000012.json").exists()
    assert (tmp_path / "puct_stats.json").exists()
    snapshot = archive.snapshot()
    assert "sample_stats" in snapshot
    assert snapshot["last_sampled_stats"][0]["state_id"] == "parent"
