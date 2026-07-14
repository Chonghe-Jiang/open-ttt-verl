import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.puct import archive_puct_score, rank_archive_nodes
from guidance_ttt.state import LibraryEntry, make_root_node


def _entry(parent_id: str, reward: float, suffix: str) -> LibraryEntry:
    return LibraryEntry(
        id=f"entry-{suffix}",
        parent_id=parent_id,
        problem_id="erdos",
        timestep=1,
        guidance=f"try idea {suffix}",
        execution_thinking=f"thinking {suffix}",
        solution=f"def run(): return {suffix!r}",
        verifier_reward=reward,
        verifier_raw_score=None,
        verifier_status="valid",
        verifier_message="ok",
        summary=f"summary {suffix}",
        reusable_idea=f"idea {suffix}",
        failure_mode=None,
        metadata={},
    )


def test_acquire_group_binds_same_group_to_same_node(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=0.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=2)

    first = library.acquire_group("0:slot-a")
    second = library.acquire_group("0:slot-a")

    assert first.id == second.id == root.id


def test_submit_child_adds_entry_child_and_updates_best(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=2.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1)
    selected = library.acquire_group("0:slot-a")

    child = library.submit_child("0:slot-a", _entry(selected.id, reward=4.0, suffix="a"))
    snapshot = library.snapshot()

    assert child.parent_id == selected.id
    assert child.entry_id == "entry-a"
    assert child.value == 4.0
    assert snapshot["best_node_id"] == child.id
    assert snapshot["nodes"][selected.id]["children"] == [child.id]


def test_attach_entry_to_root_makes_root_context_non_empty(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=2.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1)
    entry = _entry(root.id, reward=4.0, suffix="bootstrap")
    entry.timestep = 0
    entry.metadata = {"bootstrap": True, "raw_model_summary": "Bootstrap raw summary."}

    library.attach_entry_to_root(root.id, entry)
    snapshot = library.snapshot()
    context = library.context_for_node(root)

    assert snapshot["nodes"][root.id]["entry_id"] == entry.id
    assert snapshot["nodes"][root.id]["value"] == 4.0
    assert snapshot["nodes"][root.id]["metadata"]["bootstrap"] is True
    assert snapshot["entries"][entry.id]["metadata"]["raw_model_summary"] == "Bootstrap raw summary."
    assert context["selected_entry"].id == entry.id


def test_submit_child_serializes_nested_omegaconf_metadata(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="polyomino_packing", raw_score=0.0, reward=0.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1)
    selected = library.acquire_group("0:slot-a")
    entry = _entry(selected.id, reward=1.0, suffix="poly")
    entry.metadata = {
        "task": OmegaConf.create(
            {
                "id": "polyomino_packing",
                "frontiercs": {
                    "problem_id": "0",
                    "n_cases": 70,
                },
            }
        )
    }

    library.submit_child("0:slot-a", entry)
    store = json.loads(path.read_text())

    task = store["entries"][entry.id]["metadata"]["task"]
    assert task == {
        "id": "polyomino_packing",
        "frontiercs": {
            "problem_id": "0",
            "n_cases": 70,
        },
    }


def test_library_saves_via_atomic_temp_replace(tmp_path, monkeypatch):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=0.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1)

    original_write_text = Path.write_text
    original_replace = Path.replace
    write_targets: list[Path] = []
    replace_targets: list[Path] = []

    def tracked_write_text(self: Path, *args, **kwargs):
        write_targets.append(self)
        return original_write_text(self, *args, **kwargs)

    def tracked_replace(self: Path, target):
        replace_targets.append(Path(target))
        return original_replace(self, target)

    monkeypatch.setattr(Path, "write_text", tracked_write_text)
    monkeypatch.setattr(Path, "replace", tracked_replace)

    selected = library.acquire_group("0:slot-a")
    library.submit_child("0:slot-a", _entry(selected.id, reward=1.0, suffix="atomic"))

    assert path not in write_targets
    assert path in replace_targets
    assert json.loads(path.read_text())["entries"]["entry-atomic"]["summary"] == "summary atomic"


def test_puct_can_prefer_unvisited_child_over_high_value_visited_child(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=0.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1, puct_c=5.0)
    selected = library.acquire_group("0:slot-a")
    good = library.submit_child("0:slot-a", _entry(selected.id, reward=1.1, suffix="good"))
    library.acquire_group("0:slot-b")
    fresh = library.submit_child("0:slot-b", _entry(selected.id, reward=1.0, suffix="fresh"))

    library.mark_node_visited(good.id, count=10)
    library.mark_node_visited(fresh.id, count=0)
    picked = library.acquire_group("1:slot-a")

    assert picked.id == fresh.id


def test_archive_puct_score_blends_own_value_with_best_child_after_visit():
    node = make_root_node(problem_id="polyomino_packing", raw_score=100.0, reward=100.0)

    score = archive_puct_score(
        node=node,
        visit_count=1,
        best_reachable_value=50.0,
        prior=0.0,
        scale=1.0,
        total_visits=1,
        puct_c=0.0,
    )

    assert score == 90.0


def test_archive_puct_score_uses_best_child_directly_when_configured():
    node = make_root_node(problem_id="polyomino_packing", raw_score=100.0, reward=100.0)

    score = archive_puct_score(
        node=node,
        visit_count=1,
        best_reachable_value=50.0,
        prior=0.0,
        scale=1.0,
        total_visits=1,
        puct_c=0.0,
        q_mode="best_child",
    )

    assert score == 50.0


def test_archive_puct_score_uses_own_value_before_first_visit():
    node = make_root_node(problem_id="polyomino_packing", raw_score=100.0, reward=100.0)

    score = archive_puct_score(
        node=node,
        visit_count=0,
        best_reachable_value=50.0,
        prior=0.0,
        scale=1.0,
        total_visits=1,
        puct_c=0.0,
    )

    assert score == 100.0


def test_rank_archive_nodes_reports_blended_q_value_for_visited_node():
    visited = make_root_node(problem_id="polyomino_packing", raw_score=50.0, reward=50.0)
    fresh = make_root_node(problem_id="polyomino_packing", raw_score=40.0, reward=40.0)
    visited.id = "visited"
    fresh.id = "fresh"

    ranked = rank_archive_nodes(
        [visited, fresh],
        initial_ids=set(),
        visit_counts={"visited": 1},
        best_reachable_values={"visited": 100.0},
        total_visits=1,
        puct_c=0.0,
    )

    ranked_by_id = {item[2].id: item for item in ranked}
    assert ranked_by_id["visited"][4] == 60.0
    assert ranked[0][2].id == "visited"


def test_library_restores_puct_config_from_archive_config(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=2.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=4, puct_c=2.5, max_buffer_size=17, topk_children=3)
    store = library.snapshot()
    store.pop("rollout_n")
    store.pop("puct_c")
    path.write_text(json.dumps(store))

    restored = GuidanceLibrary(path, rollout_n=1, puct_c=1.0)

    assert restored.rollout_n == 4
    assert restored.puct_c == 2.5
    assert restored.max_buffer_size == 17
    assert restored.topk_children == 3
    assert restored.snapshot()["config"] == {
        "rollout_n": 4,
        "puct_c": 2.5,
        "max_buffer_size": 17,
        "topk_children": 3,
    }


def test_same_step_selection_does_not_see_child_created_in_same_step(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=1.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1, puct_c=1.0)

    selected = library.acquire_group("1:slot-a", visible_timestep_exclusive=1)
    child = library.submit_child("1:slot-a", _entry(selected.id, reward=10.0, suffix="same-step"))
    same_step = library.acquire_group("1:slot-b", visible_timestep_exclusive=1)
    same_step_context = library.context_for_node(root, visible_timestep_exclusive=1)
    next_step = library.acquire_group("2:slot-a", visible_timestep_exclusive=2)
    next_step_context = library.context_for_node(root, visible_timestep_exclusive=2)

    assert selected.id == root.id
    assert same_step.id == root.id
    assert same_step_context["global_best_entries"] == []
    assert next_step.id == child.id
    assert next_step_context["global_best_entries"][0].id == "entry-same-step"


def test_archive_dedup_uses_solution_identity_without_artifacts(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=1.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=2, topk_children=2)

    selected = library.acquire_group("1:slot-a")
    first = _entry(selected.id, reward=2.0, suffix="a")
    first.solution = "same legacy solution text"
    first.summary = "distinct summary a"
    library.submit_child("1:slot-a", first)

    second = _entry(selected.id, reward=3.0, suffix="b")
    second.solution = "same legacy solution text"
    second.summary = "distinct summary b"
    library.submit_child("1:slot-a", second)

    snapshot = library.snapshot()

    child_entry_ids = {
        node["entry_id"]
        for node in snapshot["nodes"].values()
        if node["entry_id"] in {"entry-a", "entry-b"}
    }
    assert child_entry_ids == {"entry-b"}


def test_code_bearing_selection_skips_higher_value_node_without_solution(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="polyomino_packing", raw_score=1.0, reward=1.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1, topk_children=2)
    root_entry = _entry(root.id, reward=1.0, suffix="root-code")
    library.attach_entry_to_root(root.id, root_entry)

    selected = library.acquire_group("1:slot-a", require_solution=True)
    missing_code = _entry(selected.id, reward=100.0, suffix="missing-code")
    missing_code.solution = ""
    library.submit_child("1:slot-a", missing_code)

    next_selected = library.acquire_group("2:slot-a", require_solution=True)

    assert next_selected.id == root.id
    assert library.get_entry(next_selected.entry_id).solution == root_entry.solution


def test_code_bearing_selection_fails_when_no_visible_node_has_solution(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="polyomino_packing", raw_score=0.0, reward=0.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1)

    with pytest.raises(RuntimeError, match="no visible node with solution code"):
        library.acquire_group("1:slot-a", require_solution=True)


def test_existing_group_cannot_switch_from_parent_without_solution(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="polyomino_packing", raw_score=0.0, reward=0.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=2)
    library.acquire_group("1:slot-a")

    with pytest.raises(RuntimeError, match="Existing group .* has no solution code"):
        library.acquire_group("1:slot-a", require_solution=True)


def test_archive_pruning_preserves_ancestors_of_kept_nodes(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=1.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1, topk_children=1)

    selected = library.acquire_group("1:mid")
    mid = library.submit_child("1:mid", _entry(selected.id, reward=2.0, suffix="mid"))

    library.acquire_group("2:grandchild")
    grandchild = library.submit_child("2:grandchild", _entry(mid.id, reward=4.0, suffix="grandchild"))

    selected = library.acquire_group("3:sibling")
    sibling = library.submit_child("3:sibling", _entry(selected.id, reward=3.0, suffix="sibling"))
    snapshot = library.snapshot()

    assert grandchild.id in snapshot["nodes"]
    assert mid.id in snapshot["nodes"]
    assert sibling.id in snapshot["nodes"]
    assert snapshot["nodes"][grandchild.id]["parent_id"] == mid.id
    assert grandchild.id in snapshot["nodes"][mid.id]["children"]


def test_group_finalization_updates_discover_puct_stats(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=1.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=2, puct_c=1.0)

    selected = library.acquire_group("1:slot-a", visible_timestep_exclusive=1)
    library.submit_child("1:slot-a", _entry(selected.id, reward=2.0, suffix="low"))
    mid = library.snapshot()
    library.submit_child("1:slot-a", _entry(selected.id, reward=3.0, suffix="high"))
    snapshot = library.snapshot()

    assert mid["puct_T"] == 0
    assert snapshot["puct_T"] == 1
    assert snapshot["puct_n"][selected.id] == 1
    assert snapshot["puct_m"][selected.id] == 3.0


def test_group_of_sixteen_finalizes_once_and_rejects_extra_child(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="polyomino_packing", raw_score=27.0, reward=27.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=16, puct_c=1.0)
    selected = library.acquire_group("1:slot-a", visible_timestep_exclusive=1)

    for index in range(15):
        library.submit_child(
            "1:slot-a",
            _entry(selected.id, reward=float(index + 1), suffix=f"partial-{index}"),
        )
    partial = library.snapshot()

    assert partial["groups"]["1:slot-a"]["submitted"] == 15
    assert partial["groups"]["1:slot-a"]["finalized"] is False
    assert partial["puct_T"] == 0
    assert selected.id not in partial["puct_n"]
    assert selected.id not in partial["puct_m"]

    library.submit_child("1:slot-a", _entry(selected.id, reward=16.0, suffix="final"))
    finalized = library.snapshot()

    assert finalized["groups"]["1:slot-a"]["submitted"] == 16
    assert finalized["groups"]["1:slot-a"]["finalized"] is True
    assert finalized["puct_T"] == 1
    assert finalized["puct_n"][selected.id] == 1
    assert finalized["puct_m"][selected.id] == 16.0

    with pytest.raises(RuntimeError, match="already complete"):
        library.submit_child("1:slot-a", _entry(selected.id, reward=100.0, suffix="extra"))

    assert library.snapshot() == finalized


def test_pristine_archive_can_adopt_recipe_runtime_config(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="polyomino_packing", raw_score=27.0, reward=27.0)
    library = GuidanceLibrary(
        path,
        initial_nodes=[root],
        rollout_n=1,
        puct_c=9.0,
        max_buffer_size=17,
        topk_children=1,
    )

    library.configure_pristine_archive(
        rollout_n=16,
        puct_c=1.0,
        max_buffer_size=1000,
        topk_children=2,
    )
    snapshot = library.snapshot()

    assert snapshot["config"] == {
        "rollout_n": 16,
        "puct_c": 1.0,
        "max_buffer_size": 1000,
        "topk_children": 2,
    }
    assert set(snapshot["nodes"]) == {root.id}


def test_pristine_archive_persists_best_child_q_mode(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="polyomino_packing", raw_score=27.0, reward=27.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1)

    library.configure_pristine_archive(
        rollout_n=16,
        puct_c=1.0,
        puct_q_mode="best_child",
        max_buffer_size=1000,
        topk_children=2,
    )

    restored = GuidanceLibrary(path)
    assert restored.snapshot()["config"]["puct_q_mode"] == "best_child"
    with pytest.raises(ValueError, match="puct_q_mode"):
        restored.assert_runtime_config(
            rollout_n=16,
            puct_c=1.0,
            puct_q_mode="blended",
            max_buffer_size=1000,
            topk_children=2,
        )


def test_runtime_config_validation_rejects_corrupt_group_accounting(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="polyomino_packing", raw_score=27.0, reward=27.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=2)
    selected = library.acquire_group("1:slot-a")
    library.submit_child("1:slot-a", _entry(selected.id, reward=1.0, suffix="low"))
    library.submit_child("1:slot-a", _entry(selected.id, reward=2.0, suffix="high"))

    store = json.loads(path.read_text())
    store["puct_T"] = 2
    path.write_text(json.dumps(store))
    corrupted = GuidanceLibrary(path, rollout_n=2)

    with pytest.raises(ValueError, match="puct_T=2, expected one update"):
        corrupted.assert_runtime_config(
            rollout_n=2,
            puct_c=1.0,
            max_buffer_size=1000,
            topk_children=2,
        )


def test_same_step_batch_blocks_selected_lineages(tmp_path):
    path = tmp_path / "library.json"
    root_a = make_root_node(problem_id="erdos", raw_score=0.5, reward=1.0)
    root_b = make_root_node(problem_id="erdos", raw_score=0.5, reward=0.9)
    library = GuidanceLibrary(path, initial_nodes=[root_a, root_b], rollout_n=1, puct_c=0.0)

    first = library.acquire_group("1:slot-a", visible_timestep_exclusive=1)
    second = library.acquire_group("1:slot-b", visible_timestep_exclusive=1)

    assert {first.id, second.id} == {root_a.id, root_b.id}
