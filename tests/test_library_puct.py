import json

from guidance_ttt.library import GuidanceLibrary
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


def test_archive_dedup_does_not_use_solution_text_without_artifacts(tmp_path):
    path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=1.0)
    library = GuidanceLibrary(path, initial_nodes=[root], rollout_n=1, topk_children=2)

    selected = library.acquire_group("1:slot-a")
    first = _entry(selected.id, reward=2.0, suffix="a")
    first.solution = "same legacy solution text"
    first.summary = "distinct summary a"
    library.submit_child("1:slot-a", first)

    selected = library.acquire_group("2:slot-a")
    second = _entry(selected.id, reward=3.0, suffix="b")
    second.solution = "same legacy solution text"
    second.summary = "distinct summary b"
    library.submit_child("2:slot-a", second)

    snapshot = library.snapshot()

    child_entry_ids = {
        node["entry_id"]
        for node in snapshot["nodes"].values()
        if node["entry_id"] in {"entry-a", "entry-b"}
    }
    assert child_entry_ids == {"entry-a", "entry-b"}


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


def test_same_step_batch_blocks_selected_lineages(tmp_path):
    path = tmp_path / "library.json"
    root_a = make_root_node(problem_id="erdos", raw_score=0.5, reward=1.0)
    root_b = make_root_node(problem_id="erdos", raw_score=0.5, reward=0.9)
    library = GuidanceLibrary(path, initial_nodes=[root_a, root_b], rollout_n=1, puct_c=0.0)

    first = library.acquire_group("1:slot-a", visible_timestep_exclusive=1)
    second = library.acquire_group("1:slot-b", visible_timestep_exclusive=1)

    assert {first.id, second.id} == {root_a.id, root_b.id}
