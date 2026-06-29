from guidance_ttt.tasks import get_task_spec


def test_task_registry_loads_erdos_spec():
    spec = get_task_spec("erdos_min_overlap")

    assert spec.task_id == "erdos_min_overlap"
    assert spec.solution_language == "python"
    assert "Lower raw C5 is better" in spec.guidance_objective(0.4)
    assert spec.create_root_node().problem_id == "erdos"


def test_task_registry_loads_polyomino_spec():
    spec = get_task_spec("polyomino_packing")

    assert spec.task_id == "polyomino_packing"
    assert spec.solution_language == "cpp"
    assert "Polyomino Packing" in spec.problem_prompt
    assert "Higher FrontierCS score is better" in spec.guidance_objective(0.0)
    assert "complete C++17 program" in spec.execution_solution_contract
    assert spec.create_root_node().problem_id == "polyomino_packing"
    assert spec.create_root_node().raw_score == 0.0


def test_task_registry_rejects_unknown_task():
    try:
        get_task_spec("missing_task")
    except KeyError as exc:
        assert "Unknown Guidance-TTT task" in str(exc)
    else:
        raise AssertionError("unknown task should raise KeyError")
