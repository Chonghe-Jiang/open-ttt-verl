from guidance_ttt.state import LibraryNode, make_root_node


def test_make_root_node_sets_reward_value_and_empty_tree_links():
    node = make_root_node(problem_id="erdos", raw_score=0.5, reward=2.0)

    assert isinstance(node, LibraryNode)
    assert node.problem_id == "erdos"
    assert node.parent_id is None
    assert node.children == []
    assert node.value == 2.0
    assert node.raw_score == 0.5
    assert node.visits == 0

