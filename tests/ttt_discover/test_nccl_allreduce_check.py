from scripts.ttt_discover.nccl_allreduce_check import expected_allreduce_sum


def test_expected_allreduce_sum_matches_rank_sum():
    assert expected_allreduce_sum(world_size=1) == 1.0
    assert expected_allreduce_sum(world_size=2) == 3.0
    assert expected_allreduce_sum(world_size=4) == 10.0
