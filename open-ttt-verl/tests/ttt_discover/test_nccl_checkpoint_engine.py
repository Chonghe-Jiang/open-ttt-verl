import pytest


def test_nccl_checkpoint_engine_registers_when_cupy_is_available():
    pytest.importorskip("cupy")

    from verl.checkpoint_engine.base import CheckpointEngineRegistry
    import verl.checkpoint_engine.nccl_checkpoint_engine  # noqa: F401

    assert "nccl" in CheckpointEngineRegistry._registry
