from pathlib import Path


def test_naive_update_weights_does_not_unconditionally_offload_actor_to_cpu():
    source = Path("verl/workers/engine_workers.py").read_text()

    assert "if self.actor.engine.is_param_offload_enabled:" in source
    assert "self.actor.engine.to(\"cpu\", model=True, optimizer=False, grad=False)" in source
