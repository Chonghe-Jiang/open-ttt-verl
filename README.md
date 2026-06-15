# Guidance + Execution TTT

This repository implements a Guidance-TTT prototype on top of the `open-ttt-verl`
pattern.

The rollout loop is:

```text
PUCT library node
-> trainable guidance model
-> guidance / idea
-> execution LLM
-> execution thinking + solution code + summary
-> verifier reward
-> new library child node
-> RL update on guidance tokens only
```

The first target task is Erdos minimum overlap. Execution uses `MockLLMClient`
by default so the package can be tested without a real API.

## Prepare a Smoke Run

```bash
python -m guidance_ttt.main_erdos \
  --config guidance_ttt/config/erdos_smoke.yaml \
  --prepare-only
```

This writes:

```text
outputs/guidance_ttt/erdos_smoke/library.json
outputs/guidance_ttt/erdos_smoke/ttt_slots.parquet
outputs/guidance_ttt/erdos_smoke/agent_loop.yaml
```

## Tests

```bash
pytest -q tests
```

## Design Notes

- PUCT selects one library node before prompt assembly.
- Guidance prompt attaches the selected node summary and scores, not full global history.
- Execution prompt attaches the same selected node plus the full prior solution code when available.
- Execution LLM returns `<execution_thinking>`, one Python code block, and `<summary>` in a single response.
- The verifier reward is assigned only to guidance model response tokens.
- Execution failures are environment outcomes and become library entries with reward `0.0`.
- The execution-provided summary is stored with verifier reward/status as structured library metadata; the summary should not claim verifier success before verification runs.
