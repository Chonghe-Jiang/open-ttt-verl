# Guidance + Execution TTT

This repository implements a Guidance-TTT prototype on top of the `open-ttt-verl`
pattern.

The rollout loop is:

```text
PUCT library node
-> trainable guidance model
-> guidance / idea
-> execution LLM
-> solution code
-> verifier reward
-> summarizer LLM
-> new library child node
-> RL update on guidance tokens only
```

The first target task is Erdos minimum overlap. Execution and summarization use
`MockLLMClient` by default so the package can be tested without a real API.

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
- The verifier reward is assigned only to guidance model response tokens.
- Execution/summarization failures are environment outcomes and become library entries with reward `0.0`.

