# Frozen TriMul scratch seed

`glm52_scratch_bootstrap_library.json` is the default TriMul library seed.
It was generated once by GLM-5.2 from the public TriMul task statement and then
frozen. The generation prompt explicitly supplied no parent code, prior
solution, library summary, guidance, verifier score, search history, or
TTT-Discover optimized candidate.

The seed passed the official evaluator on an NVIDIA H100:

- all 18 public correctness cases passed with zero reported maximum error;
- all seven leaderboard cases passed;
- geometric-mean runtime: `10177.396849081848 us`;
- Guidance-TTT reward: `1500 / runtime_us = 0.14738542893071152`;
- solution SHA-256: `49485cfe6f1e0f5d2ea40df2bead246b59ab8774cff3a79ab03cba4cf08995a9`.

The JSON is a complete pristine Guidance-TTT library. It retains the exact
accepted prompt, GLM response, raw model summary, generated source, API usage,
and per-case verifier artifacts. Normal TriMul runs copy this file and make no
bootstrap LLM request.

The one-off provenance workflow is:

```bash
modal run scripts/modal_trimul_h200_smoke.py \
  --action generate_scratch_seed \
  --prompt-mode summary_only
```

That command writes a newly generated candidate to the Modal run volume; it
does not overwrite this tracked frozen seed automatically. Replacing the
tracked seed requires an explicit review of the prompt provenance, code hash,
and complete official evaluator result.
