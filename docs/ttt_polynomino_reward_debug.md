# Symptom

The original Modal runs for the Frontier-CS Polynomino reproduction had rollout traffic, but no learning signal:

- `critic/score`, `critic/rewards`, `advantages`, `actor/pg_loss`, and `actor/grad_norm` were all zero.
- `rollout_debug.jsonl` showed `valid=0`, `reward_nonzero=0`, `has_python_block=0`, `has_extracted_code=0`.
- The agent-loop error was `No python code block found`.
- Decoded responses were empty or Qwen-style special fragments such as `</think>`.

After adding the Polyomino C++ path, the first attempted run still failed differently:

- chat-template generation produced only a few tokens, usually an immediate `<|im_end|>`.
- raw C++ completion with `load_format=dummy` produced long but random-looking Unicode/tool/thinking fragments.
- all candidates compiled or verified as invalid, so all rewards were the same invalid penalty and group advantages were still zero.

# Root Cause

Ranked by likelihood and confirmed evidence:

1. **P0: the original job was wired to the Erdos Python agent, not Frontier-CS Polynomino.** The old path expected a Python `run()` function and extracted ```python blocks, so every Polynomino-style candidate would fail before the verifier.
2. **P0: the rollout model was effectively not generating from real Qwen weights when `actor_rollout_ref.rollout.load_format` stayed at verl's default `dummy`.** With LoRA enabled, vLLM launched with `--load_format dummy`; debug output looked random and contained non-code multilingual fragments. Setting `actor_rollout_ref.rollout.load_format=auto` made vLLM load safetensors and immediately restored valid C++ candidates.
3. **P0: the candidate format and extractor did not match Frontier-CS Polynomino.** Polynomino expects a complete C++17 program that writes placements, not a Python function returning a data structure.
4. **P1: parse failures previously received reward `0.0`.** When every rollout in a group has the same score, `entropic_adaptive_beta` produces zero advantages and PPO has no gradient. The Polyomino loop now gives invalid candidates `-0.1`, and valid candidates get `10 * raw_ratio`.
5. **P1: chat-template generation was a poor forcing mechanism for this task.** The fixed loop uses raw completion with a C++ code-fence/header prefix and `min_tokens=512`, then extracts from the prefixed candidate text.

Implementation note: the first recovery run used the raw C++ prefix as a pragmatic
smoke path. The faithful Discover path now disables that by default for Polyomino:
the task is represented as a Discover-style `Environment`/`RewardEvaluator`, the
assistant response is parsed with the generic final-code-block parser, invalid
candidates receive Discover's default failure reward `0.0`, and PUCT visits are
updated per rollout rather than only once per finalized verl group.

# Evidence

Key implementation points:

- Polyomino agent registration and generation path: `verl_ttt_discover/agent_loop.py:215`.
- Raw C++ completion prefix: `verl_ttt_discover/agent_loop.py:237`.
- Prompt construction and raw-token prompt: `verl_ttt_discover/agent_loop.py:265`.
- Dynamic `max_tokens`, `min_tokens=512`, and code-fence stop: `verl_ttt_discover/agent_loop.py:272`.
- Candidate extraction from prefix + response: `verl_ttt_discover/agent_loop.py:296`.
- Reward assignment and archive update: `verl_ttt_discover/agent_loop.py:321`.
- Debug fields for prompt mode, extraction, rewards, and excerpts: `verl_ttt_discover/agent_loop.py:356`.
- C++ extractor: `verl_ttt_discover/cpp_sandbox.py:58`.
- C++ compile/run sandbox: `verl_ttt_discover/cpp_sandbox.py:79`.
- Polynomino verifier and ratio score: `verl_ttt_discover/polyomino_env.py:232`.
- C++ evaluation on validation cases: `verl_ttt_discover/polyomino_env.py:277`.
- Baseline seed state: `verl_ttt_discover/polyomino_env.py:304`.
- Polyomino prompt with previous best code: `verl_ttt_discover/polyomino_env.py:329`.
- Modal/verl recipe shape: `verl_ttt_discover/config/polyomino_2gpu_h200_qwen3_8b_g4_n16.yaml:1`.
- Required real rollout load: `verl_ttt_discover/config/polyomino_2gpu_h200_qwen3_8b_g4_n16.yaml:53`.

Confirmed Modal run:

- Active detached app: `ap-ka7eQBfBRnVHqdrJ7cXCSD`.
- vLLM command uses `Qwen/Qwen3-8B`, `--tensor_parallel_size 2`, bf16, `max_new_tokens=26000`.
- vLLM now uses `--load_format auto` and logs safetensors checkpoint loading.
- First training step recovered non-constant rewards:
  - `critic/score/mean = 2.3553`
  - `critic/score/max = 8.3333`
  - `critic/score/min = -0.1`
  - `actor/pg_loss = -0.01147`
  - `actor/grad_norm = 0.0204`
  - `response_length/mean = 1471.47`
- First-step debug summary from `rollout_debug.jsonl`:
  - rows: `64`
  - valid: `19`
  - invalid: `45`
  - `has_cpp_block = 64`
  - `has_extracted_code = 64`
  - reward range: `-0.1` to `8.3333`

# Broken Pipeline Path

Original broken path:

```text
Qwen response
  -> decode
  -> extract_python_code()
  -> no ```python block
  -> reward_score remains 0.0
  -> archive.submit_child(..., None)
  -> token_level_scores all 0
  -> entropic_adaptive_beta sees constant group rewards
  -> advantages = 0
  -> pg_loss = 0, grad_norm = 0
```

Intermediate broken path after adding C++ but before `load_format=auto`:

```text
raw C++ prompt
  -> vLLM dummy-weight generation
  -> random/special-token text
  -> C++ block extracted but compile fails
  -> reward_score = -0.1 for every rollout
  -> advantages = 0
```

Current faithful path:

```text
Discover Polyomino Environment prompt
  -> Qwen/Qwen3-8B chat-template generation loaded from safetensors
  -> generic final-code-block parser
  -> PolyominoRewardEvaluator
  -> compile C++ once
  -> run official Frontier-CS problem 0 cases with 16-way case concurrency
  -> judge each output with official chk.cc/testlib checker
  -> raw score = mean(100000 * official checker ratio)
  -> reward_score = raw_score / 10000
  -> valid child inserted for valid candidates
  -> failed rollout recorded for invalid candidates
  -> non-constant group rewards
  -> non-zero advantages and actor gradient
```

# Difference From Original Discover

Original Discover-style flow assumes:

- prompts enforce a candidate representation;
- generated candidates are executable by the task evaluator;
- evaluator failures are part of the search feedback;
- the library/archive receives valid improved candidates;
- bootstrapping or a previous valid candidate keeps search grounded.

The reproduction initially missed or mismatched these parts:

- It reused Erdos Python `run()` parsing for a C++ algorithmic contest problem.
- It had no Polynomino verifier adapter.
- It did not force a Frontier-CS submission format.
- It left rollout vLLM at `load_format=dummy`, which made generated candidates meaningless under LoRA.
- It gave parse failures a uniform zero score, causing GRPO/TTT collapse.

The current faithful implementation transfers those pieces by seeding a valid C++ shelf packer,
prompting with Discover's previous-state context, compiling and verifying C++ submissions,
storing valid children in the PUCT archive, and recording failed rollouts in PUCT.

# Required Fixes

## P0: Blocking Issues Preventing Any Learning

- Use the Polyomino agent loop, not Erdos: `ttt_discover_polyomino`.
- Extract C++ candidates, not Python `run()` functions.
- Verify the official Frontier-CS output format: `W H`, then `X Y R F` per piece.
- Load real rollout weights: `actor_rollout_ref.rollout.load_format=auto`.
- Use Discover failure semantics: invalid candidates receive reward `0.0`; valid candidates receive the normalized official-style score.
- Update PUCT visit counts per rollout, including failed rollouts.

## P1: Issues Affecting Reproduction Quality

- The local validation set is small; it is useful for fast training smoke tests but not a full Frontier-CS benchmark.
- Many first-step invalid candidates are near-miss compile errors. A repair pass or stricter skeleton could raise valid rate.
- Raw completion works as a debug fallback, but it bypasses Discover renderer semantics and is disabled for faithful Polyomino runs.
- Current reward is official Frontier-CS problem 0 checker score normalized by `10000`.

## P2: Architectural Improvements For Polynomino

- Add more official-style validation cases and randomized hidden cases.
- Add a syntax-repair or baseline-fallback mutation operator before assigning invalid penalty.
- Store richer archive metadata: strategy summary, validation case breakdown, compile error class.
- Make PUCT select among code strategies or normalized construction features, not just whole prior code blobs.
- Add a preflight generation-only Modal action that samples 4 candidates before launching a 50-step training job.

# PUCT / Library Interaction

The archive now starts with a valid baseline from `create_polyomino_initial_state()` and stores full C++ code plus raw ratio. During rollouts:

- `archive.acquire_group()` selects an existing state using PUCT.
- The prompt includes the selected state's previous best C++ code.
- Valid candidates call `archive.submit_child(group_uid, scored.state)`.
- Invalid candidates call `archive.submit_child(group_uid, None)`.

When every reward was zero or `-0.1`, PUCT could only update visits and could not accumulate meaningful child values. In the fixed run, 19/64 first-step candidates were valid, so the archive can now receive children with positive values and PUCT is functional rather than effectively disabled.
