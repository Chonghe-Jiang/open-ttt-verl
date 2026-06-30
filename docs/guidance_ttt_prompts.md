# Guidance-TTT Prompt Architecture

本文档按当前代码整理 `guidance-ttt` 的 prompt 架构，覆盖 guidance model、execution
model、execution 自总结、verifier 和 library writeback 的完整信息流。

代码来源：

- `guidance_ttt/prompts.py`: guidance/execution prompt 模板、raw summary attach、tag parser
- `guidance_ttt/tasks/__init__.py`: task registry、task-specific prompt/verifier/objective 配置
- `guidance_ttt/agent_loop.py`: verl agent loop、模型调用、verification、canonical summary、library 写回
- `guidance_ttt/library.py`: PUCT node selection、visible context、JSON library
- `guidance_ttt/verifier/erdos.py`: Erdos verifier、reward/raw score 计算
- `guidance_ttt/verifier/polyomino.py`: Polyomino C++ `<solution>` 提取和 FrontierCS result mapping
- `guidance_ttt/verifier/frontiercs_adapter.py`: 外部 FrontierCS/go-judge/Docker evaluator adapter

## Overall Structure

```mermaid
flowchart TD
    A[Problem prompt / task spec] --> C[PUCT library selection]
    B[(GuidanceLibrary JSON)] --> C

    C --> D[selected_node]
    C --> E[selected_entry / local_failure_entries]
    C --> X[global_best_entries<br/>target score only]

    A --> F[build_guidance_prompt]
    D --> F
    E --> F
    F --> G[Guidance model<br/>trainable actor]
    G --> H[Raw guidance response<br/>&lt;think&gt; + &lt;guidance&gt;]
    H --> I[extract_guidance_or_format_error]

    A --> J[build_execution_prompt]
    D --> J
    E --> J
    I --> J
    J --> K[Execution LLM<br/>frozen executor]
    K --> L[Execution response<br/>&lt;execution_thinking&gt; + &lt;solution&gt; + &lt;summary&gt;]

    L --> M[Task verifier]
    M --> N[reward / raw score / status / artifacts]
    L --> O[build_execution_summary]
    N --> O
    I --> O

    O --> P[LibraryEntry<br/>guidance, thinking, solution, canonical summary]
    N --> P
    P --> Q[LibraryNode<br/>value=reward, raw_score, status]
    Q --> B

    N --> R[verl reward_score]
    H --> S[response_ids / response_mask]
    R --> T[RL update on guidance tokens only]
    S --> T
```

核心隔离规则：

```text
visible_timestep_exclusive = global_step
```

每个 rollout 只能看到 `timestep < global_step` 的 library 内容。同一个 training step 中刚
产生的其它 rollout 不会进入当前 prompt。

## One Rollout Flow

```text
GuidanceLibrary.acquire_group(group_uid, visible_timestep_exclusive=global_step)
  -> selected_node chosen by PUCT

GuidanceLibrary.context_for_node(selected_node, visible_timestep_exclusive=global_step)
  -> selected_entry
  -> global_best_entries (objective target only; not prompt summary attach)
  -> local_failure_entries

build_guidance_prompt(...)
  -> guidance chat messages
  -> guidance model outputs <think>...</think> and <guidance>...</guidance>

extract_guidance_or_format_error(...)
  -> parsed guidance
  -> guidance_format_ok

build_execution_prompt(...)
  -> execution chat messages
  -> execution model outputs <execution_thinking>, <solution>, <summary>

task_spec.verify_execution_text(execution_text)
  -> verifier status / task raw score / reward / artifacts

build_execution_summary(...)
  -> canonical summary combining execution thinking, model summary, solution, guidance, verifier result

GuidanceLibrary.submit_child(...)
  -> write LibraryEntry and child LibraryNode
```

## Shared Library Context

Guidance prompt 和 execution prompt 来自同一次 PUCT selection 的同一个
`selected_node` / `selected_entry`，但 attach 的粒度不同。

共同来源：

```python
selected_node = library.acquire_group(...)
context = library.context_for_node(selected_node, ...)
selected_entry = context["selected_entry"]
global_best_entries = context["global_best_entries"]  # target score only
local_failure_entries = context["local_failure_entries"]
```

- Guidance prompt 和 execution prompt 都 attach lightweight context: raw model
  `<summary>` plus verifier score/status/message。不会直接 attach canonical
  `entry.summary` 的 6-section 大块内容，除非老 library entry 缺少 raw summary 来源。
- Guidance prompt 和 execution prompt 都不 attach global best summary。当前实现只把
  PUCT-selected entry 的 raw summary + 分数作为主历史上下文；guidance prompt 另外保留
  local failure raw summaries。
- 两个 prompt 都不 attach node id、visits、verified artifacts、previous guidance、
  reusable idea、failure mode 或 root initial construction facts。
- Execution prompt 额外 attach 当前 parsed `<guidance>`，用于把 guidance 落成具体代码。
- 当前 `lineage_entries` 会由 library context 返回，但没有直接写入 prompt。

## Summary Attach Rules

Guidance 和 execution prompt 都使用 `_raw_summary_for_prompt(entry, fallback=...)`，
但这里的 “raw summary” 指 execution model 原始 `<summary>`，不是
`LibraryEntry.summary` 里的 canonical summary。

没有 previous entry、且无法找到 raw summary 时：

```text
No previous summary is attached.
```

有 entry 时，attach helper 按优先级取 summary：

```text
1. entry.metadata["raw_model_summary"]
2. entry.metadata["execution_text"] 中的 <summary>...</summary>
3. entry.summary 作为老 library entry 的兼容 fallback
```

然后追加 verifier score block：

```text
{raw model summary}

Verifier status: {entry.verifier_status}
{task raw score label}: {entry.verifier_raw_score}
Reward: {entry.verifier_reward}
Verifier message: {entry.verifier_message}
```

该 helper 不做 clipping、code removal、implementation fact extraction 或 profile
compaction。新 library entry 的 prompt context 通常不包含 solution code，因为 execution
prompt 要求 raw `<summary>` 不包含 code；如果老 entry 只能 fallback 到 canonical
`entry.summary`，则可能仍然带上旧的 canonical 内容。

## Task Registry

当前 loop 通过 `task.id` 获取 `TaskSpec`。Erdos 和 Polyomino 共用同一个
guidance/execution/library/PUCT 流程，差异只放在 task spec 中。

```python
TaskSpec(
    task_id=...,
    problem_prompt=...,
    solution_language=...,          # "python" or "cpp"
    execution_solution_contract=...,
    score_direction=...,            # "min" or "max"
    raw_score_label=...,            # e.g. "Raw C5" or "FrontierCS score"
    create_root_node=...,
    verifier=...,
    solution_extractor=...,
    guidance_objective=...,
)
```

Configured tasks:

- `erdos_min_overlap`: execution outputs Python `run()`, lower raw C5 is better,
  reward is `1 / (1e-8 + C5)`.
- `polyomino_packing`: execution outputs a full C++17 program, evaluation always
  goes through external FrontierCS/go-judge/Docker, higher FrontierCS score is better.

## Guidance Model Prompt

Guidance model 是被 RL 训练的 actor。它不直接写最终 solution，而是输出下一步
搜索方向。verl 的 reward 最终只作用在 guidance model response tokens 上。

### Guidance System Prompt

```text
You are the Guidance Model, acting as a strategic navigator for an open-ended scientific discovery process.

Your primary objective is to provide **evolutionary guidance**. Do not write final code or focus on low-level implementation details. Instead, your task is to propose high-level directional shifts, conceptual mutations, and novel pathways to explore the search space.

Focus on how the current ideas can *evolve* to escape local optima and discover fundamentally new mechanisms.
```

### Guidance User Prompt

下面是 `build_guidance_prompt(...)` 当前构造的完整 user prompt。`{...}` 是运行时
Python f-string 插入的变量。

````text
<problem>
{problem_prompt}
</problem>

The next sections describe the current search state for this problem. Use them
as run-local context when deciding the next step.

<selected_summary>
{_raw_summary_for_prompt(selected_entry, fallback="No previous summary is attached.")}
</selected_summary>

<local_failures>
{failure_text or "No local failure summaries yet."}
</local_failures>

# Objective
{objective_text}

# Evolutionary Guidelines
1. **Analyze History, Do Not Repeat It:** Identify why the current profile plateaued based on `<selected_summary>` and `<local_failures>`.
2. **High-Level Mutations, No Low-Level Details:** Propose conceptual algorithmic shifts, structural relaxations, or novel search topologies (e.g., introducing a new mathematical constraint or hybridizing optimization frameworks). Do not write code or micromanage hyperparameters.
3. **Strict Separation of Thought and Action:** You must separate your cognitive process from the final directional output using the exact XML tags provided below.

Provide your response exactly in the following format:

<think>
</think>

<guidance>
</guidance>

The following notes explain what each block should contain:
<think>
Use this space entirely for internal reflection. Diagnose historical bottlenecks from the logs, extract lessons from local failures, and debate which conceptual shift is most likely to yield a breakthrough.
</think>
<guidance>
This must contain only your final, actionable evolutionary trajectory.
</guidance>
````

### Guidance Runtime Variables

`failure_text`:

```python
failure_text = "\n\n".join(
    _raw_summary_for_prompt(
        entry,
        fallback="No previous summary is attached.",
        raw_score_label=task_spec.raw_score_label,
    )
    for entry in local_failure_entries
)
```

`best_valid_target` / `objective_text`:

```python
target = task_spec.best_target(selected_node, selected_entry, global_best_entries)
objective_text = task_spec.guidance_objective(target)
```

所以 guidance prompt 的历史内容是 raw model summary + verifier score。objective 文案和
score 方向由 task spec 控制。Erdos 使用 “Lower raw C5 is better”；Polyomino 使用
“Higher FrontierCS score is better”。

注意：`global_best_entries` 可以继续用于 task objective 计算当前目标分数，但 global
best 的 summary 不再 attach 到 guidance prompt。

### Guidance Parsing

`extract_guidance_or_format_error(...)` 的规则：

```text
If raw output has <guidance>...</guidance>:
  submitted guidance = text inside <guidance>
  guidance_format_ok = true
Else if raw output has non-empty text outside <think>...</think>:
  submitted guidance = outside-think text
  guidance_format_ok = false
Else:
  submitted guidance = synthetic formatting-failure guidance
  guidance_format_ok = false
```

如果第一次 generation 解码后为空，agent loop 会 retry 一次，并记录
`guidance_generation_attempts`。

## Execution Model Prompt

Execution model 是冻结的执行器。它把 parsed guidance 和 library context 转成一个具体
可运行的 task-specific candidate。Erdos 输出 Python，Polyomino 输出 C++17。execution tokens 不参与 RL 更新；它的输出只通过 verifier 产生
reward，并写回 library。

实际发送给 execution model 的 chat messages：

```python
[
    {"role": "system", "content": execution_prompt.system},
    {"role": "user", "content": execution_prompt.user},
]
```

其中：

```python
execution_prompt = build_execution_prompt(
    problem_prompt=problem_prompt,
    selected_node=selected_node,
    selected_entry=selected_entry,
    guidance=guidance,
    solution_language=task_spec.solution_language,
    solution_contract=task_spec.execution_solution_contract,
    raw_score_label=task_spec.raw_score_label,
)
```

### Execution System Prompt

```text
You are the execution model. Turn guidance into one concrete runnable {Python|C++17} candidate. Output execution thinking first, then the code block, then the summary.
```

### Execution User Prompt

下面是 `build_execution_prompt(...)` 当前构造的完整 user prompt。`<problem>` 是权威任务
定义，library context 是历史证据，`<guidance>` 是当前要执行的方向。

````text
<problem>
{problem_prompt}
</problem>

The next sections describe the current search state for this problem.

<selected_summary>
{_raw_summary_for_prompt(selected_entry, fallback="No previous summary is attached.")}
</selected_summary>

<guidance>
{guidance}
</guidance>

Use the problem statement as the authoritative task specification.
Use the attached library context as historical evidence, not as code to copy blindly.
Implement one concrete solution that follows the guidance while satisfying the problem specification.
{solution_contract}

Return exactly these three blocks:

<execution_thinking>
Briefly explain how the guidance was translated into the submitted solution.
</execution_thinking>

<solution>
```{python_or_cpp}
{placeholder}
```
</solution>

<summary>
Summarize the solution and its guidance-driven diff from the previous idea in natural language. Explain how the candidate was generated, including the search, refinement, or optimization strategy used, and what specific changes were made based on the guidance. If implementation details are central to the solution, such as parameter tuning, threshold choices, normalization, perturbation design, or constraint handling, highlight them and explain why they matter. Do not include code, hard-coded arrays, copied profile values, or raw candidate parameters.
</summary>
````

### Execution Runtime Variables

因此 execution model 和 guidance model 使用相同 PUCT-selected selected summary
context；execution prompt 只额外给出当前 parsed guidance，不再 attach visible global
best valid summary。

### Execution Parsing

Execution response 预期包含三个 blocks：

```text
<execution_thinking>...</execution_thinking>
<solution>```python ... ```</solution>    # Erdos
<solution>```cpp ... ```</solution>       # Polyomino
<summary>...</summary>
```

解析方式：

- `execution_thinking = extract_tag_or_none(execution_text, "execution_thinking") or ""`
- `solution = _extract_solution_code(execution_text, task_spec=task_spec)`
- `model_summary = extract_tag_or_none(execution_text, "summary")`

`_extract_solution_code(...)` 优先读取 `<solution>...</solution>`，再调用 task-specific
extractor。Erdos 接受 Python fenced code；Polyomino 只接受 `cpp` / `c++` / `C++` fenced code。

Verifier 通过 task spec 对完整 `execution_text` 调用：

```python
verification = task_spec.verify_execution_text(
    execution_text,
    timeout_s=timeout_s,
    config=task_verifier_config,
)
```

Erdos verifier 要求 candidate code 定义 `run()`，并返回：

```python
h_values, c5_bound, n_points
```

valid 时：

```text
raw_score = verified C5
reward = 1.0 / (1e-8 + raw_score)
status = "valid"
artifacts = code, h_values, c5_bound, n_points
```

invalid/parse error/timeout 时：

```text
reward = 0.0
raw_score = None
status = "parse_error" / "invalid" / "timeout"
```

Polyomino verifier 要求 `<solution>` 中存在 C++17 fenced block，并把 C++ code 交给外部
FrontierCS evaluator：

```text
valid:
  raw_score = FrontierCS score
  reward = FrontierCS score
  status = "valid"

invalid:
  reward = 0.0
  raw_score = None
  status = "invalid"

environment unavailable:
  reward = 0.0
  raw_score = None
  status = "environment_error"
```

没有本地 smoke evaluator；FrontierCS 不 vendored 到 `guidance/`。

## Canonical Summary

Execution model 的 `<summary>` 不是最终直接写入 library 的唯一 summary。agent loop 会在
verification 之后调用 `build_execution_summary(...)`，把 model summary、execution thinking、
solution、guidance 和 verifier 真实结果合成 canonical summary。

固定 sections：

```text
Execution Interpretation
Implemented Algorithm
New Ideas Introduced
Empirical Outcome
Failure / Bottleneck Analysis
Next Guidance Delta
```

`Empirical Outcome` 由 verifier 填充：

```text
Verifier status: {verification.status}
{task_spec.raw_score_label}: {verification.raw_score}
Reward: {verification.reward}
Verifier message: {verification.message}
Verified returned profile: n_points=..., c5_bound=..., h length=..., head=[...], tail=[...]
```

如果 model summary 没有可解析 section，原始 `<summary>` 会作为 `New Ideas Introduced`。
如果 verifier invalid，`Failure / Bottleneck Analysis` 会记录真实 failure status/message。

## Library Writeback

每个 rollout 写入一个 `LibraryEntry`：

```python
LibraryEntry(
    id=str(uuid4()),
    parent_id=selected_node.id,
    problem_id=selected_node.problem_id,
    timestep=int(global_step),
    guidance=guidance,
    execution_thinking=execution_thinking,
    solution=solution,
    verifier_reward=verification.reward,
    verifier_raw_score=verification.raw_score,
    verifier_status=verification.status,
    verifier_message=verification.message,
    summary=summary,  # canonical summary returned by build_execution_summary(...)
    reusable_idea=_extract_reusable_idea(summary),
    failure_mode=None if verification.valid else verification.status,
    metadata={
        "group_uid": group_uid,
        "selected_node_id": selected_node.id,
        "guidance_format_ok": guidance_format_ok,
        "guidance_generation_attempts": guidance_generation.attempts,
        "raw_guidance_with_specials": guidance_generation.raw_text,
        "guidance_stop_reason": guidance_generation.stop_reason,
        "raw_guidance_text": guidance_text,
        "guidance_prompt": {"system": guidance_prompt.system, "user": guidance_prompt.user},
        "execution_prompt": {"system": execution_prompt.system, "user": execution_prompt.user},
        "task": task_config,
        "execution_text": execution_text,
        "raw_model_summary": raw_model_summary,
        "execution_provider": self.execution_llm_config.get("provider", "mock"),
        "execution_model": self.execution_llm_config.get("model", "mock-exec"),
        "execution_response_metadata": execution_response_metadata,
        "execution_response_usage": execution_response_usage,
        "verification_artifacts": verification.artifacts,
        "execution_fallback_used": execution_result.fallback_used,
        "execution_fallback_reason": execution_result.fallback_reason,
        "original_execution_text": execution_result.original_execution_text,
    },
)
```

同时写入一个 child `LibraryNode`：

```python
LibraryNode(
    problem_id=entry.problem_id,
    timestep=entry.timestep,
    entry_id=entry.id,
    value=entry.verifier_reward,
    raw_score=entry.verifier_raw_score,
    visits=0,
    parent_id=selected_node.id,
    children=[],
    metadata={"verifier_status": entry.verifier_status},
)
```

`submit_child(...)` 在 group 完成后更新 PUCT stats，并可能过滤 archive：

```text
puct_n: visit counts
puct_m: best reachable values
puct_T: total completed group visits
```

## Training Signal

Guidance model 是唯一被训练的模型：

```text
prompt_ids = guidance system + guidance user
response_ids = guidance model response tokens
response_mask = [1] * len(response_ids)
reward_score = verification.reward
```

Execution model、verifier、summary canonicalizer 都只提供 environment feedback。execution
失败不会被固定 fallback solution 替换；失败会以 reward `0.0` 和对应 status 写入 library，
成为后续 guidance prompt 的历史信号。
