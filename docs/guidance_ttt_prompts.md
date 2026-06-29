# Guidance-TTT Prompt Framework

This document summarizes 当前 `guidance-ttt` 中 guidance model 和 execution model
的 prompt 构造方式。代码来源以当前仓库为准：

- `guidance_ttt/prompts.py`: prompt 模板、entry summary 压缩、root initial construction facts
- `guidance_ttt/agent_loop.py`: prompt 构造、模型调用、verification、library 写回
- `guidance_ttt/library.py`: PUCT 选 node、取 selected/global/failure context
- `guidance_ttt/tasks/erdos.py`: Erdos problem prompt

## One Rollout Prompt Flow

每个 rollout 有两个模型调用：

```text
GuidanceLibrary.acquire_group(...)
  -> selected_node

GuidanceLibrary.context_for_node(selected_node, visible_timestep_exclusive=global_step)
  -> selected_entry
  -> global_best_entries
  -> local_failure_entries

build_guidance_prompt(...)
  -> guidance actor prompt
  -> guidance actor outputs <guidance>...</guidance>

build_execution_prompt(...)
  -> execution model prompt
  -> execution model outputs <execution_thinking> + <solution> + <summary>

verify_erdos_solution_text(execution_text)
  -> verifier status / raw C5 / reward / artifacts

build_execution_summary(...)
  -> canonical summary with verifier result

LibraryEntry + LibraryNode written into library.json
```

重要隔离规则：

```text
visible_timestep_exclusive = global_step
```

因此 prompt 只看 `timestep < global_step` 的 library entries。同一个 training step
内刚产生的其它 rollout 不会泄漏进当前 prompt。

## Shared Library Context

guidance model 和 execution model 的 library context 来自同一个 selected node，但
attach 的粒度不同。

共同来源：

```python
selected_node = library.acquire_group(...)
context = library.context_for_node(selected_node, ...)
selected_entry = context["selected_entry"]
global_best_entries = context["global_best_entries"]
local_failure_entries = context["local_failure_entries"]
```

差异：

- guidance prompt 看 summary-only context：selected/global/failure 部分只 attach
  `entry.summary` 经 `_summary_for_guidance()` 压缩后的文本，不 attach node id、visits、
  reward、raw score、verifier status/message、verified artifacts、previous guidance、
  reusable idea、failure mode 或 root initial construction facts。
- execution prompt 看“可执行上下文”：root initial construction facts、selected entry 的更长
  canonical summary、verified profile artifacts、global best valid entry、current guidance。

## Entry Summary Construction

`_summary_only_for_guidance(entry)` 用于 guidance prompt；`_entry_summary(entry,
include_solution=...)` 用于 execution prompt。

guidance prompt 没有 previous entry 时：

```text
No previous summary is attached.
```

guidance prompt 有 entry 时，只输出：

```text
{clipped _summary_for_guidance(entry.summary)}
```

因此 guidance 的 selected/global/failure library context 不包含 entry id、reward、raw
score、verifier 状态、verified profile artifacts、previous guidance、reusable idea 或
failure mode。`_summary_for_guidance()` 仍会移除 summary 内的大段 fenced code block，并
提取 compact attached-code facts。

execution prompt 使用 `_entry_summary(entry, include_solution=True)`。

没有 previous entry 时：

```text
No previous library entry is attached.
```

有 entry 时，基础字段包括：

```text
Entry id: {entry.id}
Reward: {entry.verifier_reward}
Raw score: {entry.verifier_raw_score}
Verifier status: {entry.verifier_status}
Verifier message: {clipped verifier_message}
Verified returned profile artifacts ...   # valid entry 且 verifier artifacts 中有 h_values 时
Canonical summary:
{clipped / compressed summary}
Previous guidance: {clipped / compacted guidance}
Reusable idea: {clipped reusable_idea}
Failure mode: {entry.failure_mode, if present}
```

root node 如果带有初始构造 metadata，还会通过 `_root_initial_construction_facts(...)`
注入 execution prompt：

```text
Current initial construction (reference state to improve): initialization=..., n_points=..., raw C5=..., c5_bound=..., h=... / h length=...
```

`include_solution=True` 时用于 execution prompt：

- summary clip 更长。
- previous guidance / verifier message / reusable idea clip 更长。
- 当前版本不再额外附加 `entry.solution` 的 backward-compatible code excerpt；可执行上下文主要来自
  canonical summary、verified artifacts、root initial construction 和 global best valid entry。

## Guidance Model Prompt

### Role

guidance model 是被 RL 训练的 actor。它不直接写最终 Python solution，而是输出下一步
搜索/修改策略。verl 的 reward 最终只作用在 guidance model response tokens 上。

### System Prompt

```text
You are the Guidance Model, acting as a strategic navigator for an open-ended scientific discovery process.

Your primary objective is to provide **evolutionary guidance**. Do not write final code or focus on low-level implementation details. Instead, your task is to propose high-level directional shifts, conceptual mutations, and novel pathways to explore the search space.

Focus on how the current ideas can *evolve* to escape local optima and discover fundamentally new mechanisms.
```

### Attached Inputs

guidance user prompt attach：

```text
<problem>
{ERDOS_PROBLEM_PROMPT}
</problem>

<selected_summary>
{_summary_only_for_guidance(selected_entry)}
</selected_summary>

<global_best>
{summary-only global best context}
</global_best>

<local_failures>
{summary-only local failure context}
</local_failures>
```

`global_best` 中如果 selected entry 本身就是当前 visible global best，会写：

```text
The selected summary is also the current global best visible summary.
```

### Complete Guidance User Prompt Template

下面是 `build_guidance_prompt()` 当前构造的完整 user prompt。`{...}` 是运行时
Python f-string 插入的变量。

````text
<problem>
{problem_prompt}
</problem>

The next sections describe the current search state for this problem. Use them
as run-local context when deciding the next step.

<selected_summary>
{_summary_only_for_guidance(selected_entry)}
</selected_summary>

<global_best>
{best_text or "No global best summary yet."}
</global_best>

<local_failures>
{failure_text or "No local failure summaries yet."}
</local_failures>

# Objective
Your task is to provide the next **evolutionary guidance** to beat the current visible target raw score ({best_valid_target}). Lower raw C5 is better.

# Evolutionary Guidelines
1. **Analyze History, Do Not Repeat It:** Identify why the current profile plateaued based on `<selected_summary>` and `<local_failures>`.
2. **High-Level Mutations, No Low-Level Details:** Propose conceptual algorithmic shifts, structural relaxations, or novel search topologies (e.g., introducing a new mathematical constraint or hybridizing optimization frameworks). Do not write code or micromanage hyperparameters.
3. **Strict Separation of Thought and Action:** You must separate your cognitive process from the final directional output using the exact XML tags provided below. 
<think>
Use this space entirely for internal reflection. Diagnose historical bottlenecks from the logs, extract lessons from local failures, and debate which conceptual shift is most likely to yield a breakthrough.
</think>
<guidance>
This must contain only your final, actionable evolutionary trajectory.
</guidance>

Provide your response exactly in the following format:

<think>
</think>

<guidance>
</guidance>
````

### Guidance Objective

当前 guidance prompt 强调：

- 目标是给出下一步 evolutionary guidance，beat 当前 visible target raw score。
- guidance 要基于 `<selected_summary>` 和 `<local_failures>` 分析历史瓶颈，避免重复失败轨迹。
- guidance 只提高层 conceptual / structural / mathematical mutation，不写代码，不 micromanage hyperparameters。
- 输出必须严格分成 `<think>` 和 `<guidance>` 两块。
- `<think>` 用于内部历史诊断和方案权衡。
- `<guidance>` 只保留最终 actionable evolutionary trajectory。

### Important Runtime Targets

`best_valid_target` 的来源：

- 如果 selected/global context 中有 valid entry，使用 visible best valid raw C5。
- 如果没有 valid entry，则使用 selected node 的 raw score。
- 这个 target 只作为 guidance objective 中的分数目标；当前 guidance user prompt 不再
  attach hardcoded Erdos seed、63-point full profile 或低层优化 schedule。

### Required Guidance Output Format

guidance model 必须输出 `<think>` 和 `<guidance>` 两个 XML blocks：

```text
<think>
</think>

<guidance>
</guidance>
```

`<think>` 是内部历史诊断和方案权衡；`<guidance>` 只包含最终 actionable evolutionary
trajectory。

### Guidance Parsing Rule

`extract_guidance_or_format_error()` 的规则：

```text
If raw output has <guidance>...</guidance>:
  submitted guidance = text inside that tag
  guidance_format_ok = true
Else if raw output has text outside <think>...</think>:
  submitted guidance = outside-think text
  guidance_format_ok = false
Else:
  submitted guidance = synthetic formatting-failure guidance
  guidance_format_ok = false
```

如果第一次 generation 解码后为空，agent loop 会 retry 一次，并记录
`guidance_generation_attempts`。

## Execution Model Prompt

### Role

execution model 是冻结的执行器。它把 guidance 转成一个具体可运行的 Python candidate。
它的 tokens 不参与 RL 更新；它的输出只通过 verifier 产生 reward，并写回 library。

### Complete Execution Chat Prompt

`GuidanceExecutionAgentLoop.run(...)` 调用 execution model 时，实际发送的是下面两条
chat messages：

```python
[
    {"role": "system", "content": execution_prompt.system},
    {"role": "user", "content": execution_prompt.user},
]
```

其中 `execution_prompt = build_execution_prompt(...)`。

### Full Execution System Prompt

这是当前代码实际发送给 execution model 的完整 system prompt：

```text
You are the execution model. Turn guidance into one concrete runnable Python candidate. Output execution thinking first, then the code block, then the summary.
```

### Full Execution User Prompt

下面是当前代码实际发送给 execution model 的完整 user prompt。它是一个薄 wrapper：
`<problem>` 是权威任务定义，`<guidance>` 给出方向，library context 只提供历史证据。
wrapper 本身不再写 Erdos/C5/SLSQP/project 等任务专用实现策略。

`{...}` 是运行时由 Python f-string 插入的变量。

````text
<problem>
{problem_prompt}
</problem>

<selected_library_node>
Node id: {selected_node.id}
Timestep: {selected_node.timestep}
Value: {selected_node.value}
Raw score: {selected_node.raw_score}
{initial_construction_text}
{_entry_summary(selected_entry, include_solution=True)}
</selected_library_node>

<global_best_valid_entry>
{best_valid_text}
</global_best_valid_entry>

<guidance>
{guidance}
</guidance>

Use the problem statement as the authoritative task specification.
Use the attached library context as historical evidence, not as code to copy blindly.
Implement one concrete solution that follows the guidance while satisfying the problem specification.

Return exactly these three blocks:

<execution_thinking>
Briefly explain how the guidance was translated into the submitted solution.
</execution_thinking>

<solution>
```python
# complete executable solution required by the problem
```
</solution>

<summary>
Use natural language to summarize the overall idea and method of the solution. Explain how the candidate was generated, including the search, refinement, or optimization strategy used. If the solution’s main contribution lies in specific implementation details, such as parameter fine-tuning, threshold adjustment, normalization choices, perturbation design, or constraint-handling tricks, explicitly emphasize those details and explain why they matter. Do not include code, hard-coded arrays, copied profile values, or raw candidate parameters.
</summary>
````

注意：execution model 的 `<summary>` 只记录 execution model 自己的解释。verifier 真实结果会在
后续 canonical summary 阶段写回。

### Execution Prompt Runtime Variables

这些变量在 `build_execution_prompt(...)` 内部生成：

```python
best_valid = _best_valid_entry(global_best_entries or [], selected_entry)
best_valid_text = (
    _entry_summary(best_valid, include_solution=True)
    if best_valid
    else "No global best valid entry yet."
)
initial_construction_text = _root_initial_construction_facts(selected_node)
```

所以 execution model 看到的 library 内容和 guidance model 看到的 library 内容来自同一次
PUCT selection；区别是 execution prompt 会 attach 更长的 selected entry summary，并额外
attach 当前 visible best valid entry。

## Canonical Summary After Verification

execution model 输出的 `<summary>` 不是最终直接写入 library 的唯一依据。

agent loop 会在 verifier 之后调用 `build_execution_summary(...)`，生成 canonical summary。
固定 section：

```text
Execution Interpretation
Implemented Algorithm
New Ideas Introduced
Empirical Outcome
Failure / Bottleneck Analysis
Next Guidance Delta
```

其中 `Empirical Outcome` 由 verifier 真实结果填充：

```text
Verifier status: {verification.status}
Raw C5: {verification.raw_score}
Reward: {verification.reward}
Verifier message: {verification.message}
Verified returned profile: n_points=..., c5_bound=..., h length=..., head=[...], tail=[...]
```

如果 verifier invalid，则 canonical summary 会记录 failure status/message；不会自动替换成
固定 fallback solution。

## Library Writeback Metadata

每个 rollout 写入一个 `LibraryEntry`：

```python
LibraryEntry(
    parent_id=selected_node.id,
    timestep=global_step,
    guidance=guidance,
    execution_thinking=execution_thinking,
    solution=solution,
    verifier_reward=verification.reward,
    verifier_raw_score=verification.raw_score,
    verifier_status=verification.status,
    verifier_message=verification.message,
    summary=canonical_summary,
    reusable_idea=extract_from_summary(summary),
    failure_mode=None if valid else verification.status,
    metadata={
        "guidance_prompt": {"system": ..., "user": ...},
        "execution_prompt": {"system": ..., "user": ...},
        "raw_guidance_text": ...,
        "raw_guidance_with_specials": ...,
        "guidance_format_ok": ...,
        "execution_text": ...,
        "execution_provider": ...,
        "execution_model": ...,
        "execution_response_metadata": ...,
        "execution_response_usage": ...,
        "verification_artifacts": ...,
        "execution_fallback_used": false,
        "execution_fallback_reason": null,
        "original_execution_text": ...,
    },
)
```

同时写入一个 lightweight `LibraryNode`：

```python
LibraryNode(
    entry_id=entry.id,
    value=verification.reward,
    raw_score=verification.raw_score,
    parent_id=selected_node.id,
    metadata={"verifier_status": verification.status},
)
```

之后的 prompt 主要从 `LibraryEntry.summary`、`verification_artifacts`、
`guidance`、`solution` 中抽取上下文；PUCT 选择主要看 `LibraryNode.value`、
`raw_score`、`puct_n`、`puct_m`、`puct_T`。

## No Automatic Execution Fallback

如果 execution model call 失败、输出为空、没有 Python code、或者 verifier invalid：

- 不会自动替换为固定 fallback solution。
- failure 会作为真实 environment outcome 写入 library。
- reward 为 `0.0`。
- `execution_fallback_used` 固定为 `false`。

这保证失败本身也成为 guidance actor 的训练信号。
