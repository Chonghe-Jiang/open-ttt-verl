# Guidance-TTT Prompt Framework

This document summarizes 当前 `guidance-ttt` 中 guidance model 和 execution model
的 prompt 构造方式。代码来源以当前仓库为准：

- `guidance_ttt/prompts.py`: prompt 模板、entry summary 压缩、known-good Erdos template
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
  -> execution model outputs <execution_thinking> + Python code + <summary>

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

- guidance prompt 看“决策摘要”：压缩 summary、reward、raw score、verifier status、
  short guidance、reusable idea、global best、local failures。
- execution prompt 看“可执行上下文”：selected entry 的更长 canonical summary、verified
  profile artifacts、possible solution code excerpt、global best valid solution、current
  guidance、known-good Erdos implementation skeleton。

## Entry Summary Construction

`_entry_summary(entry, include_solution=...)` 被同时用于 guidance prompt 和 execution
prompt。

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

`include_solution=False` 时用于 guidance prompt：

- summary 会通过 `_summary_for_guidance()` 去掉大段 code block，并提取实现 facts。
- clip 更短，避免 actor prompt 被历史代码占满。
- long profile array 会被 compact。

`include_solution=True` 时用于 execution prompt：

- summary clip 更长。
- previous guidance / verifier message / reusable idea clip 更长。
- 如果 `entry.summary` 中没有 fenced Python code，并且 `entry.solution` 存在，会附加：

````text
Previous solution code excerpt (backward-compatible):
```python
{entry.solution clipped to 1800 chars}
```
````

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

<selected_library_node>
Node id: {selected_node.id}
Timestep: {selected_node.timestep}
Value: {selected_node.value}
Raw score: {selected_node.raw_score}
Visits: {selected_node.visits}
{_entry_summary(selected_entry, include_solution=False)}
</selected_library_node>

<global_best>
{compressed global best summaries}
</global_best>

<local_failures>
{compressed local failure summaries}
</local_failures>
```

`global_best` 中如果 selected entry 本身就是当前 visible global best，会写：

```text
The selected library node is also the current global best visible entry; use its summary above.
```

### Complete Guidance User Prompt Template

下面是 `build_guidance_prompt()` 当前构造的完整 user prompt。`{...}` 是运行时
Python f-string 插入的变量。

````text
<problem>
{problem_prompt}
</problem>

<selected_library_node>
Node id: {selected_node.id}
Timestep: {selected_node.timestep}
Value: {selected_node.value}
Raw score: {selected_node.raw_score}
Visits: {selected_node.visits}
{_entry_summary(selected_entry, include_solution=False)}
</selected_library_node>

<global_best>
{best_text or "No global best entry yet."}
</global_best>

<local_failures>
{failure_text or "No local failure entries yet."}
</local_failures>

# Objective
Your task is to provide the next **evolutionary guidance** to beat the current best valid raw score ({best_valid_target}). Lower raw C5 is better.

# Evolutionary Guidelines
1. **Analyze History, Do Not Repeat It:** Identify why the current profile plateaued based on `<selected_library_node>` and `<local_failures>`.
2. **High-Level Mutations, No Low-Level Details:** Propose conceptual algorithmic shifts, structural relaxations, or novel search topologies (e.g., introducing a new mathematical constraint or hybridizing optimization frameworks). Do not write code or micromanage hyperparameters.
3. **Strict Separation of Thought and Action:** You must separate your cognitive process from the final directional output using the exact XML tags provided below.
    * **The `<think>` block:** Use this space entirely for internal reflection. Diagnose historical bottlenecks from the logs, extract lessons from local failures, and debate which conceptual shift is most likely to yield a breakthrough.
    * **The `<guidance>` block:** This must contain only your final, actionable evolutionary trajectory. It should clearly outline:
        - The **Evolutionary Mutation**: The new structural or mathematical property being explored.
        - The **Directional Search Strategy**: The high-level algorithmic mechanism to execute the mutation.
        - The **Progress Target**: The explicit structural change that indicates successful mutation from {selected_node.raw_score} towards {best_valid_target} or lower.

Provide your response exactly in the following format:

<think>
</think>

<guidance>
</guidance>
````

### Guidance Objective

当前 guidance prompt 强调：

- 目标是给出下一步 evolutionary guidance，beat 当前 best valid raw score。
- guidance 要基于 `<selected_library_node>` 和 `<local_failures>` 分析历史瓶颈，避免重复失败轨迹。
- guidance 只提高层 conceptual / structural / mathematical mutation，不写代码，不 micromanage hyperparameters。
- 输出必须严格分成 `<think>` 和 `<guidance>` 两块。
- `<think>` 用于内部历史诊断和方案权衡。
- `<guidance>` 只保留最终 actionable evolutionary trajectory，并明确：
  - Evolutionary Mutation
  - Directional Search Strategy
  - Progress Target

### Important Runtime Targets

`best_valid_target` 的来源：

- 如果 selected/global context 中有 valid entry，使用 visible best valid raw C5。
- 如果没有 valid entry，且当前 raw score 为空或弱于 `KNOWN_GOOD_ERDOS_RAW_C5`，则使用
  `KNOWN_GOOD_ERDOS_RAW_C5 = 0.3810181186942784`。
- 这个 target 只作为 guidance objective 中的分数目标；当前 guidance user prompt 不再
  attach 63-point full profile 或低层优化 schedule。

### Required Guidance Output Format

guidance model 必须输出 `<think>` 和 `<guidance>` 两个 XML blocks：

```text
<think>
</think>

<guidance>
</guidance>
```

`<think>` 是内部历史诊断和方案权衡；`<guidance>` 只包含最终 actionable evolutionary
trajectory，应该清楚覆盖 Evolutionary Mutation、Directional Search Strategy、Progress
Target。

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

### System Prompt

```text
You are the execution model. Turn guidance into one concrete runnable Python candidate. Output execution thinking first, then the code block, then the summary.
```

### Attached Inputs

execution user prompt attach：

````text
<problem>
{ERDOS_PROBLEM_PROMPT}
</problem>

<selected_library_node>
Node id: {selected_node.id}
Timestep: {selected_node.timestep}
Value: {selected_node.value}
Raw score: {selected_node.raw_score}
{_entry_summary(selected_entry, include_solution=True)}
</selected_library_node>

<global_best_valid_solution>
{_entry_summary(best_valid_entry, include_solution=True) or "No global best valid entry yet."}
</global_best_valid_solution>

<guidance>
{parsed guidance}
</guidance>
````

然后 prompt 给出 execution constraints、implementation direction 和 known-good template。

### Execution Objective

execution prompt 要求：

- 目标是 produce valid candidate with raw C5 lower than target。
- constant `h[i] = 0.5` 只能当 baseline，不能 unchanged 返回。
- previous solution code 是 reference point to beat，不是 copy target。
- 如果 current best 已接近 target，不能只 copy 或 rerun exact same SLSQP。
- 必须 implement deterministic search，evaluate candidates，再让 verifier 判分。
- 每次 perturbation 后显式 project/repair，保证：
  - `0 <= h[i] <= 1`
  - `sum(h) == n_points / 2`
  - final `c5_bound` 由 final h 重新计算
- guidance 或 previous summary 提到 plateau 时，至少实现两个 candidate families：
  - smaller multi-scale coordinate/pair sweep
  - active-lag/top-contributor mass-transfer sweep
- inherited profile raw C5 `<= 0.38103` 时：
  - preserve 63-point incumbent
  - first try active-lag transfers
  - if no strict improvement, try deterministic finite-difference Adam / smooth-max escape
- 默认 preserve inherited `n_points`；known-good template 对应 `n_points=63`。
- 输出 summary 必须说明是否 beat inherited raw C5，以及哪个 perturbation family 贡献最好。

### Known-Good Template

execution prompt 嵌入：

````text
<known_good_minimax_template>
```python
{ERDOS_MINIMAX_EXECUTION_TEMPLATE}
```
</known_good_minimax_template>
````

当前 template 是 verifier-compatible scoring/repair reference，核心包括：

- `project_to_box_sum(h, target)`
- `c5_score(h) = max(np.correlate(h, 1-h, mode="full") * (2.0 / n))`
- 63-point known-good initialization
- optional SLSQP bounded local refinement
- final return:

```python
return [float(x) for x in best_h], float(c5_bound), int(n_points)
```

### Required Execution Output Format

execution model 必须按顺序输出三块：

````text
<execution_thinking>
brief reasoning
</execution_thinking>

```python
def run(seed=42, budget_s=1, **kwargs):
    # final runnable solution
```

<summary>
Execution Interpretation
...

Implemented Algorithm
...

New Ideas Introduced
...

Empirical Outcome
Pending verifier execution.

Failure / Bottleneck Analysis
...

Next Guidance Delta
If no strict improvement was found, name the exact perturbation families and delta
scales that failed, then propose at least two changed knobs for the next step. Do not
recommend copying the same profile unchanged.
</summary>
````

注意：execution model 的 `<summary>` 不能声称 verifier success，因为 verifier 还没运行。

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
known-good template。

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

- 不会自动替换为 known-good template。
- failure 会作为真实 environment outcome 写入 library。
- reward 为 `0.0`。
- `execution_fallback_used` 固定为 `false`。

这保证失败本身也成为 guidance actor 的训练信号。
