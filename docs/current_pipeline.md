# Guidance-TTT 当前流水线说明

本文档描述当前 `guidance-ttt` 的 Erdos Guidance-TTT 训练流水线，重点覆盖：

- 旧长跑任务 `67845` 的运行方式和状态解释
- 新提交的 summary-canonical unlimited 版本任务 `80637`
- actor、execution model、verifier、library 之间的数据流
- prompt 构建逻辑和当前 prompt 模板原文
- 新版 `summary` 作为 library canonical 字段的写回规则

代码对应仓库：

```text
/work/mit/ppliang_mit/lsy/guidance-ttt
```

旧长跑配置：

```text
guidance_ttt/config/erdos_gpt_oss_20b_8gpu_long.yaml
```

新版 summary-canonical 配置：

```text
guidance_ttt/config/erdos_gpt_oss_20b_8gpu_summary_canonical.yaml
```

## 当前任务状态

截至最近一次检查：

- 旧任务 `67845` 仍在运行，作业名 `guidance_ttt_gptoss_8gpu`
- 新任务 `80637` 已提交并开始运行，作业名 `guidance_ttt_summary_unlimited`
- 两个任务使用不同输出目录，不会同时写同一个 `library.json`

旧任务输出目录：

```text
outputs/guidance_ttt/erdos_gpt_oss_20b_8gpu_long
```

新版任务输出目录：

```text
outputs/guidance_ttt/erdos_gpt_oss_20b_8gpu_summary_canonical_unlimited
```

旧任务日志：

```text
/work/mit/ppliang_mit/lsy/logs/guidance_ttt_gptoss_8gpu-67845.out
/work/mit/ppliang_mit/lsy/logs/guidance_ttt_gptoss_8gpu-67845.err
```

新版任务日志：

```text
/work/mit/ppliang_mit/lsy/logs/guidance_ttt_summary_unlimited-80637.out
/work/mit/ppliang_mit/lsy/logs/guidance_ttt_summary_unlimited-80637.err
```

监控命令：

```bash
squeue -j 67845,80637
```

## 高层角色

这套系统有四个核心角色。

1. guidance actor

   被训练的模型，当前是：

   ```text
   /work/mit/ppliang_mit/lsy/models/Qwen3-8B
   ```

   它不直接输出最终 Python 解法，而是输出 `<guidance>...</guidance>` 形式的搜索建议。verl 只训练这个 actor 的输出 token。

2. execution model

   冻结的执行模型，当前是：

   ```text
   models/gpt-oss-20b
   ```

   它读取 actor 的 guidance、历史 library context、当前 best/failure 信息，然后生成可运行 Python。

3. verifier

   沙箱执行 execution model 生成的 Python，检查返回值是否合法，并计算 raw C5。raw C5 越低越好，训练 reward 是 `1 / (1e-8 + raw_C5)`。

4. library

   JSON 持久化搜索状态，保存每次 rollout 的 prompt、guidance、execution output、solution、verifier result、summary，以及 PUCT tree node。

## 训练配置

长跑和 summary-canonical 新任务使用同一组主要训练参数，只是输出目录和实验名不同。

关键参数：

- `groups_per_batch: 8`
- `group_size: 32`
- 每个训练 step 共 `8 * 32 = 256` 个 rollout
- actor response length: `run.max_response_length: 1024`
- execution model max tokens: `llm.execution.max_tokens: null`
- execution vLLM max model length: `llm.execution.max_model_len: 32768`
- execution vLLM tensor parallelism: `8`
- execution vLLM GPU memory utilization: `0.35`
- execution vLLM max concurrent sequences: `32`

actor rollout 本身保持较短上下文：

```text
actor_rollout_ref.rollout.max_model_len=4096
actor_rollout_ref.rollout.max_num_seqs=8
actor_rollout_ref.rollout.max_num_batched_tokens=4096
```

execution model 使用更长上下文，原因是 execution prompt 会嵌入 selected node、global best、summary、可选历史代码片段和 known-good scoring skeleton。新版 unlimited 配置不再设置固定 completion token 上限；本地 vLLM client 会根据 tokenizer 计算 prompt tokens，并把 `max_model_len - prompt_tokens` 作为生成上限。因此它仍受模型 context window 限制，但不再受人为的 `26000` 输出长度限制。

## 准备阶段

入口：

```bash
python -m guidance_ttt.main_erdos \
  --config guidance_ttt/config/erdos_gpt_oss_20b_8gpu_summary_canonical.yaml
```

`guidance_ttt.main_erdos.prepare_run()` 会创建 run directory，并写出三类关键文件：

- `library.json`: 持久化 PUCT/search library
- `ttt_slots.parquet`: 训练数据槽位，每个 batch slot 一行
- `agent_loop.yaml`: verl agent-loop 配置，包括 execution LLM 参数

如果 `library.json` 不存在，会用 Erdos baseline root node 初始化。baseline 是常数 `h = 0.5` profile，raw C5 由 verifier 公式计算，root reward 为：

```python
1.0 / (1e-8 + raw_c5)
```

`ttt_slots.parquet` 当前有 `groups_per_batch = 8` 行。每行带 `extra_info`：

- `library_path`
- `rollout_n`
- `group_size`
- `puct_c`
- `slot_id`

这些字段在 rollout 时传入 `GuidanceExecutionAgentLoop.run()`。

## Verl Trainer 设置

`main_erdos.py` 会把 recipe YAML 转换成 hydra overrides。关键 overrides：

```text
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
actor_rollout_ref.actor.policy_loss.loss_mode=ttt_reinforce_is
actor_rollout_ref.rollout.n=32
actor_rollout_ref.rollout.agent.default_agent_loop=guidance_execution_erdos
actor_rollout_ref.rollout.agent.agent_loop_config_path=<run>/agent_loop.yaml
```

`guidance_ttt.verl_ext` 注册自定义 `ttt_reinforce_is` policy loss。rollout engine 会计算 actor token 的 logprob，训练时用 importance-sampling corrected token rewards。

## 一个训练 Step 内发生什么

每个训练 step 有 8 个 batch slots。每个 slot 采样 32 个 actor responses，所以一个 step 总共 256 个 rollouts。

每个 rollout 执行 `GuidanceExecutionAgentLoop.run()`：

1. 从 `extra_info` 读取 `library_path`、`rollout_n`、`puct_c`。
2. 构造 `group_uid = <global_step>:<slot_uid>`。
3. 打开 `library.json`。
4. 调用 `GuidanceLibrary.acquire_group(group_uid, visible_timestep_exclusive=global_step)`。
5. 选中一个 parent node；同一 group 的 32 个 rollouts 共用这个 parent node。
6. 读取 selected entry、global best entries、local failure entries。
7. 调用 `build_guidance_prompt(...)` 构造 actor prompt。
8. actor 生成 guidance tokens。
9. 用 `extract_guidance_or_format_error(...)` 提取 `<guidance>...</guidance>`。
10. 调用 `build_execution_prompt(...)` 构造 execution prompt。
11. execution model 生成 `<execution_thinking>`、Python code、`<summary>`。
12. verifier 提取 Python code 并沙箱执行。
13. `build_execution_summary(...)` 把 thinking、code、model summary、verifier result 合成 canonical summary。
14. 写入 `LibraryEntry` 和 child `LibraryNode`。
15. 返回 verifier reward 给 verl，用来训练 actor。

重要隔离规则：

```text
visible_timestep_exclusive = global_step
```

构建 prompt 时只看 `timestep < global_step` 的 library entries。这样同一个训练 step 内的其他 rollout 不会泄漏到当前 prompt。

## Group 语义

一个 group 是一个 selected parent node 加最多 `group_size` 个 children。当前 `group_size=32`。

第一次 rollout 到达某个 slot 时：

1. `acquire_group()` 用 PUCT 选择 parent node。
2. parent node 的 `visits` 加 1。
3. `groups[group_uid]` 记录 selected node。

同一个 slot 的后续 31 个 rollout 复用这个 selected parent node。每次 child 写回时：

1. `group["submitted"] += 1`
2. child node 加到 parent 的 `children`
3. 若 `submitted >= 32`，group 标记为 `finalized`

所以一个完整 step 通常写入 256 个 entries。

## Guidance Prompt 构建

位置：

```text
guidance_ttt/prompts.py::build_guidance_prompt
```

用途：

- 让 actor 输出高层 guidance，不输出最终代码
- actor 的输出 token 是唯一进入训练梯度路径的部分
- prompt 会包含 problem、selected/global/failure raw summary-only context、目标 raw score、
  输出格式和推荐搜索策略
- selected/global/failure context 只来自 raw `entry.summary`；不做 clipping、
  code removal、fact extraction 或 profile compaction；不 attach node id、visits、reward、raw score、verifier 状态、
  verified artifacts、previous guidance、reusable idea、failure mode 或 root 初始构造 facts

### Guidance System Prompt 原文

```text
You are the Guidance Model, acting as a strategic navigator for an open-ended scientific discovery process.

Your primary objective is to provide **evolutionary guidance**. Do not write final code or focus on low-level implementation details. Instead, your task is to propose high-level directional shifts, conceptual mutations, and novel pathways to explore the search space.

Focus on how the current ideas can *evolve* to escape local optima and discover fundamentally new mechanisms.
```

### Guidance User Prompt 模板原文

下面是当前 `build_guidance_prompt(...)` 中 user prompt 的原文模板。`{...}` 是运行时插入的 Python f-string 变量。

````text
<problem>
{problem_prompt}
</problem>

The next sections describe the current search state for this problem. Use them
as run-local context when deciding the next step.

<selected_summary>
{_raw_summary_for_prompt(selected_entry, fallback="No previous summary is attached.")}
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

### Guidance 输出提取规则

位置：

```text
guidance_ttt/prompts.py::extract_guidance_or_format_error
```

规则：

1. 如果输出有 `<guidance>...</guidance>`，使用标签内文本，`guidance_format_ok=True`。
2. 如果没有 `<guidance>`，删除 `<think>...</think>` 内文本后，使用剩余文本，`guidance_format_ok=False`。
3. 如果只输出了 thinking，没有可用文本，则合成 formatting failure guidance。

这保证 actor 即使格式不完美，也能产生一个可训练的 rollout；但 metadata 会记录格式失败。

## Raw Summary Context 构建

位置：

```text
guidance_ttt/prompts.py::_raw_summary_for_prompt
```

guidance 和 execution prompt 行为：

- `entry is None` 时输出 `No previous summary is attached.`
- `entry.summary is None` 时输出 fallback 文本
- `entry.summary == ""` 时输出 fallback 文本
- 有 entry 且 summary 非空时直接输出 raw `entry.summary`
- 不调用 `.strip()`、`_clip()`、code block removal、implementation fact extraction 或 profile compaction
- 不输出 `Entry id`、reward、raw score、verifier status/message、verified artifacts、
  previous guidance、reusable idea 或 failure mode

execution prompt 额外 attach parsed `<guidance>`，并 attach visible best valid entry 的
raw summary；不再 attach selected node metadata、root initial construction facts 或
verified artifacts。

## Execution Prompt 构建

位置：

```text
guidance_ttt/prompts.py::build_execution_prompt
```

用途：

- 给冻结 execution model 一个完整任务上下文
- 强制输出顺序为 thinking、solution code、summary
- 具体 code interface 由 `<problem>{problem_prompt}</problem>` 定义
- 要求模型输出自然语言 `<summary>`；verifier 后再规范化为 canonical sections 写回 library

### Execution System Prompt 原文

```text
You are the execution model. Turn guidance into one concrete runnable Python candidate. Output execution thinking first, then the code block, then the summary.
```

### Execution User Prompt 模板原文

下面是当前 `build_execution_prompt(...)` 中 user prompt 的原文模板。`{...}` 是运行时插入的 Python f-string 变量。

````text
<problem>
{problem_prompt}
</problem>

The next sections describe the current search state for this problem. Use them
as run-local context when implementing the guided candidate.

<selected_summary>
{_raw_summary_for_prompt(selected_entry, fallback="No previous summary is attached.")}
</selected_summary>

<global_best_valid_summary>
{best_valid_text}
</global_best_valid_summary>

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

## Execution 输出合同

新版 execution model 必须按顺序输出：

1. `<execution_thinking>...</execution_thinking>`
2. `<solution>...</solution>`，内部包含一个 fenced Python code block，code 实现 `<problem>` 要求的完整 solution
3. `<summary>...</summary>`

也就是：

````text
<execution_thinking>
...
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

注意：execution model 自己不能声称 verifier success，因为 verifier 尚未运行。真正的 verifier outcome 会在代码验证后由 `build_execution_summary(...)` 覆盖或填入。

## Summary Canonical 写回

位置：

```text
guidance_ttt/agent_loop.py::build_execution_summary
```

新版 library 写回把 `summary` 作为 canonical 字段。写入 `LibraryEntry.summary` 前会规范化成六个 sections：

1. `Execution Interpretation`
2. `Implemented Algorithm`
3. `New Ideas Introduced`
4. `Empirical Outcome`
5. `Failure / Bottleneck Analysis`
6. `Next Guidance Delta`

规范化规则：

- `Execution Interpretation` 必定合并提取出的 `<execution_thinking>`。
- `Implemented Algorithm` 必定附上提取出的 solution code，格式为 fenced Python block。
- `Empirical Outcome` 必定使用 verifier 实际结果，不信任模型自报。
- 如果 execution model 的 `<summary>` 缺失或 malformed，会合成完整六段 summary。
- 如果 verifier valid，failure analysis 默认写 `No verifier failure reported.`
- 如果 verifier 失败，failure analysis 默认写 `Verifier reported <status>: <message>`。
- `LibraryEntry.solution` 和 `LibraryEntry.execution_thinking` 仍然写入，主要用于 backward compatibility。

`Empirical Outcome` 的实际格式：

```text
Verifier status: <verification.status>
Raw C5: <verification.raw_score or None>
Reward: <verification.reward>
Verifier message: <verification.message>
```

## Reusable Idea 提取

位置：

```text
guidance_ttt/agent_loop.py::_extract_reusable_idea
```

新版优先级：

1. `Next Guidance Delta`
2. `New Ideas Introduced`
3. 旧格式 `Reusable idea: ...`

这保证新 library entries 的可复用信息来自 canonical summary，而老 library entries 仍然可读。

## Verifier

位置：

```text
guidance_ttt/verifier/erdos.py
guidance_ttt/verifier/sandbox.py
```

verifier 先从 execution output 中提取 Python code，然后在单独 sandbox process 中执行。程序必须定义：

```python
def run(seed=42, budget_s=..., **kwargs):
    return (h_values, c5_bound, n_points)
```

检查项：

- `h_values` 是一维有限数组
- `len(h_values) == n_points`
- 每个值在 `[0, 1]`
- `sum(h_values) == n_points / 2`
- 返回的 `c5_bound` 和 verifier 重新计算的 C5 一致

raw C5 公式：

```python
max(np.correlate(h, 1.0 - h, mode="full") * (2.0 / n_points))
```

reward：

```python
1.0 / (1e-8 + raw_score)
```

如果 parse、execution、timeout 或 validation 失败：

- entry 仍写入 library
- `verifier_raw_score = null`
- `verifier_reward = 0.0`
- `failure_mode = verifier.status`

## Library 存储结构

`library.json` 主要包含：

- `entries`: 完整 rollout 记录
- `nodes`: PUCT/search tree nodes
- `groups`: group bookkeeping
- `best_node_id`: 当前最高 reward node

一个 `entries` item 包含：

- actor `guidance`
- extracted `execution_thinking`
- extracted `solution`
- verifier reward、raw C5、status、message
- canonical `summary`
- `reusable_idea`
- `failure_mode`
- metadata，包括 prompts、raw generation text、execution model/provider、usage、selected node、group id

一个 `nodes` item 包含：

- `entry_id`
- `value`，也就是 verifier reward
- `raw_score`，也就是 raw C5
- `visits`
- `children`
- `parent_id`

`best_node_id` 通过最大化 node `value` 更新。因为 reward 是 `1 / C5`，所以正常情况下 `best_node_id` 对应最低 valid raw C5。

## PUCT 选择

新 group 创建时，library 从可见 children 中用 PUCT 选 parent。node value 来自 verifier reward，visits 用于 exploration。

当前实现选的是 best root 的 immediate children，不做任意深度 tree traversal。完整 lineage 仍通过 `parent_id` 保留，后续可以扩展 prompt context。

## 训练信号

agent loop 返回给 verl 的 reward 来自 verifier。verl 把这个 reward 贴到 actor response tokens 上，用 `ttt_reinforce_is` 更新 actor。

不会训练：

- execution model token
- verifier
- library selection logic

会训练：

- guidance actor，即 `/work/mit/ppliang_mit/lsy/models/Qwen3-8B`

所以真实学习目标是：

```text
产生能让冻结 execution model 写出更低 raw C5、且 verifier-valid Python 的 guidance。
```

## 旧任务当前分数解释

旧任务 `67845` 是修改前启动的进程，不会自动加载 summary-canonical 代码。它仍使用旧 prompt/summary 行为，直到该 Slurm 作业结束或重启。

最近一次 library 统计显示旧任务已经写到 timestep 20，其中 step 20 仍未满 256 条。当前最好结果：

```text
timestep: 20
raw C5: 0.3811454579668518
reward: 2.623670183812785
status: valid
```

训练日志里的 `critic/score` 是 reward，不是 raw C5。实验真正关心的 C5 分数在 `library.json` 的 `verifier_raw_score`。

## 新任务说明

新任务 `80637` 使用当前代码，也就是 summary-canonical unlimited 版本。它的配置文件是：

```text
guidance_ttt/config/erdos_gpt_oss_20b_8gpu_summary_canonical.yaml
```

提交脚本：

```text
/work/mit/ppliang_mit/lsy/jobs/guidance_ttt_gpt_oss_20b_8gpu_summary_canonical.sbatch
```

它写入独立 output directory：

```text
outputs/guidance_ttt/erdos_gpt_oss_20b_8gpu_summary_canonical_unlimited
```

因此旧任务和新任务不会共享 library，也不会互相污染 prompt context。

## 常用检查命令

查看 Slurm 状态：

```bash
squeue -j 67845,80637
```

查看旧任务 library best：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("outputs/guidance_ttt/erdos_gpt_oss_20b_8gpu_long/library.json")
data = json.loads(p.read_text())
best_node = data["nodes"][data["best_node_id"]]
best_entry = data["entries"][best_node["entry_id"]]
print(best_entry["timestep"])
print(best_entry["verifier_status"])
print(best_entry["verifier_raw_score"])
print(best_entry["verifier_reward"])
print(best_entry["verifier_message"])
PY
```

按 step 聚合 best C5：

```bash
python - <<'PY'
import json
from collections import defaultdict, Counter
from pathlib import Path

p = Path("outputs/guidance_ttt/erdos_gpt_oss_20b_8gpu_long/library.json")
data = json.loads(p.read_text())
by_step = defaultdict(list)
for entry in data["entries"].values():
    by_step[int(entry["timestep"])].append(entry)

cum_best = None
for step in sorted(by_step):
    rows = by_step[step]
    statuses = Counter(row["verifier_status"] for row in rows)
    valid_scores = [
        float(row["verifier_raw_score"])
        for row in rows
        if row["verifier_status"] == "valid" and row["verifier_raw_score"] is not None
    ]
    best = min(valid_scores) if valid_scores else None
    if best is not None:
        cum_best = best if cum_best is None else min(cum_best, best)
    print(step, len(rows), dict(statuses), best, cum_best)
PY
```

## 重要注意事项

- 旧 job `67845` 不会因为代码文件被修改而改变行为。
- 新 job `80637` 才会使用 summary-canonical prompt、summary writeback 和 unlimited execution output 配置。
- 不要让两个正在运行的 job 写同一个 `library.json`。
- `critic/score` 是 reward；raw C5 需要看 `verifier_raw_score`。
- 部分 step 可能少于 256 个 valid raw scores，因为 parse/validation 失败也会写 entry。
- stderr 中偶发 multiprocessing temp directory cleanup 的 `Device or resource busy` 不一定代表训练失败；要结合 `squeue` 和 `library.json` 是否继续增长判断。
