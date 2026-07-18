# Top-k guidance 实验完整结果与平台分析

## 一句话结论

本次 8×8 top-k guidance 实验实际完成 **24 / 50 step**，历史最高分在 step 16 达到 **81.3115**；step 17–24 没有突破。平台的主要原因是 guidance 与 top-2 reference 在语义和分支来源上都趋同：大量候选持续给同一套 aspect-ratio / LER / look-ahead 贪心评分加项，而不是探索不同搜索范式。作业随后因同卡 actor 与 vLLM 的显存峰值叠加而 OOM 退出，并非达到训练终止条件。

## 实验设置

| 项目 | 配置 |
|---|---|
| 代码分支 | `topk-guidance` |
| 任务 | `polyomino_packing`，70 个 case |
| 训练模型 | Qwen3-8B + LoRA rank 32 |
| Guidance / execution 模型 | GPT-OSS-120B，execution temperature 0.0 |
| 批设置 | `groups_per_batch=8`，`group_size=8`（8×8） |
| 搜索 | `puct_c=1.0`，`puct_q_mode=best_child` |
| 上下文 | prompt 16,384；response 8,192；vLLM max model len 24,576 |
| 计划 / 实际 | 50 step / 24 step |
| 作业 | `162397`，最终状态 `FAILED (1:0)`，运行 5:55:57 |

本版本的 guidance prompt 输入为：main parent 的完整代码与分数、其父节点摘要（根节点时省略）、以及两条 PUCT reference 的摘要和分数。execution 输出完整候选解，而不是 patch。

配置来源：[topk_guidance_2gpu_g8_context16k_50step.yaml](../../guidance_ttt/config/topk_guidance_2gpu_g8_context16k_50step.yaml)。

## 全部已完成 step 的分数

| Step | 该步最高分 | 历史最高分 |
|---:|---:|---:|
| 1 | 56.6862 | 56.6862 |
| 2 | 68.6888 | 68.6888 |
| 3 | 74.9560 | 74.9560 |
| 4 | 77.1759 | 77.1759 |
| 5 | 78.4575 | 78.4575 |
| 6 | 77.4111 | 78.4575 |
| 7 | 78.7507 | 78.7507 |
| 8 | 79.3825 | 79.3825 |
| 9 | 79.6057 | 79.6057 |
| 10 | 79.8703 | 79.8703 |
| 11 | 80.4205 | 80.4205 |
| 12 | 81.2093 | 81.2093 |
| 13 | 81.2383 | 81.2383 |
| 14 | 81.2383 | 81.2383 |
| 15 | 81.2697 | 81.2697 |
| 16 | **81.3115** | **81.3115** |
| 17 | 81.3115 | 81.3115 |
| 18 | 81.2991 | 81.3115 |
| 19 | 81.1661 | 81.3115 |
| 20 | 81.0982 | 81.3115 |
| 21 | 81.1241 | 81.3115 |
| 22 | 81.1228 | 81.3115 |
| 23 | 81.1668 | 81.3115 |
| 24 | 81.2193 | 81.3115 |

可以分成三个阶段：step 1–12 快速增长（56.69 → 81.21）；step 13–16 缓慢提升（81.24 → 81.31）；step 17–24 平台期。后 8 步的最优点从未高于 step 16，且落后 `0.00–0.21`。

原始数据可在 [Excel](../topk_guidance_score_history.xlsx) 和 [PNG 曲线图](../topk_guidance_score_history.png) 中查看；完整训练记录在 [SLURM 日志](../../outputs/slurm/entropic-best-child-162397.out)。

## 平台期：step 17–22 的证据

| Step | 该步最高分 | 距历史最高分 | 有效样本数 / 64 |
|---:|---:|---:|---:|
| 17 | 81.3115 | 0.0000 | 52 |
| 18 | 81.2991 | -0.0124 | 56 |
| 19 | 81.1661 | -0.1454 | 49 |
| 20 | 81.0982 | -0.2134 | 50 |
| 21 | 81.1241 | -0.1875 | 45 |
| 22 | 81.1228 | -0.1887 | 46 |

有效率从 step 18 的 87.5% 降至 step 21 的 70.3%、step 22 的 71.9%。这表明后续加复杂机制没有稳定地产生高质量可执行解。

### Guidance 的重复程度

这 6 步共有 384 条 guidance。没有空白归一化后的逐字重复，但主题高度集中：

| 主题关键词 | 覆盖数 | 占比 |
|---|---:|---:|
| ratio | 338 / 384 | 88.0% |
| aspect | 328 / 384 | 85.4% |
| orientation / pruning | 295 / 384 | 76.8% |
| look-ahead / predict | 273 / 384 | 71.1% |
| irregularity / boundary | 213 / 384 | 55.5% |
| gap | 208 / 384 | 54.2% |
| LER | 165 / 384 | 43.0% |
| restart | 132 / 384 | 34.4% |

最高分候选的方向依次是：shape compatibility + aspect orientation（17）、动态 aspect 阈值 / pruning（18）、多阶段 aspect adaptation（19）、下一块预测（20）、2–5 块 look-ahead（21）、irregularity/boundary 加权 aspect 和 gap 评估（22）。它们在表述上不同，但都在给同一个贪心 placement scoring function 继续添加 aspect、未来空间或阈值项；没有尝试排序、有限回溯、局部重排、候选空洞生成或小规模精确搜索等正交方向。

### Top-k reference 的收敛

每个 step 有 8 个 group，main node 可以不同，但 top-2 reference 由同一轮的 PUCT 排名决定，因此大量复用：

- step 17 的 8 个 group 中，7 个同时引用分数 **81.3115** 与 **81.2743** 的同一对节点；
- step 18 的多数 group 又集中引用 **81.3115** 与 **81.2991**；
- step 19 的多个 group 同时引用两条分数都为 **81.2991** 的 reference；
- step 20–22 仍由每轮两条约 `81.05–81.17` 的高分支反复担任 reference。

`best_child` 用最佳子代作为父价值，会放大已有高分子代的祖先。再加上全局 top-2 reference，8 个 group 即使 main parent 不同，仍持续被同一高分家族的摘要锚定。这是语义收敛的直接结构性来源。

### 上下文与实现复杂度

step 17–24 的平均 prompt 长度由 7,347 增至 8,056 token，单步时间由约 1,082 秒升至约 1,140 秒。高分候选实现约为 536–654 行、22k–27k 字符；在完整代码基础上不断叠加启发式，容易使各项互相抵消或仅在少数条件触发。这里尚未触及 16k prompt 截断，但长代码与三个历史上下文会降低 prompt 中“切换研究方向”的显著性。

## OOM 终止

step 24 完成、记录最高分 81.2193 后，下一步在 actor 的 `compute_log_prob` / entropy 计算阶段失败：GPU 0 需要新增 11.01 GiB，而剩余仅 10.71 GiB。报错同时显示 actor 进程已用约 45.25 GiB，co-located vLLM 进程已用约 122.36 GiB。

因此直接原因是**同卡峰值显存不足**。增长的序列长度（step 24 prompt 平均 8,056、最大 8,513 token）和 vLLM 仍占用的 KV cache 都会增加风险，但日志不能证明仅由 prompt 增长单独造成 OOM。此次 16k context 配置已经避免了旧 8k 配置的 prompt 长度不一致错误；新的限制是 actor 与 rollout 的联合显存峰值。

## 下一轮的建议

1. **先改善 reference 多样性。** reference-1 保留 PUCT 第一名；reference-2 从不同祖先分支中选分数足够高、summary 与前者最不相似的节点，并排除 main 的祖先/后代。
2. **加入 novelty guard。** prompt 明确列出 main、parent、reference 已尝试的机制；要求新 guidance 不得仅再加入 aspect/LER/look-ahead/threshold 项，且每次只提出一个可证伪假设与回退条件。
3. **预留正交探索配额。** 每个 8-way group 至少 2 条候选强制尝试：有限回溯/局部重排、piece ordering、候选 gap 生成或末段精确搜索。
4. **控制 mutation 范围。** 要求 execution 做单机制消融式改动、保留 baseline fallback，并说明新项实际进入哪个决策路径，避免多项启发式耦合后无法归因。
5. **重启前处理显存。** 将 actor/ref 的 log-prob micro batch 从 2 降到 1，或降低 vLLM `gpu_memory_utilization` / 启用更积极的 cache 释放；保留 step 24 checkpoint 再续跑。应先做一次短恢复验证，确认峰值显存安全后再完成剩余 step。

## 关联材料

- [此前的 step 17–22 专项分析](../topk_guidance_plateau_analysis_steps17_22_zh.md)
- [分数 Excel](../topk_guidance_score_history.xlsx)
- [SVG 曲线图](../topk_guidance_score_history.svg)
- [PNG 曲线图](../topk_guidance_score_history.png)

## 后续实验计划：top-k guidance 消融

为验证平台期是否主要来自 reference 同质化与 guidance 的局部启发式收敛，后续四组实验保持同一训练预算，仅改变 guidance/search 配置。每组均使用 Polynomino 8×8、8 张 B200、Qwen3-8B TTT guidance、GPT-OSS-120B execution、50 step、`groups_per_batch=8`、`group_size=8`、`puct_c=1.0`、`puct_q_mode=best_child`；verifier、execution prompt 及其余训练超参不变。

| 实验 | Guidance/search 改动 | 验证目标 | 已提交作业 |
|---|---|---|---:|
| Exp 1：Diverse References | reference 改为 PUCT Top-1 + 前列高分候选中策略文本最不相似、且不同祖先分支的节点 | 验证 reference 同质化是否导致收敛 | 163661 |
| Exp 2：Search-Aware Prompt | prompt 注入已探索策略、失败方向、近期试验及策略频率；高频方向被明确标记为已探索 | 验证搜索历史能否提升探索多样性 | 163662 |
| Exp 3：Policy Guidance Prompt | guidance 输出一个可证伪的 search policy hypothesis：搜索方向、机制、预期收益、风险/证伪条件 | 验证 guidance 作为树搜索策略是否优于局部 solution optimizer | 163663 |
| Exp 4：Combined | 同时启用 Diverse References、Search-Aware 和 Policy Guidance | 验证完整搜索策略是否突破平台 | 163664 |

作业使用完整 8 卡 B200 节点：GPU 0–3 用于 TTT，GPU 4–7 以 4-way tensor parallel 运行 GPT-OSS execution。这样保留 8×8 的 64 条训练样本，并满足训练 batch 必须被 4 张训练卡整除的约束。作业会提交到 `b200-batch` 队列，在完整节点空闲后自动启动。

每个运行结束后会自动写入 `topk_guidance_metrics.json`，统一记录：

- best verifier score；
- valid execution rate；
- guidance strategy diversity（当前为 guidance 文本的词汇 Jaccard 多样性代理，保留原始 guidance/summary 以供后续 embedding 分析）；
- Aspect/LER/look-ahead 等策略标签覆盖率；
- reference diversity（不同祖先分支数、reference strategy similarity、lineage relation）；
- 首次严格超过当前最佳分数 81.3115 的 step；若未突破则记录为 `null`。

本轮实现版本为 `topk-guidance` 分支 commit `28db605c`。
