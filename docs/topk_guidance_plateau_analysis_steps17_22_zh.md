# Top-k guidance：step 17–22 未提升分析

## 结论

step 17 将历史最高分带到 **81.3115** 后，step 18–22 均未超过它。这更像是**搜索方向收敛后的局部平台**，而不是单次 verifier 波动：guidance 没有逐字复制，但反复围绕同一组启发式（aspect ratio、LER、look-ahead、orientation pruning）做叠加式微调；与此同时，PUCT 的两个 reference 在同一轮的多数 group 中相同，进一步把生成引向相近的改动。

## 观测数据

| Step | 该步最高分 | 距历史最高分 | 有效样本数 / 64 |
|---:|---:|---:|---:|
| 17 | 81.3115 | 0.0000 | 52 |
| 18 | 81.2991 | -0.0124 | 56 |
| 19 | 81.1661 | -0.1454 | 49 |
| 20 | 81.0982 | -0.2134 | 50 |
| 21 | 81.1241 | -0.1875 | 45 |
| 22 | 81.1228 | -0.1887 | 46 |

step 18 起最优候选不仅没有提高，反而稳定地落在历史最优点下方约 `0.01–0.21`。有效样本比例也从 step 18 的 87.5% 降到 step 21 的 70.3%、step 22 的 71.9%，说明新增复杂度没有稳定转化为更优可执行解。

数据来源：训练 [SLURM 日志](../outputs/slurm/entropic-best-child-162397.out) 与 [library 快照](../outputs/guidance_ttt/topk_guidance_2gpu_g8_context16k_50step_20260717/library.json)。本报告只分析已完整记录的 step 17–22。

## Guidance 是否重复？

不是文本级重复：这 6 个 step 的 384 条 guidance 中，没有两条在空白归一化后完全相同。但主题重复非常明显：

| 主题关键词 | 覆盖 guidance 数 | 占比 |
|---|---:|---:|
| ratio | 338 / 384 | 88.0% |
| aspect | 328 / 384 | 85.4% |
| orientation / pruning | 295 / 384 | 76.8% |
| look-ahead / predict | 273 / 384 | 71.1% |
| irregularity / boundary | 213 / 384 | 55.5% |
| gap | 208 / 384 | 54.2% |
| LER | 165 / 384 | 43.0% |
| restart | 132 / 384 | 34.4% |

最高分候选的建议也呈连续递进、但同质的模式：

| Step | 最高分候选的主改动 |
|---:|---|
| 17 | shape compatibility + aspect-ratio orientation 选择 |
| 18 | 动态 aspect 阈值、orientation pruning、restart 阈值 |
| 19 | 多阶段 aspect adaptation + context-aware restart |
| 20 | 下一块的 predictive aspect alignment / look-ahead |
| 21 | 2–5 块的 multi-step look-ahead |
| 22 | irregularity/boundary 加权 aspect 与 multi-stage gap 评估 |

因此问题不在措辞相同，而在于这些 proposal 都在继续给同一个贪心评分函数添加 aspect、未来空间和阈值项。它们缺少彼此排斥的假设或真正不同的搜索范式，例如排序策略、回溯策略、局部重排、候选生成或精确小规模子问题求解。

## Top-k reference 为什么会放大收敛

当前配置是 `puct_q_mode: best_child`、`puct_c: 1.0`、`group_size: 8`。每一步有 8 个 group；每个 group 各自选 main node，但 reference 取同一时刻按 PUCT 排名靠前的两个节点。

这造成 reference 的高度复用。例如：

- 在 step 17 的 8 个 group 中，7 个同时看到分数 **81.3115** 和 **81.2743** 的同一对 reference。
- 在 step 18 的多数 group 中，reference 又集中到 **81.3115** 和 **81.2991**。
- step 19 的多个 group 甚至同时看到两个分数都为 **81.2991** 的 reference；它们是不同节点，却没有提供足够清晰的策略差异。
- step 20–22 也分别由每轮的两条约 `81.05–81.17` 的最高分支反复充当全局 reference。

`best_child` 会把父节点价值近似为其最优子代，从而强化已经出现高分子代的祖先。配合全局 top-2 reference，这使得 8 个 group 即使 main node 不同，仍被相似的高分经验锚定。当前每步仍有 8 个不同的有效 parent，但探索主要是在同一高分家族附近作小改动，而非跨家族尝试。

## 其他放大因素

1. **增量启发式堆叠。** 高分候选的实现规模约 536–654 行、22k–27k 字符。下一轮继续在完整代码上添加新评分项，容易使不同 term 相互抵消，或让新 term 只在少数条件触发。step 17–22 的 prompt 平均长度已约 7.3k–7.7k token；并未触发 16k 截断，但大量实现细节会弱化“下一步应当换方向”的信号。
2. **单步回报对微调友好。** 目前 score 直接偏好即时紧凑度/面积改进，模型自然重复提出“更精细的 aspect 预测”。这在早期有效，接近 81 后则更需要能够跳出局部构形的操作。
3. **复杂度上升而失败率上升。** look-ahead 从下一块扩到 3–5 块、再扩到全体 remaining pieces 的同时，有效样本数下降；这说明更长的前瞻近似并没有带来稳定收益，反而增加实现与调参风险。

## 建议的下一次改动

优先级从高到低：

1. **让两个 reference 强制多样化。** 保留 PUCT 第一名；第二名不要直接取第二高分，而在不同祖先分支中选择与 main/reference-1 的 summary 相似度最低、且分数足够高的节点。还应排除 main 的祖先/后代，避免“同一代码家族的两个摘要”。
2. **在 guidance prompt 加入“已尝试主题”与 novelty 约束。** 从 main parent、previous parent 和两个 reference 的 summary 中抽取最近已用的机制；明确要求新 guidance 不得再次仅加入 aspect/LER/look-ahead/threshold 项。每次只提出一个可证伪假设，并写出预期改善和回退条件。
3. **把一部分 rollout 固定为正交策略。** 例如每个 8-way group 至少保留 2 个给：局部重排/有限回溯、piece ordering、候选空洞生成、末段精确搜索。不要让它们都从同一 reference 对推导评分函数微调。
4. **从 `best_child` 改为带访问均值的价值估计，或提高新分支探索权重。** 这样分数接近的高分祖先不会因单个最佳子代长期垄断 reference；需做对照实验确认，不建议在当前运行中热改。
5. **限制一次 mutation 的范围。** 要求 execution 对单个机制做消融式改动，保留 baseline fallback，并要求说明新增项真正进入了哪个决策路径。这样能区分“机制无效”与“多项耦合后退化”。

## 推荐的最小验证实验

下一轮可只改前两项：`diverse top-2 reference + prompt novelty guard`，其余超参保持不变。观察 10–15 step：

- guidance 中 aspect/ratio 的覆盖率是否显著低于当前 85%–88%；
- 每轮两个 reference 的祖先是否不同；
- 有效样本率是否回到约 80% 以上；
- 是否能超过 81.3115。

这能直接验证平台是否主要来自 reference/prompt 同质化，而不混入过多训练配置变量。
