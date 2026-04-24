# SPEC_L2_V3.5 — 物种共存（Speciation, Assortative HGT, Niche-aware Consensus）

> 状态：v3.5 · 2026-04 · 在 v3.4 三相杂交协议依然失败之后的根本性转向。
> 一句话：**放弃「单细胞双修」的执念，接受 AND-菌 / NOT-菌作为两个共存物种，让 L2 由菌落整体（专家投票）完成。**

---

## 0. 起源 — v3.4 失败的诊断

v3.4 把 v3.0 的「单一爆发窗口」拆成「共栖 → 受控交换 → 恢复」三相，目的是
压制「权重幅值大的 NOT D1 学家一口吞掉 AND 学家」。日志显示协议本身正确
执行（phase 切换、blend 衰减都符合预期），但 **AND 命中率仍然崩溃**，且
NOT 命中率也跟着掉——「both_drop」。

User 的洞察一击命中：

> 我在想，原因是不是 AND 和 NOT 的进化完全不一样，应该是生殖隔离，无法
> 杂交，或者说杂交无意义。

这是对的。从 weights 层观察：

| 维度 | AND-学家（自然演化） | NOT-学家（D1 种子衍生） |
|---|---|---|
| 主导编码 | 分布式（多隐元小幅值正权重叠加） | 位置式（A-detector / S-tonic 两个子电路） |
| 典型权重幅值 | ±2 以内 | 可达 ±5 |
| 关键结构 | 全是兴奋通路，输入越多越兴奋 | 兴奋 + 抑制并存，依靠 S 信号门控 |
| 静息行为 | 弱 spike | 受 S 控制的强 tonic |

`blend_weights(w_r, w_d, η)` 是凸组合 `(1-η)·self + η·donor`，无论 η 多
小，**只要权重符号不同、幅值悬殊，混血总是同时破坏两个父本的关键电路**。
v3.4 的 η=0.05 让一次伤害变小，但累积起来仍然把双方的电路都磨平。

这其实就是经典生物学里的**基因组不相容（genome incompatibility）**——
两个分别走完 "AND 适应峰" 和 "NOT 适应峰" 的菌株，在基因层面已经不可
通婚了。

## 1. 哲学转向：从「Hybrid」到「Colony」

User 把这个问题升到了演化史：

> 你试图让 AND 菌群和 NOT 菌群通过权重杂交（HGT）变成一个"全才"。但
> 在生物史上，当两个功能完全不同且难以兼容的系统相遇时，它们走的是
> 内共生路线。

可选三条路径：

1. **基因复制（Gene Duplication）**：增加单 agent 的隐藏神经元数，提供
   冗余位置让两个电路并存。架构改动重，与 SPEC v1.0 的 220 权重契约
   冲突，留给未来的 L-level。
2. **内共生（Endosymbiosis）**：多 agent 嵌套，host 调度 sub-network。
   接近 multi-agent 范式，本仓不是该方向的演化。
3. **物种分化（Speciation）**：两群独立物种共栖于同一菌落，<u>菌落整体</u>
   呈现复合逻辑能力。 ← **本 SPEC 选择的路径。**

设计原则一句话：

> **群体即生命**。L2 的成功判据不是"找到一个能同时做 AND 和 NOT 的细
> 胞"，而是"菌落里既有专做 AND 的居民、又有专做 NOT 的居民，且两类
> 居民都能稳定再生产"。

## 2. 设计

### 2.1 Niche（生态位）= 一维标量

每个 agent 一个 `niche ∈ [-1, +1]` 标量：

```
niche(agent) = acc_AND_individual − acc_NOT_individual
```

- niche → +1：AND-专家
- niche → −1：NOT-专家
- niche ≈ 0：通才（罕见）或 novice（更常见）

`Population._niche_slot(slot)` / `_niche_array_living(idx)` 实现于
`archaea/population.py`。

### 2.2 物种四象限分类

`Population._species_of_slot(slot) → int`：

| 编号 | 标签 | 条件 |
|---|---|---|
| 0 | NOVICE | 任一 mode 上未达 `NICHE_SPECIALIST_THRESHOLD=0.65`，**或** 该 mode 样本数 `< NICHE_MIN_SAMPLES=10` |
| 1 | AND_EXPERT | acc_AND ≥ 0.65 且 ≥ 10 个 AND 样本，且 NOT 不达标 |
| 2 | NOT_EXPERT | 对偶 |
| 3 | DUAL_EXPERT | 两边都达标。理论上仍可能出现，是 **bonus** 而非要求 |

样本数门槛极重要：在 `and_only` 培养皿里，agent 的 `_acc_not_n` 永远 0，
任何"运气好的输出"都不能算 NOT 能力。

### 2.3 Assortative HGT（同类相吸基因流）

`hgt_pairs()` 增加两个可选参数：

```python
hgt_pairs(
    ..., 
    niche: np.ndarray | None = None,                      # 长度同 alive
    assortative_temperature: float = math.inf,            # T
)
```

行为：

- `niche is None` 或 `T == ∞`：**bit-identical to v3.4** — 候选 donor 中
  挑最富裕者，平局随机。
- 否则：候选 donor 仍按"半径 + credit ≥ donor_ratio·mine"过滤；但最终
  donor 通过 softmax 加权抽样，权重 `∝ exp(-|niche[d] - niche[i]| / T)`：
  - `T → 0`：严格物种隔离。只有最近 niche 的同类能转入基因。
  - `T = 0.3`（推荐默认）：柔性偏好——同物种概率几个数量级大于异物种，
    但偶发跨界基因仍可能出现。
  - `T → ∞`：恢复 legacy 行为。

数值稳定性：先减去 `logits.max()` 再 `exp`；如果所有权重下溢到 0
（极小 T + 极远 niche），fallback 到「niche 最近的 donor」。

`assortative_temperature` 流：

```
SimConfig.assortative_temperature: float | None
  None → cfg → Population(assortative_temperature = ∞)   # legacy
  0.0  → strict speciation
  0<T<∞→ soft preference
```

`None` 在 JSON 里更友好；`∞` 仅作为 Python 内部 sentinel。

### 2.4 Niche-aware Consensus（专家投票）

旧的 `acc_and_pop` / `acc_not_pop` 是**所有存活 agent 的逐 mode 准确率均
值**。在物种共存设定下这是错的——一个 NOT 专家在 AND 题上的"准确率"是
噪声，不是知识；包括它会污染指标。

新增三个 telemetry 字段：

```python
acc_and_swarm   = mean of acc_AND  over agents in {AND_EXPERT, DUAL_EXPERT}
acc_not_swarm   = mean of acc_NOT  over agents in {NOT_EXPERT, DUAL_EXPERT}
colony_dual_acc = (acc_and_swarm + acc_not_swarm) / 2
                  IFF |AND_voters| ≥ 10 AND |NOT_voters| ≥ 10
                  ELSE 0.0
```

`colony_dual_acc` 是 v3.5 的**唯一 L2 成功判据**：它非零意味着菌落里两个
物种都达到了"投票门槛"，且各自能稳定解决自己擅长的任务。一个 100% AND
的菌落 `colony_dual_acc = 0`，正确反映了"它没有完成 L2"。

### 2.5 物种结构 telemetry

```python
species_counts: {
  "novice":      int,
  "and_expert":  int,
  "not_expert":  int,
  "dual_expert": int,
}
```

总和等于 `n_living`。前端在 ObservePage（L2v2 Dashboard）渲染为四色堆
叠条 + 四象限计数 + 三个 swarm 指标徽章。

## 3. 实现清单

| 模块 | 改动 |
|---|---|
| `archaea/slime.py` | `hgt_pairs(..., niche=None, assortative_temperature=∞)` 增加 niche 加权抽样路径；常量 `DEFAULT_ASSORTATIVE_TEMPERATURE = ∞` |
| `archaea/population.py` | 常量 `NICHE_SPECIALIST_THRESHOLD=0.65`、`NICHE_MIN_SAMPLES=10`、`SPECIES_*`；`_niche_slot` / `_species_of_slot` / `_niche_array_living`；`step_window` 在调用 `hgt_pairs` 时按 task=L2v2 + finite T 注入 niche；新增 `_last_acc_and_swarm` / `_last_acc_not_swarm` / `_last_colony_dual_acc` / `_last_species_counts` 缓存；telemetry 字段 `acc_and_swarm` / `acc_not_swarm` / `colony_dual_acc` / `species_counts` / `assortative_temperature` |
| `archaea/runtime.py` | `SimConfig.assortative_temperature: float | None = None`；normalized + start_sim 透传；telemetry 把 `∞` → `None` 后送上 websocket |
| `archaea/server.py` | `StartBody.assortative_temperature: float | None = Field(None, ge=0, le=10)` |
| `tests/test_speciation.py` | 10 个新 case：assortative HGT 同源偏置 / 富裕 vs 同类二选一 / legacy 兼容 / 物种分类四象限 / 样本不足→novice / step_window emit 物种 telemetry / colony_dual_acc 单物种=0 / 双物种>0.5 / SimConfig 圆环 / Population 烟雾测试 |
| `webui/src/types.ts` | `SimConfig.assortative_temperature?: number | null`；`TelemetryEvent.{acc_and_swarm,acc_not_swarm,colony_dual_acc,species_counts,assortative_temperature}`；新 interface `SpeciesCounts` |
| `webui/src/pages/MixerPage.tsx` | 「物种隔离」section：三个一键预设（off / soft / strict）+ T 滑块 0..2；底部 summary 加 `T_niche` |
| `webui/src/colonies/l2v2/Dashboard.tsx` | `SpeciesPanel` 组件——四色堆叠条、四象限计数、三个 swarm 指标徽章（colony_dual_acc 高亮）；说明文字按是否达到双物种门槛切换 |

## 4. 推荐使用 protocol

杂交 AND-学家 + NOT-学家时：

```
v3.4 三相协议：commensal=60s, exchange=120s, phase2_blend=0.05, phase2_prob_mul=1.0
v3.5 同类相吸：assortative_temperature = 0.30   ← 默认推荐
```

观察重点（按时序）：

1. t < 60s：物种结构面板应该出现两个明显的色块（AND 蓝 + NOT 洋红），中间
   小段 novice。`colony_dual_acc` 仍为 0，因为协议关闭了 HGT，两个物种各
   过各的。
2. 60s ≤ t < 180s：受控交换期 + 同类相吸。观察 hgt_count 增长，但 AND
   和 NOT 区块大小应该**保持稳定**——证明 HGT 没有跨物种污染。
3. t ≥ 180s：恢复期。`colony_dual_acc` 应该稳定在 0.7+。如果两个物种相
   对比例失衡（如 AND : NOT = 9:1），考虑下次起 sim 调整 founder fraction。

## 5. 不做（明确放在 future work）

- **基因复制 / 隐藏层扩容**：留给 L3 重新启动时考虑。本 SPEC 在 220
  权重契约下解决问题。
- **内共生 / 嵌套 agent**：multi-agent 范式与本仓不一致，不在 L 级别
  range 内。
- **per-cell crossover** 替代 `blend_weights`：v3.0 ERRATA 列在 §7.5；
  v3.5 用 prezygotic isolation 解决，所以这条进一步 deprioritize。
- **跨物种性选择（让 dual_expert 享有繁殖优势）**：会再次把 fitness
  pressure 引向"双修陷阱"。如果想试，是 v3.6 的事。

## 6. 兼容性

- 旧 strain 文件：不受影响。`Strain` 不携带 niche。
- 旧 SimConfig（不带 assortative_temperature）：默认 None → ∞ → legacy
  行为，bit-identical to v3.4。所有 v3.4 测试继续通过。
- L1 colony：`step_window` 内对非 `TASK_L2V2` 直接走 `niche=None` 分支，
  无任何额外开销。

## 7. ERRATA v3.5b — 评测层 / 考核 / 使用层对齐物种共存

### 7.1 问题诊断

v3.5 演化机制本身是成功的——物种面板上能清晰看到 AND-experts、NOT-experts
两个色块，`colony_dual_acc` 能稳定到 0.8+。但用户报告"好像不行"。

抽日志后看明白了：**演化层成功**，**评测/使用层还在用 v3.4 的"单体双修"
假设**，导致同一份成功被错读为失败：

| 指标 | v3.4 物理意义 | v3.5 实际行为 | 用户感受 |
|---|---|---|---|
| `consensus_acc` | 全员投票准确率 | NOT 专家在 AND 题上沉默，被算"不答=不对"，拉低均值 | "AND 命中率才 33%" |
| `acc_and_11_pop` | (1,1)→1 命中率 | 同上：NOT 专家沉默拉低 | "1∧1 学不会" |
| `both_pass_pct` | 同时过 AND+NOT 的个体比例 | 严格生殖隔离下基本为 0（dual_expert 罕见） | "0% 双修" |
| Use 页 `target=best/ensemble` | 选 fitness 最高 | fitness=mean(acc_AND, acc_NOT)，物种化群里 best 通常是某一专精 | 6 题考试一面倒 |

### 7.2 修法（v3.5b 三件套）

**A. niche-aware consensus** —— `Population` 多算两份并行指标：

- `consensus_acc_swarm` / `consensus_bit_swarm` / `consensus_voters_swarm`：
  当前 oracle mode 对应的专家投票（AND 题问 AND/dual 专家，NOT 题问
  NOT/dual 专家）。voters_swarm == 0 时显式置 `bit_swarm=None`，UI 显示
  "尚无 X 专家"，不再误算成"沉默=错"。
- `acc_and_11_swarm` / `acc_not_0_swarm` + `row_acc_swarm` / `row_n_swarm`：
  6 行真值表的 swarm 版本，与 `_row_buf_correct/total` 同 ring 索引。

**B. inference 路由（`SimulationRuntime._resolve_query_target`）** ——
集中 dispatch，新增四个 target：

- `colony`：根据 `f_s_hz` 自动路由（≤ mid → `and_expert`，> mid → `not_expert`）；
  `mid = (S_AND_HZ + S_NOT_HZ) / 2 = 50 Hz`。
- `and_expert` / `not_expert` / `dual_expert`：调 `Population.top_k_slots_by_niche`，
  按对应 mode 的 `_logic_acc_slot` 排序。

**Fallback 契约**：niche 池为空 → 自动回退 `ensemble` + 设置
`target_degraded = "no_<niche>_specialist"`。响应里同时返回
`target_resolved`（运行时实际用的 target）和 `target_degraded`（回退原因 or null），
让前端能渲染"问了 AND 专家"或"↩ 已回退 ensemble"的徽章。

**C. UI 对齐** ——

- Dashboard `Translator`：头条改成"专家共识"（swarm bit + acc），全员均值
  demote 成灰字脚注。
- Dashboard `SpecificAccuracy`：`accAnd11Swarm`/`accNot0Swarm` 当头条，
  全员均值放灰字。提示语从"沉默搭便车"改成"物种共存的预期表征"。
- Dashboard `TruthTableMatrix`：每格头条 = swarm 命中率（带 niche 投票数），
  下方灰字 = 全员均值。
- Use 页 `LogicTester`：默认 target 改成 `colony`，新增 chip 选项
  `colony / and_expert / not_expert`；OneShotResult 顶上加路由徽章
  ("→ 问了 AND 专家 · 5 票" 或 "↩ 菌落尚无 NOT 专家，已回退 ensemble")；
  6 题成绩单加"问了谁"列。

### 7.3 哲学一致性

`acc_and_pop` 等"全员均值"**保留不删**——它们是诊断"全菌落是否被某物种独占"
的有效工具（如果 `acc_and_pop` 接近 `acc_and_swarm` 说明几乎没有 NOT 专家）。
但默认 UI 突出 swarm 指标，因为它们才是 v3.5 物种共存模型下"L2 是否成功"
的正确尺子。

### 7.4 兼容性

- 旧 SimConfig / 旧 strain 不变。
- 旧 inference target (`best/ensemble/random/swarm`) 行为完全保留，
  只是 `target_resolved` 和 `target_degraded` 字段对它们恒等于原值/null。
- 老 telemetry 字段全部保留；`*_swarm` 字段是新增。前端旧版只显示老字段
  仍然能跑（数字会"虚低"，但是这就是当初的问题，不影响 API 契约）。
