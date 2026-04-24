# SPEC_L2_V3.0 — 杂交皿（Admixture Experiment）

> 状态：v3.0 草案 · 2026-04 · 用于打通 L2v2 演化中长期卡死的「单一逻辑陷阱」（NOT 学不会）。
> 一句话：**两个独立培养皿分别养出 AND / NOT 学家，再倒在一起让 HGT 把基因混合，自然涌现双修个体。**

---

## 0. 起源

L2v2（SPEC_L2_V2.0）已经稳定跑通 AND（acc_AND ≈ 92%），但 NOT 长期停留在 25–30%（≈ 瞎猜），无论怎么改 fitness 公式、奖励表、初始权重偏置都解决不了。诊断结论：

> **NOT 不是奖励问题，是演化路径问题。**
>
> AND 在 SNN 拓扑里是一跳电路（全正权重 + 阈值累积），NOT 需要 2–3 跳的抑制 + disinhibition 电路。在同一个种群里，AND 学家的 fitness gradient 短得多，永远先占领；占领之后 σ 收敛、突变窄化，NOT 所在的「窄缝」永远到不了。

这是 Sewall Wright 经典的 **fitness valley** 问题。

### 0.1 生物学借鉴

复杂的新功能在细菌世界里几乎从来不是「点突变出来的」，而是「借来的」：

| 案例 | 怎么获得新功能 |
|---|---|
| 大肠杆菌 Cit+（柠檬酸利用） | LTEE 31000 代点突变出现一次；自然界 99% 通过 HGT 从克雷伯氏菌借 |
| β-内酰胺酶抗药性 | 没有任何病原菌独立演化，全部 HGT 传播 |
| 日本人肠道菌吃海带多糖 | 直接从海洋细菌 HGT 来 porphyranase 基因簇（Hehemann 2010） |
| 酸奶发酵 | 保加利亚乳杆菌 + 嗜热链球菌**分别培养再混合** |

这就是「**异域演化 + 横向基因转移 (allopatric evolution + HGT)**」—— 实验微生物学家做了百年的标准 protocol，叫 **admixture experiment**。

### 0.2 设计原则

1. **不动 fitness 公式、不动奖励表、不动 SNN 拓扑** —— 这些都是「上帝改规则」。
2. **只增加用户能做的「实验布置」** —— 像 Lenski 一样：决定开几个培养皿、什么时候相遇、按什么比例。
3. **复用已有的 HGT 基础设施**（`slime.py` / `population.py`）—— 不重新发明轮子。
4. **Slime reward bonus 永远关闭** —— v2.x 的故障调查证明 `pheromone_bonus_k > 0` 会污染 oracle 评分形成 silent attractor。Admixture 模式只用 slime 的「空间 + HGT」，不用「奖励加成」。

---

## 1. 三个核心概念

### 1.1 培养皿 (Colony / Petri Dish)
现有概念。一次仿真 = 一个培养皿 + 一段演化历史。本 SPEC 不动培养皿模型本身。

### 1.2 菌株 (Strain)
**新概念**。一个菌株 = 某个培养皿在某个时刻的「全部活体快照」（或 top-K 子集）。

字段（v3.0）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string (UUID) | 文件名锚点 |
| `name` | string | 用户命名（可重） |
| `task` | `"l1" \| "l2v2_ctrl"` | 来源 task — 跨 task 不能混 |
| `weights` | float64 (K, 220) | K 个 agent 的 SNN 权重（K = 保存时活体数，最多 pop_max） |
| `fitness_at_save` | float64 (K,) | 保存时各 agent 的 fitness（NaN 允许） |
| `t_sim` | float64 | 来源 sim 的仿真时间 |
| `source_colony_id` | string | 来源 colony id |
| `source_seed` | int64 | 来源种子 |
| `source_difficulty` | string \| null | L2v2 才有 |
| `acc_and_pop_at_save` | float64 \| null | L2v2：保存时的种群 AND 准确率 |
| `acc_not_pop_at_save` | float64 \| null | L2v2：保存时的种群 NOT 准确率 |
| `note` | string | 用户备注 |
| `created_at` | ISO8601 string | 落盘时间 |
| `spec_version` | `"L2.V3.0"` | 字段升级时 bump |

存储：`checkpoints/strains/<id>.npz`（单文件，复用 `np.savez_compressed`，metadata 以标量数组形式存进去），同名 `<id>.json` 缓存元数据用于快速列表浏览。

**菌株不存 credit、年龄、空间位置** —— 倒进新培养皿后这些字段一律重置。这是「孢子状态」：基因组冻干，环境信息丢弃。

### 1.3 杂交 (Admixture)
新培养皿启动时，初始 slot 不再随机生成，而是从一组菌株按比例采样 weights 灌入。语义：

```
初始 slot 共 n_initial 个，按各 founder 的 fraction 分配：
  founder[0] (strain_A, fraction=0.5) → 0.5 × n_initial 个 slot 用 A 的 weights 采样
  founder[1] (strain_B, fraction=0.5) → 0.5 × n_initial 个 slot 用 B 的 weights 采样
若 sum(fraction) < 1.0，剩余 slot 用 task 默认 random init 填补 (保留 evolvability)
若某 strain 的 K 不够，用替换抽样（with replacement）
```

启动后，**N_INITIAL 之后的 slot 全空**，正常按繁殖填满（和现在一致）。

### 1.4 杂交期 (Admixture Window)
启动后前 N 秒（默认 30s）临时放大 HGT 概率（默认 ×5），模拟「两个种群刚相遇时菌群密度高、基因交换爆发」的真实生物学窗口。期满后回归正常 HGT 率。

实现方式：`SimConfig.admixture_window_s` (float, ≥0) + `admixture_hgt_multiplier` (float, ≥1)。当 `t_sim < admixture_window_s` 时，HGT 概率是 `slime.hgt_prob × admixture_hgt_multiplier`，否则就是 `slime.hgt_prob`。

杂交期内**默认强制开启 slime spatial + HGT**（否则 HGT 根本不会被触发），但 `pheromone_bonus_k` 强制设为 0（避免污染 L2v2 reward）。期满后用户可以保留这些设置或手动关掉。

---

## 2. 后端 API

### 2.1 新增端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/strains` | 列出所有菌株（meta only，不返回 weights） |
| `POST` | `/api/strains/save` | 把当前 sim 的活体保存为新菌株 |
| `DELETE` | `/api/strains/{id}` | 删除菌株（同时删 .npz 和 .json） |

**`POST /api/strains/save` body**：
```json
{ "name": "AND-only 跑 30min", "note": "p_mode_and=1.0, t=1800s, acc_AND=0.95" }
```
返回完整的 Strain 元数据（含 id、created_at）。

### 2.2 启动 API 扩展

`POST /api/start` 的 body 增加可选字段：

```json
{
  "...原有字段...": "...",
  "founders": [
    { "strain_id": "uuid-of-AND-strain", "fraction": 0.5 },
    { "strain_id": "uuid-of-NOT-strain", "fraction": 0.5 }
  ],
  "admixture_window_s": 30.0,
  "admixture_hgt_multiplier": 5.0
}
```

约束：
- 所有 founders 的 `task` 字段必须等于 SimConfig 的 `task`（不能拿 L1 菌株进 L2v2）
- `sum(fraction)` 不必等于 1（小于 1 时余量随机填）
- 若 `founders == None or []`，行为与 v2.x 完全一致（向后兼容）
- 若 `admixture_window_s > 0` 但 `slime_mold == False`，server 自动把 `slime_mold` 翻为 True 且强制 `pheromone_bonus_k = 0`（杂交需要空间 + HGT）

---

## 3. 前端

### 3.1 新群落卡片：⚗️ 杂交皿 · Mixer

加在 `colonies/registry.ts` 的第三张卡片。**不是一个新 task**（task 仍是 l1 或 l2v2_ctrl），而是「以 admixture 模式启动既有 task」的入口。卡片描述：

> ⚗️ 杂交皿 · Mixer  
> SPEC_L2_V3.0 · 难度 ★★★★  
> 一句话：选两个已经长大的菌株（一个 AND 学家、一个 NOT 学家），按比例倒在一起 + HGT 杂交期 → 双修个体涌现。

### 3.2 MixerPage（新页面）

布局：
```
┌──────────────────────────────────────────────┐
│ 顶部：菌株库（卡片网格）                        │
│   每张卡显示：name / task / acc_AND / acc_NOT  │
│   / t_sim / source_difficulty / 删除按钮        │
├──────────────────────────────────────────────┤
│ 中部：实验配置                                  │
│   - 选定的 founder 列表（可加可减）              │
│   - 各 founder 的 fraction 滑块                 │
│   - 目标 task（必须所有 founder 一致）            │
│   - 目标 task_difficulty（默认 balanced）         │
│   - 杂交期长度 admixture_window_s                │
│   - HGT 倍数 admixture_hgt_multiplier            │
│   - pop_max / n_initial / seed                  │
├──────────────────────────────────────────────┤
│ 底部：启动按钮 → 进入新培养皿的 observe 页       │
└──────────────────────────────────────────────┘
```

### 3.3 ObservePage 加「💾 保存为菌株」按钮

在生态仪表盘附近加一个按钮。点击后弹窗：name / note 输入框 → POST /api/strains/save → toast 「✓ 已保存为 <name>，K=N agents」。

---

## 4. 测试

### 4.1 单元测试

- `test_strain_save_load_roundtrip`: 一个 strain 保存到 npz 后加载，weights 字节相同
- `test_strain_meta_json_consistent`: .npz 和 .json 元数据一致
- `test_strain_pop_save_then_spawn_preserves_weights`: 保存全部 K 个活体，再 spawn 到一个 n_initial=K 的新 pop，**前 K 个 slot 的 weights 与原 pop 字节相同**（无替换抽样的特殊情况）
- `test_strain_spawn_from_two_strains_fraction_correct`: A.fraction=0.5, B.fraction=0.5, n_initial=100 → 50 个 slot 来自 A weights 池，50 个来自 B
- `test_strain_spawn_fraction_lt_1_remainder_random`: A.fraction=0.3 + B.fraction=0.4 + n_initial=100 → 30 + 40 + 30 random
- `test_admixture_window_boosts_hgt`: 同一 seed 下，前 30s 的 hgt_count 显著高于后 30s（用 mock 的 slime 配置 + 大种群）
- `test_strain_cross_task_rejected`: 用 L1 strain 启动 L2v2 任务 → server 拒绝

### 4.2 集成测试（手工）

1. 用 L2v2 + `task_difficulty="extreme"` + `p_mode_and=1.0`（实际通过 `task_difficulty` 一个新预设 `pure_and`）跑 5 分钟，保存为 strain `AND-pure`
2. 同样 sim，`pure_not` 预设，5 分钟，保存为 `NOT-pure`
3. MixerPage 选 `AND-pure (50%) + NOT-pure (50%) + balanced + 30s 杂交期`，启动
4. **预期**：前 30s 看到大量 HGT 事件；之后 5–15min 内 `acc_not_pop` 应至少升到 60%（vs 当前长期卡在 25%）

> 引入 `pure_and` / `pure_not` 两个新 difficulty 预设是 v3.0 的副产品 —— 没有它们用户没法养纯化菌株。

---

## 5. 不在本 SPEC 范围

- 多于 2 个菌株的混合（API 一开始就支持 N 个，但 UI 第一版只优化 2 个）
- 谱系树可视化
- 菌株自动分类 / 标签
- 跨 SPEC 版本的菌株迁移（v3.0 → v3.1 时再设计）
- 真正的「有性生殖」（基因杂交，不是种群混居）—— 这是另一种生物学模型，留给 L4

---

## 6. 哲学声明

杂交皿不是为了「让 NOT 一定能涌现」。它是把演化失败的责任从「演化算法不行」推回到「环境布置不到位」—— **用户决定开几个独立培养皿、按什么比例混合、给多长杂交期**，这些决策本身就是演化实验的一部分。

如果用户选错（比如两个都是 AND 菌株），他会得到一个还是只会 AND 的种群 —— 算法没错，环境没设好。这正是 Lenski 实验的精神：**实验生物学家的工作是设计实验，不是干预演化**。

---

## 7. ERRATA — v3.4：3 相生态杂交协议（2026-04-22）

### 7.1 v3.0 单窗口模型的失败

按 v3.0 §1.4 的设计，杂交期 = 启动后 N 秒内 `eff_hgt_prob = base × multiplier`。
实测一旦把 D1 NOT 学家（structural seed，权重幅值 ±2~±5）和 AND 学家
（自然演化出来，权重幅值通常 ±0.5~±2）倒进同一培养皿，会出现「单边
污染」：

- HGT 用 `blend_weights(w_recipient, w_donor, η=0.30)` 做凸组合
- NOT 学家是 donor 时，强幅值的负权重一次就把 AND 学家的对应正权重
  「翻号」——AND 逻辑当场失效
- 反过来 AND 学家做 donor 时，弱幅值对 NOT 学家的强结构种子几乎没影响
- 加上 admixture 期 HGT × 5 的爆发，几十秒内整群被 NOT 表型扫荡，
  `acc_and_11_pop` 从 0.85 跌到 0.27（实测截图，2026-04-22）

根本原因是 v3.0 的「立即放大基因流」与生物学相反：真实的双菌共栖
（Lenski LTEE、Synechococcus / Prochlorococcus 海洋混合）总是先有
**长时间共栖期**，让两群各自适应新环境，然后才会出现可观测的 HGT；
HGT 一旦发生，**单次转移的基因量很小**（几个 ORF / 一段质粒），
不会一次替换掉半个基因组。

### 7.2 v3.4 协议：3 相

把 `admixture_window_s` + `admixture_hgt_multiplier` 两个旧参数替换成
4 个新参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `admixture_commensal_s` | `60.0` | Phase 1（共栖期）长度，HGT 完全关闭 |
| `admixture_exchange_s` | `120.0` | Phase 2（受控交换期）长度 |
| `admixture_phase2_blend` | `0.05` | Phase 2 期间的 `hgt_blend`（基线 0.30） |
| `admixture_phase2_prob_mul` | `1.0` | Phase 2 期间的 `hgt_prob` 倍率 |

阶段判别由 `Population._admixture_phase()` 实现：

```
t < commensal_s                           → Phase 1（HGT 全关）
commensal_s ≤ t < commensal_s + exchange_s → Phase 2（低 blend HGT）
t ≥ commensal_s + exchange_s              → Phase 3（基线 HGT）
commensal_s == 0 且 exchange_s == 0       → Phase 3（协议禁用）
```

- **Phase 1（共栖期）**：HGT 完全关闭。两个 founder 群体共享同一物理
  培养皿（slime grid）、共享代谢经济（食物 / 信息素），但基因池**完全
  隔离**。每个菌株各自在新环境里完成 founder collapse 抗性测试，
  确认能在新难度预设下存活。
- **Phase 2（受控交换期）**：HGT 启用，但 `hgt_blend` 降到 0.05（基线的
  1/6）。每次转移只渗透几条权重，不会一口替换掉对方半个基因组。即使
  D1 种子的强负权重被转移给 AND 学家，η=0.05 意味着 AND 学家的正权重
  只被拉低 5%，不会翻号。给演化足够多的尝试次数找到「真正有用的
  crossover」（保留两边表型的杂交个体）。
- **Phase 3（恢复期）**：协议结束，HGT 回到 colony 基线。从这里开始
  就是普通 colony，已经长出来的双修个体凭借代谢竞争力继续传播。

### 7.3 Telemetry 变化

`step_window` 返回的 info 字典新增：

```python
"admixture_phase":          1 | 2 | 3,
"admixture_active":         bool,        # phase ∈ {1, 2}
"admixture_commensal_s":    float,
"admixture_exchange_s":     float,
"admixture_phase2_blend":   float,
"admixture_phase2_prob_mul":float,
"eff_hgt_prob":             float,       # 当前 phase 实际生效
"eff_hgt_blend":            float,       # 当前 phase 实际生效
```

UI（StrainBar）按 phase 渲染不同徽章：青色 = ① 共栖期、洋红 + 脉动 = 
② 受控交换期、Phase 3 不显示徽章（已是普通 colony）。

### 7.4 向后兼容

旧字段 `admixture_window_s` / `admixture_hgt_multiplier` **完全移除**。
保存的 strain 文件不受影响（meta 里没有 admixture 字段）；只有 UI 表单
和 `POST /api/start` 的 body schema 变了。

### 7.5 没做（暂列）

- 把 HGT 的 `blend_weights`（凸组合）改成「per-cell crossover」（每条
  权重独立从 donor 或 recipient 二选一抽样）。这是更接近真实细菌
  接合 / 转化的 B 方案。如果 v3.4 的 phase 2 仍然出现单边污染再考虑。
- 三个或更多菌株的协议化杂交：当前协议对所有 founder 一视同仁，没有
  「strain A 先和 B 共栖、再和 C 杂交」的高阶序列。

