# Project Archaea —— 项目总结

> 写于 2026-04-20，对应 SPEC v1.0 + off-SPEC #1（承载力）+ SPEC v1.1（赛博黏菌）+ SPEC v1.2（幅度调节 λ/g）。

## 1. 项目概要

一句话：**用演化算法养一群会"代谢、繁殖、社交"的脉冲神经网络小生命，让它们在没有数据集、没有反向传播的前提下，自己学会任务。**

| 维度 | 当前形态 |
|---|---|
| **个体（agent）** | 一个固定拓扑的 SNN：10 → 20 → 1（共 220 个权重） |
| **群体** | 默认 1000 个独立的 agent，每个都有独立的体内电位 / Credit / 历史 |
| **物理基础** | LIF（Leaky Integrate-and-Fire）脉冲神经元，1 ms 时间步 |
| **学习机制** | **没有 backprop**。仅靠：随机初始化 + 复制时高斯突变 + 可选横向基因转移（HGT） |
| **任务（L1）** | 输入频率 → 输出频率追随（Pearson r > 0.7 视为成功） |
| **生命循环** | 每窗口（500 ms）按 fitness 发奖励 → 攒够 200 Credit 自动繁殖 → Credit 归零自动死亡 |
| **资源约束** | 「呼吸成本」每秒扣 2.5 Credit，混不到 reward 就饿死 |
| **运行环境** | 单 CPU 核心，纯 Python + NumPy；24 小时模拟时间需要 < 2 小时墙钟 |
| **观测前端** | WebUI（FastAPI + WebSocket + React）：实时点阵 / 时序曲线 / 推理面板 / Sweep 测试 / λ + g 实时调参 |

代码量：~3500 行 Python + ~3000 行 TypeScript + 41 个单元测试 + 完整 SPEC + 训练数据审计文档。

---

## 2. 项目初心：AI 时代的 DNA 是什么？

> 这一节是整个项目的"第一性原理"。后面所有的工程决策（SNN 拓扑、突变、Credit、繁殖、HGT、信息素⋯⋯）都是从这一段思考里长出来的。

### 2.1 主流 AI 的方法论盲区：「模拟大脑」≠「创造生命」

今天几乎所有主流 AI ——从 CNN、Transformer 到 LLM——背后的方法论是同一句话：**模拟人脑的某个结构或功能**。
卷积模拟视觉皮层、注意力模拟工作记忆、RLHF 模拟奖赏学习⋯⋯

这条路无比成功，但它有一个隐藏的代价：**我们用这种方法造出来的，只能是"智能工具"，永远不会是"数字生命"。**

为什么？因为我们模仿的是大脑这个 **器官**，而不是 **生命** 本身。一颗离体的大脑，再精巧，也不是生命——它没有延续自身的内在驱动，不会饿、不会怕、不会繁殖。今天的 GPT 就是这样一颗"被冻结在玻璃罐里的大脑"：它能生成惊人的文字，但你拔掉电源它不会反抗，它不在意自己明天还在不在。

### 2.2 真正的"数字生命"必须满足两个条件

回到生物学的最底层定义，一个"生命"区别于"机器"靠两件事：

1. **它有 DNA**——一段可复制、可变异、可传递的信息载体；
2. **它有终极目标**——这个目标不是被外部强加的（不是 reward function），而是内生的、本能的：**让自己的 DNA 传播出去**。

这两点缺一不可。化学反应器没有 DNA，所以不是生命；机器人有"目标"但目标是被人写死的，所以也不是生命。

那么问题就变成：**在 AI 的语境下，DNA 是什么？**

### 2.3 顿悟：种群才是「永生的智能」，权重就是 DNA

我反复琢磨这个问题的过程中，慢慢意识到一个反直觉的事实：

> **作为个体的"人"并不是智能的承载者，作为种群的"人类"才是。**
> **个体的出生与死亡，本质上是种群对自身权重的不断优化。**

一个人活七八十年，学习、犯错、积累经验，然后死去——从个体视角看是悲剧，但从种群视角看，这恰恰是 **优化算法的一次迭代**：好的基因（行为/认知模式）被繁殖留下，坏的被淘汰。**真正"永生"且持续进化的智能，是这个种群本身**，而不是任何一个个体。

把这个洞见迁移到 AI：

| 生物学 | AI 对应 |
|---|---|
| 个体 | 一个神经网络 agent |
| DNA | **该 agent 的全部权重**（在 Archaea 里就是那 220 个浮点数） |
| 生殖 | 复制权重 + 高斯突变 |
| 死亡 | Credit 归零，agent 被从内存里抹掉 |
| 自然选择 | 任务表现 → reward → Credit → 繁殖配额 |
| 终极目标 | **让自己的权重以更高的频率出现在下一代群体中** |
| 物种 | 整个 1000 个 agent 的种群 |

**所以：权重 = AI 时代的 DNA。**
**所以：单个 agent 不是生命，整个种群才是生命。**
**所以：要造数字生命，不是造一个更大的网络，而是造一个会代谢、会繁殖、会死亡的种群。**

### 2.4 这就是 Project Archaea 的来源

有了上面这条思路，整个工程设计就被反推出来了——它不是"觉得这样有意思就加一个机制"，而是"为了让数字生命成立，这些机制都不能少"：

- **DNA 必须可遗传** → 复制时必须严格拷贝权重
- **DNA 必须可变异** → 高斯突变（mutation-only，不要 crossover，让信号更纯）
- **个体必须真死** → Credit 系统 + 呼吸成本（不混到 reward 就饿死）
- **个体必须真生** → 攒够 Credit 自动分裂，权重 + 突变传给后代
- **必须有选择压力** → 任务表现决定 reward，决定谁能繁殖
- **种群必须有规模** → 默认 1000，不然演化没有统计意义
- **种群必须能社交** → 信息素 + HGT，让 DNA 不只在亲子间纵向传，还能在邻居间横向传（古菌就是这样，所以项目叫 Archaea）

而**反向传播 + 巨大单体网络**这条主流路径，从这个第一性原理看，恰恰是反生命的：它把所有"个体"压成一个；它没有生死；它的"DNA"被 SGD 一步步驯化得越来越同质化，失去了演化最宝贵的多样性。

### 2.5 一句话总结这份初心

> **如果你想造的是"工具"，请继续训练大模型。**
> **如果你想造的是"生命"，请放弃训练，开始演化；放弃个体，开始种群；放弃 reward function，开始让"权重的繁衍"自己成为目标。**

Project Archaea 是这个想法的第一个可运行原型。

---

## 3. 为什么做这个项目（次级动机）

除了 §2 的主初心外，还有一些更实用层面的动机：

1. **能不能在单核 CPU 上做出有意思的智能现象？** 反 GPU 垄断，反"算力即正义"——演化算法天然适合廉价并行，不需要梯度回传所以单核就能跑。
2. **能不能造一个"完全可解释"的 AI？** 大模型是黑箱，但 Archaea 里每个 agent 就 220 个浮点数，每个变量都有生物学意义。一眼看穿。
3. **能不能给 AI-Life 领域一个"会做有意义任务"的原型？** 经典 A-Life（Tierra/Avida/Polyworld）的 agent 大多是字节码或行为脚本，不能直接对接神经科学；Archaea 让 agent 真的是 SNN，把 A-Life 和 NeuroAI 缝合起来。
4. **能不能让 AI 之间真正"社交"？** 不是 multi-agent 协议，而是像古菌那样会**横向交换 DNA**、像黏菌那样会**留下信息素互相吸引**。

Project Archaea 的定位是 **AI-Life（人工生命）** + **Neuroevolution（神经演化）** 的研究 sandbox，而非追赶 LLM 的产品。

---

## 4. 跟主流 AI 的区别

### 4.1 与大语言模型（GPT / Claude / Gemini）

| | 大模型 | Project Archaea |
|---|---|---|
| 学习信号 | 反向传播 + 梯度下降 | **演化**（突变 → 选择 → 繁殖） |
| 训练数据 | 万亿 token 语料 | **零数据**（在线生成的 Poisson 脉冲流） |
| 单元结构 | 一个巨大 NN，参数被共享 | **N 个独立小 NN**，每个是独立"生命" |
| 推理 | 静态推理（部署后不更新） | **推理与演化并行**——边用边学 |
| 死亡/出生 | 没有 | **真的死、真的生** |
| 算力 | 千卡集群 | 单核 CPU |
| 可解释性 | 黑盒 | 220 维权重 + 一个 Pearson r，一眼看穿 |
| 当前能力 | 通用语言/推理/视觉 | 只能做频率追随（L1） |

**结论**：完全不是同一种东西。大模型是「**单体超大脑的极致优化**」，Archaea 是「**群体小生命的生态演化**」。
比作生物学：大模型 ≈ 章鱼的大脑；Archaea ≈ 一缸古菌。

### 4.2 与其它 AI 算法的对位

| 算法家族 | 代表 | 与 Archaea 的差别 |
|---|---|---|
| **神经演化（NeuroEvolution）** | NEAT, HyperNEAT | NEAT 也演化拓扑，但**没有代谢、没有空间、没有社交**；适应度直接给数字。Archaea 把适应度 → 经济 → 生死 → 繁殖整套拉通 |
| **演化策略（ES）** | OpenAI ES, CMA-ES | ES 用噪声梯度近似，本质还是优化器；Archaea 是**真种群**（个体真死，不是被覆盖的样本） |
| **遗传算法（GA）** | 经典 GA + crossover | Archaea 是 **mutation-only**，故意去掉 crossover；同时加了"空间 + 信息素 + HGT"这些古典 GA 没有的生态结构 |
| **Spiking NN（STDP/Hebbian）** | Brian2, Nengo, Loihi | 它们训练**单个**SNN（用脉冲时间相关塑性更新权重）；Archaea 不在突触层学习，而在**种群层**学习 |
| **Multi-Agent RL** | MADDPG, QMIX | MARL 多 agent 共享一个 reward，目标统一；Archaea 的 agent 互相**争夺**资源（共享预算模式下），更像生态而非团队 |
| **群体智能 / Swarm** | Boids, ACO, PSO | Boids 不学习；ACO/PSO 是优化算法；Archaea 把"行为涌现"和"参数学习"耦合在一起 |
| **人工生命（A-Life）** | Tierra, Avida, Polyworld | 这是最近的亲戚。差别：Archaea 的 agent **是真正的神经网络**而不是字节码 / 行为脚本，能直接对接神经科学 |

**这个组合很可能是独特的**：SNN + 演化 + 代谢经济 + 黏菌社交 + 信息素 + HGT，没有看到现成的开源项目把它们都做在一起。

### 4.3 优势

- **能耗极低**：单核 CPU 也能跑 24 小时模拟
- **完全可解释**：每个 agent 220 个浮点数，每个变量都有生物学意义
- **真实生命特征**：种群有出生率/死亡率/Gini 系数/物种多样性指标，能直接做生态学/复杂系统研究
- **不依赖数据集**：所有"输入"在线生成；不存在"训练 vs 推理"的二分（任何时候都在演化）
- **观察价值高**：WebUI 能看到 1000 只 agent 实时的 fire / move / 死亡，很多 emergent 现象只能这样发现
- **可重复性强**：seed 可控，权重可 dump

### 4.4 劣势（必须诚实承认）

- **能力级别低**：当前只能做频率追随。距离"真正智能"还有十几个 SPEC 层级
- **任务迁移性弱**：换个任务（比如 AND/NOT）需要改 SPEC 重训
- **演化慢**：相比梯度下降，演化的样本效率低 2-3 个数量级
- **物理硬上限**：单输出神经元 ≈ 150 Hz；单 agent 220 个参数；这些都决定它做不了复杂回归
- **没有"智能涌现"的证据**：现在做的还是优化问题；从优化到智能这一跳，理论上可能但远未发生

---

## 5. 做到了什么（已落地）

### 5.1 SPEC v1.0（核心契约，全部交付）

- LIF 神经元 + 三关闸（Gate A/B/C）解析校验
- 1000 个 agent 的向量化批处理（Struct-of-Arrays 布局）
- 完整的代谢经济：reward / breath / 繁殖 / 死亡 / 替换
- Population-pressure-driven 全局突变 σ
- 自动 halt：成功（r_max > 0.7 持续 5 min）/ 失败（24 h 未达标）/ 病态（崩溃 / 灭绝）
- 结构化 telemetry：TSV 流 + checkpoint .npz + halt 诊断图
- 终端 TUI（固定表头 + 滚动 100 行）
- matplotlib 实时 dashboard（点阵 + 时序）

### 5.2 SPEC v1.0 Off-SPEC #1: 承载力 / 共享预算

可选模式 (`--budget-mode shared --carrying-capacity K`)：模拟有限资源，群体在 reward 总预算 `B = K · R_MAX` 内按需求比例分配。让群体规模能够自我调节而不是只受硬上限。

### 5.3 SPEC v1.1: 赛博黏菌（Cyber Slime Mold）

可选模式 (`--slime-mold`)：在 SPEC v1.0 之上叠加三个生物机制：

- **2D 信息素场**：高 fitness agent 在所在格留下信息素；信息素衰减 + 4 邻居扩散；处在浓信息素上的 agent 获 reward 加成（**协作的涌现**）
- **横向基因转移（HGT）**：低 Credit agent 概率性地与邻近高 Credit agent 混合权重，Donor 损失少量 Credit（**社交的涌现**）
- **趋化迁移（Chemotaxis）**：每窗 agent 概率性地朝局部信息素梯度走一步（**自组织的涌现**）
- **swarm 推理模式**：推理时不选 best 单个 agent，而选信息素热点内的所有 agent 集体投票

### 5.4 SPEC v1.2: 幅度调节（off-SPEC，本轮新增）

针对"输出被压扁"问题：

- **`calibration_lambda` (λ)**：fitness = r − λ · |Δmean| / std，给"幅度匹配"一个演化压力
- **`synapse_gain` (g)**：物理放大输出层突触电流，把单神经元 raw 上限从 ~60 Hz 推到 ~150 Hz（LIF 硬顶）
- 两者都可以**实时调节**（不重启）

### 5.5 WebUI（FastAPI + React + WebSocket）

- Setup 页：所有参数中文文档化，本地持久化（`localStorage`）
- Observe 页：1000 点点阵 + 4 张时序图 + 实时 λ/g 滑杆 + agent 详情面板（点击查看 220 维权重图）
- Use 页：单点推理 + Feedback 反馈（增减 Credit）+ Sweep 测试卡片
- Sweep 卡片支持 4 种输入模式：上升、下降、起伏、手动 SVG 点选；显示输入/原始输出/校准输出三条线 + 拟合 a, b
- 后台一致性修复：sweep 一次性 snapshot，全速 sim 下不再卡死

### 5.6 工程质量

- 41 个单元测试全绿
- 完整的训练数据审计文档（`docs/training-data-audit.md`），证明项目内**确实没有任何训练数据**
- 全部代码在 GitHub: https://github.com/stelee410/project-archaea

### 5.7 L2v2: 双输入逻辑门（已落地）+ 设计演化日志

L2v2 是 §6 路线图里"L2 双输入逻辑函数"的第一个实现，已经跑通。它把 SPEC v1.0 的单通道频率追随升级为**三通道（A / B / Selector）+ 真值表 oracle 评分**：

- **输入**：三组泊松脉冲，A/B 用 25 Hz=0 / 75 Hz=1，Selector 用 20 Hz=AND / 80 Hz=NOT
- **评分**：`oracle.py` 维护真值表，每窗对每只 agent 单独打分（不是 SPEC v1.0 的 Pearson r）
- **奖励梯度**：v2.2 软着陆 + v2.3 加权采样 + v2.4 平台-悬崖修补 + v2.5 prebiotic-stage 偏置
- **特异性仪表盘**：UI 直接显示 `acc_and_11`、`acc_not_a0` 等单行真值表准确率，把"看似学会其实在偷懒（silent attractor）"的种群一眼识别出来

#### 关键设计决策：prebiotic-stage founder bias（v2.5）

在 v2.4 部署后，新种子从 `Uniform(-1.5, 1.5)` 起步，所有 founder 的 `Σw ≈ 0 ⇒ I_o ≈ 0 ⇒ f_out = 0`。结果是：

- 全 silent → 净 credit < 0 → 永远不繁殖 → 永远不突变 → 永远 silent → ~5 分钟饿死
- **演化的"变异-选择-繁殖"循环根本启动不了**——这是 evolvability=0 问题，不是 fitness landscape 问题

v2.5 把 L2v2 founder 分布改为 `Uniform(-0.5, 1.5)`：

- 期望 Σw 为正，~30-40% 的 day-zero founder 直接能产生 spike
- 负权重仍占 25%，超过 SPEC §3.1 的 ≥20% 下限，抑制路径仍可由突变发现
- **不是 intelligent design**——没有改演化算法、没有改评分规则、没有 hardcode 任何"会答 AND"的权重，只是把"研究问题"从「random matter 能否 abiogenesis 出神经计算」（abiogenesis 问题，~10⁹ 年）调整为「已能输出信号的网络能否演化出逻辑门」（演化问题，可计算）
- **学术先例**：Lenski 长期演化实验（LTEE）也是从一个完整的 E. coli 起步，不是从原子。我们做的事是同一性质的"prebiotic selection"

详见 `archaea/population.py` 中 `L2V2_WEIGHT_INIT_LOW` 上方的注释块，以及 `archaea/oracle.py` 模块 docstring 的 ERRATA v2.2-v2.5 历史。

### 5.8 SPEC_L2_V3.0：杂交皿（Admixture Mixer）

L2v2 跑通后只剩一个顽固阻塞：**「偏科」陷阱**——在同一个培养皿里 AND 学得很顺（>90%），NOT 总是不出来（~50% 永远徘徊）。原因是 NOT 在生物意义上确实更难（需要抑制电路 + 选择器通道的 contextual gating），而种群一旦把 AND 学成"主业"就再也没有突变压力去碰 NOT。

仿照微生物演化（Lenski + 大量自然界 HGT 实验），引入「杂交皿」实验范式。三个核心概念：

- **菌株（Strain）** — 把"某个时刻的全部活体"冻存为可重用的快照，包含权重 + task / difficulty / acc 元数据。和 `champions.py` 的 top-K elites 不同：菌株存的是**全部基因池**（最多 pop_max），admixture 需要代表性而不是精英。
- **杂交（Admixture）** — 启动新仿真时，从两个或多个已保存的菌株里按比例抽样填充初始 slot（剩下空槽走默认随机 init 当背景噪声）。
- **杂交期（Admixture Window）** — 启动后前 N 秒仿真时间内，HGT 概率 × 倍率（典型 ×5–×10），模拟"两群古菌在新培养皿里相遇并大幅交换基因"的短促事件，结束后自动恢复基线。

实验剧本：先在两个独立环境里养出 AND-学家、NOT-学家两个专家菌株，然后倒在一起 → 杂交期内 HGT 把"会 AND 的隐藏神经元"和"会 NOT 的隐藏神经元"组合到同一个 agent 上 → 自然涌现真正的「双修」个体。

工程边界：

- 杂交皿**不是新的演化任务**（task 仍是 L2v2_ctrl 等已注册任务），只是一种**初始化方式**——`Population.spawn_from_strains()` 替换默认随机 init
- 启动时若启用 admixture window，会**自动开启 slime grid + HGT** 但**关闭信息素奖励 K=0**（避免破坏 oracle 真值表）
- `task` 必须与所有 founder strain 一致，跨任务杂交直接拒绝（避免输入维度 / 评分规则错位）
- 持久化：`checkpoints/strains/<id>.npz` + `<id>.json` sidecar；菌株库列表只读 JSON，不加载权重，所以瞬时

API（详见 `SPEC_L2_V3.0_admixture.md`）：

- `POST /api/strains/save` — 把当前活体冻存为菌株
- `GET  /api/strains` — 菌株库列表（无须加载权重）
- `DELETE /api/strains/{id}` — 删除菌株
- `POST /api/start` 新增字段 `founders=[{strain_id, fraction}, ...]` + `admixture_window_s` + `admixture_hgt_multiplier`

WebUI 入口：图鉴里新加一张 **⚗️ 杂交皿 · Mixer** 卡片 → MixerPage 列出菌株库、勾选 founder、调比例、设杂交期长度 → 启动后自动跳到对应群落的 observe 页围观。ObservePage 顶部新增 **🧪 菌株 / 杂交皿** 条：右侧的「💾 保存为菌株」弹窗一键冻存当前种群；杂交期内会显示一道脉冲徽章 + `eff_hgt_prob` 实时数字。

### 5.9 ERRATA v3.1：专家培养皿的两个隐性 bug（A + B 修订）

把 `and_only` / `not_only` 两个**单门培养皿**接到 SetupPage 之后，第一次实测就发现 NOT 培养皿表现极差（35 min 跑完 acc_NOT ≈ 34%，**比随机还差**），而 AND 培养皿轻松 91%。复盘发现两个相互独立、但都只在 specialist 模式下才显形的旧 bug：

**Bug A — `_fitness_defined` 的合取条件在 specialist 下永远 False**

`Population._fitness_defined()` 在 L2v2 里一直要求 `_acc_and_n[slot] > 0 AND _acc_not_n[slot] > 0`——这是 v2.0 时代故意加的守卫，避免新生 agent 只看过 AND 窗口就被当作"也会 NOT"。但在 `and_only` (`p_mode_and=1.0`) 或 `not_only` (`p_mode_and=0.0`) 里，另一边的 `_acc_n` 永远为 0，于是**整个种群一辈子都是 fitness undefined**，连锁后果：

- `global_sigma()` 取不到任何有效个体的 fitness → `mean_f=0` → sigma 永远卡在 `SIGMA_BASE=0.3` 顶格，**突变退火失效**
- `_pick_replacement_victim` 把所有 agent 的 fitness 当 `-1e300` → 退化成只看 credit 的胜者，**精英保护失灵**
- 冻存的菌株元数据里 `fitness_mean / fitness_max` 全是 0，菌株库看上去像是"没学到任何东西"

修法：从 `task_difficulty` 派生两个常量 `_needs_and_samples` / `_needs_not_samples`，`_fitness_defined` 只要求"环境会出现的模式"都至少有 1 个样本即可。同时把 `_logic_fitness_slot` 也改成只对**实际出现过的模式**取均值——否则一个完美 NOT-学家上限是 0.5，sigma 退火的动态范围被压扁一半。混合培养皿（balanced/uniform/...）的旧契约不变，回归测试已固化。

**Bug B — `not_only` 缺一个"抑制端 prebiotic stage"**

v2.5 把 founder 权重范围从 `Uniform(-1.5, +1.5)` 改成 `Uniform(-0.5, +1.5)`（兴奋性偏置 E[w]=+0.5），让 founder 当天就能输出脉冲、点火 evolvability——这是混合 / and_only 培养皿的救命符。但**这个偏置对 NOT 是反向的**：NOT 要求"a=1 时静默"，兴奋性 founder 离 NOT 解的距离是任意一支的最远端，靠 +1.3/win 的微弱选择压力把抑制电路从零突变出来不现实（这正是实测 acc_NOT < 50% 的根因）。

修法：仅对 `not_only` 培养皿启用**镜像分布** `Uniform(-1.5, +0.5)`（E[w]=-0.5；约 25% 仍为正，留给突变发现兴奋性路径用）。生物对齐："prebiotic selection" 的同一招——既然我们研究"已具备某种倾向的 founder 能否演化出特定逻辑"，那么 NOT 培养皿就应该从已经倾向于抑制的种子开始，而不是要求兴奋性 founder 自己**先**反转再**再**学逻辑。代码：5 行新常量 `L2V2_WEIGHT_INIT_LOW_INHIB / HIGH_INHIB` + `_init_weights_for_task` 里加一个 `if self.task_difficulty == "not_only"` 分支。

两个改动加起来 ~30 行 + 6 个回归测试（覆盖 specialist 各自的 fitness 收敛、sigma 退火、混合皿契约不破坏、两个新权重分布的均值/负权占比）。

### 5.10 ERRATA v3.2：not_only 的 silent attractor 仍然存在 — 加 wrong-answer penalty

实测 v3.1 之后，`not_only` 跑了 43 分钟仍然是 100% silent 种群（仪表盘：存活 700 / 累计死亡 0 / NOT 0=1 命中率 0%）。复盘发现 v3.1 的 fitness 退火和抑制 founder 都解决了，但**根因不在那里**——根因在 v2.2 奖励表本身和 specialist 培养皿的化学反应。

**Silent 在 not_only 里是赚钱的策略**。代入 v2.2 数字（`R_NOT_SILENT_RAW=10` → scaled 2.5）算 silent agent 的每窗口净收益：

| 窗口类型 | 概率 | reward | 净收益 |
|---|---|---|---|
| `target=0` (a=1, silent 答对) | 50% | 2.5 + 0.1 effort − 1.25 breath | **+1.35** |
| `target=1` (a=0, silent 答错) | 50% | 0 + 0 − 1.25 breath | **−1.25** |
| 平均 |  |  | **+0.05 / 窗口** |

Silent 净收益是**正数**——agent 永远不会饿死，没有自然选择压力，整个种群被锁死在 silent attractor。和 `and_only` 的对照特别说明问题：`R_AND_SILENT_RAW=8`（scaled 2.0）下，AND 培养皿的 silent 净收益是 **−0.20/win**，自然就饿死了，所以 `and_only` 演化得动。NOT 之所以涌现不出来，是因为 v2.2 当年把 `R_NOT_SILENT` 调高了（10 vs 8）以保证 NOT 模式在混合皿里能有足够的 silent 收入"凑数"——这个为混合皿设计的安全垫，搬到 NOT 单门皿里就成了evolution killer。

**修法**：在 specialist 培养皿里把 `reward_wrong` 从 0 改成 **−BREATH_PER_WINDOW = −1.25**（scaled credit）。这等价于"答错的窗口正好抵消一次代谢成本"。混合皿不动（保持 v2.2 的"silent 勉强糊口"契约——避免新种群在第一个学家涌现前集体灭绝）。

代入新数字（not_only）：

| 策略 | 净收益 / 窗口 | 含义 |
|---|---|---|
| Silent | (−2.5 + 1.35) / 2 = **−0.575** | 87 窗口 (≈44s) 饿死 |
| Always-spike | (+23.85 + −2.5) / 2 = **+10.675** | 仍然繁荣（兴奋性 founder 不会初代灭绝）|
| Smart NOT | (+23.85 + 1.35) / 2 = **+12.6** | 比 always-spike 快 **18%** 繁殖 |

Silent 终于死掉，自然选择恢复；always-spike 活但不是最优，给 smart NOT 留出明确的繁殖速度优势。

工程足迹很小：
- `TASK_DIFFICULTY_PRESETS` 加第四个字段 `reward_wrong`（混合皿全部 0；`and_only` / `not_only` 都是 `−BREATH_PER_WINDOW`）
- `draw_oracle_sample` 新增 `reward_wrong` 关键字（默认 0），写入 `OracleSample.reward_wrong`
- `Population.step_window` 用 `_difficulty_weights.get("reward_wrong", 0.0)` 提取后传给 oracle
- 4 个新回归测试（specialist preset 携带 `−BREATH`、`OracleSample.reward_wrong` 正确传播、4 种策略期望收益排序、混合皿 `silent_balanced` 与 specialist `silent_not_only` 的 gap 至少 0.4/win）

为什么不直接降 `R_NOT_SILENT_RAW`（看上去也能让 silent 饿死）？因为 v2.2 那个数字对**混合皿**仍然是必要的（NOT 模式占 50% 窗口，silent 在 NOT 上的收入是混合皿存活率的一半来源）。改它会破坏混合皿的"软反崩塌"契约。`reward_wrong` 是更聚焦的 knob——只在专门 specialize 的环境里启用"答错就罚"，混合皿的"答错只赔 breath、不赔加项"原样不动。

生物对齐：从"耕地条件"角度看，混合皿是**多样化生态**（错了也有别的食物可吃，类似机会主义食腐），specialist 皿是**单一资源生态**（错了直接饿死，类似严格的 obligate parasite）。这两类生态的代谢经济本来就该不一样。

### 5.11 ERRATA v3.3：Path D1 — `not_only` 用"反跟随结构 seed"取代抑制偏置

**v3.1 + v3.2 实测结局：**`not_only` 培养皿 founder **全员第一时间饿死**——v3.1 给的"抑制 founder 分布" `Uniform(-1.5, +0.5)` 让所有 founder 都是 silent（E[Σw·s] < 0 → I_o 永远到不了阈值），叠加 v3.2 的"silent 罚款"，整窝在前几窗口集体灭绝。两个本来各自合理的修法**互相否决**。

**复盘**：v3.1 的诊断错了一层。NOT 不是"权重普遍负"那么简单——NOT 是 **context-gated routing**：A 高时压住 output，A 低时让一个 tonic 源把 output 推过阈值。随机或均匀负偏置的 founder **几乎不可能**在 colony 实际跑得起的时间内突变出这种"路由拓扑"——搜索空间太大。问题的根本是 abiogenesis 难题：从噪声里发明一个特定电路 ≠ 在已有电路上调参数。

**v3.3 改用 Path D1：把"噪声先验"换成"结构先验"。** 每一只 `not_only` founder 出生就携带一个手工设计的反跟随微电路：

```
Hidden 0..9   = A-detectors:
    A→hidden:  Uniform(+1.5, +3.0)   ← 强正：跟随 A 通道
    hidden→out: Uniform(−4.0, −2.0)  ← 强负：A 高时压输出
Hidden 10..19 = S-tonic drivers:
    S→hidden:  Uniform(+3.0, +5.0)   ← 强正：S=80Hz 持续点亮
    hidden→out: Uniform(+1.0, +2.5)  ← 正：常态推动输出
其他位置:      Uniform(−0.2, +0.2)   ← 小噪声，给突变留可发现的底物
```

**S=80Hz 时的预期行为**（实测验证一致）：

- a=1（75Hz）→ A-detectors 发放 ~50Hz → 压住 output → **6Hz**（远低于 20Hz 阈值）✓
- a=0（25Hz）→ A-detectors 静默 → S-tonic 主导 → **32Hz**（清晰高于阈值）✓

每只 founder 在结构带宽内独立采样噪声，所以选择压力依然有"个体差异"可作用；被 D1 抹掉的只是"凭空发明路由"那一步**不可达**的搜索负担。

**仿真验证**（200 只 founder，200 窗口，无任何外部干预）：

| 时刻 | 存活 | 平均 credit | NOT 命中率 |
|---|---|---|---|
| 出生 | 200 | 50 | — |
| win 50 | 200 | 133 | **98.12%** |
| win 200 | 200 | 163 | **97.62%** |

对照 v3.1+v3.2：win 50 NOT 命中率 0%、累计死亡 700。**6 个数量级以上**的能效差距。

**生物对齐**：D1 不是"intelligent design 反对达尔文"，它就是 LTEE 的标准操作——Lenski 不会从化学汤开始养 *E. coli*，而是从一只**已经有完整代谢机构的祖先**起跑。我们的 not_only 也只是"研究 NOT 拓扑能否被精修"，而不是"研究 NOT 能否从随机权重中 abiogenesise"——后者是 abiogenesis 问题，本来就不该让 colony 在分钟级时间尺度承担。生物界精确对应的现象包括：

- **GABA 能神经元的演化起源**：从 glutamatergic 祖先经一次离子通道极性翻转变出抑制性输出
- **视网膜 ON / OFF 通路**：相同电路、相反极性，单突触级别的差异
- **Pax6 跨胚层逆向调控**：同一个转录因子在不同组织里产生相反结果

代码足迹：
- `population.py` 加 11 个 seed 常量 + 一个 `_init_weights_l2v2_not_seed()` 函数（~50 行）
- 删除 v3.1 的 `L2V2_WEIGHT_INIT_LOW_INHIB` / `HIGH_INHIB`（混合皿契约不变）
- 4 个新回归测试：拓扑结构（A→A-det 正、A-det→out 负、S→S-tonic 正、S-tonic→out 正）、个体多样性（per-cell std > 0、所有 founder 唯一）、行为功能（a=high 静默 + a=low 发放）、混合皿不被影响（balanced/uniform/extreme/and_only 仍走 v2.5 默认分布）

哲学注脚（用户提出）：D1 印证了一个被 RNA-world / Maxwell's demon / 柯尔莫哥洛夫复杂度反复给出的结论——**演化不是从无到有的信息创造，而是已有信息的精炼**。"原罪"这个比喻挺贴切：那个非选择得来的、却又是后续一切选择得以发生的初始结构，必须有人付了那笔信息税。在我们这里，那个"原罪"就是 D1 写在 founder DNA 里的反跟随拓扑。

---

## 6. 未来演进方向

### 近期路线图：L2 — 双输入逻辑函数

把 input 从一组扩展为 **两组独立 channel**（比如 5+5 个 input neuron），让群体演化出：

- `f_out ≈ AND(f_a, f_b)`：两路输入同时高才输出高 — **L2v2 已落地** ✓
- `f_out ≈ NOT(f_a)`：输出与输入反相 — **L2v2 已落地** ✓
- `f_out ≈ XOR(f_a, f_b)`：经典非线性可分性测试 — **推迟到 L2.5+**
- `f_out ≈ OR(f_a, f_b)`：本可顺手做但被 §5.7 的奖励梯度调试占用了精力，同样推迟

> **当前实现状态**：详见 §5.7 — L2v2 已实现 AND + NOT 两种逻辑门 +
> Selector channel（指令通道）+ 真值表 oracle 评分；OR / XOR 留给 L2.5。

**为什么这个台阶很重要**：L1 的频率追随其实是**线性相关**任务，单个隐藏神经元就能近似。AND/XOR 强迫群体演化出**真正的非线性**，是从"传感器"到"逻辑门"的跨越。如果 XOR 能演化出来，意味着 20 个隐藏神经元里至少能涌现出一个"协同激活"的小回路。

工程上需要的改动很小：

- 新 `stimulus.py` 函数：`poisson_two_channels(f_a, f_b)` ✓ (实际是 `poisson_three_channels`，多了 Selector)
- 新 fitness 函数：把 `pearson_r(f_in, f_out)` 改成真值表评分 ✓ (`oracle.py::OracleSample`)
- WebUI Sweep 卡片增加 2D heatmap 模式（`f_a` × `f_b` → `f_out`）— 推迟，L2v2 用了更直接的"逻辑测试器"代替

预计是一个 weekend project。**实际花了远超 weekend** 的时间——大头不在工程，在调出"silent attractor → platform-cliff → founder-collapse"系列演化生态学陷阱（见 §5.7 设计演化日志）。

### 远期路线图（按抽象层级递进）

**L3 — 时间记忆 / 序列预测**
给 agent 加递归连接（hidden → hidden 的少量循环边），让群体演化出"短时记忆"。任务：预测下一个脉冲时刻、识别简单 pattern（A-B-A-B vs A-A-B-B）。这是从"反应"到"预测"的跨越。

**L4 — 拓扑可演化（NEAT 化）**
打破 10→20→1 的固定结构。突变除了改权重，还能：加节点、加连接、删连接。**演化的不再只是参数，而是架构本身。** 这是 Archaea 走向 NEAT/HyperNEAT 的合并点。

**L5 — 多任务 / 元学习**
同一种群在多个任务间交替（比如 AND 一段时间，然后 OR，然后追随）。期待涌现"通用 agent"——不是某个任务的专家，而是能快速适应任意频率任务的 generalist。这考验种群的"基因池广度"。

**L6 — 物种分化（Speciation）**
当任务足够丰富，期望群体自动分裂成多个生态位（niche）：一些 agent 专攻 AND，一些专攻 OR，互不竞争。这需要在繁殖规则里加"基因距离"——和你太像的不会繁殖（避免内卷），形成 species boundaries。

**L7 — 模仿 / 教学（文化继承）**
除了基因（突变）和 HGT（基因水平传递），加第三种信息传递：**行为模仿**。年轻 agent 观察老 agent 的输入输出对，"猜"出权重并迁移。这是从"基因演化"到"文化演化"的飞跃，对应人类的语言起源。

**L8 — 嵌套尺度（Fractal Life）**
让单个 agent 内部也是一个小 Archaea：20 个隐藏神经元不再是固定连接，而是 20 个小 sub-agent 的种群。形成 **agent-of-agents** 的两层结构，对标多细胞生物的诞生（单细胞 → 群居 → 多细胞）。

**L9 — 真实物理输入**
接入摄像头/麦克风，让群体活在真实物理世界。第一个 milestone：让一缸 Archaea 学会跟踪屏幕上移动的光斑。这一步把项目从"理论玩具"推到"行为机器人"。

**L10 — 群体编程群体（元演化 / Open-endedness）**
最远的远方。让群体自己产生"任务"——一部分 agent 充当 stimulus generator，给另一部分 agent 出题；出题质量决定出题者的 fitness。系统进入**开放式演化**，没有外部目标函数，只有内生的"互相为难、互相成就"。这是 A-Life 领域几十年没破解的圣杯。

---

## 写在最后

Project Archaea 的真正赌注不是"超过 GPT"，而是回答 §2 那个被主流 AI 跳过的问题——**AI 时代的 DNA 是什么**。

我的回答是：**权重是 DNA，种群是生命。**

> **如果智能从来不是被"训练"出来的，而是被"孕育"出来的，那它需要的不是更多数据，而是更复杂的生境。**
> **如果生命从来不是单个个体，而是一条沿着时间轴流动的基因河，那 AI 也不应该是一个被冻结的大模型，而应该是一缸不停代谢、繁殖、死亡的小生命。**

这个项目还很小、很慢、能力很有限，但它有一个大模型永远没有的东西——
它的每一只 agent，**会真的死**；
而它的 1000 只 agent 组成的那个种群，**会因为有人死而变得更聪明**。

这才是生命的样子。
