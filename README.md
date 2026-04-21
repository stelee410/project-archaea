# Project Archaea (L1)

Python ≥ 3.11 实现 `SPEC.md` 的 L1 工程契约：在新陈代谢经济压力下，对 10→20→1 的脉冲神经网络（SNN）个体做**仅突变**演化，目标是让某个个体的输出脉冲发放率与输入发放率的 Pearson 相关 r ≥ 0.7。

> 本 README 中文 + 英文混排：命令/字段名保持英文以便复制；解释性段落用中文。

---

## 你究竟在观测什么？

> **「一群随机布线的小神经网络，在『听到输入脉冲发得越快，自己输出脉冲也发得越快』就能多挣 Credit、否则饿死」的经济压力下，仅靠**繁殖时的随机权重突变**演化下去——能不能在 24 小时仿真时间内，**至少有一个个体**让自己输出脉冲率与输入脉冲率的 Pearson 相关 `r ≥ 0.7`。**

整个 L1 实验只做一件事：**测「mutation-only 进化 + 这套经济参数」能不能产出 rate-tracking**。所有日志、图、表都只是观测它有没有发生。

### 核心指标：Pearson r

每个 agent 自己保留**最近 40 个 500 ms 窗**的 `(f_in, f_out)`，算 Pearson **r**（输出方差为 0 时强制 r=0，封掉「躺平输出常数」漏洞）。

| r | 「智商」 | 经济结果 |
|---|---|---|
| ≤ 0 | 完全不跟踪输入，沉默或恒定输出 | 净 −2.5 Credit/s，必死 |
| ≈ 0.25 | 弱跟踪 | 收支打平 |
| 0.5  | 中等 | +2.5/s，会快速繁殖 |
| **≥ 0.7** | **达成目标** | 程序记 `t_first_success`，存归档 |
| 0.8  | 强跟踪 | +5.5/s，约 27 s 仿真就能繁殖 |

---

## 怎么判断「成功 / 失败 / 卡住」

直接对照 `SPEC.md` §7，程序也已实现自动 halt：

### 成功（success）
- 仿真任意时刻 **`r_max ≥ 0.7`**（曲线面板上的 `r_max` 触到 0.7）。
- stdout 最终打印 `SUCCESS recorded at t_first_success=...s`，并把曲线/直方图存到 `diagnostics/`。

### 半程检查（必看）
- **T+2h（仿真时间 7200 s）**：若 `r_max < 0.3` → **直接 fail**，存 `diagnostics/t2h_*` 退出。
- 也就是说，**最迟仿真 2h** 必须看到 r 明显抬头，否则后面也基本没希望。

### 病理性 halt
- **灭绝**：存活 < 10。
- **单一文化**：top-100 个体的权重标准差 < 0.01（基因池塌缩）。
- **停滞**：`r_max` 在最近 2h 仿真内的最大值都没涨过 0.05，且仍 < 0.7。

> 注意：「成功」**不是**工程契约的硬要求；契约只要求**装置可运行、可复现、有遥测、自动 halt**。能否真的演化出来是科学问题。

---

## 控制台列在告诉你什么

每行一个 500 ms 仿真窗，Tab 分隔：

`t_sim`, `pop_size`, `births`, `deaths`, `r_max`, `r_mean`, `credit_mean`, `credit_gini`, `weight_std`, `sigma`, `budget_pressure`

| 列 | 看它做什么决定 |
|---|---|
| `t_sim` | 推进多少仿真时间；和 wall-clock 对比可估总耗时 |
| `pop_size` | 应稳定接近 `pop_max=1000`；持续下滑 → 经济压力过大或正在灭绝 |
| `births` / `deaths` | **进化引擎的脉搏**：长期为 0 = 没有人达到繁殖线，等于停摆 |
| **`r_max`** | **核心指标**：要看到它**整体抬升**，最终 ≥ 0.7 |
| `r_mean` | 群体水平；若 `r_max` 涨但 `r_mean` 不动，只有少数人在学，多数搭便车 |
| `credit_mean` | 群体「饱腹度」。若一直 < 50，说明大家普遍营养不良 |
| `credit_gini` | Credit 不平等。**0.5+** 表示「能量大户」（少数高 r 个体在垄断繁殖资源），其实是**正常且健康**的进化信号 |
| `weight_std` | **基因多样性**。从初始 ≈ 1.73 应缓慢下降；急速跌到 < 0.5 要警惕 monoculture |
| `sigma` | 突变步长 `σ = 0.3 × exp(−2 × max(0, r_mean))`；从 0.30 往下 = 平均 r 在涨，**隐含进度条** |
| `budget_pressure` | **[off-SPEC]** 共享预算模式下的 `D/B`。`0` = 该模式关闭；`< 1` = 资源富余，纯 meritocracy；`= 1` = 刚好踩在承载力上；`> 1` = 全员被等比削减奖励，群体停止扩张并自发收敛 |

补充：
- `r_max` / `r_mean` 只统计**已定义适应度**的个体（≥ 40 窗历史）。
- `weight_std` 是 220 个权重位置上「跨个体 std」再取平均。

---

## 图形面板对应观测什么

启动方式见后文 *Live dashboard*。

**上：点阵**
- 颜色越来越绿 → 群体普遍 Credit 高 → 经济在赚，整体在适应。
- 越来越多深红/灰 → 经济收紧或正在灭绝。
- 粉点频繁出现 → 繁殖事件密集 → 演化引擎在转。
- 粉点很少 → 几乎没人达到 200 Credit → 学习卡住。

**中：曲线（这就是判官）**
- **`r_max` 向 0.7 攀升** = **正在成功**。
- `r_max` 长期贴 0、`r_mean` 也不动 = **没有任何学习信号**。
- `N` 阶梯线掉下来 = 种群在塌；红条变密 = 死亡浪潮。
- `sigma` 紫虚线慢慢下移 = 平均适应度在抬，**进化收敛中**。
- 底部绿/红条带的密度比 = 净增减；持续红 > 绿 = 走向灭绝。

**下：预算紧张度（仅启用 shared-budget 时有意义）**
- 红线 `D/B`、灰虚线在 `y=1`。
- `0` = 没启用 / 没人有适应度；`< 1` = 资源富余，行为同 SPEC；`> 1` = 群体被等比削减，开始自我抑制。
- 看到红线**贴着 1 来回小幅波动** = 已找到承载力均衡。
- 红线**单调爬升** = 群体仍在尝试扩张，但每人份额下降；点阵中红色（低 Credit）会变多。

---

## 两条最简单的「肉眼判定法」

1. **第一遍跑（几分钟仿真）**：盯 `r_max` 那条蓝线。
   - **从 0 慢慢往上爬**（哪怕到 0.2、0.3）→ 引擎活着，继续等。
   - **始终 ≈ 0**，但 `births` 不为 0 → 引擎在转、但选择压力没找到方向；看 `weight_std` 是否还在 1+，否则就是过早收敛。
   - **`births` 从某点起一直是 0**：要么大家全饿死了，要么没人到 200 Credit，**实验事实上停摆**。
2. **过仿真 2h（7200 s）那一刻**：程序自己会判官——若没自动 halt，说明 `r_max ≥ 0.3`，**还有戏**。

---

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 主程序

```bash
python -m archaea.run --seed 42 --duration 86400 --pop-max 1000 --log run.log
```

在**真实终端（TTY）**上，stdout 把**表头固定在屏幕最底一行**，数据在其上方滚动重绘（绝对光标定位；缓冲最多 `--console-rows` 行，默认 100）。使用备用屏 `\033[?1049h/l`，退出时自动恢复。`run.log` 始终是**纯 TSV 追加**。

强制传统逐行输出或写入文件时：

```bash
python -m archaea.run --seed 42 --duration 60 --pop-max 50 --log run.log --plain-stdout
python -m archaea.run --seed 42 --duration 60 --pop-max 50 --log run.log --console-rows 50
```

### 通用参数

| 参数 | 说明 |
|---|---|
| `--seed` | RNG 种子（默认 `42`）。同种子 + 同代码/依赖 → `run.log` 应**逐字节一致**。 |
| `--duration` | 仿真秒数（默认 `86400` = 24 h）。 |
| `--pop-max` | 种群上限（默认 `1000`；Gate C 测试用 `100`）。 |
| `--n-initial` | 初始存活个体数（默认等于 `--pop-max`）。可单独设小值做「最小启动」。 |
| `--log` | TSV 日志路径（默认 `run.log`）。 |
| `--plain-stdout` | TTY 上禁用滚动表，回到「一窗一行」模式。 |
| `--console-rows` | 滚动表保留的最大行数（默认 100）。 |
| `--carrying-capacity K` | **[off-SPEC]** 共享预算模式的承载力 `K`。`0` = 关闭（默认，等价 SPEC §4.4）。 |
| `--budget-mode {none,shared}` | **[off-SPEC]** `none`（默认）= SPEC；`shared` 启用承载力模型，需配合 `--carrying-capacity K>0`。详见下方专节。 |

工作目录会被切到项目根目录，使 `diagnostics/`、`checkpoints/`、`run.log` 路径与 SPEC §10 对齐。

---

### 最小启动（minimal seed → grow to cap）

`--n-initial` 与 `--pop-max` 完全独立，可以从 **1 个祖先**起步、上限设到 1000：

```bash
# 单祖先起步，扩张到 1000；24h 仿真
python -m archaea.run --seed 42 --duration 86400 --pop-max 1000 --n-initial 1 --log run.log

# 10 个起步，扩张到 1000；演化更稳健
python -m archaea.run --seed 42 --duration 7200 --pop-max 1000 --n-initial 10 --log run.log
```

注意事项：

1. 单个体的初始 Credit = 50，呼吸 1.25/窗，**前 20 s 完全没有适应度**（Pearson 需要满 40 窗历史），所以**前 20 s 必死 0.4 倍 Credit**——`50 → 2.9`，几乎贴零；这是在等「第一次评分」。一旦 r > 0.25 就反向赚回。
2. **灭绝判据已自适应**：原 SPEC §7.4 的「`n_living < 10` → halt」改为「**先到过 10 才算**」（高水位线）。所以 `--n-initial 1` 不会立刻 halt；只有人口曾经长到 10+、之后又崩到 < 10 才算灭绝。`n_living = 0` 始终算灭绝。
3. **σ 与 sigma 进度条仍然有效**：单祖先时 `r_mean = r_max`，σ 一旦下降就说明祖先已学会跟踪，能开始繁殖。
4. **可视化**：点阵会显示出绝大多数空槽，新生儿粉点会从某个角落开始扩散，很有「殖民」的观感；用 `python -m archaea.visualize --pop-max 1000 --n-initial 1 --duration 600 --every 5` 直接看。

---

### 承载力 / 共享预算（off-SPEC，可选）

> **动机**：SPEC §4.4 的奖励是「**每人独立**拿 `max(0, r) × R_MAX`」，资源**无限**——意味着 `pop_max` 是一堵硬墙，群体只会一路指数膨胀直到撞墙，没有任何「**资源稀缺**」的反馈。一个真实的生态系统应该在硬顶之前**自发**收敛到某个**承载力 `K`**。
>
> 本节描述一个可选的扩展，**默认关闭**；启用后 SPEC §4.4 的语义被扩展（其它常数不变）。

**模型**：每窗整个群体共享固定奖励预算

```
B = K × R_MAX           （= K × 5 单位 / 窗）
demand_i = R_MAX × max(0, r_i)   （未定义适应度按 0）
D = Σ demand_i
ratio = min(1, B/D)              （供大于求时不打折）
reward_i = demand_i × ratio
```

- 当 `D ≤ B`（资源富余）：每人吃满 demand，行为与 SPEC 完全一致。
- 当 `D > B`（资源紧张）：**所有人**按 `B/D` 同比例打折，**强者依然拿得多但绝对额下降**——meritocracy 与稀缺并存。
- `budget_pressure = D/B` 写进日志的第 11 列；可视化第三块面板有曲线 + `y=1` 参考线。

**均衡态的数学直觉**：群体会停在「平均个体的 reward ≈ 呼吸费」的 `N*` 处。粗略估算

```
N* ≈ K × R_MAX × r_mean / BREATH_PER_WINDOW
   = K × 5 × r_mean / 1.25
   = 4 × K × r_mean
```

举例 `K=30`、`r_mean=0.85` → `N* ≈ 102`，远小于 `pop_max=1000`。**这就是「聪明族群在硬顶之前自发收敛」的样子**。

**启用**：

```bash
# K=30，群体上限留 1000 当安全阀
python -m archaea.run --seed 42 --duration 86400 \
    --pop-max 1000 --n-initial 50 \
    --carrying-capacity 30 --budget-mode shared \
    --log run.log
```

可视化也对称支持：

```bash
python -m archaea.visualize --seed 42 --duration 120 \
    --pop-max 200 --n-initial 100 \
    --carrying-capacity 30 --budget-mode shared \
    --no-show --save diagnostics/dash_budget.png
```

**怎么调 K**：
- `K` 越**小** → 资源越紧 → 均衡 `N*` 越小 → 选择压**更强**，但繁殖事件可能完全停摆（reward 全部贴在呼吸费上）。
- `K` 越**大** → 接近 SPEC（`K = pop_max` 时基本无差别）。
- 经验起点：`K ≈ pop_max / 5`，配合 `--n-initial ≈ K`。

**与 SPEC 的关系**：本扩展**不破坏** SPEC §2 / §3 / §6 的任何契约（神经元、刺激、Gates 全部按原样跑）；只是替换了 §4.4 的奖励分配规则，且通过 `--budget-mode none`（默认）即可完全回退到原 SPEC。

---

### 🍄 赛博黏菌 / Cyber Slime Mold（SPEC v1.1，off-SPEC，可选）

> **动机**：SPEC v1.0 跑出来的"种群"本质上是**并行评估池**——所有 agent 看同样的输入，互相不通信，仅通过资源/槽位**间接竞争**。"群体"的存在感很弱：去掉除了 top-10 之外的所有个体，几乎不损失任何东西。
>
> v1.1 在不破坏 v1.0 evolutionary core 的前提下，加入**三个相互耦合的群体机制**，把这个"评估池"改造成一个**真正的去中心化协作系统**。设计哲学：保留 agent 个体性（不是单一大脑），通过**间接通信信道**让群体涌现出"觅食网络"般的自组织结构。

#### 三个机制 = Agent 三要素的两条短板补全

| 机制 | 对应 | 生物学原型 |
|---|---|---|
| **信息素场** (pheromone field) | **协作** (stigmergy) | *Physarum* 的化学轨迹 |
| **HGT** (horizontal gene transfer) | **社交** (lateral learning) | 古菌真实生物机制 |
| **趋化迁移** (chemotaxis) | **社交 + 协作** | 黏菌沿梯度移动 |

繁殖（突变）已经在 v1.0 完整实现；v1.1 补全的是社交和协作。

#### 1. 信息素场（协作 / 共识形成）

把种群放在 G×G 的环面网格上（默认 16×16）。每窗顺序：

1. **感知 → 奖励放大**：站在浓痕格子上的 agent，奖励 ×(1 + K · P_local / P_max)。这是**正反馈**：高 fitness → 高 reward → 富足 → 留下浓痕 → 吸引更多 agent → 更高浓痕 → ...
2. **HGT** 执行（见下）。
3. **释放**：每个有定义 fitness 的 agent 在自己格子留下 `emit × max(0, r)` 的信息素。
4. **挥发 + 扩散**：`P ← (1−decay) · P` 然后 `P ← P + diffusion · Laplacian(P)`（4-邻域、周期边界）。默认 decay=0.05（半衰期约 14 窗 = 7s）、diffusion=0.20。
5. **趋化迁移** 执行。

#### 2. HGT 横向基因转移（社交 / 横向学习）

每个低 credit agent 每窗有 `hgt_prob`（默认 2%）概率，在 Chebyshev 半径内查找 credit ≥ `hgt_donor_ratio`（默认 2×）倍于自己的邻居。匹配则：

```
W_self ← (1 − η) · W_self + η · W_donor       η = hgt_blend = 0.30
credit_self ← credit_self − hgt_cost          hgt_cost = 5.0
fitness_history_self ← cleared (40 窗重新累计)
```

历史清零是关键：旧的 (f_in, f_out) 记录已经不能反映新权重，必须从头测量。

这是**非繁殖的横向学习**——SPEC v1.0 里只有"父→子"的纵向传递，HGT 让群体级别的"知识"以非血缘方式扩散，可以救回低 credit agent，避免基因池过早收敛。

#### 3. 趋化迁移（社交 / 自组织）

每个 agent 每窗 `migrate_prob`（默认 30%）概率沿信息素梯度走一格（3×3 Moore 邻域 argmax，含原地不动）。配合信息素场，群体会自发形成"觅食网络"——和真实 *Physarum* 在培养皿里寻找最短路径几乎一模一样。

#### CLI 启用

```bash
# 100 个个体在 16×16 网格上跑赛博黏菌，30 秒
python -m archaea.run --seed 42 --duration 30 \
    --pop-max 200 --n-initial 100 \
    --slime-mold \
    --log run.log

# 全套调参（保留默认即可，下面只是展示开关位置）
python -m archaea.run --seed 42 --duration 300 \
    --pop-max 200 --n-initial 100 \
    --slime-mold --grid-size 16 \
    --pheromone-decay 0.05 --pheromone-diffusion 0.20 \
    --pheromone-emit 0.5 --pheromone-bonus 0.5 \
    --hgt-prob 0.02 --hgt-blend 0.30 \
    --migrate-prob 0.30 \
    --log run.log
```

可与共享预算叠加（"资源稀缺 + 自组织觅食"）：

```bash
python -m archaea.run --seed 42 --duration 600 \
    --pop-max 500 --n-initial 100 \
    --slime-mold \
    --carrying-capacity 30 --budget-mode shared \
    --log run.log
```

#### 日志新增 3 列

`run.log` 在原有列后追加：

```
... weight_std  sigma  budget_pressure  phero_max  hgt  moves
```

- `phero_max`：当前最强信息素细胞值（衡量"觅食网络"的强度）。
- `hgt`：本窗 HGT 事件数。
- `moves`：本窗趋化迁移次数。

#### WebUI 视觉

- **观测页**自动检测 slime 模式 → 点阵区域切换为**真实空间布局**：
  - 背景 = 信息素热力图（深蓝 → 紫 → 琥珀色，越亮信息素越强）
  - dot 按 (x, y) 放在对应网格里，多 agent 同格自动散布
  - 上方信息条多了 P_max / HGT / 移动次数三个实时数字
- **统计表**自动追加 4 行（pheromone_max / pheromone_mean / hgt_count / migrations）
- **Agent 详情**面板显示该 agent 当前 (x, y) 与 local pheromone

#### 与 SPEC 的关系

- 默认关闭。`--slime-mold` 不开 → 行为与 SPEC v1.0 **byte-identical**（由 `tests/test_slime.py::test_population_slime_disabled_matches_default_behaviour` 守护）。
- 开启后属于 SPEC v1.1 扩展（formal spec 见 `SPEC.md` §13）。
- 三个机制都不修改 §1–§6 的 agent 拓扑、SNN 动力学、繁殖突变规则。仅在 §4.4 的奖励分配上叠了一层乘子，加了 §5 的繁殖位置约束（child 落在 parent 1 格内），并新增了 HGT/迁移/信息素三个独立子系统。
- 接受度判据 (§7.2 的 r ≥ 0.7) **不变**。这一扩展是研究装置：**问的是"去中心化协作能不能跑赢纯个体选择，会涌现什么样的结构"**，而不是工程交付目标。

---

## Live dashboard（可选实时图形）

两块面板：

1. **Dots** — 槽位点阵。颜色 = Credit 健康度；亮粉 = 本窗新生儿；浅粉粗边 = 本窗亲代；灰粗边 = 本窗饿死。
2. **Curves** — 滑动时间窗（默认 600 s 仿真时间）：
   - 左轴：`r_max`、`r_mean`、`sigma`（Pearson r 范围 [0,1]，`sigma` 紫虚线）。
   - 右轴：`N` 存活数（橙、阶梯线）、`credit_mean`（棕）。
   - 底部绿/红条：该窗有 `births` / `deaths`。

独立查看器（自带仿真循环，不读 `run.log`）：

```bash
# 弹窗（小种群更流畅）
python -m archaea.visualize --seed 33 --duration 60 --pop-max 100 --every 1 --tail 600

# 无界面：保存最后一帧 -> diagnostics/visualize_last.png
python -m archaea.visualize --no-show --duration 30 --pop-max 100 --every 2

# 自定义 PNG；大种群请加大 --every
python -m archaea.visualize --no-show --save diagnostics/dots.png --duration 10 --pop-max 1000 --every 20 --tail 900
```

把同一仪表盘挂到主实验上，**实时刷新**（同时仍写 `run.log`）：

```bash
python -m archaea.run --seed 42 --duration 3600 --pop-max 200 --log run.log \
  --visual --visual-every 5 --visual-tail 600
```

| 可视化参数 | 说明 |
|---|---|
| `--visual` | 打开实时面板。 |
| `--visual-every N` | 每 N 个仿真窗重绘一次（`pop-max 1000` 或 24h 长跑务必加大，例如 20–100）。 |
| `--visual-tail T` | 横轴只保留最近 T 秒仿真时间。 |

> 同时使用 `--visual` 和默认的 TTY 滚动表时，备用屏与 matplotlib 在个别终端可能彼此干扰；如有异常，请加 `--plain-stdout`。

可视化扩展思路写在 `archaea/visualize.py` 里的 `VISUALIZATION_IDEAS`：脉冲栅格、权重 PCA、Gini 时间序列、视频导出等。

---

## 测试（Gates A–C）

```bash
pytest tests/
```

- **Gate A**：单个 LIF 在 100 Hz×10 输入下输出落在 [20, 120] Hz。
- **Gate B**：经济模型对「完美/打平/废柴」三类 mock agent 的 Credit 轨迹符合 SPEC §6.2。
- **Gate C**：60 s 仿真、`pop_max=100`，种群规模始终在 [50, 100]，至少 1 次 birth、1 次 death，无 NaN。
  - 由于动力学含随机性，测试用搜索得到的固定 RNG `seed=33` 满足该带宽（见 `tests/test_short_run.py` 注释）。

---

## 输出物

| 路径 | 内容 |
|---|---|
| `run.log` | 与 stdout 同步的 TSV 遥测（每窗一行）。 |
| `checkpoints/t_<sec>.npz` | 每 600 s 仿真：存活权重、Credit、历史、RNG 状态。 |
| `diagnostics/fitness_curve.png` | `r_max` / `r_mean` 时序曲线（halt 时）。 |
| `diagnostics/weight_hist.png` | 末态权重分组直方图（halt 时）。 |
| `diagnostics/top10.csv` | top-10 个体的权重与 fitness。 |
| `diagnostics/t2h_*` | T+2h 失败时额外 dump（详见 SPEC §7.3）。 |
| `diagnostics/visualize_last.png` | 可视化最后一帧（headless 模式）。 |
| `diagnostics/champions_first.npz` | **首次** `r_max ≥ 0.7` 时的 top-10 权重存档（DNA）。 |
| `diagnostics/champions_final.npz` | 实验结束（任何 halt 路径）时的 top-10 权重存档。 |

---

## Champions：把成功的「群体 DNA」拿出来用

群体跨过 `r_max ≥ 0.7` 时，主程序会**自动**把 top-10 个体的权重打包进 `diagnostics/champions_first.npz`；实验结束再写一个 `champions_final.npz`。每个 `.npz` 含：

- `weights` `(K, 220)` — 每个个体的全部突触权重（10×20 + 20×1）。
- `fitness` `(K,)` — Pearson r（dump 时点）。
- `credit` `(K,)`、`source_slot` `(K,)`、`t_sim`、`seed`、`pop_max`、`spec_version`、`created_at`。

> 这就是你说的「集体 DNA」。`spec_version="L1.0"` 用来挡住 SPEC §2 常数（`I_in`、`τ`、`V_threshold` 等）改动后还盲加载的情况。

### Python 内对外服务

```python
from archaea.champions import ChampionEnsemble

ens = ChampionEnsemble.load("diagnostics/champions_first.npz")
print(ens.k, "champions, best fitness =", ens.fitness[ens.best_index()])

# 单点查询：给定输入发放率（Hz），跑 500 ms，返回最佳个体输出发放率（Hz）
hz_out = ens.rate_for(f_in_hz=80.0, duration_ms=500.0, seed=0, warmup_ms=100.0)

# 批量查询：返回所有 K 个个体的 f_out（Hz） -> shape (K,)
rates = ens.rates_for(f_in_hz=80.0, duration_ms=500.0, seed=0)

# f_in 扫描：(rates_in, rates_out_best)
fin, fout = ens.sweep([10, 30, 50, 70, 90], duration_ms=500.0, seed=0)

# 直接喂自己造的脉冲：input_spikes 形状 (T, 10)，0/1
out = ens.run_spikes(input_spikes, reset=True)   # -> (T, K)
```

调用约定（与 SPEC §3 / §1.3 一致）：
- 输入是 **10 个独立 Poisson 脉冲序列**，**同一发放率 `f_in` Hz**。
- `warmup_ms`：先跑一段（默认建议 100 ms）让 LIF 膜电位脱离冷启动瞬态再开始计数。
- `reset=True`：每次请求重置膜电位和不应期；做有状态服务时设 `False`。

### CLI 服务

```bash
# 元数据 + 每个个体的 fitness / credit / 原始 slot
python -m archaea.serve --champions diagnostics/champions_first.npz info

# 单点：给 50 Hz 输入，看最佳个体输出
python -m archaea.serve --champions diagnostics/champions_first.npz rate \
    --f-in 50 --duration 500 --seed 0

# 扫描 10..100 Hz，输出 TSV：f_in, f_out_best, f_out_mean, f_out_max
python -m archaea.serve --champions diagnostics/champions_first.npz sweep \
    --start 10 --stop 100 --step 10 --duration 500 --seed 0
```

`sweep` 的输出可以直接 `>` 到文件画 I/O 曲线；理想情况下 `f_out_best` 应随 `f_in` 单调上升（这就是「rate-tracking」生效的最直观体现）。

> 如果你想再包一层 HTTP/WebSocket，对外暴露 REST 接口（POST `/rate {f_in: 50}` → `{f_out: 32.0}`），把 `ChampionEnsemble` 当成内部模型即可——只是新增 web 依赖（FastAPI / Flask），属于 L2 范畴。

---

## WebUI（React + WebSocket，可选）

> 一个现代化的 Web 控制台，把 CLI 的所有事都用图形化做一遍：参数化启动、实时观测点阵 + 多面板曲线 + agent 内部拓扑、以及外部使用 + Credit 反馈闭环。**与 SPEC 行为完全解耦**——后端 (`archaea/server.py`) 只是把 `Population` 包了一层 thread-safe runtime + FastAPI；CLI/`run.py`/Gates 不受任何影响。

### 后端：FastAPI + WebSocket

```bash
pip install -r requirements.txt   # 现在含 fastapi / uvicorn / websockets
python -m archaea.server --host 127.0.0.1 --port 8000
```

REST：

| 方法 + 路径 | 说明 |
|---|---|
| `GET /api/status` | 当前仿真状态。 |
| `POST /api/start` | body 为 `SimConfig`（见下表），启动一个新仿真（替换现有的）。 |
| `POST /api/stop` | 停止当前仿真。 |
| `POST /api/inference` | `{f_in_hz, target, top_k, duration_ms, warmup_ms}` → 返回每个被查询 agent 的 `f_out_hz`。**不会暂停主仿真**——内部复制权重离线推理。 |
| `POST /api/feedback` | `{slots, delta_per_slot, label, f_in_hz?, f_out_hz?}`，**直接修改主仿真里那些 agent 的 Credit**。Credit ≤ 0 立即饿死。 |
| `GET /api/agent/{slot}` | agent 详情 + 内部 10→20→1 拓扑（用于详情面板渲染突触图）。 |
| `GET /api/feedback-log?limit=N` | 反馈历史。 |

WebSocket：

| 路径 | 说明 |
|---|---|
| `GET /ws/telemetry` | 每仿真窗一帧 JSON：与 CLI log 等价的 11 列 + 每槽位的 `alive/credit/fitness` 数组 + 本窗 `dead/parent/child` 槽位列表。 |

`SimConfig` 字段全部在 UI 的「设置」页有中文解释，对应 CLI 参数：`seed`、`pop_max`、`n_initial`、`carrying_capacity`、`budget_mode`、新增 `target_speed_hz`（每实际秒推进多少个仿真窗，0 = 全速）。

### 前端：React 18 + TypeScript + Tailwind + Recharts

```bash
cd webui
npm install                  # 第一次
npm run dev                  # 开发：http://127.0.0.1:5173 （Vite 代理 /api 与 /ws 到 8000）
npm run build                # 产出 webui/dist；FastAPI 会自动 serve
```

**生产部署**（一个进程同时供 API + UI）：

```bash
cd webui && npm install && npm run build
cd .. && python -m archaea.server --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000
```

`archaea/server.py` 检测到 `webui/dist/` 存在就自动挂载 SPA + 静态资源，不需要额外配置。

### 三个 Tab

1. **设置 / 启动**：每个参数有「简短描述 + 详细解释」两段中文，并明确标注 `off-SPEC` 项。
2. **观测**：
   - 大块 Canvas 点阵（颜色 = Credit 健康度，描边 = 本窗事件）。
   - **点击任一 dot → 右侧弹出该 agent 自己的 10→20→1 拓扑图**——蓝青边 = 正权重、红边 = 负权重，粗细 ∝ |w|。**注意：SPEC 中 agent 之间没有直接连接**，整个种群的耦合只发生在「共享槽位 / 共享预算」上，所以主面板上没有跨 agent 连线；详情图展示的才是真实的突触结构。
   - 4 张实时曲线：r 趋势（含 sigma + r=0.7 参考线）、种群规模 + 平均 Credit、预算 D/B（含 y=1 参考线）、births/deaths。
   - 右侧「实时遥测」表格 = CLI 日志的 11 列。
3. **使用**：
   - 设 `f_in`、target（best / ensemble top-K / random）、duration、warmup → 点「发送」。
   - 看到 `f_out` 后，按 ✓ 或 ✗ 给被查询 agent **加/扣 Credit**（默认 ±5，可调）。
   - 错答多次直接饿死那个 agent → 下一窗就会有人替补 → 你**手动塑造**了演化方向。
   - 右侧表格保留最近 80 次交互，含每行最终的判定与饿死数。

> **隐喻**：「一个族群的存在意义就是占用更多的 Credit」——你给输入、给反馈，整个族群就在为「**让你判它对**」而进化。这也是为什么 `--budget-mode shared` 和这套 UI 配合特别好玩：资源有限时，正确率高的 agent 才能在「外部反馈」加成下挤掉竞争者。

### 一键体验

```bash
# 终端 1：后端
python -m archaea.server --host 127.0.0.1 --port 8000

# 终端 2：前端（dev 模式，热重载）
cd webui && npm install && npm run dev
# 浏览器打开 http://127.0.0.1:5173
```

在「设置」页选 `pop_max=200`、`n_initial=80`、`budget_mode=shared`、`carrying_capacity=30`、`target_speed_hz=20`，启动后切到「观测」看点阵和曲线，最后切到「使用」喂几个 `f_in` 试试 ✓/✗ 反馈。

---

## 项目结构

参见 `SPEC.md` §10 的目录树。新增：

- `archaea/runtime.py`、`archaea/server.py` — WebUI 后端
- `webui/` — React 前端工程（`webui/dist/` 是 build 产物，被 FastAPI 自动 serve）
