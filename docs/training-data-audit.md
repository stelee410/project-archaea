# 训练数据审计：Project Archaea 是否存在"训练数据"？

> **结论先行**：本仓库实现的是 **mutation-only 演化算法**（外加可选 HGT），**不存在任何传统意义上的"训练"，也没有任何外部训练数据集**。所有"输入"都是每个 500 ms 窗口在线随机生成的 Poisson 脉冲串；权重的全部变化只来自三个写入点：随机初始化、繁殖时的高斯突变、可选的横向基因转移（HGT）线性混合。fitness（Pearson r）只作为选择压力影响 Credit / 繁殖 / 死亡，**从不直接修改权重**。

审计时间：2026-04-20  
审计人：复审脚本（Cursor / Claude Opus 4.7）  
审计范围：`archaea/` 全部模块 + `tests/` + `webui/` 中与权重/数据相关接口

---

## 1. 权重究竟是怎么变化的？只有 3 个写入点

`pop.weights[...]` 在整个 `archaea/` 包里只在以下 3 个地方被赋值（已 grep 验证），没有第 4 个。

### (a) 初始化 — 均匀随机

```142:144:archaea/population.py
    def spawn_initial_slot(self, slot: int) -> None:
        self.alive[slot] = True
        self.weights[slot] = self.rng.uniform(-3.0, 3.0, size=N_WEIGHTS)
```

1000 个 agent 在 `t=0` 各自抽一份 `Uniform(-3, 3)` 的 220 维向量。完全无信息输入。

### (b) 繁殖突变 — 父代权重 + 高斯噪声

```376:376:archaea/population.py
            w_child = self.weights[parent] + self.rng.normal(0.0, sigma, size=N_WEIGHTS)
```

新生儿的权重 = 父代权重 + `N(0, σ²)`。`σ` 是按全种群平均 fitness 算出来的全局标量（`SPEC §5.3`），不携带任何样本信息。

### (c) HGT 横向基因转移（仅 `--slime-mold` 模式）

```341:343:archaea/population.py
                self.weights[r_slot] = blend_weights(
                    self.weights[r_slot], self.weights[d_slot], self.slime.hgt_blend
                )
```

两个 agent 之间做权重的线性混合 `W_r ← (1-η) W_r + η W_d`。还是 agent 之间互抄，没有任何外部数据进入。

### 反向证据（grep）

- `rg "backward|grad|loss|train" archaea/`：**零命中**（除了 `target_speed_hz` 这种与"目标"无关的命名巧合）。
- `requirements.txt` 只有 `numpy` / `matplotlib`，无任何深度学习框架。
- `SPEC.md §0` 明确禁止：

  ```20:20:SPEC.md
  **Stack (mandatory):** Python ≥ 3.11, `numpy` only for simulation math. ... **No PyTorch, JAX, TensorFlow, Brian2, or Nengo.**
  ```

---

## 2. fitness（学习信号）从来不进权重

`r_agent = Pearson(f_in_history, f_out_history)` 的作用链路：

```
fitness  →  reward  →  Credit  →  (Credit≥200 触发繁殖) ∨ (Credit≤0 死亡)  →  谁突变 / 谁淘汰
```

它**只调控选择压力**（谁能复制、谁被替换），不会乘到权重上、不会反向传播、也不会作为残差加到 W：

```79:82:archaea/agent.py
    def reward_delta(self) -> float:
        """ΔCredit_reward = max(0, r) * R_max (SPEC §4.2)."""
        r = self.fitness() if self.fitness_defined() else 0.0
        return max(0.0, r) * 5.0
```

这是经典的 **mutation-only Evolutionary Strategies**，不是监督学习也不是 policy-gradient RL（即使 `reward` 这个词出现）。

---

## 3. 输入数据是**在线随机采样**，不是数据集

`step_window` 每个 500 ms 窗口现场生成输入：

```261:262:archaea/population.py
        f_in = draw_input_rate(self.rng)
        spikes = poisson_spikes_window(self.rng, f_in, 500.0, N_INPUT)
```

```31:33:archaea/stimulus.py
def draw_input_rate(rng: np.random.Generator) -> float:
    """f_in ~ Uniform(10, 100) Hz (SPEC §3.1)."""
    return float(rng.uniform(10.0, 100.0))
```

也就是说：

- 频率 `f_in` 每窗都是 `Uniform(10, 100) Hz` 现抽的；
- 10 个输入神经元各自现采一条 Poisson spike train；
- 所有 1000 个 agent **共享同一组**输入（这是为了"只比较权重差异"用的工程技巧，不是数据来源）；
- **整个仓库里没有任何数据文件、没有 `dataset/` 目录、没有 dataloader，全部由 `rng` 现场合成**。

Repo 根目录只有 `archaea/ tests/ webui/ checkpoints/ diagnostics/ docs/`，没有 `data/`。`checkpoints/*.npz` 是仿真过程产物（用于断点续跑），不是输入数据。

---

## 4. 唯一可能让人担心的"外部信号"：WebUI 的 `/api/feedback`

这是 `webui` 加进来的 UX 功能，调用链：

```449:497:archaea/runtime.py
    def feedback(
        self,
        slots: list[int],
        delta_per_slot: float,
        label: str,
        ...
    ):
        ...
                old = float(pop.credit[s])
                new = max(0.0, old + float(delta_per_slot))
                pop.credit[s] = new
```

它做的事是：**人手动给某些 agent 的 Credit 加 / 减一个数值**。

- `label`（`"correct"|"wrong"|"manual"`）只是写进 `feedback_log` 用于审计/展示，**没有进入任何权重更新**；
- 它**不修改 weights**，只调整 Credit；
- 影响通路依然是 Credit → 繁殖/死亡 → 间接的选择压力。

所以即便用户在 WebUI 上点 "correct/wrong" 按钮，也**不构成传统意义上的 training data**——它等价于人为施加一次额外的"奖励/惩罚事件"，再让进化机制去筛选。

**严格"零外部信号"复现条件**：只走 CLI 入口 `python -m archaea.run`，不调用 `/api/feedback`。CLI 路径里完全没有这条接口。

---

## 5. 一句话对照表

| 项目 | 这个代码 |
|---|---|
| 损失函数 | ❌ 无 |
| 反向传播 / 梯度 | ❌ 无 |
| 训练数据集 | ❌ 无（输入是每窗在线 Poisson 采样） |
| 标签 | ❌ 无（fitness 是 agent 自己的输出对自己的输入历史的 Pearson r） |
| STDP / Hebbian | ❌ 无 |
| 权重变化机制 | ✅ 仅初始化随机 + 繁殖时父代权重加高斯噪声（+ 可选 HGT 线性混合） |
| 学习信号怎么作用 | ✅ 仅作为选择压力（Credit 算术 → 繁殖/死亡），不进权重 |

> 在这个项目里，"模型怎么训练"的准确说法应该是 **"种群怎么演化"**——而演化的全部信息来源就是 `rng` 和"哪些 agent 的 (f_in, f_out) 相关性高"这一选择规则本身。没有任何外部知识被注入。

---

## 6. 复现本次审计的方法

如果未来需要重新核验本结论（例如代码改动后），按以下步骤即可：

### 6.1 检查"无深度学习/反传"

```bash
rg -n "backward|grad|loss|\.train\(|from torch|import torch|tensorflow|jax" archaea/
cat requirements.txt
```

期望：`archaea/` 下零命中（仅 `target_speed_hz`、`label="..."` 之类的命名巧合可被忽略）；`requirements.txt` 仅含 `numpy` / `matplotlib`。

### 6.2 检查权重写入点（应当只有 3 个）

```bash
rg -n "self\.weights\[" archaea/population.py
rg -n "weights\s*=" archaea/population.py
```

期望命中：`spawn_initial_slot` 中的 `rng.uniform(-3, 3)`、`step_window` 中的 `w_child = self.weights[parent] + self.rng.normal(...)`、`step_window` 中的 `blend_weights(...)`（HGT，slime 模式）。

### 6.3 检查输入是否在线生成（无外部数据）

```bash
rg -n "open\(|np\.load|pd\.read|\.csv|\.npz|dataset" archaea/
ls /Users/stelee/Dev/project-archaea
```

期望：`archaea/` 下与"读数据"相关的命中应仅限于 `checkpoints/*.npz`（写出，仿真断点）和 `diagnostics/`（写出，halt 诊断）。仓库根目录无 `data/`、无 `datasets/`。

### 6.4 检查 fitness 是否参与权重更新

```bash
rg -n "fitness|pearson|r_agent" archaea/
```

期望：fitness 只出现在以下用途——
- `agent.py` / `population.py`：用于计算 reward / Credit；
- `population.py` `global_sigma()`：用于计算全局突变尺度 `σ`；
- `telemetry.py` / `visualize.py`：用于日志和可视化。

不应有任何 `weights ... fitness ...` 形式的赋值。

### 6.5 检查 `/api/feedback` 是否改权重

```bash
rg -n "def feedback" archaea/runtime.py -A 50
```

期望：函数体内只对 `pop.credit[s]` 赋值，不对 `pop.weights[...]` 赋值。

### 6.6 当前审计依据的 SPEC 章节

- `SPEC.md §0`：技术栈限制（禁用所有深度学习框架）。
- `SPEC.md §1.2`：初始化规则（`Uniform(-3, 3)`）。
- `SPEC.md §3`：输入是 `Uniform(10, 100) Hz` 的 Poisson 脉冲串，每窗现场抽。
- `SPEC.md §4.1–4.4`：fitness → reward → Credit 的纯标量算术。
- `SPEC.md §5.2–5.3`：繁殖 = 父代权重 + `N(0, σ²)`；`σ` 由全局平均 fitness 决定。
- `SPEC.md §11`："Mutation-only. No STDP, no Hebbian learning, no eligibility traces."
- `SPEC.md §13.3`：HGT 是 agent 间权重的线性混合，不是外部数据。
