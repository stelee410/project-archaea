# Project Archaea — L1 Engineering Contract (SPEC v1.0)

> **Scope discipline.** This document specifies L1 only: evolving a population of spiking-neural-network agents such that their single output neuron's firing rate tracks the input firing rate (Pearson r > 0.7). It does **not** cover arithmetic, topology evolution, or crossover. Those are later phases and are explicitly out of scope.

> **Implementation note (off-SPEC extension).** The runner ships an opt-in *shared-budget / carrying-capacity* mode (CLI: `--budget-mode shared --carrying-capacity K`) that **extends** §4.4 to model finite resources (`B = K × R_MAX` per window, distributed proportionally to demand with a haircut when oversubscribed). It is **disabled by default** so the SPEC behavior is preserved bit-for-bit; see `README.md` § "承载力 / 共享预算" for the full description, motivation, and equilibrium math.

---

## 0. What the deliverable is

A single Python package, runnable from the command line, that:

1. Simulates a population of spiking-neural-network (SNN) agents under a metabolic economy.
2. Applies mutation-only evolution (no backprop, no crossover, no topology search).
3. Emits structured telemetry to stdout and a log file.
4. Halts automatically on defined success, failure, or pathology conditions.

**Stack (mandatory):** Python ≥ 3.11, `numpy` only for simulation math. `matplotlib` only for diagnostic plots emitted at halt. **No PyTorch, JAX, TensorFlow, Brian2, or Nengo.** If a dependency feels necessary, stop and ask.

**Runtime budget:** Must complete 24 hours of simulated time (not wall-clock) on a single CPU core in under 2 hours wall-clock. If the first 1 minute of simulated time takes more than 5 seconds wall-clock, profile and optimize before continuing.

---

## 1. Entity & topology

### 1.1 The agent

An **agent** is one complete feedforward SNN with fixed topology `[10 input → 20 hidden → 1 output]`, fully connected between adjacent layers.

- Total evolvable weights per agent: `10 × 20 + 20 × 1 = 220`.
- Agents do **not** share neurons. 1000 agents = 1000 independent networks.
- Topology is immutable in L1. Only weight values evolve.

### 1.2 Population

- `POP_MAX = 1000` agents.
- **Initialization:** all 1000 agents spawned at t=0 with weights `W ~ Uniform(-3.0, 3.0)` independently per weight. (Note the range: wider than the intuitive [-1, 1] — this is deliberate, see §2.4.)
- **Replacement rule:** when a birth would exceed POP_MAX, the new agent replaces the living agent with the lowest Credit. This is the only mechanism that keeps population bounded.

### 1.3 Output definition

For each agent, `f_out` is the firing rate of the single output neuron, computed as:

```
f_out(t) = (number of output spikes in the window [t-500ms, t]) / 0.5s
```

The window slides every 500 ms, non-overlapping. Each agent emits one `(f_in_mean, f_out)` sample per window.

---

## 2. Physical simulation (LIF neurons)

### 2.1 Neuron model

Leaky integrate-and-fire, discrete-time Euler integration:

```
V[t+dt] = V[t] + (dt / τ) * (V_rest - V[t]) + (dt / τ) * R * I[t]
```

Spike condition: if `V[t+dt] ≥ V_threshold`, emit spike at `t+dt`, set `V[t+dt] = V_reset`, enter refractory period.

### 2.2 Constants (mandatory, do not change without updating this SPEC)

| Symbol | Value | Meaning |
|---|---|---|
| `dt` | 1 ms | Simulation step |
| `τ` | 20 ms | Membrane time constant |
| `R` | 1.0 | Membrane resistance |
| `V_rest` | 0.0 | Resting potential |
| `V_threshold` | 1.0 | Spike threshold |
| `V_reset` | 0.0 | Post-spike reset potential |
| `t_refractory` | 2 ms | Absolute refractory period |
| `I_in` | **2.5** | Current injected per presynaptic spike (see §2.4) |

### 2.3 Synaptic integration

When a presynaptic neuron spikes at time `t`, each postsynaptic neuron `j` receives an **instantaneous current impulse** of magnitude `W_ij × I_in` at time `t` only (no synaptic decay; the leak in the membrane equation is the only decay). Summing over all presynaptic spikes in the current step:

```
I_j[t] = I_in × Σ_i (W_ij × spike_i[t])
```

where `spike_i[t] ∈ {0, 1}`.

### 2.4 Why `I_in = 2.5`

At `dt/τ = 0.05` and initial `|W| ≈ 1.5` (mean absolute value of Uniform(-3, 3)), a single positive-weight spike contributes `ΔV ≈ 0.05 × 2.5 × 1.5 ≈ 0.19`. This allows an output neuron to reach `V_threshold = 1.0` from ~6 coincident or closely-timed spikes — consistent with biological plausibility and sufficient for 100 Hz input to drive non-trivial output rates.

**This value is load-bearing.** Changing any of `I_in`, weight init range, `τ`, or `V_threshold` requires re-running the calibration test in §6.1 and confirming the acceptance criterion there holds.

---

## 3. Input encoding (Pulsar)

### 3.1 Stimulus generation

At each 500 ms evaluation window:

1. Draw `f_in ~ Uniform(10, 100)` Hz. This is the target rate for all 10 input neurons in this window.
2. For each input neuron independently, generate a Poisson spike train at rate `f_in` over the 500 ms window.
3. All 1000 agents in the population see the **same** 10 input spike trains during the same window. This eliminates per-agent stimulus variance as a confound during fitness comparison.

### 3.2 Rationale

Using identical stimuli across the population means fitness differences reflect weight differences only. This is standard evolutionary-strategies practice and reduces the sample size needed to distinguish good weights from lucky draws by roughly two orders of magnitude.

---

## 4. Fitness, reward, and economy

### 4.1 Fitness metric (identical to acceptance metric)

Each agent maintains a **rolling history** of its last `N_history = 40` windows' `(f_in, f_out)` pairs (= 20 seconds of memory).

Its **fitness** `r_agent` is the Pearson correlation coefficient between those 40 `f_in` values and 40 `f_out` values.

- If fewer than 40 windows have elapsed for this agent, fitness is undefined; treat as 0 for reward purposes.
- If the variance of `f_out` across those 40 windows is zero (agent always fires at the same rate, or never), fitness is exactly 0. **This is what closes the "lazy constant-output" loophole** — Pearson is undefined for zero-variance outputs and we treat it as minimum fitness.

### 4.2 Reward per window

At the end of each 500 ms window, every living agent receives:

```
ΔCredit_reward = max(0, r_agent) × R_max
```

with `R_max = 5.0`.

Agents with negative correlation (anti-tracking) get zero reward but are not penalized beyond the breath cost.

### 4.3 Breath (metabolic cost)

Every agent pays `C_breath = 0.0025` Credit per simulation step (`dt = 1 ms`), i.e. `2.5 Credit/s`, i.e. `1.25 Credit/window`.

### 4.4 Credit dynamics

```
Credit_new = Credit_old + ΔCredit_reward - C_breath × (steps in window)
           = Credit_old + max(0, r_agent) × 5.0 - 1.25
```

Initial Credit per agent: `C_init = 50.0`.

### 4.5 Economic closure (worked numbers)

| Agent behavior | r | Net Credit / second |
|---|---|---|
| Silent or constant output | 0 | −2.5 |
| Weak tracking | 0.25 | **0.0** (break-even) |
| Moderate tracking | 0.5 | +2.5 |
| Strong tracking | 0.8 | +5.5 |

A break-even-or-better agent (r ≥ 0.25) survives indefinitely. A strong tracker (r = 0.8) reaches the 200-Credit reproduction threshold from initial 50 in `(200 − 50) / 5.5 ≈ 27` seconds of simulation. This is deliberately fast — we want the evolutionary loop to turn over many times within the 24-hour budget.

---

## 5. Reproduction & mutation

### 5.1 Death

When `Credit ≤ 0`, the agent is removed from the population. No energy recycling in L1 (the returned-to-pool mechanic from earlier drafts is deferred).

### 5.2 Birth trigger

When an agent's `Credit ≥ C_repro = 200.0` at the end of a window:

1. Deduct `C_cost_repro = 100.0` from the parent (parent continues with Credit = 100).
2. Spawn one child with weights `W_child = W_parent + ε`, where `ε ~ Normal(0, σ² × I_220)`.
3. Child's initial Credit: `C_init = 50.0`.
4. Child inherits an empty fitness history (must accrue its own 40 windows before its fitness is defined).

### 5.3 Mutation scale (population-pressure driven)

σ is a **global** value recomputed every window from the whole population, not per-agent:

```
mean_fitness = mean(r_agent for all living agents with defined fitness)
σ = σ_base × exp(−2.0 × max(0, mean_fitness))
σ_base = 0.3
```

Intuition:
- Population baseline (`mean_fitness ≈ 0`): `σ = 0.3` — aggressive exploration.
- Population adapting (`mean_fitness = 0.5`): `σ ≈ 0.11` — moderate.
- Population converged (`mean_fitness = 0.8`): `σ ≈ 0.06` — refinement.

This replaces the earlier per-agent `σ ∝ 1/Credit` formulation, which was dead code (high-Credit reproducing agents always had tiny σ).

### 5.4 Population cap enforcement

When `POP_MAX = 1000` is full and a birth occurs, the new child replaces the living agent with the lowest current Credit. Ties broken by lowest fitness, then arbitrarily. A parent cannot replace itself — if the parent is the lowest-Credit agent, the child replaces the second-lowest.

---

## 6. Required pre-flight tests (gates)

**Each test is a gate. Failing test → halt, produce diagnostic output, do not proceed to the next.**

### 6.1 Gate A — Single-neuron firing calibration

**Script:** `tests/test_single_neuron.py`

Instantiate one LIF output neuron. Connect 10 input neurons with weights all = `+1.5`. Drive each input neuron with a 100 Hz Poisson spike train for 1.0 s of simulated time.

**Acceptance:** output neuron fires at rate `∈ [20, 120] Hz`. Fewer than 20 Hz means the network is too quiet (most agents will be silent); more than 120 Hz means runaway (rate coding will saturate).

**On failure:** dump the membrane-potential trace to `diagnostics/gate_a_vtrace.png` and print the observed rate. Do not proceed. Suggested calibration knob: adjust `I_in` in the range [1.0, 5.0].

### 6.2 Gate B — Economic closure

**Script:** `tests/test_economy.py`

Mock three agents by bypassing the SNN and directly stipulating their fitness:

- Agent *Perfect*: `r_agent = 1.0` every window.
- Agent *Breakeven*: `r_agent = 0.25` every window.
- Agent *Dead-weight*: `r_agent = 0.0` every window.

Run 60 seconds of simulated time. **Acceptance:**

- *Perfect* reaches Credit ≥ 200 within 35 s and triggers at least one reproduction.
- *Breakeven* remains alive with Credit in [30, 80] throughout.
- *Dead-weight* reaches Credit ≤ 0 within [15, 30] s.

**On failure:** print Credit trajectories. Numbers off by > 20% indicate a bug in the economy; off by factors indicate a wrong constant.

### 6.3 Gate C — Short population run

**Script:** `tests/test_short_run.py`

Full system, 60 seconds of simulated time, `POP_MAX = 100` (not 1000, for speed). **Acceptance:**

- Population size stays in `[50, 100]` throughout (not collapsing, not pinned at cap).
- At least one birth event occurs.
- At least one death event occurs.
- No exceptions, no `NaN` or `±inf` in any Credit, weight, or fitness value at any point.

**On failure:** dump the telemetry log and the final weight distribution histogram.

---

## 7. Main experiment & acceptance

### 7.1 Run configuration

After Gates A–C pass, run the full experiment:

- `POP_MAX = 1000`
- Simulated duration: up to 24 hours (= 86,400 s = 172,800 windows)
- Random seed: configurable via `--seed`, default 42. **A given seed must produce bit-identical output across runs.**

### 7.2 Success criterion (T+24h)

At least one agent in the population has achieved `r_agent ≥ 0.7` over its trailing 40-window history at some point during the run. Record the timestamp of first crossing as `t_first_success`.

### 7.3 Intermediate checkpoint (T+2h)

At `t = 7200 s` of simulated time, evaluate:

- If `max(r_agent) < 0.3` across the population: **halt and declare failure**. Dump:
  - Final weight distribution per layer (`diagnostics/t2h_weights.png`)
  - Fitness history of the top-10 agents (`diagnostics/t2h_top10.csv`)
  - Full telemetry log
- If `max(r_agent) ≥ 0.3`: continue.

### 7.4 Pathology halts (may trigger any time)

Halt immediately and dump diagnostics if any of the following occur:

- **Extinction:** living population < 10 agents.
- **Monoculture:** `std(weights)` across the population drops below `0.01` for the top-100 agents simultaneously (genetic diversity collapse).
- **Stagnation:** `max(r_agent)` has not improved by at least `0.05` in the last 2 hours of simulated time *and* current `max(r_agent) < 0.7`.

---

## 8. Telemetry

### 8.1 Per-window log line (stdout + `run.log`)

Emitted every window (500 ms simulated). Single line, tab-separated:

```
t_sim  pop_size  births  deaths  r_max  r_mean  credit_mean  credit_gini  weight_std  sigma
```

- `t_sim`: simulated time in seconds, float, 3 decimals.
- `births`, `deaths`: counts in this window.
- `r_max`, `r_mean`: over agents with defined fitness only.
- `credit_gini`: Gini coefficient of Credit across the living population. Indicates energy monopoly.
- `weight_std`: mean over all 220 weight positions of the std-dev across agents. Indicates genetic diversity.
- `sigma`: current global mutation scale.

### 8.2 Checkpoint dump

Every 600 s of simulated time, write `checkpoints/t_<sim_seconds>.npz` containing:

- All living agents' weight arrays (shape `[pop_size, 220]`)
- All living agents' Credit values
- All living agents' fitness histories
- RNG state (for reproducibility)

### 8.3 Halt dump

On any halt (success, failure, or pathology), in addition to the telemetry:

- Line plot of `r_max` and `r_mean` vs. `t_sim` (`diagnostics/fitness_curve.png`)
- Histogram of final weight values grouped by layer (`diagnostics/weight_hist.png`)
- CSV of the top-10 agents' weights and fitness (`diagnostics/top10.csv`)

---

## 9. Implementation order (mandatory)

Complete and verify each step before moving to the next. No skipping.

1. **LIF neuron class + single-neuron unit test.** Verify membrane dynamics numerically (analytic solution under constant current).
2. **Network forward pass.** 10→20→1, fixed weights, driven by Poisson inputs. Verify shapes and firing rates.
3. **Gate A.** (§6.1)
4. **Economy class + Gate B.** (§6.2)
5. **Population class** (birth, death, replacement, mutation).
6. **Gate C.** (§6.3)
7. **Telemetry and checkpointing.**
8. **Full 24 h run.**

**Anti-pattern warning for the implementer:** If Gate A, B, or C fails, do not adjust constants quietly and re-run. Report the observed numbers, propose a specific change, and wait for confirmation. The constants in this SPEC were chosen through numerical analysis; silent retuning defeats the point.

---

## 10. Project layout

```
archaea/
├── SPEC.md                      # this file
├── README.md                    # how to run; generated after implementation
├── requirements.txt             # numpy, matplotlib; nothing else
├── archaea/
│   ├── __init__.py
│   ├── neuron.py                # LIF_Neuron, Network
│   ├── agent.py                 # Agent (wraps Network + Credit + history)
│   ├── economy.py               # Credit update rules, reproduction
│   ├── population.py            # Population, replacement rule, global σ
│   ├── stimulus.py              # Poisson spike-train generator
│   ├── telemetry.py             # log writer, checkpoint writer, plots
│   └── run.py                   # CLI entry point; orchestrates the main loop
├── tests/
│   ├── test_single_neuron.py    # Gate A
│   ├── test_economy.py          # Gate B
│   └── test_short_run.py        # Gate C
├── diagnostics/                 # created at runtime
└── checkpoints/                 # created at runtime
```

CLI:

```
python -m archaea.run --seed 42 --duration 86400 --pop-max 1000 --log run.log
```

---

## 11. Explicit non-goals for L1

So there's no ambiguity about what *not* to build:

- No crossover, no sexual reproduction. Mutation-only.
- No topology evolution. Fixed 10→20→1.
- No STDP, no Hebbian learning, no eligibility traces. Weights change **only** via mutation at birth.
- No arithmetic task. Rate tracking only.
- No GUI, no dashboard, no web server. stdout + log file + post-hoc plots.
- No distributed compute. Single process, single core.
- No hyperparameter search. Use the constants given. If they don't work, the SPEC is wrong — come back and fix it, don't grid-search around it.

---

## 12. Definition of done

The L1 contract is complete when all of the following hold:

1. All three gates (§6) pass on a fresh checkout with `python -m pytest tests/`.
2. The full 24-hour run (§7.1) executes to completion or halts on a defined condition (§7.2–7.4) without unhandled exceptions.
3. Telemetry and halt dumps are produced as specified (§8).
4. A `README.md` exists documenting: how to install, how to run, how to interpret the log format, and how to reproduce a given seed.
5. Re-running with the same seed produces bit-identical `run.log`.

Success on the biological objective (r > 0.7 achieved) is **not** part of the engineering contract. The contract delivers the apparatus; whether the apparatus produces evolution is the scientific question the apparatus exists to answer.
