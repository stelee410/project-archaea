import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { api } from "../api";
import { useStore } from "../store";
import type { FounderSpec, SimConfig, SimTask, StrainMeta, TaskDifficulty } from "../types";
import { COLONIES } from "../colonies/registry";

/**
 * SPEC_L2_V3.0 §3.3 — ⚗️ Mixer / 杂交皿启动页。
 *
 * 用户旅程：
 *   1. 在某个普通群落里跑出 AND-学家、NOT-学家两个菌株（ObservePage 上 💾 保存）。
 *   2. 进入这里 → 选两个菌株 + 各自比例（默认 0.5/0.5）→ 选 task / 难度 / pop_max。
 *   3. 启动后台后端用 founders 灌入初始种群、放大前 N 秒 HGT，自动跳到对应群落
 *      observe 页围观。
 *
 * 设计原则：
 * - 杂交皿不是一个新的「task」（演化任务还是 L2v2_ctrl），它只是一种「初始化方式」。
 *   所以这里不动 colonies/registry，启动完成后让 App 把视图切到 L2v2 colony 的
 *   observe 页就行。
 * - 任意时刻只允许 sum(fraction) ≤ 1.0；剩余空槽用默认随机 init 兜底
 *   （同一 task 的「随机祖先」当背景噪声）。
 * - 起码要选 1 个菌株才允许启动；选 2 个时是经典「双修」实验。
 */
const DIFFICULTY_HINT: Record<TaskDifficulty, string> = {
  uniform: "uniform — SPEC 原版 (P(target=1)=37.5%)",
  balanced: "balanced — v2.3 推荐 (P(target=1)=50%)，杂交后默认就用这个",
  hard: "hard — 强压力 (沉默上限 30%)",
  extreme: "extreme — 实验室 (founder 弱时高崩)",
  and_only: "and_only — 100% 考 AND（杂交后选这个等于把 NOT 学家退化）",
  not_only: "not_only — 100% 考 NOT（同理 AND 学家会退化）",
};

export function MixerPage() {
  const status = useStore((s) => s.status);
  const setStatus = useStore((s) => s.setStatus);
  const resetHistory = useStore((s) => s.resetHistory);

  const [strains, setStrains] = useState<StrainMeta[] | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [reloadAt, setReloadAt] = useState(0);

  // Default to first L2v2 colony task (the only one supporting strains today).
  const defaultTask: SimTask = "l2v2_ctrl";
  const [task, setTask] = useState<SimTask>(defaultTask);
  const [difficulty, setDifficulty] = useState<TaskDifficulty>("balanced");
  const [popMax, setPopMax] = useState(1000);
  const [nInitial, setNInitial] = useState(800);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1_000_000));
  const [synapseGain, setSynapseGain] = useState(2.0);

  const [picks, setPicks] = useState<FounderSpec[]>([]);
  // SPEC_L2_V3.4 — 3-phase ecological admixture protocol defaults.
  // commensal=60s lets each strain stabilize in the new dish; exchange=120s
  // gives a long, low-intensity gene-flow window (blend=0.05 ≪ baseline 0.30
  // so the larger-magnitude strain can't sweep the smaller-magnitude one in
  // a single transfer).
  const [commensalS, setCommensalS] = useState(60.0);
  const [exchangeS, setExchangeS] = useState(120.0);
  const [phase2Blend, setPhase2Blend] = useState(0.05);
  const [phase2ProbMul, setPhase2ProbMul] = useState(1.0);
  // SPEC_L2_V3.5 — assortative HGT (prezygotic isolation by niche similarity).
  // null  = legacy / disabled (richest neighbour wins, v3.4 bit-identical)
  // 0     = strict speciation (only the closest-niche donor)
  // 0.05–1.0 = soft niche preference (smaller = stronger bias)
  // Default: 0.30 — moderate assortative bias, recommended after the §5.13
  // ERRATA documented that single-pool admixture between AND-experts and
  // D1-seeded NOT-experts collapses both lineages without it.
  const [assortativeT, setAssortativeT] = useState<number | null>(0.30);

  const [busy, setBusy] = useState(false);
  const [launchErr, setLaunchErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStrains(null);
    setLoadErr(null);
    api
      .listStrains()
      .then((rs) => {
        if (cancelled) return;
        setStrains(rs);
      })
      .catch((e) => {
        if (cancelled) return;
        setLoadErr(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [reloadAt]);

  const strainsByTask = useMemo(() => {
    if (!strains) return new Map<SimTask, StrainMeta[]>();
    const m = new Map<SimTask, StrainMeta[]>();
    for (const s of strains) {
      const arr = m.get(s.task) ?? [];
      arr.push(s);
      m.set(s.task, arr);
    }
    return m;
  }, [strains]);

  const eligible = strainsByTask.get(task) ?? [];
  const totalFrac = picks.reduce((a, p) => a + p.fraction, 0);
  const hasOver = totalFrac > 1.0001;
  const colony = COLONIES.find((c) => c.id === task);

  function togglePick(strainId: string) {
    setPicks((prev) => {
      const exists = prev.find((p) => p.strain_id === strainId);
      if (exists) return prev.filter((p) => p.strain_id !== strainId);
      // Default: split evenly with whatever's already picked.
      const n = prev.length + 1;
      const each = Math.min(1 / n, 0.5);
      return [
        ...prev.map((p) => ({ ...p, fraction: each })),
        { strain_id: strainId, fraction: each },
      ];
    });
  }

  function setFrac(strainId: string, f: number) {
    setPicks((prev) =>
      prev.map((p) =>
        p.strain_id === strainId ? { ...p, fraction: clamp(f, 0.01, 1) } : p
      )
    );
  }

  async function launch() {
    if (picks.length === 0) {
      setLaunchErr("至少要挑 1 个菌株作为 founder");
      return;
    }
    if (hasOver) {
      setLaunchErr(
        `比例之和 ${totalFrac.toFixed(2)} > 1.0；请下调某个 founder 的比例`
      );
      return;
    }
    // Confirm if another sim is running.
    if (status?.running) {
      const ok = window.confirm(
        `当前已有仿真在跑（task=${status.config?.task} · t=${status.t_sim.toFixed(0)}s · N=${status.n_living}）。\n\n` +
          `启动 ⚗️ Mixer 会强制结束它，演化进度会丢失。\n\n确定要替换吗？`
      );
      if (!ok) return;
    }
    setBusy(true);
    setLaunchErr(null);
    try {
      resetHistory();
      const cfg: SimConfig = {
        seed,
        pop_max: popMax,
        n_initial: nInitial,
        carrying_capacity: null,
        budget_mode: "none",
        target_speed_hz: 20,
        slime_mold: true,           // admixture needs spatial HGT
        grid_size: 32,
        pheromone_decay: 0.05,
        pheromone_diffusion: 0.2,
        pheromone_emit: 0.5,
        pheromone_bonus_k: 0.0,     // SPEC_L2_V3.0: pheromone reward off (would distort oracle)
        hgt_enabled: true,
        hgt_prob: 0.02,
        hgt_blend: 0.30,
        migrate_enabled: true,
        migrate_prob: 0.30,
        calibration_lambda: 0.0,
        synapse_gain: synapseGain,
        task,
        task_difficulty: difficulty,
        founders: picks,
        admixture_commensal_s: commensalS,
        admixture_exchange_s: exchangeS,
        admixture_phase2_blend: phase2Blend,
        admixture_phase2_prob_mul: phase2ProbMul,
        assortative_temperature: assortativeT,
      };
      const next = await api.start(cfg);
      setStatus(next);
      window.dispatchEvent(
        new CustomEvent("archaea:mixer-launched", { detail: { colonyId: task } })
      );
    } catch (e) {
      setLaunchErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function deleteOne(strainId: string, name: string) {
    if (!window.confirm(`确定删除菌株「${name}」（id=${strainId}）？该操作不可撤销。`)) return;
    try {
      await api.deleteStrain(strainId);
      setPicks((prev) => prev.filter((p) => p.strain_id !== strainId));
      setReloadAt((x) => x + 1);
    } catch (e) {
      alert(String(e));
    }
  }

  return (
    <div className="max-w-[1200px] mx-auto p-6 space-y-6">
      <div className="rounded-lg border border-cyan-700/40 bg-cyan-950/15 p-5">
        <div className="flex items-baseline gap-3 mb-2">
          <span className="text-3xl">⚗️</span>
          <div>
            <h1 className="text-xl font-semibold text-cyan-100">
              杂交皿 · Admixture Mixer
            </h1>
            <div className="text-xs text-cyan-200/60 font-mono">
              SPEC_L2_V3.0 · 把多个菌株倒进新培养皿 · 杂交期内放大 HGT
            </div>
          </div>
        </div>
        <p className="text-sm text-cyan-100/85 leading-relaxed">
          这不是一个新的演化任务，只是<b>初始化方式</b>不同：用你之前 💾 保存的菌株
          作为 founder 灌入初始种群（其余空槽走默认随机 init），并在前 N 秒
          临时放大 HGT 概率，模拟两群古菌在新培养皿里相遇 → 基因水平转移 →
          诞生「双修」个体。
        </p>
      </div>

      {/* Strain library */}
      <section className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-base font-semibold text-slate-100">
            🧪 菌株库 (Strain library)
          </h2>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <button
              onClick={() => setReloadAt((x) => x + 1)}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200"
            >
              🔄 刷新
            </button>
            <span className="font-mono">
              共 {strains?.length ?? "?"} 个 · 当前 task：
              {colony?.emoji} {task}
            </span>
          </div>
        </div>

        {/* Task selector */}
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-400">演化任务：</span>
          {COLONIES.filter((c) => !c.locked).map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => {
                setTask(c.id);
                setPicks([]); // task changed → clear picks (cross-task不允许)
              }}
              className={clsx(
                "px-2.5 py-1 rounded text-xs",
                task === c.id
                  ? "bg-cyan-500/20 text-cyan-100 ring-1 ring-cyan-400/60"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              )}
            >
              {c.emoji} {c.name}
            </button>
          ))}
        </div>

        {loadErr && (
          <div className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/40 rounded px-2 py-1.5 mb-2">
            加载菌株库失败：{loadErr}
          </div>
        )}

        {strains == null ? (
          <div className="text-sm text-slate-400 py-4">读取中…</div>
        ) : eligible.length === 0 ? (
          <div className="text-sm text-slate-400 py-6 text-center bg-slate-950/50 rounded border border-dashed border-slate-700">
            <div className="text-2xl mb-2 opacity-60">🪴</div>
            <div>当前 task 下还没有菌株。</div>
            <div className="text-xs text-slate-500 mt-1">
              先去对应群落跑一阵 → 在「观测」页 💾 保存为菌株 → 回来杂交。
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            {eligible.map((s) => {
              const picked = picks.find((p) => p.strain_id === s.id);
              return (
                <StrainRow
                  key={s.id}
                  strain={s}
                  picked={!!picked}
                  fraction={picked?.fraction ?? 0.5}
                  onToggle={() => togglePick(s.id)}
                  onSetFrac={(f) => setFrac(s.id, f)}
                  onDelete={() => deleteOne(s.id, s.name)}
                />
              );
            })}
          </div>
        )}
      </section>

      {/* Composition summary */}
      <section className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
        <h2 className="text-base font-semibold text-slate-100 mb-2">
          🧬 初始种群配方
        </h2>
        <div className="text-xs text-slate-400 mb-2">
          每个 founder 的比例是「占 n_initial 的份额」（按 fraction 抽样替换）。
          Sum &lt; 1 时剩余空槽用默认随机 init 兜底。
        </div>
        <div className="bg-slate-950 rounded h-5 overflow-hidden flex border border-slate-800">
          {picks.map((p, i) => {
            const meta = strains?.find((s) => s.id === p.strain_id);
            return (
              <div
                key={p.strain_id}
                className={clsx(
                  "h-full text-[10px] text-slate-950 font-mono px-1 flex items-center",
                  PICK_COLORS[i % PICK_COLORS.length]
                )}
                style={{ width: `${(p.fraction * 100).toFixed(2)}%` }}
                title={`${meta?.name ?? p.strain_id}: ${(p.fraction * 100).toFixed(0)}%`}
              >
                {p.fraction >= 0.06
                  ? `${meta?.name ?? p.strain_id.slice(0, 6)} ${(p.fraction * 100).toFixed(0)}%`
                  : ""}
              </div>
            );
          })}
          {totalFrac < 1 && (
            <div
              className="h-full bg-slate-700 text-[10px] text-slate-300 font-mono px-1 flex items-center"
              style={{ width: `${((1 - totalFrac) * 100).toFixed(2)}%` }}
              title={`随机 init: ${((1 - totalFrac) * 100).toFixed(0)}%`}
            >
              {1 - totalFrac >= 0.06
                ? `random ${((1 - totalFrac) * 100).toFixed(0)}%`
                : ""}
            </div>
          )}
        </div>
        <div className="mt-2 text-xs font-mono text-slate-300 numeric">
          total founder fraction = {totalFrac.toFixed(2)} · random fill ={" "}
          {Math.max(0, 1 - totalFrac).toFixed(2)}
          {hasOver && (
            <span className="text-rose-300 ml-2">⚠ &gt; 1.0，必须 ≤ 1</span>
          )}
        </div>
      </section>

      {/* Sim params */}
      <section className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
        <h2 className="text-base font-semibold text-slate-100 mb-3">
          ⚙️ 培养皿参数
        </h2>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          <NumField label="seed" value={seed} onChange={setSeed} step={1} />
          <NumField label="pop_max" value={popMax} onChange={setPopMax} step={100} min={50} />
          <NumField label="n_initial" value={nInitial} onChange={setNInitial} step={50} min={1} />
          <NumField
            label="synapse_gain (g)"
            value={synapseGain}
            onChange={setSynapseGain}
            step={0.1}
            min={0.1}
          />
          {task === "l2v2_ctrl" && (
            <div className="col-span-2 lg:col-span-3">
              <label className="block text-xs text-slate-300 mb-1">
                难度 task_difficulty
              </label>
              <div className="flex flex-wrap gap-1.5">
                {(Object.keys(DIFFICULTY_HINT) as TaskDifficulty[]).map((d) => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => setDifficulty(d)}
                    className={clsx(
                      "px-2.5 py-1 rounded text-xs",
                      difficulty === d
                        ? "bg-amber-500/20 text-amber-100 ring-1 ring-amber-400/60"
                        : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                    )}
                    title={DIFFICULTY_HINT[d]}
                  >
                    {d}
                  </button>
                ))}
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                {DIFFICULTY_HINT[difficulty]}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Admixture protocol — SPEC_L2_V3.4 */}
      <section className="rounded-lg border border-fuchsia-700/40 bg-fuchsia-950/15 p-4">
        <h2 className="text-base font-semibold text-fuchsia-100 mb-2">
          💞 杂交协议 (SPEC_L2_V3.4 · 3 相)
        </h2>
        <p className="text-xs text-fuchsia-200/80 leading-relaxed mb-3">
          按生物学的「双菌共栖 → 受控基因流 → 自由融合」分三段。
          v3.0 的「立即放大 HGT」会让权重幅值大的一方瞬间扫荡培养皿（已经
          实测到 AND 学家被 NOT D1 种子覆盖）；v3.4 用<b>慢热协议</b>替代——
          先各自适应，再小步渗透，最后才回到基线。仿真自动启用 slime grid + HGT、
          但<b>关闭信息素奖励</b>（避免破坏 oracle 真值表）。
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div className="rounded border border-cyan-700/50 bg-cyan-950/20 p-3">
            <div className="text-xs font-semibold text-cyan-100 mb-1">
              ① 共栖期 (commensal)
            </div>
            <div className="text-[11px] text-cyan-200/70 leading-snug mb-2">
              0 .. {commensalS.toFixed(0)}s · HGT <b>完全关闭</b>。
              两群菌株共享同一培养皿、独立繁殖、独立竞争——
              先确认各自能在新环境里活下来。
            </div>
            <NumField
              label="共栖期长度 (s)"
              value={commensalS}
              onChange={setCommensalS}
              step={5}
              min={0}
            />
          </div>

          <div className="rounded border border-fuchsia-700/50 bg-fuchsia-950/20 p-3">
            <div className="text-xs font-semibold text-fuchsia-100 mb-1">
              ② 受控交换期 (controlled exchange)
            </div>
            <div className="text-[11px] text-fuchsia-200/70 leading-snug mb-2">
              {commensalS.toFixed(0)} .. {(commensalS + exchangeS).toFixed(0)}s ·
              HGT 启用，但 blend 仅 <b>{phase2Blend.toFixed(2)}</b>
              （基线 0.30）。每次只小幅渗透几条权重，
              避免大幅值一方一口吞掉另一方。
            </div>
            <div className="grid grid-cols-2 gap-2">
              <NumField
                label="交换期长度 (s)"
                value={exchangeS}
                onChange={setExchangeS}
                step={5}
                min={0}
              />
              <NumField
                label="phase2 blend (η)"
                value={phase2Blend}
                onChange={setPhase2Blend}
                step={0.01}
                min={0}
              />
              <NumField
                label="phase2 prob ×"
                value={phase2ProbMul}
                onChange={setPhase2ProbMul}
                step={0.5}
                min={0}
              />
            </div>
          </div>
        </div>

        <div className="rounded border border-slate-700/60 bg-slate-900/40 p-2.5">
          <div className="text-xs font-semibold text-slate-200 mb-0.5">
            ③ 恢复期 (restored, t ≥ {(commensalS + exchangeS).toFixed(0)}s)
          </div>
          <div className="text-[11px] text-slate-400 leading-snug">
            协议结束，HGT blend / prob 恢复 colony 基线（0.30 / 0.02）。
            从这里开始就是普通的 colony 演化了。
          </div>
        </div>

        <div className="text-[11px] text-fuchsia-200/60 mt-2">
          想跳过协议直接走基线，把两个长度都设为 0；想做「永远低 blend」实验
          可以把交换期开很大（如 3600s）。
        </div>
      </section>

      {/* Speciation / assortative HGT — SPEC_L2_V3.5 */}
      <section className="rounded-lg border border-emerald-700/40 bg-emerald-950/15 p-4">
        <h2 className="text-base font-semibold text-emerald-100 mb-2">
          🧬 物种隔离 · 同类相吸 HGT (SPEC_L2_V3.5)
        </h2>
        <p className="text-xs text-emerald-200/80 leading-relaxed mb-3">
          v3.4 仍然失败是因为 AND 学家和 NOT D1 学家的「基因组不相容」——
          权重幅值 / 编码方式根本不一样，强行 blend 会两败俱伤。v3.5 接受
          <b>「群体即生命」</b>原则：不再追求单体「双修」，而是让两个物种<b>
          共存于同一菌落</b>，AND 个体只跟 AND 换基因、NOT 只跟 NOT 换基因；
          L2 任务由<b>菌落整体</b>（专家投票）完成，而不是任何单细胞。
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mb-3">
          <button
            type="button"
            onClick={() => setAssortativeT(null)}
            className={clsx(
              "rounded border p-2 text-left text-xs leading-snug",
              assortativeT === null
                ? "border-slate-400/70 bg-slate-700/30 text-slate-100 ring-1 ring-slate-300/60"
                : "border-slate-700 bg-slate-950/30 text-slate-300 hover:bg-slate-900"
            )}
          >
            <div className="font-semibold mb-0.5">关闭 (legacy)</div>
            <div className="text-[10px] opacity-80">
              v3.4 行为：HGT 永远挑「最富裕」的邻居。
              已知会让 D1 学家单边扫荡 AND 学家。
            </div>
          </button>
          <button
            type="button"
            onClick={() => setAssortativeT(0.30)}
            className={clsx(
              "rounded border p-2 text-left text-xs leading-snug",
              assortativeT !== null && assortativeT > 0
                ? "border-emerald-400/70 bg-emerald-700/20 text-emerald-100 ring-1 ring-emerald-300/60"
                : "border-slate-700 bg-slate-950/30 text-slate-300 hover:bg-slate-900"
            )}
          >
            <div className="font-semibold mb-0.5">柔性偏好 (推荐)</div>
            <div className="text-[10px] opacity-80">
              T &gt; 0：donor 抽样 ∝ exp(-|Δniche|/T)。
              小 T = 强偏好，大 T = 弱偏好。仍允许偶发跨界基因。
            </div>
          </button>
          <button
            type="button"
            onClick={() => setAssortativeT(0)}
            className={clsx(
              "rounded border p-2 text-left text-xs leading-snug",
              assortativeT === 0
                ? "border-fuchsia-400/70 bg-fuchsia-700/20 text-fuchsia-100 ring-1 ring-fuchsia-300/60"
                : "border-slate-700 bg-slate-950/30 text-slate-300 hover:bg-slate-900"
            )}
          >
            <div className="font-semibold mb-0.5">严格物种隔离</div>
            <div className="text-[10px] opacity-80">
              T = 0：只有 niche 最近的同类才能转入基因。
              彻底封死跨物种 HGT，保护各自专长。
            </div>
          </button>
        </div>

        {assortativeT !== null && (
          <div className="rounded border border-emerald-700/40 bg-emerald-950/20 p-3">
            <label className="block text-xs text-emerald-200 mb-1">
              assortative_temperature (T) ={" "}
              <span className="font-mono numeric">{assortativeT.toFixed(2)}</span>
              <span className="text-emerald-300/60 ml-2">
                niche = acc_AND − acc_NOT ∈ [-1, +1]
              </span>
            </label>
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={assortativeT}
              onChange={(e) => setAssortativeT(Number(e.target.value))}
              className="w-full accent-emerald-400"
            />
            <div className="flex justify-between text-[10px] text-emerald-200/60 font-mono mt-0.5">
              <span>0 = 严格隔离</span>
              <span>0.3 = 推荐</span>
              <span>1 = 弱偏好</span>
              <span>2 = 几乎随机</span>
            </div>
          </div>
        )}

        <div className="text-[11px] text-emerald-200/60 mt-2">
          ObservePage 上会出现「物种结构」面板和 colony_dual_acc 指标——
          后者只有当 AND 专家和 NOT 专家都达到一定数量时才非零，是 v3.5 真正
          的 L2 成功判据。
        </div>
      </section>

      {launchErr && (
        <div className="px-4 py-2 rounded bg-rose-500/15 text-rose-200 text-sm border border-rose-500/30">
          {launchErr}
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={launch}
          disabled={busy || picks.length === 0 || hasOver}
          className={clsx(
            "px-5 py-2 rounded-md font-medium transition-colors",
            "bg-cyan-500 hover:bg-cyan-400 text-slate-950",
            "disabled:opacity-40 disabled:cursor-not-allowed"
          )}
          title={
            picks.length === 0
              ? "至少挑 1 个菌株"
              : hasOver
                ? "比例之和 > 1，必须 ≤ 1"
                : "启动杂交皿"
          }
        >
          {busy ? "启动中…" : "🚀 倒进培养皿"}
        </button>
        <span className="text-xs text-slate-400 numeric">
          founders = {picks.length} · sum_frac = {totalFrac.toFixed(2)} ·
          commensal = {commensalS.toFixed(0)}s · exchange ={" "}
          {exchangeS.toFixed(0)}s · η = {phase2Blend.toFixed(2)} · T_niche ={" "}
          {assortativeT === null ? "off" : assortativeT.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

const PICK_COLORS = [
  "bg-cyan-300",
  "bg-fuchsia-300",
  "bg-amber-300",
  "bg-emerald-300",
  "bg-rose-300",
];

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function StrainRow({
  strain,
  picked,
  fraction,
  onToggle,
  onSetFrac,
  onDelete,
}: {
  strain: StrainMeta;
  picked: boolean;
  fraction: number;
  onToggle: () => void;
  onSetFrac: (f: number) => void;
  onDelete: () => void;
}) {
  const dt = strain.created_at ? new Date(strain.created_at) : null;
  return (
    <div
      className={clsx(
        "rounded border p-3 transition-colors",
        picked
          ? "border-cyan-400/60 bg-cyan-950/30"
          : "border-slate-800 bg-slate-900/40 hover:bg-slate-900/70"
      )}
    >
      <div className="flex items-baseline gap-3">
        <input
          type="checkbox"
          checked={picked}
          onChange={onToggle}
          className="h-4 w-4 accent-cyan-400"
        />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-slate-100 truncate">
            {strain.name}
            <span className="ml-2 text-[10px] font-mono text-slate-500">
              {strain.id}
            </span>
          </div>
          <div className="text-[11px] font-mono text-slate-400 numeric mt-0.5">
            n={strain.n_agents} · t_sim={strain.t_sim.toFixed(0)}s · seed=
            {strain.source_seed}
            {strain.source_difficulty && ` · diff=${strain.source_difficulty}`}
            {strain.acc_and_pop_at_save != null && (
              <>
                {" "}· AND={(strain.acc_and_pop_at_save * 100).toFixed(0)}%
              </>
            )}
            {strain.acc_not_pop_at_save != null && (
              <>
                {" "}· NOT={(strain.acc_not_pop_at_save * 100).toFixed(0)}%
              </>
            )}
            {dt && ` · ${dt.toLocaleString()}`}
          </div>
          {strain.note && (
            <div className="text-[11px] text-slate-300/80 mt-1 italic line-clamp-2">
              "{strain.note}"
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={onDelete}
          className="text-[11px] text-rose-300/70 hover:text-rose-200 px-2 py-0.5 rounded hover:bg-rose-500/10"
          title="删除菌株（不可撤销）"
        >
          ✕ 删除
        </button>
      </div>
      {picked && (
        <div className="mt-3 flex items-center gap-3">
          <label className="text-xs text-cyan-200">比例</label>
          <input
            type="range"
            min={0.01}
            max={1}
            step={0.01}
            value={fraction}
            onChange={(e) => onSetFrac(Number(e.target.value))}
            className="flex-1 accent-cyan-400"
          />
          <span className="text-xs font-mono text-cyan-100 numeric w-14 text-right">
            {(fraction * 100).toFixed(0)}%
          </span>
          <input
            type="number"
            min={0.01}
            max={1}
            step={0.05}
            value={fraction}
            onChange={(e) => onSetFrac(Number(e.target.value))}
            className="w-20 bg-slate-950 border border-slate-700 rounded px-1.5 py-0.5 text-xs font-mono numeric"
          />
        </div>
      )}
    </div>
  );
}

function NumField({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <div className="rounded border border-slate-800 bg-slate-950/40 p-2">
      <label className="block text-xs text-slate-300 mb-1">{label}</label>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-sm font-mono numeric"
      />
    </div>
  );
}
