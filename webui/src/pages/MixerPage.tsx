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
  const [windowS, setWindowS] = useState(20.0);
  const [hgtMul, setHgtMul] = useState(8.0);

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
        admixture_window_s: windowS,
        admixture_hgt_multiplier: hgtMul,
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

      {/* Admixture window */}
      <section className="rounded-lg border border-fuchsia-700/40 bg-fuchsia-950/15 p-4">
        <h2 className="text-base font-semibold text-fuchsia-100 mb-2">
          💞 杂交期 (admixture window)
        </h2>
        <p className="text-xs text-fuchsia-200/80 leading-relaxed mb-3">
          启动后前 <b>{windowS.toFixed(0)} 秒</b>仿真时间内，HGT 概率 ×{" "}
          <b>{hgtMul.toFixed(1)}</b>，鼓励两群菌株在新培养皿里大幅交换基因；
          窗口结束后自动恢复基线。仿真会自动启用 slime grid + HGT、
          但<b>关闭信息素奖励</b>（避免破坏 oracle 真值表）。
        </p>
        <div className="grid grid-cols-2 gap-3">
          <NumField
            label="窗口长度 (s)"
            value={windowS}
            onChange={setWindowS}
            step={1}
            min={0}
          />
          <NumField
            label="HGT 倍率 ×"
            value={hgtMul}
            onChange={setHgtMul}
            step={0.5}
            min={1}
          />
        </div>
        <div className="text-[11px] text-fuchsia-200/60 mt-2">
          eff_hgt_prob 在窗口里 = base_hgt_prob × multiplier （上限 1.0）。
          想直接让两群「混匀」可以加大窗口；想观察「短促相遇 + 长期独立」就缩短。
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
          window = {windowS.toFixed(0)}s × {hgtMul.toFixed(1)}
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
