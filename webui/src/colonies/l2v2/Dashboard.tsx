import { useEffect, useMemo, useRef } from "react";
import clsx from "clsx";
import { useStore, type ChartPoint } from "../../store";
import type {
  OracleSnapshot,
  RowAccuracies,
  TaskDifficulty,
  TelemetryEvent,
} from "../../types";

/**
 * SPEC_L2_V2.0 §4 — Ecological dashboard for non-technical observers.
 *
 * Three panels:
 *   ① Vitality ECG   — total Credit (pop × mean) over time, heartbeat-style.
 *   ② Logic Progress — AND / NOT accuracy bars + "both-pass" elite badge.
 *   ③ Translator     — current oracle question, population consensus, evaluation comment.
 *
 * Mounted by ObservePage only when the running task is "l2v2_ctrl".
 */
export function L2Dashboard({ ev }: { ev: TelemetryEvent | null }) {
  const history = useStore((s) => s.history);
  const totalBorn = useStore((s) => s.totalBorn);
  const totalDied = useStore((s) => s.totalDied);

  if (!ev || ev.task !== "l2v2_ctrl") return null;

  const oracle = ev.oracle ?? null;
  const accAnd = ev.acc_and_pop ?? 0;
  const accNot = ev.acc_not_pop ?? 0;
  const bothPass = ev.both_pass_pct ?? 0;
  const diversity = ev.logic_diversity ?? 0;
  const consensus = ev.consensus_bit ?? null;
  const consAcc = ev.consensus_acc ?? 0;
  const accAnd11 = ev.acc_and_11_pop ?? 0;
  const accNot0 = ev.acc_not_0_pop ?? 0;
  const rowAcc = ev.row_acc ?? null;
  const rowN = ev.row_n ?? null;
  const difficulty = ev.task_difficulty ?? "balanced";

  return (
    <div className="rounded-lg border border-amber-700/40 bg-amber-950/10 p-4 space-y-4">
      <div className="flex items-baseline justify-between">
        <h2 className="text-base font-semibold text-amber-100">
          🧬 L2v2 生态仪表盘
        </h2>
        <span className="text-[11px] text-amber-200/60 font-mono">
          SPEC_L2_V2.0 · 三通道逻辑门控
        </span>
      </div>

      <Translator
        oracle={oracle}
        consensus={consensus}
        consensusAcc={consAcc}
        accAnd={accAnd}
        accNot={accNot}
      />

      <SpecificAccuracy
        accAnd11={accAnd11}
        accNot0={accNot0}
        accAnd={accAnd}
        accNot={accNot}
        difficulty={difficulty}
      />

      <LogicProgress
        accAnd={accAnd}
        accNot={accNot}
        bothPass={bothPass}
        diversity={diversity}
      />

      {rowAcc && rowN && <TruthTableMatrix rowAcc={rowAcc} rowN={rowN} />}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <SurvivalCounter
          alive={ev.pop_size}
          totalBorn={totalBorn}
          totalDied={totalDied}
          tSim={ev.t_sim}
        />
        <VitalityECG history={history} />
        <DiversityBadge
          diversity={diversity}
          bothPass={bothPass}
          accAnd={accAnd}
          accNot={accNot}
        />
      </div>
    </div>
  );
}

// ── Translator strip ─────────────────────────────────────────────────────

function Translator({
  oracle,
  consensus,
  consensusAcc,
  accAnd,
  accNot,
}: {
  oracle: OracleSnapshot | null;
  consensus: 0 | 1 | null;
  consensusAcc: number;
  accAnd: number;
  accNot: number;
}) {
  if (!oracle) {
    return (
      <div className="rounded border border-amber-800/40 bg-amber-950/30 px-3 py-2 text-sm text-amber-200/70">
        等待第一个 Oracle 信号…
      </div>
    );
  }
  const modeText = oracle.mode_name === "AND" ? "与门 (AND)" : "取反 (NOT)";
  const inputs =
    oracle.mode === 1
      ? `[A=${oracle.bit_a}]`
      : `[A=${oracle.bit_a}, B=${oracle.bit_b}]`;
  const correct = consensus !== null && consensus === oracle.target_bit;

  return (
    <div className="rounded border border-amber-800/40 bg-amber-950/30 px-3 py-3 space-y-2">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-sm font-mono">
        <div>
          <span className="text-amber-200/60">指令: </span>
          <span
            className={clsx(
              "font-semibold",
              oracle.mode === 0 ? "text-emerald-300" : "text-rose-300"
            )}
          >
            {modeText}
          </span>
          <span className="text-amber-200/40 ml-1 text-[11px]">
            (S≈{oracle.f_s_hz.toFixed(0)}Hz)
          </span>
        </div>
        <div>
          <span className="text-amber-200/60">输入: </span>
          <span className="text-amber-100">{inputs}</span>
        </div>
        <div>
          <span className="text-amber-200/60">预期: </span>
          <span className="text-amber-100 font-bold">{oracle.target_bit}</span>
        </div>
        <div className="ml-auto">
          <span className="text-amber-200/60">族群共识: </span>
          {consensus === null ? (
            <span className="text-slate-400">—</span>
          ) : (
            <span
              className={clsx(
                "font-bold text-base",
                correct ? "text-emerald-300" : "text-rose-300"
              )}
            >
              [{consensus}]
            </span>
          )}
          <span className="text-amber-200/40 ml-2 text-[11px]">
            准确率 {(consensusAcc * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      <div className="text-[12px] text-amber-100/80 italic leading-relaxed">
        {evaluationComment(oracle, consensus, consensusAcc, accAnd, accNot)}
      </div>
    </div>
  );
}

function evaluationComment(
  oracle: OracleSnapshot,
  consensus: 0 | 1 | null,
  consAcc: number,
  accAnd: number,
  accNot: number
): string {
  if (consensus === null) return "种群尚未对此题做出回答。";
  const correct = consensus === oracle.target_bit;
  if (oracle.mode_name === "NOT") {
    if (correct && consAcc > 0.85) return "种群已掌握『拒绝』逻辑——这是高难溢价题。";
    if (correct && consAcc > 0.55) return "种群正在对抗高难度环境，部分个体已学会取反。";
    if (!correct && accNot < 0.2) return "种群对 NOT 指令仍无反应，演化压力可能不足。";
    return `NOT 题正确率 ${(accNot * 100).toFixed(0)}%，仍在攻坚。`;
  }
  if (correct && consAcc > 0.85) return "种群对 AND 已经形成稳固共识。";
  if (correct && consAcc > 0.55) return "AND 已被多数掌握，少数个体仍在迷路。";
  if (!correct && accAnd < 0.3) return "AND 共识尚未形成，奖励信号在等待第一只『启蒙者』。";
  return `AND 题正确率 ${(accAnd * 100).toFixed(0)}%。`;
}

// ── Specific-accuracy dashboard (the "真学会" gauge) ─────────────────────
//
// Pinned right under the Translator on purpose: this is THE gauge that tells
// "看到演化在做什么" apart from the silent attractor.
// - acc_and_11_pop = fraction of (1,1)-AND windows the population gets right
// - acc_not_0_pop  = fraction of (a=0)-NOT windows the population gets right
// Both are zero for a permanently-silent population, regardless of how the
//混合 acc_and_pop / acc_not_pop look on the温度计 below.

const SILENT_CEILING_BY_DIFFICULTY: Record<TaskDifficulty, { and: number; not: number }> = {
  uniform:  { and: 0.75, not: 0.5 },
  balanced: { and: 0.50, not: 0.5 },
  hard:     { and: 0.30, not: 0.3 },
  extreme:  { and: 0.10, not: 0.1 },
  // SPEC_L2_V3.0 §2.4 specialist dishes — only one mode is sampled, so the
  // OTHER mode's accuracy is undefined; we just mirror the active mode's
  // silent ceiling and let the dashboard show 0.0 for the inactive side.
  and_only: { and: 0.50, not: 0.0 },
  not_only: { and: 0.0,  not: 0.5 },
};

function SpecificAccuracy({
  accAnd11,
  accNot0,
  accAnd,
  accNot,
  difficulty,
}: {
  accAnd11: number;
  accNot0: number;
  accAnd: number;
  accNot: number;
  difficulty: TaskDifficulty;
}) {
  const ceil = SILENT_CEILING_BY_DIFFICULTY[difficulty] ?? SILENT_CEILING_BY_DIFFICULTY.balanced;
  // "Silent gap" — how much of the headline acc is *not* explained by真学会.
  // Big gap on a healthy-looking温度计 = silent attractor in disguise.
  const andGap = Math.max(0, accAnd - accAnd11 * 0.25);
  const notGap = Math.max(0, accNot - accNot0 * 0.5);
  const verdict = verdictFor(accAnd11, accNot0, accAnd, accNot);
  return (
    <div className="rounded border border-cyan-700/40 bg-cyan-950/20 p-3 space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-semibold text-cyan-100">
          🎯 真学会率（特异性仪表盘）
        </span>
        <span className="text-[11px] text-cyan-200/60 font-mono">
          v2.3.1 · 难度={difficulty}
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <SpecBar
          label="1 AND 1 = 1 命中率"
          value={accAnd11}
          mixed={accAnd}
          mixedLabel="acc_AND 混合"
          silentCeiling={ceil.and}
          color="emerald"
          hint="真学会的唯一证据 — 这一项不上升，AND 温度计的高分都是错觉。"
        />
        <SpecBar
          label="NOT 0 = 1 命中率"
          value={accNot0}
          mixed={accNot}
          mixedLabel="acc_NOT 混合"
          silentCeiling={ceil.not}
          color="rose"
          hint="NOT 模式的『启蒙』指标 — 沉默个体在此恒等于 0。"
        />
      </div>
      <div className="text-[12px] text-cyan-100/85 italic leading-relaxed">
        {verdict}
        {(andGap > 0.15 || notGap > 0.15) && (
          <span className="text-amber-300/90 not-italic">
            {" "}· ⚠ 温度计 vs. 真学会率出现明显落差，怀疑「沉默搭便车」。
          </span>
        )}
      </div>
    </div>
  );
}

function verdictFor(
  accAnd11: number,
  accNot0: number,
  accAnd: number,
  accNot: number
): string {
  if (accAnd11 < 0.05 && accNot0 < 0.05) {
    if (accAnd > 0.4 || accNot > 0.4) {
      return "种群仍处于『沉默搭便车』状态 — 温度计虚高，未真懂任一种 1=输出。";
    }
    return "种群尚未对 target=1 的题目做出有效回答；演化火苗未点燃。";
  }
  if (accAnd11 > 0.7 && accNot0 > 0.7) {
    return "种群已掌握 1∧1=1 与 NOT 0=1 — 这是『活过来』的硬证据。";
  }
  if (accAnd11 > 0.5) {
    return "AND 已破冰 — 1∧1=1 命中过半，演化通道打开。";
  }
  if (accNot0 > 0.5) {
    return "NOT 已破冰 — 拒绝指令开始被掌握。";
  }
  return "在 0%↔50% 之间挣扎 — 部分个体已尝试 1 输出，但还不稳。";
}

function SpecBar({
  label,
  value,
  mixed,
  mixedLabel,
  silentCeiling,
  color,
  hint,
}: {
  label: string;
  value: number;
  mixed: number;
  mixedLabel: string;
  silentCeiling: number;
  color: "emerald" | "rose";
  hint: string;
}) {
  const pct = Math.max(0, Math.min(1, value));
  const mixedPct = Math.max(0, Math.min(1, mixed));
  const colors = {
    emerald: "bg-emerald-400",
    rose: "bg-rose-400",
  } as const;
  return (
    <div className="rounded border border-cyan-800/30 bg-cyan-950/30 p-3">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-semibold text-cyan-100">{label}</span>
        <span className="text-base font-mono numeric text-cyan-200">
          {(pct * 100).toFixed(1)}%
        </span>
      </div>
      <div className="mt-2 relative h-3 rounded bg-slate-900/80 overflow-hidden">
        <div
          className={clsx("h-full transition-[width] duration-300", colors[color])}
          style={{ width: `${pct * 100}%` }}
        />
        {/* 混合温度计参考虚线 — 看落差用 */}
        <div
          className="absolute top-0 bottom-0 border-l border-dashed border-cyan-300/40"
          style={{ left: `${mixedPct * 100}%` }}
          title={`${mixedLabel} = ${(mixedPct * 100).toFixed(1)}%`}
        />
        {/* 沉默上限——这条线不能解释为「学会」 */}
        <div
          className="absolute top-0 bottom-0 border-l-2 border-amber-400/70"
          style={{ left: `${silentCeiling * 100}%` }}
          title={`沉默策略上限 ≈ ${(silentCeiling * 100).toFixed(0)}%`}
        />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-cyan-200/60 numeric">
        <span>{mixedLabel} {(mixedPct * 100).toFixed(0)}%</span>
        <span>沉默上限 {(silentCeiling * 100).toFixed(0)}%</span>
      </div>
      <div className="mt-1 text-[11px] text-cyan-200/60 leading-snug">{hint}</div>
    </div>
  );
}

// ── Truth-table matrix (all 6 rows) ──────────────────────────────────────

function TruthTableMatrix({
  rowAcc,
  rowN,
}: {
  rowAcc: RowAccuracies;
  rowN: RowAccuracies;
}) {
  const rows: Array<{
    key: keyof RowAccuracies;
    label: string;
    target: 0 | 1;
  }> = [
    { key: "and_00", label: "AND (0,0) → 0", target: 0 },
    { key: "and_01", label: "AND (0,1) → 0", target: 0 },
    { key: "and_10", label: "AND (1,0) → 0", target: 0 },
    { key: "and_11", label: "AND (1,1) → 1", target: 1 },
    { key: "not_a0", label: "NOT (a=0) → 1", target: 1 },
    { key: "not_a1", label: "NOT (a=1) → 0", target: 0 },
  ];
  return (
    <details className="rounded border border-amber-800/30 bg-amber-950/20 p-3">
      <summary className="cursor-pointer text-xs font-semibold text-amber-100 hover:text-amber-50">
        🔬 完整真值表（6 行细分准确率） — 看每一题真学会了多少
      </summary>
      <div className="mt-3 grid grid-cols-2 md:grid-cols-3 gap-2">
        {rows.map((r) => {
          const acc = rowAcc[r.key];
          const n = rowN[r.key];
          const isTargetOne = r.target === 1;
          const dim = n < 5;
          return (
            <div
              key={r.key}
              className={clsx(
                "rounded border px-2 py-1.5",
                isTargetOne
                  ? "border-emerald-700/40 bg-emerald-950/20"
                  : "border-slate-700/50 bg-slate-900/30",
                dim && "opacity-50"
              )}
            >
              <div className="flex items-baseline justify-between">
                <span
                  className={clsx(
                    "text-[11px] font-mono",
                    isTargetOne ? "text-emerald-200" : "text-slate-300"
                  )}
                >
                  {r.label}
                </span>
                <span className="text-sm font-mono numeric text-amber-100">
                  {(acc * 100).toFixed(0)}%
                </span>
              </div>
              <div className="mt-1 h-1.5 rounded bg-slate-900/80 overflow-hidden">
                <div
                  className={clsx(
                    "h-full transition-[width] duration-300",
                    isTargetOne ? "bg-emerald-400" : "bg-slate-500"
                  )}
                  style={{ width: `${Math.max(0, Math.min(1, acc)) * 100}%` }}
                />
              </div>
              <div className="text-[10px] text-amber-200/50 mt-0.5 numeric">
                n={n} 窗 {dim && "(样本不足)"}
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-2 text-[11px] text-amber-200/60 leading-relaxed">
        绿底 = target=1 的题目（沉默上限 0%）；灰底 = target=0 的题目（沉默上限 100%）。
        想确认演化「真懂」，盯绿底两栏的进度。
      </div>
    </details>
  );
}

// ── Logic progress bars ──────────────────────────────────────────────────

function LogicProgress({
  accAnd,
  accNot,
  bothPass,
  diversity,
}: {
  accAnd: number;
  accNot: number;
  bothPass: number;
  diversity: number;
}) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      <ProgressBar
        label="AND 混合温度计"
        value={accAnd}
        color="emerald"
        sub="所有 AND 题的平均正确率（含 target=0 的 3 行 + target=1 的 1 行）。沉默策略可虚高。"
      />
      <ProgressBar
        label="NOT 混合温度计"
        value={accNot}
        color="rose"
        sub="所有 NOT 题的平均正确率（target=0 与 target=1 各占一半）。"
      />
      <ProgressBar
        label="精英双通过率"
        value={bothPass}
        color="amber"
        sub="同时 AND≥70% 且 NOT≥70% 的个体占比 (SPEC §5.3 收敛标准)"
        targetLine={0.05}
        targetLabel="5% 验收线"
      />
      <ProgressBar
        label="逻辑多样性"
        value={diversity}
        color="violet"
        sub="1 - |acc_AND - acc_NOT| / max → 越接近 1 = 越平衡"
      />
    </div>
  );
}

function ProgressBar({
  label,
  value,
  color,
  sub,
  targetLine,
  targetLabel,
}: {
  label: string;
  value: number;
  color: "emerald" | "rose" | "amber" | "violet";
  sub: string;
  targetLine?: number;
  targetLabel?: string;
}) {
  const pct = Math.max(0, Math.min(1, value));
  const colors = {
    emerald: "bg-emerald-500",
    rose: "bg-rose-500",
    amber: "bg-amber-500",
    violet: "bg-violet-500",
  } as const;
  return (
    <div className="rounded border border-amber-800/30 bg-amber-950/20 p-3">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-semibold text-amber-100">{label}</span>
        <span className="text-sm font-mono numeric text-amber-200">
          {(pct * 100).toFixed(1)}%
        </span>
      </div>
      <div className="mt-2 relative h-3 rounded bg-slate-900/80 overflow-hidden">
        <div
          className={clsx("h-full transition-[width] duration-300", colors[color])}
          style={{ width: `${pct * 100}%` }}
        />
        {targetLine != null && (
          <div
            className="absolute top-0 bottom-0 border-l-2 border-amber-300/70"
            style={{ left: `${targetLine * 100}%` }}
            title={targetLabel}
          />
        )}
      </div>
      <div className="mt-1 text-[11px] text-amber-200/50">{sub}</div>
    </div>
  );
}

// ── Survival counter ─────────────────────────────────────────────────────

function SurvivalCounter({
  alive,
  totalBorn,
  totalDied,
  tSim,
}: {
  alive: number;
  totalBorn: number;
  totalDied: number;
  tSim: number;
}) {
  return (
    <div className="rounded border border-amber-800/30 bg-amber-950/20 p-3">
      <div className="text-xs font-semibold text-amber-100 mb-2">存活计数器</div>
      <div className="grid grid-cols-3 gap-2 text-center">
        <Stat label="存活" value={alive.toString()} accent="emerald" />
        <Stat label="累计出生" value={totalBorn.toString()} accent="sky" />
        <Stat label="累计死亡" value={totalDied.toString()} accent="rose" />
      </div>
      <div className="mt-2 text-[11px] text-amber-200/50 numeric">
        t_sim={tSim.toFixed(1)}s · 自仿真启动起累计
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: "emerald" | "sky" | "rose";
}) {
  const colors = {
    emerald: "text-emerald-300",
    sky: "text-sky-300",
    rose: "text-rose-300",
  } as const;
  return (
    <div>
      <div className={clsx("text-xl font-bold numeric", colors[accent])}>
        {value}
      </div>
      <div className="text-[10px] text-amber-200/60 mt-0.5">{label}</div>
    </div>
  );
}

// ── Vitality ECG ─────────────────────────────────────────────────────────

function VitalityECG({ history }: { history: ChartPoint[] }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  // Use last 200 points (~100s @ 2Hz windows or ~10s @ 20Hz)
  const slice = useMemo(() => history.slice(-200), [history]);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const W = cv.width;
    const H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, W, H);

    if (slice.length < 2) return;
    const vmin = 0;
    const vmax = Math.max(1, ...slice.map((p) => p.vitality));
    const tmin = slice[0].t;
    const tmax = slice[slice.length - 1].t;
    const tspan = Math.max(1e-6, tmax - tmin);

    // Grid line at the median for "heartbeat" reference.
    ctx.strokeStyle = "rgba(245, 158, 11, 0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.stroke();

    // Polyline.
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    slice.forEach((p, i) => {
      const x = ((p.t - tmin) / tspan) * (W - 4) + 2;
      const y = H - ((p.vitality - vmin) / (vmax - vmin)) * (H - 4) - 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Last-point pulse dot.
    const last = slice[slice.length - 1];
    const lx = W - 2;
    const ly = H - ((last.vitality - vmin) / (vmax - vmin)) * (H - 4) - 2;
    ctx.fillStyle = "#fbbf24";
    ctx.beginPath();
    ctx.arc(lx, ly, 3, 0, Math.PI * 2);
    ctx.fill();
  }, [slice]);

  return (
    <div className="rounded border border-amber-800/30 bg-amber-950/20 p-3">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-semibold text-amber-100">种群健康度</span>
        <span className="text-[11px] text-amber-200/60 numeric">
          Σ Credit ≈{" "}
          {slice.length > 0
            ? slice[slice.length - 1].vitality.toFixed(0)
            : "—"}
        </span>
      </div>
      <canvas
        ref={ref}
        width={320}
        height={70}
        className="w-full h-[70px] mt-2 rounded"
      />
    </div>
  );
}

// ── Diversity badge ──────────────────────────────────────────────────────

function DiversityBadge({
  diversity,
  bothPass,
  accAnd,
  accNot,
}: {
  diversity: number;
  bothPass: number;
  accAnd: number;
  accNot: number;
}) {
  // L2v2 only has two logic gates (AND / NOT) — see SPEC_L2_V2.0 §2.
  // The third "OR/XOR" gate from the original §6 roadmap is deferred to L2.5+,
  // so this badge speaks in terms of偏科 (skewed specialisation), not "三选一".
  const trapped = diversity < 0.3 && bothPass < 0.05;
  const elite = bothPass >= 0.05;

  // Which side is the population偏科 toward?  Tells the user where to push next
  // (e.g. raise NOT sampling weight if "只会 AND, 不会 NOT").
  let skewLabel = "";
  if (trapped) {
    const gap = Math.abs(accAnd - accNot);
    if (gap < 0.1) {
      skewLabel = "AND/NOT 双低";
    } else if (accAnd > accNot) {
      skewLabel = `偏科 AND (会 ${(accAnd * 100).toFixed(0)}%, 不会 NOT ${(accNot * 100).toFixed(0)}%)`;
    } else {
      skewLabel = `偏科 NOT (会 ${(accNot * 100).toFixed(0)}%, 不会 AND ${(accAnd * 100).toFixed(0)}%)`;
    }
  }

  return (
    <div
      className={clsx(
        "rounded border p-3 flex flex-col justify-between",
        trapped
          ? "border-rose-700/50 bg-rose-950/30"
          : elite
            ? "border-emerald-700/50 bg-emerald-950/30"
            : "border-amber-800/30 bg-amber-950/20"
      )}
    >
      <div className="text-xs font-semibold text-amber-100">演化诊断</div>
      <div
        className={clsx(
          "text-sm font-medium mt-1",
          trapped
            ? "text-rose-300"
            : elite
              ? "text-emerald-300"
              : "text-amber-200"
        )}
      >
        {elite
          ? "✓ 已突破 5% 精英线 — 验收达标"
          : trapped
            ? `⚠ 单一逻辑陷阱：${skewLabel}`
            : "⏳ 进行中…"}
      </div>
      <div className="text-[11px] text-amber-200/60 mt-1">
        Logic_Diversity_Score = {diversity.toFixed(3)} · Both-Pass ={" "}
        {(bothPass * 100).toFixed(1)}%
      </div>
    </div>
  );
}
