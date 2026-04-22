import { useEffect, useMemo, useRef } from "react";
import clsx from "clsx";
import { useStore, type ChartPoint } from "../../store";
import type { OracleSnapshot, TelemetryEvent } from "../../types";

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

      <LogicProgress
        accAnd={accAnd}
        accNot={accNot}
        bothPass={bothPass}
        diversity={diversity}
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <SurvivalCounter
          alive={ev.pop_size}
          totalBorn={totalBorn}
          totalDied={totalDied}
          tSim={ev.t_sim}
        />
        <VitalityECG history={history} />
        <DiversityBadge diversity={diversity} bothPass={bothPass} />
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
        label="AND 进化温度计"
        value={accAnd}
        color="emerald"
        sub="平均逻辑正确率"
      />
      <ProgressBar
        label="NOT 进化温度计"
        value={accNot}
        color="rose"
        sub="高难溢价题正确率（稀缺）"
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
}: {
  diversity: number;
  bothPass: number;
}) {
  const trapped = diversity < 0.3 && bothPass < 0.05;
  const elite = bothPass >= 0.05;
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
            ? "⚠ 陷入『三选一』陷阱 (logic_diversity 过低)"
            : "⏳ 进行中…"}
      </div>
      <div className="text-[11px] text-amber-200/60 mt-1">
        Logic_Diversity_Score = {diversity.toFixed(3)} · Both-Pass ={" "}
        {(bothPass * 100).toFixed(1)}%
      </div>
    </div>
  );
}
