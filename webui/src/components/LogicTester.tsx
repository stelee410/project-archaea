import { useState } from "react";
import clsx from "clsx";
import { api } from "../api";
import type { InferenceResponse } from "../types";

/**
 * SPEC_L2_V2.0 — human-friendly "ask the swarm a logic question" panel.
 *
 * Replaces the L1 "type a Hz number" UX for the L2v2_ctrl task:
 *
 *   1. User picks instruction (AND / NOT) and bit values (0 / 1)
 *   2. Frontend translates → (f_a_hz, f_b_hz, f_s_hz) per oracle.py constants
 *   3. Backend runs the chosen agent(s) for ``duration_ms`` and returns f_out_hz
 *   4. Frontend classifies (f_out > 50 Hz ? 1 : 0), compares to expected bit,
 *      and renders a verdict + expected vs actual + a plain-Chinese comment.
 *
 * One-shot mode: one question.
 * Battery mode : runs all 6 distinct questions (AND × 4 + NOT × 2) and
 *                shows a pass/fail table + total pass rate.
 *
 * Constants below MUST stay in sync with archaea/oracle.py.
 */

// ── Constants mirrored from archaea/oracle.py (SPEC_L2_V2.0 §2) ────────────
const LOGIC_LOW_HZ = 25.0;
const LOGIC_HIGH_HZ = 75.0;
const S_AND_HZ = 20.0;
const S_NOT_HZ = 80.0;
const OUT_SPIKING_THRESHOLD_HZ = 50.0;

type Mode = "AND" | "NOT";
type Bit = 0 | 1;

interface QuestionSpec {
  mode: Mode;
  a: Bit;
  b: Bit;
  expected: Bit;
}

interface QuestionResult extends QuestionSpec {
  fOutHz: number;
  outBit: Bit;
  correct: boolean;
  fA: number;
  fB: number;
  fS: number;
  durationMs: number;
}

function expectedBit(mode: Mode, a: Bit, b: Bit): Bit {
  if (mode === "AND") return ((a & b) as 0 | 1);
  // NOT operates on A only; B is a distractor in oracle.py
  return ((1 - a) as 0 | 1);
}

function bitToHz(bit: Bit): number {
  return bit ? LOGIC_HIGH_HZ : LOGIC_LOW_HZ;
}

function modeToSHz(mode: Mode): number {
  return mode === "AND" ? S_AND_HZ : S_NOT_HZ;
}

function questionLabel(q: QuestionSpec): string {
  if (q.mode === "AND") return `${q.a} AND ${q.b}`;
  return `NOT ${q.a}`;
}

function ALL_QUESTIONS(): QuestionSpec[] {
  return [
    { mode: "AND", a: 0, b: 0, expected: 0 },
    { mode: "AND", a: 0, b: 1, expected: 0 },
    { mode: "AND", a: 1, b: 0, expected: 0 },
    { mode: "AND", a: 1, b: 1, expected: 1 },
    { mode: "NOT", a: 0, b: 0, expected: 1 },
    { mode: "NOT", a: 1, b: 0, expected: 0 },
  ];
}

interface Props {
  target: "best" | "ensemble" | "random" | "swarm";
  topK: number;
  swarmRadius: number;
  durationMs: number;
  warmupMs: number;
}

export function LogicTester({
  target,
  topK,
  swarmRadius,
  durationMs,
  warmupMs,
}: Props) {
  const [mode, setMode] = useState<Mode>("AND");
  const [a, setA] = useState<Bit>(1);
  const [b, setB] = useState<Bit>(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [oneShot, setOneShot] = useState<QuestionResult | null>(null);
  const [battery, setBattery] = useState<QuestionResult[] | null>(null);

  const expected = expectedBit(mode, a, b);
  const currentSpec: QuestionSpec = { mode, a, b, expected };

  async function askOne(spec: QuestionSpec): Promise<QuestionResult> {
    const fA = bitToHz(spec.a);
    const fB = bitToHz(spec.b);
    const fS = modeToSHz(spec.mode);
    const r: InferenceResponse = await api.inference({
      f_in_hz: fA,
      f_b_hz: fB,
      f_s_hz: fS,
      target,
      top_k: topK,
      duration_ms: durationMs,
      warmup_ms: warmupMs,
      swarm_radius: swarmRadius,
    });
    const outBit: Bit = r.f_out_hz > OUT_SPIKING_THRESHOLD_HZ ? 1 : 0;
    return {
      ...spec,
      fOutHz: r.f_out_hz,
      outBit,
      correct: outBit === spec.expected,
      fA,
      fB,
      fS,
      durationMs,
    };
  }

  async function runOne() {
    setBusy(true);
    setError(null);
    setBattery(null);
    try {
      const out = await askOne(currentSpec);
      setOneShot(out);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function runBattery() {
    setBusy(true);
    setError(null);
    setOneShot(null);
    try {
      const results: QuestionResult[] = [];
      for (const q of ALL_QUESTIONS()) {
        results.push(await askOne(q));
      }
      setBattery(results);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-lg border border-amber-700/40 bg-amber-950/10 p-5">
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="text-base font-semibold text-amber-100">
          🧠 给种群出一道逻辑题 (L2v2)
        </h2>
        <span className="text-[11px] text-amber-200/60 font-mono">
          SPEC_L2_V2.0 · 通过 S 通道频率切换指令
        </span>
      </div>

      <p className="text-xs text-amber-200/70 mb-4 leading-relaxed">
        点选下面的「指令 + 输入比特」，前端会自动翻译成三通道电平
        （A/B 用 25 Hz=0 / 75 Hz=1，S 用 20 Hz=AND / 80 Hz=NOT），
        喂给当前种群里的目标 agent，再把它的输出 f_out 用 50 Hz 阈值翻译回 0/1。
        和「使用页 = 输 Hz 数字」相比，这里你直接说人话。
      </p>

      {/* Question builder */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
        <div>
          <div className="text-[11px] text-amber-200/70 mb-1">指令 (S 通道)</div>
          <div className="inline-flex rounded-md bg-slate-950/60 ring-1 ring-amber-800/40 p-0.5 w-full">
            <ModeBtn label="与门 AND" active={mode === "AND"} onClick={() => setMode("AND")} />
            <ModeBtn label="取反 NOT" active={mode === "NOT"} onClick={() => setMode("NOT")} />
          </div>
          <div className="text-[10px] text-amber-200/50 mt-1 numeric">
            {mode === "AND" ? `S ≈ ${S_AND_HZ.toFixed(0)} Hz` : `S ≈ ${S_NOT_HZ.toFixed(0)} Hz`}
          </div>
        </div>

        <BitPicker
          label="输入 A (Channel A, 4 神经元)"
          value={a}
          onChange={setA}
          subtitle={`f_a = ${bitToHz(a).toFixed(0)} Hz`}
        />

        <BitPicker
          label={`输入 B (Channel B, 4 神经元)${mode === "NOT" ? " · NOT 时被忽略" : ""}`}
          value={b}
          onChange={setB}
          subtitle={`f_b = ${bitToHz(b).toFixed(0)} Hz`}
          dimmed={mode === "NOT"}
        />
      </div>

      {/* Question preview */}
      <div className="rounded border border-amber-800/40 bg-amber-950/30 px-3 py-2 mb-3 text-sm font-mono flex items-baseline gap-3 flex-wrap">
        <span className="text-amber-200/60">题目:</span>
        <span className="text-amber-100 font-semibold text-base">
          {questionLabel(currentSpec)}
        </span>
        <span className="text-amber-200/60">期望:</span>
        <span className="text-amber-100 font-bold text-base">{expected}</span>
        <span className="ml-auto text-[11px] text-amber-200/40 numeric">
          f_a={bitToHz(a).toFixed(0)}Hz · f_b={bitToHz(b).toFixed(0)}Hz · f_s={modeToSHz(mode).toFixed(0)}Hz
        </span>
      </div>

      {/* Action buttons */}
      <div className="flex flex-wrap gap-3 mb-4">
        <button
          onClick={runOne}
          disabled={busy}
          className="px-4 py-2 rounded-md bg-amber-500 hover:bg-amber-400 text-slate-950 font-medium disabled:opacity-40"
        >
          {busy ? "提问中…" : "▶ 问这一题"}
        </button>
        <button
          onClick={runBattery}
          disabled={busy}
          className="px-4 py-2 rounded-md bg-violet-500 hover:bg-violet-400 text-slate-950 font-medium disabled:opacity-40"
          title="跑全部 6 道题：AND × 4 + NOT × 2"
        >
          {busy ? "测试中…" : "▶▶ 一键跑完 6 道题"}
        </button>
        <span className="text-[11px] text-amber-200/50 self-center">
          目标 agent: <code className="px-1 bg-slate-950 rounded">{target}</code>
          {target === "ensemble" && ` (top-${topK})`}
          {target === "swarm" && ` (黏菌 ±${swarmRadius})`}
          · duration={durationMs}ms
        </span>
      </div>

      {error && (
        <div className="mb-3 px-3 py-1.5 rounded bg-rose-500/15 text-rose-200 text-xs border border-rose-500/30">
          {error}
        </div>
      )}

      {oneShot && <OneShotResult r={oneShot} />}
      {battery && <BatteryResult results={battery} />}
    </div>
  );
}

// ── helpers ───────────────────────────────────────────────────────────────

function ModeBtn({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "flex-1 px-3 py-1.5 text-xs rounded transition-colors",
        active ? "bg-amber-500 text-slate-950 font-semibold" : "text-amber-200 hover:bg-amber-900/40"
      )}
    >
      {label}
    </button>
  );
}

function BitPicker({
  label,
  value,
  onChange,
  subtitle,
  dimmed,
}: {
  label: string;
  value: Bit;
  onChange: (v: Bit) => void;
  subtitle: string;
  dimmed?: boolean;
}) {
  return (
    <div className={clsx(dimmed && "opacity-50")}>
      <div className="text-[11px] text-amber-200/70 mb-1">{label}</div>
      <div className="inline-flex rounded-md bg-slate-950/60 ring-1 ring-amber-800/40 p-0.5 w-full">
        <button
          onClick={() => onChange(0)}
          className={clsx(
            "flex-1 px-3 py-1.5 text-base font-bold rounded transition-colors",
            value === 0 ? "bg-amber-500 text-slate-950" : "text-amber-200 hover:bg-amber-900/40"
          )}
        >
          0
        </button>
        <button
          onClick={() => onChange(1)}
          className={clsx(
            "flex-1 px-3 py-1.5 text-base font-bold rounded transition-colors",
            value === 1 ? "bg-amber-500 text-slate-950" : "text-amber-200 hover:bg-amber-900/40"
          )}
        >
          1
        </button>
      </div>
      <div className="text-[10px] text-amber-200/50 mt-1 numeric">{subtitle}</div>
    </div>
  );
}

function OneShotResult({ r }: { r: QuestionResult }) {
  return (
    <div
      className={clsx(
        "rounded-md border-2 p-4",
        r.correct
          ? "border-emerald-500/60 bg-emerald-950/30"
          : "border-rose-500/60 bg-rose-950/30"
      )}
    >
      <div className="flex items-baseline justify-between gap-3 mb-3">
        <div className="text-sm">
          <span className="text-slate-400">题目: </span>
          <span className="font-mono font-semibold text-base text-slate-100">
            {questionLabel(r)}
          </span>
        </div>
        <div className={clsx("text-2xl font-bold", r.correct ? "text-emerald-300" : "text-rose-300")}>
          {r.correct ? "✓ 正确" : "✗ 错误"}
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm font-mono">
        <Cell label="种群输出 f_out" value={`${r.fOutHz.toFixed(1)} Hz`} accent="emerald" />
        <Cell
          label="阈值翻译 (>50Hz=1)"
          value={r.outBit.toString()}
          accent={r.outBit ? "emerald" : "slate"}
        />
        <Cell label="期望" value={r.expected.toString()} accent="amber" />
        <Cell
          label="判定"
          value={r.correct ? "对" : "错"}
          accent={r.correct ? "emerald" : "rose"}
        />
      </div>
      <p className="text-[12px] text-slate-300 mt-3 italic leading-relaxed">
        {oneShotComment(r)}
      </p>
    </div>
  );
}

function oneShotComment(r: QuestionResult): string {
  if (r.correct) {
    if (r.mode === "NOT" && r.expected === 1) {
      return `种群正确给出了「取反结果 = 1」的高难溢价答案 (NOT ${r.a} = 1)。这是 SPEC_L2_V2.0 §2.2 中奖励最丰厚 (+50/scale) 的一类题。`;
    }
    if (r.mode === "AND" && r.expected === 1) {
      return `种群正确识别了「两个真才为真」的与门语义。`;
    }
    return `种群在该输入下保持沉默，恰好是正确的——它学会了「该不发声就不发声」。`;
  }
  // wrong
  if (r.outBit === 1 && r.expected === 0) {
    return `种群发声了 (f_out=${r.fOutHz.toFixed(0)}Hz > 50Hz)，但本题期望沉默。误激活——可能是 S 指令信号没被识别，或者抑制路径还没演化出来。`;
  }
  return `种群保持了沉默 (f_out=${r.fOutHz.toFixed(0)}Hz < 50Hz)，但本题期望发声。漏激活——synapse_gain (g) 可能太小，或者该子任务还没被覆盖到。`;
}

function BatteryResult({ results }: { results: QuestionResult[] }) {
  const passed = results.filter((r) => r.correct).length;
  const total = results.length;
  const pct = total > 0 ? passed / total : 0;
  const elite = pct >= 5 / 6; // SPEC §5.3 elite threshold = pass both modes
  return (
    <div className="rounded-md border border-violet-500/40 bg-violet-950/20 p-4">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm font-semibold text-violet-100">
          全套 6 题成绩单
        </h3>
        <div className="text-sm">
          <span className="text-violet-200/70">通过率: </span>
          <span
            className={clsx(
              "text-xl font-bold numeric",
              elite ? "text-emerald-300" : pct >= 0.5 ? "text-amber-300" : "text-rose-300"
            )}
          >
            {passed}/{total}
          </span>
          <span className="text-violet-200/50 ml-2 text-[11px]">
            ({(pct * 100).toFixed(0)}%)
            {elite && " · ⭐ 精英级"}
          </span>
        </div>
      </div>
      <table className="w-full text-xs font-mono">
        <thead className="text-violet-200/60">
          <tr>
            <th className="text-left px-2 py-1">题目</th>
            <th className="text-right px-2 py-1">f_out (Hz)</th>
            <th className="text-center px-2 py-1">输出</th>
            <th className="text-center px-2 py-1">期望</th>
            <th className="text-center px-2 py-1">判定</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr
              key={i}
              className={clsx(
                "border-t border-violet-500/10",
                r.correct ? "bg-emerald-950/10" : "bg-rose-950/10"
              )}
            >
              <td className="px-2 py-1.5 text-slate-100 font-semibold">
                {questionLabel(r)}
              </td>
              <td
                className={clsx(
                  "px-2 py-1.5 text-right numeric",
                  r.outBit ? "text-emerald-300" : "text-slate-400"
                )}
              >
                {r.fOutHz.toFixed(1)}
              </td>
              <td className="px-2 py-1.5 text-center">{r.outBit}</td>
              <td className="px-2 py-1.5 text-center text-amber-300">{r.expected}</td>
              <td className="px-2 py-1.5 text-center">
                {r.correct ? (
                  <span className="text-emerald-300">✓</span>
                ) : (
                  <span className="text-rose-300">✗</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] text-violet-200/60 mt-3 leading-relaxed">
        {elite
          ? "⭐ 该 agent / 子群已通过全部 6 题。SPEC §5.3 验收要求 5% 个体能同时通过 AND 与 NOT — 你刚刚找到了一个。"
          : pct >= 4 / 6
            ? "已掌握大部分逻辑，仍有少数边角题未通过。可以多跑几次种群（让演化继续），或换 target 试更精英的子群。"
            : "种群还在学习阶段。先回设置页跑长一点（>10 分钟）让 fitness 收敛，再回来测试。"}
      </p>
    </div>
  );
}

function Cell({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: "emerald" | "amber" | "rose" | "slate";
}) {
  const colors = {
    emerald: "text-emerald-300",
    amber: "text-amber-300",
    rose: "text-rose-300",
    slate: "text-slate-300",
  } as const;
  return (
    <div className="rounded bg-slate-950/40 px-2 py-1.5">
      <div className="text-[10px] text-slate-400">{label}</div>
      <div className={clsx("text-base font-bold", colors[accent])}>{value}</div>
    </div>
  );
}
