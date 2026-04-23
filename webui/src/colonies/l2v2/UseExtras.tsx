import { useState } from "react";
import clsx from "clsx";
import { api } from "../../api";
import { useStore } from "../../store";
import { usePersistentState } from "../../hooks/usePersistentState";
import type { InferenceResponse } from "../../types";

/**
 * SPEC_L2_V2.0 — human-friendly "ask the swarm a logic question" panel.
 *
 * Replaces the L1 "type a Hz number" UX for the L2v2_ctrl task:
 *
 *   1. User picks instruction (AND / NOT) and bit values (0 / 1)
 *   2. Frontend translates → (f_a_hz, f_b_hz, f_s_hz) per oracle.py constants
 *   3. Backend runs the chosen agent(s) for ``duration_ms`` and returns f_out_hz
 *   4. Frontend classifies (f_out > OUT_SPIKING_THRESHOLD_HZ ? 1 : 0), compares to expected bit,
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
// v2.4 platform-cliff fix (oracle.py ERRATA v2.4): lowered 50→20 so the
// near-spike cohort (~26-32 Hz on (1,1)) is correctly judged as '1'.
// MUST match OUT_SPIKING_THRESHOLD_HZ in archaea/oracle.py.
const OUT_SPIKING_THRESHOLD_HZ = 20.0;

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

// Self-contained query parameters — LogicTester is the *only* place L2v2
// users tweak target / topK etc., so it owns its own persistent form (no
// props, no shared "use-form" key with the L1 legacy panel).
type QueryTarget = "best" | "ensemble" | "random" | "swarm";

interface QueryFormState {
  target: QueryTarget;
  topK: number;
  swarmRadius: number;
  durationMs: number;
  warmupMs: number;
}

const QUERY_FORM_DEFAULTS: QueryFormState = {
  target: "ensemble",   // L2v2 推荐 ensemble 而非 best — 单个 best 常陷"塌陷个体"
  topK: 10,
  swarmRadius: 1,
  durationMs: 500,      // SPEC §2 评估窗口
  warmupMs: 100,
};

export function LogicTester() {
  const [queryForm, setQueryForm] = usePersistentState<QueryFormState>(
    "l2v2-logic-tester-query",
    QUERY_FORM_DEFAULTS,
  );
  const { target, topK, swarmRadius, durationMs, warmupMs } = queryForm;
  const setQueryField = <K extends keyof QueryFormState>(
    k: K,
    v: QueryFormState[K],
  ) => setQueryForm((s) => ({ ...s, [k]: v }));

  const [mode, setMode] = useState<Mode>("AND");
  const [a, setA] = useState<Bit>(1);
  const [b, setB] = useState<Bit>(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [oneShot, setOneShot] = useState<QuestionResult | null>(null);
  const [battery, setBattery] = useState<QuestionResult[] | null>(null);

  // sim health snapshot — used by SimHealthBanner to warn the user before
  // they get confused by f_out=0 results.
  const status = useStore((s) => s.status);
  const latest = useStore((s) => s.latest);

  const expected = expectedBit(mode, a, b);
  const currentSpec: QuestionSpec = { mode, a, b, expected };
  const isHardNotPremium = mode === "NOT" && a === 0; // NOT(0)=1 高难溢价

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

      <p className="text-xs text-amber-200/70 mb-3 leading-relaxed">
        点选下面的「指令 + 输入比特」，前端会自动翻译成三通道电平
        （A/B 用 25 Hz=0 / 75 Hz=1，S 用 20 Hz=AND / 80 Hz=NOT），
        喂给当前种群里的目标 agent，再把它的输出 f_out 用 {OUT_SPIKING_THRESHOLD_HZ} Hz 阈值翻译回 0/1
        （v2.4：阈值由 50 Hz 下调到 20 Hz，让「快要会发声」的近阈个体也能被算作 1）。
      </p>

      {/* Sim health snapshot — surface g / t_sim / current pop accuracy so
          the user can interpret f_out=0 results before getting confused. */}
      <SimHealthBanner
        gain={status?.config?.synapse_gain ?? 1}
        tSim={status?.t_sim ?? 0}
        accAnd={latest?.acc_and_pop ?? 0}
        accNot={latest?.acc_not_pop ?? 0}
        bothPass={latest?.both_pass_pct ?? 0}
      />

      {/* Query target selector — L2v2 自带，不再依赖 L1 的「底层接口」面板 */}
      <QuerySettings
        form={queryForm}
        setField={setQueryField}
        slimeOn={!!status?.config?.slime_mold}
      />

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
      <div className="rounded border border-amber-800/40 bg-amber-950/30 px-3 py-2 mb-1 text-sm font-mono flex items-baseline gap-3 flex-wrap">
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
      {isHardNotPremium && (
        <div className="mb-3 px-3 py-1.5 text-[11px] text-rose-200/90 bg-rose-950/30 border border-rose-700/40 rounded leading-relaxed">
          ⚠️ <b>高难溢价题 (ERRATA v2.3 + v2.4)</b>（reward=+25 + 加权采样占 NOT 题 50%）：输入电平很低（A=25Hz）却要求输出
          <b> 反向激活 </b>到 {OUT_SPIKING_THRESHOLD_HZ}Hz 以上。这违反 SNN 的「输入越多 → 输出越多」直觉，
          需要演化出抑制路径。<b>新启动种群 99% 错</b>，f_out=0 是预期现象 — 但学会的精英能拿半个繁殖代价。
        </div>
      )}

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

/**
 * Query target settings — picks WHICH agent(s) the LogicTester asks the
 * question to. SPEC §5.3 elite检测推荐 ensemble (top-K=10)，因为 best 单点
 * 容易卡在「AND 模式全 silent」的塌陷局部最优（v2.2 acc_AND=75% 但 1 AND 1 必错；
 * v2.3 加权采样后 silent 天花板已降到 50%，best 也更可靠）。
 *
 * UI 用 chip 按钮组而非 native <select>，因为 macOS 受控 select 在某些
 * Electron / Safari 版本下 onChange 会丢事件，导致用户「点了 best 没反应」。
 */
function QuerySettings({
  form,
  setField,
  slimeOn,
}: {
  form: QueryFormState;
  setField: <K extends keyof QueryFormState>(k: K, v: QueryFormState[K]) => void;
  slimeOn: boolean;
}) {
  const { target, topK, swarmRadius, durationMs, warmupMs } = form;
  const swarmDisabled = !slimeOn;

  const targetOptions: { value: QueryTarget; label: string; disabled?: boolean }[] = [
    { value: "best", label: "best · fitness 最高" },
    { value: "ensemble", label: "ensemble · top-K 平均" },
    { value: "random", label: "random · 随机活体" },
    { value: "swarm", label: `🍄 swarm${swarmDisabled ? "（需开黏菌）" : ""}`, disabled: swarmDisabled },
  ];

  const topKPresets = [1, 5, 10, 20, 50];
  const swarmRadiusPresets = [1, 2, 3, 5, 8];
  const durationPresets = [200, 500, 1000, 2000];

  return (
    <div className="rounded border border-amber-800/30 bg-slate-950/40 p-3 mb-3">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-[11px] text-amber-200/80 font-semibold">
          🎯 提问目标（决定让哪些 agent 回答）
        </div>
        <div className="text-[10px] text-amber-200/50">
          推荐 <code className="px-1 bg-slate-900 rounded">ensemble · top-K=10</code>
        </div>
      </div>

      {/* Row 1: target chip bar — full width, prominent */}
      <div className="mb-3">
        <label className="block text-amber-200/70 text-[11px] mb-1.5">target</label>
        <div className="flex flex-wrap gap-1.5">
          {targetOptions.map((opt) => (
            <ChipBtn
              key={opt.value}
              label={opt.label}
              active={target === opt.value}
              disabled={opt.disabled}
              onClick={() => setField("target", opt.value)}
            />
          ))}
        </div>
        <div className="text-[10px] text-slate-500 mt-1.5 leading-snug min-h-[2.4em]">
          {target === "best" &&
            "只问 fitness 最高那一个。v2.3 加权后，silent 假精英会被天花板暴露，best 比 v2.2 时代靠谱很多。"}
          {target === "ensemble" &&
            `问 fitness 前 ${topK} 个 agent，输出取平均。最稳的"看群体真实水平"方式，对单点塌陷有抵抗力。`}
          {target === "random" && "随机活体——抽样看普通成员什么水平（多半还在学习）。"}
          {target === "swarm" &&
            (swarmDisabled
              ? "需要在设置页勾选「赛博黏菌模式」并重启才能用。"
              : `问信息素峰值格 ±${swarmRadius} 内的所有活体——共识最强子群。`)}
        </div>
      </div>

      {/* Row 2: per-target params — show only the relevant one */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-2">
        {target === "ensemble" && (
          <ChipRow
            label="Top-K（合议人数）"
            presets={topKPresets}
            value={topK}
            onPick={(v) => setField("topK", v)}
            hint="参与平均的 fitness 排名前 K 个 agent。K 越大越稳但越被平庸稀释。"
          />
        )}
        {target === "swarm" && !swarmDisabled && (
          <ChipRow
            label="Swarm 半径 ±R 格"
            presets={swarmRadiusPresets}
            value={swarmRadius}
            onPick={(v) => setField("swarmRadius", v)}
            hint={`信息素 hotspot 周围 ${swarmRadius} 格内所有活体。R=1 共识最强。`}
          />
        )}
        <ChipRow
          label="duration_ms（评估窗口）"
          presets={durationPresets}
          value={durationMs}
          onPick={(v) => setField("durationMs", v)}
          hint="SPEC §2 评估窗口=500ms。短窗口更快但波动大；长窗口更稳。"
        />
      </div>

      <details className="mt-1">
        <summary className="text-[10px] text-slate-500 cursor-pointer hover:text-slate-300 select-none">
          ▶ 高级：warmup_ms = {warmupMs}
        </summary>
        <div className="mt-2 flex items-center gap-2">
          {[0, 50, 100, 200, 500].map((w) => (
            <ChipBtn
              key={w}
              label={`${w}ms`}
              active={warmupMs === w}
              onClick={() => setField("warmupMs", w)}
              small
            />
          ))}
          <span className="text-[10px] text-slate-500 ml-2">
            冷启动膜电位预热；训练时连续，所以 0 即可，推理时建议 ≥100ms。
          </span>
        </div>
      </details>
    </div>
  );
}

function ChipBtn({
  label,
  active,
  onClick,
  disabled,
  small,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  small?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "rounded-md font-mono transition-colors border",
        small ? "px-2 py-0.5 text-[10px]" : "px-3 py-1.5 text-[11px]",
        active
          ? "bg-amber-500 text-slate-950 font-semibold border-amber-400 shadow-sm shadow-amber-900/40"
          : disabled
            ? "bg-slate-900/40 text-slate-600 border-slate-800 cursor-not-allowed"
            : "bg-slate-900/60 text-amber-200 border-slate-700 hover:border-amber-500 hover:bg-slate-800",
      )}
    >
      {label}
    </button>
  );
}

function ChipRow({
  label,
  presets,
  value,
  onPick,
  hint,
}: {
  label: string;
  presets: number[];
  value: number;
  onPick: (v: number) => void;
  hint: string;
}) {
  const isCustom = !presets.includes(value);
  return (
    <div>
      <label className="block text-amber-200/70 text-[11px] mb-1">{label}</label>
      <div className="flex flex-wrap gap-1.5 items-center">
        {presets.map((p) => (
          <ChipBtn
            key={p}
            label={p.toString()}
            active={value === p}
            onClick={() => onPick(p)}
          />
        ))}
        {isCustom && (
          <span className="px-2 py-0.5 text-[10px] rounded bg-amber-500/20 text-amber-200 font-mono border border-amber-500/40">
            自定义={value}
          </span>
        )}
      </div>
      <div className="text-[10px] text-slate-500 mt-1 leading-snug">{hint}</div>
    </div>
  );
}

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
          label={`阈值翻译 (>${OUT_SPIKING_THRESHOLD_HZ}Hz=1)`}
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
      return `种群正确给出了「取反结果 = 1」的高难溢价答案 (NOT ${r.a} = 1)。这是奖励最丰厚 (+25 scaled) 的一类题，v2.3 加权后还占 NOT 题的 50%。`;
    }
    if (r.mode === "AND" && r.expected === 1) {
      return `种群正确识别了「两个真才为真」的与门语义。`;
    }
    return `种群在该输入下保持沉默，恰好是正确的——它学会了「该不发声就不发声」。`;
  }
  // wrong
  if (r.outBit === 1 && r.expected === 0) {
    return `种群发声了 (f_out=${r.fOutHz.toFixed(0)}Hz > ${OUT_SPIKING_THRESHOLD_HZ}Hz)，但本题期望沉默。误激活——可能是 S 指令信号没被识别，或者抑制路径还没演化出来。`;
  }
  // 漏激活 — 区分完全死寂 vs 部分激活
  if (r.fOutHz <= 0.5) {
    return (
      `网络完全沉默 (f_out=0Hz)，但本题期望发声。这通常意味着该 agent 的输出层电流 I_o 小于 LIF 阈值 V_th=1.0 ` +
      `——膜电位永远爬不到点火门槛。常见原因：` +
      `① 输出层增益 g 太小（观测页右上 SynapseGainSlider 调到 2~3）；` +
      `② 种群刚启动，权重还没演化（看上方健康条 t_sim）；` +
      `③ 这是 NOT(0)=1 高难题，对新种群本就是预期失败。`
    );
  }
  if (r.fOutHz < OUT_SPIKING_THRESHOLD_HZ * 0.5) {
    return (
      `网络在低强度发放 (f_out=${r.fOutHz.toFixed(0)}Hz)，离 ${OUT_SPIKING_THRESHOLD_HZ}Hz 阈值还远。` +
      `agent 的相关路径已经有微弱响应，但电流不足以稳定触发输出。` +
      `v2.4 effort bonus 会给这种「方向对、强度不够」的尝试一点正向积分，` +
      `继续演化或微调 g 通常能让它压过阈值。`
    );
  }
  return (
    `网络在中强度发放 (f_out=${r.fOutHz.toFixed(0)}Hz)，但还差一点没过 ${OUT_SPIKING_THRESHOLD_HZ}Hz。` +
    `离正确答案非常近——再演化几分钟、或者把 g 微调高一档，下次很可能就过了。`
  );
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

/**
 * Sim health snapshot — surfaces the three things that most often explain
 * "why is f_out always 0?" before the user asks:
 *   - synapse_gain g  (输出层电流缩放，<1.5 容易让 LIF 永远不点火)
 *   - t_sim          (太短 → fitness 还没 defined → best 选到未演化祖先)
 *   - acc_AND / acc_NOT 群体准确率 (< 0.5 = 还在猜)
 *
 * Each row colours itself green / amber / rose based on whether it's likely
 * to be the bottleneck.
 */
function SimHealthBanner({
  gain,
  tSim,
  accAnd,
  accNot,
  bothPass,
}: {
  gain: number;
  tSim: number;
  accAnd: number;
  accNot: number;
  bothPass: number;
}) {
  const gainBad = gain < 1.5;
  const gainOk = gain >= 2.0;
  const youngSim = tSim < 60;
  const matureSim = tSim >= 600;
  const accAvg = 0.5 * (accAnd + accNot);
  const learning = accAvg >= 0.6;
  const struggling = accAvg < 0.4;

  return (
    <div className="rounded border border-amber-800/30 bg-slate-950/40 px-3 py-2 mb-3 grid grid-cols-1 md:grid-cols-3 gap-3 text-[11px]">
      <HealthRow
        label="输出增益 g"
        value={gain.toFixed(2)}
        status={gainOk ? "ok" : gainBad ? "bad" : "warn"}
        hint={
          gainBad
            ? "g<1.5：LIF 输出层很可能永远不点火 → f_out=0。建议到「观测」页右上把 g 调到 2~3。"
            : gainOk
              ? "g≥2，输出层有足够电流穿透 V_th=1.0 阈值。"
              : "g 适中。如果常看到 f_out=0，可以再抬一点。"
        }
      />
      <HealthRow
        label="演化时长 t_sim"
        value={`${tSim.toFixed(0)}s`}
        status={matureSim ? "ok" : youngSim ? "bad" : "warn"}
        hint={
          youngSim
            ? "t_sim<60s：fitness 多半还没 defined（要求 AND/NOT 两种 mode 都见过）；best 会退化为「未演化的随机权重」，输出几乎随机。"
            : matureSim
              ? "已演化≥10min，fitness 基本都 defined。"
              : "中段演化中，部分 agent 已 defined。"
        }
      />
      <HealthRow
        label="群体准确率 (AND+NOT)/2"
        value={`${(accAvg * 100).toFixed(0)}%`}
        status={learning ? "ok" : struggling ? "bad" : "warn"}
        hint={
          struggling
            ? `acc_AND=${(accAnd * 100).toFixed(0)}% / acc_NOT=${(accNot * 100).toFixed(0)}%。<40% 接近随机猜。`
            : learning
              ? `acc_AND=${(accAnd * 100).toFixed(0)}% / acc_NOT=${(accNot * 100).toFixed(0)}%；精英率 ${(bothPass * 100).toFixed(1)}% 同时通过两种。`
              : `acc_AND=${(accAnd * 100).toFixed(0)}% / acc_NOT=${(accNot * 100).toFixed(0)}%。在学习中。`
        }
      />
    </div>
  );
}

function HealthRow({
  label,
  value,
  status,
  hint,
}: {
  label: string;
  value: string;
  status: "ok" | "warn" | "bad";
  hint: string;
}) {
  const dotColor = {
    ok: "bg-emerald-400",
    warn: "bg-amber-400",
    bad: "bg-rose-400",
  }[status];
  const valueColor = {
    ok: "text-emerald-300",
    warn: "text-amber-300",
    bad: "text-rose-300",
  }[status];
  return (
    <div className="flex items-start gap-2">
      <span className={clsx("inline-block w-2 h-2 rounded-full mt-1.5 shrink-0", dotColor)} />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className="text-slate-400">{label}</span>
          <span className={clsx("font-mono font-semibold", valueColor)}>{value}</span>
        </div>
        <div className="text-[10px] text-slate-500 leading-snug mt-0.5">{hint}</div>
      </div>
    </div>
  );
}
