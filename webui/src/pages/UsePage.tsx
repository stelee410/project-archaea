import { useState } from "react";
import clsx from "clsx";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import { useStore } from "../store";
import { usePersistentState } from "../hooks/usePersistentState";
import { LiveTrackCard } from "../components/LiveTrackCard";
import type { ColonyMeta } from "../colonies/registry";
import type { InferenceResponse, SweepResponse } from "../types";

interface InteractionRow {
  ts: number;
  f_in_hz: number;
  f_out_hz: number;
  target: string;
  slots: number[];
  label?: "correct" | "wrong";
  delta?: number;
  killed?: number;
}

interface UseFormState {
  fIn: number;
  target: "best" | "ensemble" | "random" | "swarm";
  topK: number;
  durationMs: number;
  warmupMs: number;
  swarmRadius: number;
  creditCorrect: number;
  creditWrong: number;
}

const USE_FORM_DEFAULTS: UseFormState = {
  fIn: 50,
  target: "best",
  topK: 5,
  durationMs: 500,
  warmupMs: 100,
  swarmRadius: 1,
  creditCorrect: 5,
  creditWrong: 5,
};

type SweepMode = "rampUp" | "rampDown" | "wave" | "manual";

interface SweepFormState {
  mode: SweepMode;
  fInMin: number;
  fInMax: number;
  nPoints: number;
  repeats: number;
  cycles: number;
  manual: number[];
  calibrate: boolean;
}

const SWEEP_FORM_DEFAULTS: SweepFormState = {
  mode: "rampUp",
  fInMin: 0,
  fInMax: 200,
  nPoints: 15,
  repeats: 1,
  cycles: 2,
  manual: [],
  calibrate: false,
};

function buildSequence(form: SweepFormState): number[] {
  const { mode, fInMin: lo, fInMax: hi, nPoints: n, cycles, manual } = form;
  const N = Math.max(2, Math.min(64, n));
  const a = Math.max(0, Math.min(lo, hi));
  const b = Math.max(0, Math.max(lo, hi));
  if (mode === "manual") return manual.map((v) => Math.max(0, v));
  if (mode === "rampUp")
    return Array.from({ length: N }, (_, i) => a + (b - a) * (i / (N - 1)));
  if (mode === "rampDown")
    return Array.from({ length: N }, (_, i) => b - (b - a) * (i / (N - 1)));
  // wave: sine, mid=(a+b)/2, amp=(b-a)/2
  const mid = (a + b) / 2;
  const amp = (b - a) / 2;
  const c = Math.max(0.25, Math.min(8, cycles));
  return Array.from({ length: N }, (_, i) => {
    const t = i / (N - 1);
    return mid + amp * Math.sin(2 * Math.PI * c * t);
  });
}

interface UsePageProps {
  colony: ColonyMeta;
}

export function UsePage({ colony }: UsePageProps) {
  const status = useStore((s) => s.status);
  const ColonyUseExtras = colony.UseExtras;
  const demoteLegacy = !!colony.demoteLegacyUseInput;
  const hideLegacy = !!colony.hideLegacyUseTools;

  const [form, setForm] = usePersistentState<UseFormState>(
    "use-form",
    USE_FORM_DEFAULTS,
  );
  const { fIn, target, topK, durationMs, warmupMs, swarmRadius, creditCorrect, creditWrong } =
    form;
  const setField = <K extends keyof UseFormState>(k: K, v: UseFormState[K]) =>
    setForm((s) => ({ ...s, [k]: v }));

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latest, setLatest] = useState<InferenceResponse | null>(null);
  const [log, setLog] = useState<InteractionRow[]>([]);

  const [sweepForm, setSweepForm] = usePersistentState<SweepFormState>(
    "sweep-form",
    SWEEP_FORM_DEFAULTS,
  );
  const setSweepField = <K extends keyof SweepFormState>(k: K, v: SweepFormState[K]) =>
    setSweepForm((s) => ({ ...s, [k]: v }));
  const [sweepBusy, setSweepBusy] = useState(false);
  const [sweepError, setSweepError] = useState<string | null>(null);
  const [sweep, setSweep] = useState<SweepResponse | null>(null);

  async function runSweep() {
    const seq = buildSequence(sweepForm);
    if (seq.length < 1) {
      setSweepError("没有规划任何输入点。请切换模式或在画板上点几个点。");
      return;
    }
    setSweepBusy(true);
    setSweepError(null);
    try {
      const r = await api.sweep({
        f_in_min: sweepForm.fInMin,
        f_in_max: sweepForm.fInMax,
        n_points: seq.length,
        target,
        top_k: topK,
        duration_ms: durationMs,
        warmup_ms: warmupMs,
        swarm_radius: swarmRadius,
        repeats: sweepForm.repeats,
        f_in_seq: seq,
        calibrate: sweepForm.calibrate,
      });
      setSweep(r);
    } catch (e) {
      setSweepError(String(e));
    } finally {
      setSweepBusy(false);
    }
  }

  async function query() {
    setBusy(true);
    setError(null);
    try {
      const r = await api.inference({
        f_in_hz: fIn,
        target,
        top_k: topK,
        duration_ms: durationMs,
        warmup_ms: warmupMs,
        swarm_radius: swarmRadius,
      });
      setLatest(r);
      setLog((prev) => [
        {
          ts: Date.now(),
          f_in_hz: r.f_in_hz,
          f_out_hz: r.f_out_hz,
          target: r.target,
          slots: r.agents.map((a) => a.slot),
        },
        ...prev,
      ].slice(0, 80));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function judge(label: "correct" | "wrong") {
    if (!latest) return;
    const slots = latest.agents.map((a) => a.slot);
    const delta = label === "correct" ? Math.abs(creditCorrect) : -Math.abs(creditWrong);
    setBusy(true);
    setError(null);
    try {
      const r = await api.feedback({
        slots,
        delta_per_slot: delta,
        label,
        f_in_hz: latest.f_in_hz,
        f_out_hz: latest.f_out_hz,
      });
      const killed = r.results.filter((x) => x.killed).length;
      setLog((prev) => {
        const head = prev[0];
        if (head && head.label == null) {
          return [
            { ...head, label, delta, killed },
            ...prev.slice(1),
          ];
        }
        return prev;
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!status?.running) {
    return (
      <div className="max-w-[900px] mx-auto p-8 text-slate-400 text-sm">
        仿真当前未运行——「使用」需要一个活着的种群。先去「设置 / 启动」页启动，
        然后回来给它输入并打分。
      </div>
    );
  }

  // L2v2（demoteLegacy）下，下面 4 块 L1 频率工具与本任务都无关：
  //   - 底层接口（直接喂 Hz）  : 上方 LogicTester 已能按 SPEC 翻译成多通道电平
  //   - 打分（手动 ±Credit）   : Oracle 真值表已自动评分，手动会扰乱演化方向
  //   - 实时跟随（鼠标驱动）   : 频率跟随的可视化，逻辑题没有"跟随"概念
  //   - Sweep 测试（扫频图）   : 逻辑题输出只关心 bit (>50Hz)，扫频曲线无意义
  // 但保留它们便于偶尔做底层探针 → 整体打包成默认折叠的 <details>。
  const legacyTools = (
    <>
      <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="text-base font-semibold mb-1">
            {demoteLegacy ? "🔬 底层接口：直接喂 Hz (高级 / 调试用)" : "查询种群"}
          </h2>
          {demoteLegacy ? (
            <p className="text-[11px] text-slate-400 mb-3 leading-relaxed">
              这是 L1 时代的「单通道频率输入」接口。当前群落（{colony.emoji} {colony.name}）
              推荐用上方的任务专属面板，它会按 SPEC 自动把语义化输入翻译成多通道电平。
              这里仅用于手动探针：改 f_in_hz 时其他通道走默认值。
            </p>
          ) : (
            <div className="mb-2" />
          )}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Field label="输入发放率 f_in (Hz)" hint="10 个独立 Poisson 通道，同一速率">
              <input
                type="number"
                min={0}
                max={500}
                value={fIn}
                onChange={(e) => setField("fIn", Number(e.target.value))}
                className="num-input"
              />
            </Field>
            <Field label="查询目标 target" hint="best=fitness 最高；ensemble=top-K 平均；swarm=黏菌 hotspot 集体投票">
              <select
                value={target}
                onChange={(e) => setField("target", e.target.value as typeof target)}
                className="num-input"
              >
                <option value="best">best (默认)</option>
                <option value="ensemble">ensemble (top-K)</option>
                <option value="random">random</option>
                <option value="swarm">🍄 swarm (黏菌 hotspot)</option>
              </select>
            </Field>
            <Field label="Top-K (仅 ensemble)" hint="ensemble 模式下取多少个 agent 平均">
              <input
                type="number"
                min={1}
                max={50}
                disabled={target !== "ensemble"}
                value={topK}
                onChange={(e) => setField("topK", Number(e.target.value))}
                className="num-input"
              />
            </Field>
            <Field
              label="Swarm 半径 (仅 swarm)"
              hint="信息素峰值格周围 ±R 格 (Chebyshev) 内的活体一起投票；R=1 → 3×3 邻域"
            >
              <input
                type="number"
                min={1}
                max={8}
                disabled={target !== "swarm"}
                value={swarmRadius}
                onChange={(e) => setField("swarmRadius", Number(e.target.value))}
                className="num-input"
              />
            </Field>
            <Field label="duration_ms" hint="单次推理的脉冲时长，500 ms 与训练窗一致">
              <input
                type="number"
                min={50}
                max={5000}
                step={50}
                value={durationMs}
                onChange={(e) => setField("durationMs", Number(e.target.value))}
                className="num-input"
              />
            </Field>
            <Field label="warmup_ms" hint="先跑一段让 LIF 膜电位脱离冷启动瞬态">
              <input
                type="number"
                min={0}
                max={2000}
                step={50}
                value={warmupMs}
                onChange={(e) => setField("warmupMs", Number(e.target.value))}
                className="num-input"
              />
            </Field>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
            <button
              onClick={query}
              disabled={busy}
              className="shrink-0 px-4 py-2 rounded-md bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-medium disabled:opacity-40"
            >
              发送输入 → 取输出
            </button>
            {latest && (
              <div className="text-sm text-slate-300 numeric min-w-0">
                f_in <b className="text-slate-100">{latest.f_in_hz.toFixed(1)}</b> Hz →
                f_out <b className="text-emerald-300">{latest.f_out_hz.toFixed(1)}</b> Hz
                <span className="text-slate-500 ml-2">· {latest.agents.length} agent</span>
              </div>
            )}
          </div>
          {latest && <SlotsLine slots={latest.agents.map((a) => a.slot)} />}
          {latest?.target === "swarm" && (
            <div className="mt-3 px-3 py-2 rounded bg-fuchsia-950/30 border border-fuchsia-700/30 text-xs text-fuchsia-200">
              {latest.swarm_degraded ? (
                <>
                  ⚠️ swarm 模式已退化为 top-K（原因：
                  <code className="px-1 mx-0.5 bg-slate-900 rounded">
                    {latest.swarm_degraded}
                  </code>
                  ），下面这个结果其实来自 fitness 选择器。
                  {latest.swarm_degraded === "slime_disabled" &&
                    " 想真正用 swarm，请到设置页勾上 🍄 启用赛博黏菌模式后重启。"}
                  {latest.swarm_degraded === "pheromone_empty" &&
                    " 信息素场还没建立 — 让仿真多跑几窗，让强者留下轨迹。"}
                </>
              ) : (
                <>
                  🍄 黏菌触手位于格 (
                  <b>{latest.swarm_hotspot?.[0]}</b>, <b>{latest.swarm_hotspot?.[1]}</b>
                  )，半径 ±{latest.swarm_radius_used} 内
                  <b className="text-fuchsia-100"> {latest.swarm_size} </b>
                  个活体共同回答了你 — 这是当前共识最强的子群。
                </>
              )}
            </div>
          )}
          {error && (
            <div className="mt-3 px-3 py-1.5 rounded bg-rose-500/15 text-rose-200 text-xs border border-rose-500/30">
              {error}
            </div>
          )}
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="text-base font-semibold mb-1">打分（喂经济压力）</h2>
          <p className="text-xs text-slate-400 mb-3">
            ✓ 增加被查询 agent 的 Credit（鼓励），✗ 扣 Credit（惩罚）。
            Credit 跌到 0 会立即<span className="text-rose-300">饿死</span>该 agent。
            这就是「外部使用 → 经济反馈 → 演化方向」的闭环。
          </p>
          <div className="grid grid-cols-2 gap-3 mb-4">
            <Field label="✓ 正确奖励 (+Credit)">
              <input
                type="number"
                min={0}
                max={200}
                step={1}
                value={creditCorrect}
                onChange={(e) => setField("creditCorrect", Number(e.target.value))}
                className="num-input"
              />
            </Field>
            <Field label="✗ 错误惩罚 (−Credit)">
              <input
                type="number"
                min={0}
                max={200}
                step={1}
                value={creditWrong}
                onChange={(e) => setField("creditWrong", Number(e.target.value))}
                className="num-input"
              />
            </Field>
          </div>
          {demoteLegacy ? (
            <div className="px-3 py-2 rounded bg-slate-950/60 border border-slate-700/60 text-xs text-slate-400 leading-relaxed">
              ⓘ 当前群落（{colony.emoji} {colony.name}）已有 <b className="text-slate-200">Oracle 真值表自动评分</b>
              （NOT(0)=+25 / AND(1,1)=+15 / silent正确=+0.5~1.0 Credit），手动 ±5 会和 Oracle 争夺信用分配、
              <span className="text-rose-300">扰乱演化方向</span>，所以这里禁用。
              如果只是想看 agent 输出对不对，用上方的「<b>问这一题</b>」面板。
            </div>
          ) : (
            <div className="flex gap-3">
              <button
                disabled={!latest || busy}
                onClick={() => judge("correct")}
                className={clsx(
                  "flex-1 px-4 py-2.5 rounded-md font-medium",
                  "bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-200 ring-1 ring-emerald-500/40",
                  "disabled:opacity-40"
                )}
              >
                ✓ 正确（+{Math.abs(creditCorrect)} Credit）
              </button>
              <button
                disabled={!latest || busy}
                onClick={() => judge("wrong")}
                className={clsx(
                  "flex-1 px-4 py-2.5 rounded-md font-medium",
                  "bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 ring-1 ring-rose-500/40",
                  "disabled:opacity-40"
                )}
              >
                ✗ 错误（−{Math.abs(creditWrong)} Credit）
              </button>
            </div>
          )}
        </div>

        <LiveTrackCard
          target={target}
          topK={topK}
          swarmRadius={swarmRadius}
        />

        <SweepCard
          form={sweepForm}
          setField={setSweepField}
          target={target}
          busy={sweepBusy}
          error={sweepError}
          result={sweep}
          onRun={runSweep}
        />

        <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4 text-xs text-slate-300 leading-relaxed">
          <h3 className="font-semibold text-slate-100 text-sm mb-1">这页背后做了什么</h3>
          <ol className="list-decimal list-inside space-y-1">
            <li>「发送输入」会从仿真里**复制**所选 agent 的权重，离线跑 warmup + duration ms 的 Poisson 脉冲，统计 f_out。</li>
            <li>不会暂停或干扰主仿真线程；同样的输入再问一次仍是 Poisson 抽样，结果会有微小波动，这是正常的。</li>
            <li>「打分」直接修改主仿真里那 agent 的 Credit 字段，下一窗的经济结算立刻可见，可能触发饿死或加速繁殖。</li>
            <li>「Sweep 测试」会在 [f_in_min, f_in_max] 之间均匀采样若干点，逐点查询并画出 f_in→f_out 关系曲线，用红色 y=x 直线作为「完美匹配」参照。</li>
          </ol>
        </div>
      </>
  );

  return (
    <div className="max-w-[1300px] mx-auto p-6 grid gap-6 grid-cols-12">
      <section className={clsx("space-y-5", hideLegacy ? "col-span-12" : "col-span-12 lg:col-span-7")}>
        {ColonyUseExtras && <ColonyUseExtras />}
        {hideLegacy ? null : demoteLegacy ? (
          <details className="rounded-lg border border-slate-800 bg-slate-900/30 group">
            <summary className="cursor-pointer px-5 py-3 text-sm text-slate-400 hover:text-slate-200 select-none flex items-center gap-2">
              <span className="text-slate-500 group-open:rotate-90 inline-block transition-transform">▶</span>
              🔧 L1 频率工具（与本任务无关 · 高级 / 调试用）
              <span className="ml-auto text-[11px] text-slate-600">点击展开</span>
            </summary>
            <div className="px-5 pb-5 space-y-5 border-t border-slate-800">
              <div className="pt-5 px-3 py-2 rounded bg-amber-950/20 border border-amber-700/30 text-xs text-amber-200/80 leading-relaxed">
                ⚠️ 下面这些是 L1「频率跟随」时代的工具：单通道 Hz 输入、扫频画 f_in/f_out 曲线、
                鼠标拖拽实时跟随。逻辑门控任务的输出只关心 bit（{">"}/{"<"} 50Hz），这些可视化对它没有意义；
                打分按钮已禁用避免与 Oracle 自动评分冲突。仅在你想做底层探针（验证 g 是否够、看 swarm 是否在工作）时展开。
              </div>
              {legacyTools}
            </div>
          </details>
        ) : (
          legacyTools
        )}
      </section>

      {/* 历史（仅 L1 工具产生条目；L2v2 等隐藏 L1 工具的群落整段不渲染） */}
      {!hideLegacy && (
      <aside className="col-span-12 lg:col-span-5">
        <div className="rounded-lg border border-slate-800 bg-slate-900/50 overflow-hidden">
          <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-800">
            交互历史（最新在上，最多 80 条）
          </div>
          <div className="max-h-[700px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="text-slate-500 sticky top-0 bg-slate-900">
                <tr>
                  <th className="text-left px-2 py-1.5">时刻</th>
                  <th className="text-right px-2 py-1.5">f_in</th>
                  <th className="text-right px-2 py-1.5">f_out</th>
                  <th className="text-left px-2 py-1.5">target</th>
                  <th className="text-left px-2 py-1.5">slots</th>
                  <th className="text-left px-2 py-1.5">label</th>
                  <th className="text-right px-2 py-1.5">Δcredit</th>
                  <th className="text-right px-2 py-1.5">饿死</th>
                </tr>
              </thead>
              <tbody>
                {log.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="text-center text-slate-500 py-6">
                      还没有交互。给它一个 f_in 试试。
                    </td>
                  </tr>
                ) : (
                  log.map((row, i) => (
                    <tr key={i} className="border-t border-slate-800/50">
                      <td className="px-2 py-1 text-slate-400 whitespace-nowrap">
                        {new Date(row.ts).toLocaleTimeString()}
                      </td>
                      <td className="px-2 py-1 text-right numeric">{row.f_in_hz.toFixed(1)}</td>
                      <td className="px-2 py-1 text-right numeric text-emerald-300">
                        {row.f_out_hz.toFixed(1)}
                      </td>
                      <td className="px-2 py-1">{row.target}</td>
                      <td className="px-2 py-1 font-mono text-[11px] text-slate-400 truncate max-w-[120px]">
                        [{row.slots.join(",")}]
                      </td>
                      <td className="px-2 py-1">
                        {row.label === "correct" && <span className="text-emerald-300">✓</span>}
                        {row.label === "wrong" && <span className="text-rose-300">✗</span>}
                      </td>
                      <td
                        className={clsx(
                          "px-2 py-1 text-right numeric",
                          row.delta && row.delta > 0 && "text-emerald-300",
                          row.delta && row.delta < 0 && "text-rose-300"
                        )}
                      >
                        {row.delta == null ? "—" : (row.delta > 0 ? "+" : "") + row.delta}
                      </td>
                      <td className="px-2 py-1 text-right numeric text-rose-300">
                        {row.killed != null && row.killed > 0 ? row.killed : ""}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </aside>
      )}
      <style>{`.num-input{display:block;width:100%;background:#020617;border:1px solid #334155;border-radius:6px;padding:6px 8px;font-family:ui-monospace,monospace;font-size:13px;color:#e2e8f0;}`}</style>
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="block text-xs text-slate-300 mb-1">{label}</label>
      {children}
      {hint && <div className="text-[10px] text-slate-500 mt-1 leading-snug">{hint}</div>}
    </div>
  );
}

const SWEEP_MODES: { id: SweepMode; label: string; desc: string }[] = [
  { id: "rampUp", label: "↗ 上升", desc: "min → max 线性递增" },
  { id: "rampDown", label: "↘ 下降", desc: "max → min 线性递减" },
  { id: "wave", label: "〰 起伏", desc: "正弦波，cycles 个周期" },
  { id: "manual", label: "✏ 手动", desc: "在画板上点选规划" },
];

function SweepCard({
  form,
  setField,
  target,
  busy,
  error,
  result,
  onRun,
}: {
  form: SweepFormState;
  setField: <K extends keyof SweepFormState>(k: K, v: SweepFormState[K]) => void;
  target: string;
  busy: boolean;
  error: string | null;
  result: SweepResponse | null;
  onRun: () => void;
}) {
  const planned = buildSequence(form);
  const totalCalls = planned.length * form.repeats;
  const estSec = ((100 + 500) * totalCalls) / 1000;

  const chartData = result?.points.map((p, i) => ({
    step: i + 1,
    input: p.f_in_hz,
    output: p.f_out_hz_mean,
    calibrated: p.f_out_hz_calibrated ?? null,
  })) ?? [];

  const calApplied = result?.calibration?.applied ?? false;
  const calA = result?.calibration?.a ?? null;
  const calB = result?.calibration?.b ?? null;
  const calSkip = result?.calibration?.skipped_reason ?? null;

  // simple time-domain fidelity: corr(input, output) and per-step lag-0 RMSE-like error
  const metrics = (() => {
    if (!result || result.points.length < 2) return null;
    const xs = result.points.map((p) => p.f_in_hz);
    const ys = result.points.map((p) => p.f_out_hz_mean);
    const n = xs.length;
    const mx = xs.reduce((a, b) => a + b, 0) / n;
    const my = ys.reduce((a, b) => a + b, 0) / n;
    let num = 0, dxx = 0, dyy = 0, sse = 0;
    for (let i = 0; i < n; i++) {
      const dx = xs[i] - mx, dy = ys[i] - my;
      num += dx * dy;
      dxx += dx * dx;
      dyy += dy * dy;
      const e = ys[i] - xs[i];
      sse += e * e;
    }
    const slope = dxx > 0 ? num / dxx : 0;
    const r = dxx > 0 && dyy > 0 ? num / Math.sqrt(dxx * dyy) : 0;
    const rmse = Math.sqrt(sse / n);
    return { slope, r, rmse };
  })();

  const planMaxY = Math.max(form.fInMax, ...planned, ...(form.manual ?? []), 10);

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-5">
      <h2 className="text-base font-semibold mb-1">📈 Sweep 测试（输入信号规划）</h2>
      <p className="text-xs text-slate-400 mb-3">
        规划一段 f_in 时间序列，逐步输入种群，把 <span className="text-sky-300">输入</span> 和{" "}
        <span className="text-emerald-300">输出</span> 画在同一张时序图上对比。
        每步独立查询（warmup + duration ms），不同步之间种群继续在后台演化。复用上方
        <code className="mx-1 px-1 bg-slate-800 rounded">target = {target}</code>。
      </p>

      {/* mode tabs */}
      <div className="inline-flex rounded-md bg-slate-950/60 ring-1 ring-slate-800 p-0.5 mb-3">
        {SWEEP_MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setField("mode", m.id)}
            title={m.desc}
            className={clsx(
              "px-3 py-1.5 text-xs rounded transition-colors",
              form.mode === m.id
                ? "bg-sky-500 text-slate-950 font-semibold"
                : "text-slate-300 hover:bg-slate-800"
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* mode-specific params */}
      {form.mode !== "manual" ? (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <Field label="f_in 起点 (Hz)">
            <input
              type="number" min={0} max={500}
              value={form.fInMin}
              onChange={(e) => setField("fInMin", Number(e.target.value))}
              className="num-input"
            />
          </Field>
          <Field label="f_in 终点 (Hz)">
            <input
              type="number" min={0} max={500}
              value={form.fInMax}
              onChange={(e) => setField("fInMax", Number(e.target.value))}
              className="num-input"
            />
          </Field>
          <Field label="采样点数 n_points" hint="2–64">
            <input
              type="number" min={2} max={64}
              value={form.nPoints}
              onChange={(e) => setField("nPoints", Number(e.target.value))}
              className="num-input"
            />
          </Field>
          {form.mode === "wave" && (
            <Field label="周期数 cycles" hint="正弦波在窗口内的完整周期数 (0.25–8)">
              <input
                type="number" min={0.25} max={8} step={0.25}
                value={form.cycles}
                onChange={(e) => setField("cycles", Number(e.target.value))}
                className="num-input"
              />
            </Field>
          )}
          <Field label="每点重复 repeats" hint=">1 用来压 Poisson 噪声">
            <input
              type="number" min={1} max={10}
              value={form.repeats}
              onChange={(e) => setField("repeats", Number(e.target.value))}
              className="num-input"
            />
          </Field>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <Field label="Y 轴最大值 (Hz)" hint="画板纵向范围 0..max">
            <input
              type="number" min={10} max={500}
              value={form.fInMax}
              onChange={(e) => setField("fInMax", Number(e.target.value))}
              className="num-input"
            />
          </Field>
          <Field label="每点重复 repeats" hint=">1 用来压 Poisson 噪声">
            <input
              type="number" min={1} max={10}
              value={form.repeats}
              onChange={(e) => setField("repeats", Number(e.target.value))}
              className="num-input"
            />
          </Field>
          <div className="self-end text-[11px] text-slate-400 leading-snug">
            已规划 <b className="text-slate-200">{form.manual.length}</b> 个点
            （上限 256）。点画板添加，点已有点删除。
          </div>
        </div>
      )}

      {/* preview / manual editor */}
      <div className="mt-4">
        {form.mode === "manual" ? (
          <ManualPlanner
            points={form.manual}
            yMax={form.fInMax}
            onChange={(arr) => setField("manual", arr)}
          />
        ) : (
          <PlanPreview seq={planned} yMax={planMaxY} />
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
        <button
          onClick={onRun}
          disabled={busy || planned.length < 1}
          className="shrink-0 px-4 py-2 rounded-md bg-sky-500 hover:bg-sky-400 text-slate-950 font-medium disabled:opacity-40"
        >
          {busy
            ? `运行中…（约 ${estSec.toFixed(0)} s）`
            : `▶ 运行 (${planned.length} 步 × ${form.repeats} ≈ ${estSec.toFixed(0)} s)`}
        </button>
        <label
          className="inline-flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer select-none"
          title="方案 A：不动演化，事后用最小二乘拟合 y=ax+b 反校准 f_out。立刻看到斜率≈1 的曲线。"
        >
          <input
            type="checkbox"
            checked={form.calibrate}
            onChange={(e) => setField("calibrate", e.target.checked)}
            className="accent-amber-400"
          />
          推理层校准 (calibrate)
        </label>
        {metrics && (
          <span className="text-sm text-slate-300 numeric">
            corr <b className={metrics.r >= 0.9 ? "text-emerald-300" : "text-amber-300"}>
              {metrics.r.toFixed(3)}
            </b>
            <span className="mx-2 text-slate-600">·</span>
            斜率 <b className={metrics.slope >= 0.5 ? "text-emerald-300" : "text-amber-300"}>
              {metrics.slope.toFixed(3)}
            </b>
            <span className="mx-2 text-slate-600">·</span>
            RMSE <b className="text-slate-200">{metrics.rmse.toFixed(1)} Hz</b>
          </span>
        )}
      </div>
      {error && (
        <div className="mt-3 px-3 py-1.5 rounded bg-rose-500/15 text-rose-200 text-xs border border-rose-500/30">
          {error}
        </div>
      )}
      {result?.swarm_degraded && (
        <div className="mt-3 px-3 py-1.5 rounded bg-amber-500/15 text-amber-200 text-xs border border-amber-500/30">
          ⚠️ swarm 退化：<code className="px-1 bg-slate-900 rounded">{result.swarm_degraded}</code>
          ；下面的曲线实际由 fitness 选择器产出。
        </div>
      )}
      {result && (
        <div className="mt-4 rounded border border-slate-800 bg-slate-950/60 p-2">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 18 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="2 4" />
              <XAxis
                dataKey="step"
                type="number"
                domain={[1, chartData.length || 1]}
                tick={{ fontSize: 11, fill: "#94a3b8" }}
                label={{ value: "step", position: "insideBottom", offset: -8, fill: "#64748b", fontSize: 11 }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: "#94a3b8" }}
                label={{ value: "Hz", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
                labelFormatter={(v) => `step ${v}`}
                formatter={(v, name) => {
                  const label =
                    name === "input"
                      ? "输入 f_in"
                      : name === "output"
                      ? "输出 f_out (raw)"
                      : "输出 f_out (calibrated)";
                  if (v == null || typeof v !== "number") return ["—", label];
                  return [v.toFixed(2) + " Hz", label];
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: 11, paddingTop: 4 }}
                formatter={(v) =>
                  v === "input"
                    ? "输入 f_in (规划)"
                    : v === "output"
                    ? "输出 f_out (raw)"
                    : "输出 f_out (校准后)"
                }
              />
              <ReferenceLine y={0} stroke="#475569" />
              <Line
                type="monotone"
                dataKey="input"
                stroke="#38bdf8"
                strokeDasharray="4 4"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="output"
                stroke="#34d399"
                strokeWidth={2}
                dot={{ r: 3, fill: "#34d399" }}
                isAnimationActive={false}
              />
              {calApplied && (
                <Line
                  type="monotone"
                  dataKey="calibrated"
                  stroke="#fbbf24"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "#fbbf24" }}
                  isAnimationActive={false}
                  connectNulls
                />
              )}
            </LineChart>
          </ResponsiveContainer>
          <div className="text-[10px] text-slate-500 mt-1 leading-snug px-2">
            {result.n_points} 步 × {result.repeats} 次 / 步 ·
            target=<b className="text-slate-300">{result.target}</b>
            {result.synapse_gain != null && (
              <> · g=<b className="text-fuchsia-300">×{result.synapse_gain.toFixed(2)}</b></>
            )}
            {result.target === "swarm" && result.swarm_hotspot && !result.swarm_degraded && (
              <> · hotspot=({result.swarm_hotspot[0]}, {result.swarm_hotspot[1]})
                · 首步 swarm_size={result.swarm_size_first}</>
            )}
            {calApplied && calA != null && calB != null && (
              <>
                {" · "}
                <span className="text-amber-300">
                  校准 y={calA.toFixed(3)}·x{calB >= 0 ? "+" : ""}{calB.toFixed(2)}
                </span>
                ；橙线 = (raw − {calB.toFixed(2)}) / {calA.toFixed(3)}
              </>
            )}
            {form.calibrate && !calApplied && calSkip && (
              <> · <span className="text-amber-300">校准跳过：{calSkip}</span></>
            )}
            ：蓝虚线 = 规划输入；绿实线 = 种群原始响应；橙实线 = 推理层校准后。
            <br />
            <b>校准</b>是「事后放大器」（方案 A，不动演化）；
            <b>幅度校准惩罚 λ</b>（设置页 / 观测页右上角）才是让种群「学会幅度」的演化压力（方案 C）。
          </div>
        </div>
      )}
    </div>
  );
}

/** Tiny SVG preview of the planned input sequence (read-only). */
function PlanPreview({ seq, yMax }: { seq: number[]; yMax: number }) {
  const W = 720, H = 90, ML = 36, MR = 8, MT = 6, MB = 18;
  const innerW = W - ML - MR, innerH = H - MT - MB;
  const yScale = (v: number) => MT + innerH * (1 - Math.max(0, Math.min(1, v / Math.max(1, yMax))));
  const xScale = (i: number) =>
    ML + (seq.length <= 1 ? innerW / 2 : (innerW * i) / (seq.length - 1));
  const path = seq.map((v, i) => `${i === 0 ? "M" : "L"} ${xScale(i).toFixed(1)} ${yScale(v).toFixed(1)}`).join(" ");
  return (
    <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
      <div className="text-[10px] text-slate-500 mb-1">规划预览（输入 f_in 随 step 变化）</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[90px]" preserveAspectRatio="none">
        <rect x={ML} y={MT} width={innerW} height={innerH} fill="#020617" stroke="#1e293b" />
        {/* y ticks */}
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={ML} x2={ML + innerW} y1={MT + innerH * (1 - t)} y2={MT + innerH * (1 - t)} stroke="#1e293b" strokeDasharray="2 4" />
            <text x={ML - 4} y={MT + innerH * (1 - t) + 3} fontSize="9" fill="#64748b" textAnchor="end">
              {Math.round(yMax * t)}
            </text>
          </g>
        ))}
        <text x={ML + innerW / 2} y={H - 4} fontSize="9" fill="#64748b" textAnchor="middle">
          step (1 → {seq.length})
        </text>
        <path d={path} fill="none" stroke="#38bdf8" strokeWidth="1.5" />
        {seq.length <= 32 && seq.map((v, i) => (
          <circle key={i} cx={xScale(i)} cy={yScale(v)} r={2} fill="#38bdf8" />
        ))}
      </svg>
    </div>
  );
}

/** Click-to-add / click-to-delete SVG planner for the manual mode. */
function ManualPlanner({
  points,
  yMax,
  onChange,
}: {
  points: number[];
  yMax: number;
  onChange: (next: number[]) => void;
}) {
  const W = 720, H = 220, ML = 40, MR = 12, MT = 10, MB = 26;
  const innerW = W - ML - MR, innerH = H - MT - MB;
  const yScale = (v: number) => MT + innerH * (1 - Math.max(0, Math.min(1, v / Math.max(1, yMax))));
  const stepCount = Math.max(points.length, 1);
  // x positions: each existing point sits at index i/(N-1). New click appends at the end.
  const xScale = (i: number, n: number = stepCount) =>
    ML + (n <= 1 ? innerW / 2 : (innerW * i) / (n - 1));

  function handleClick(e: React.MouseEvent<SVGSVGElement>) {
    if (points.length >= 256) return;
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const py = ((e.clientY - rect.top) / rect.height) * H;
    if (px < ML || px > W - MR || py < MT || py > H - MB) return;
    const ratio = 1 - (py - MT) / innerH;
    const hz = Math.max(0, Math.round(ratio * yMax * 10) / 10);
    onChange([...points, hz]);
  }

  function removeAt(i: number, e: React.MouseEvent) {
    e.stopPropagation();
    onChange(points.filter((_, k) => k !== i));
  }

  const path =
    points.length >= 2
      ? points.map((v, i) => `${i === 0 ? "M" : "L"} ${xScale(i, points.length).toFixed(1)} ${yScale(v).toFixed(1)}`).join(" ")
      : null;

  return (
    <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
      <div className="flex items-center justify-between mb-1">
        <div className="text-[11px] text-slate-400">
          ✏ 在画板上<b className="text-sky-300">点击添加</b>步骤；点已有的<b className="text-rose-300">圆点删除</b>。
          顺序就是 step 1, 2, 3, ...
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => onChange([])}
            className="px-2 py-0.5 text-[11px] rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
          >
            清空
          </button>
        </div>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-[220px] cursor-crosshair touch-none select-none"
        onClick={handleClick}
        preserveAspectRatio="none"
      >
        <rect x={ML} y={MT} width={innerW} height={innerH} fill="#020617" stroke="#1e293b" />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1={ML} x2={ML + innerW}
              y1={MT + innerH * (1 - t)} y2={MT + innerH * (1 - t)}
              stroke="#1e293b" strokeDasharray="2 4"
            />
            <text x={ML - 4} y={MT + innerH * (1 - t) + 3} fontSize="9" fill="#64748b" textAnchor="end">
              {Math.round(yMax * t)}
            </text>
          </g>
        ))}
        <text x={ML + innerW / 2} y={H - 6} fontSize="10" fill="#64748b" textAnchor="middle">
          step ({points.length} 个) — Y: f_in (Hz)
        </text>
        {path && <path d={path} fill="none" stroke="#38bdf8" strokeWidth="1.5" pointerEvents="none" />}
        {points.map((v, i) => (
          <g key={i} onClick={(e) => removeAt(i, e)} className="cursor-pointer">
            <circle
              cx={xScale(i, points.length)}
              cy={yScale(v)}
              r={5}
              fill="#38bdf8"
              stroke="#0c4a6e"
              strokeWidth="1"
            />
            {points.length <= 24 && (
              <text
                x={xScale(i, points.length)}
                y={yScale(v) - 8}
                fontSize="9"
                fill="#7dd3fc"
                textAnchor="middle"
              >
                {v.toFixed(0)}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}

function SlotsLine({ slots }: { slots: number[] }) {
  const [expanded, setExpanded] = useState(false);
  const PREVIEW = 12;
  const long = slots.length > PREVIEW;
  const shown = expanded || !long ? slots : slots.slice(0, PREVIEW);
  return (
    <div className="mt-2 text-[11px] text-slate-500 font-mono leading-snug break-words">
      <span className="text-slate-600">slot=[</span>
      <span className="text-slate-400">{shown.join(", ")}</span>
      {long && !expanded && (
        <>
          <span className="text-slate-600">, … {slots.length - PREVIEW} more</span>
        </>
      )}
      <span className="text-slate-600">]</span>
      {long && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300"
        >
          {expanded ? "收起" : "展开全部"}
        </button>
      )}
    </div>
  );
}
