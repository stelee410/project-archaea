import { useState } from "react";
import clsx from "clsx";
import { api } from "../api";
import { useStore } from "../store";
import type { InferenceResponse } from "../types";

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

export function UsePage() {
  const status = useStore((s) => s.status);

  const [fIn, setFIn] = useState(50);
  const [target, setTarget] = useState<"best" | "ensemble" | "random">("best");
  const [topK, setTopK] = useState(5);
  const [durationMs, setDurationMs] = useState(500);
  const [warmupMs, setWarmupMs] = useState(100);

  const [creditCorrect, setCreditCorrect] = useState(5);
  const [creditWrong, setCreditWrong] = useState(5);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latest, setLatest] = useState<InferenceResponse | null>(null);
  const [log, setLog] = useState<InteractionRow[]>([]);

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

  return (
    <div className="max-w-[1300px] mx-auto p-6 grid gap-6 grid-cols-12">
      <section className="col-span-12 lg:col-span-7 space-y-5">
        <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="text-base font-semibold mb-3">查询种群</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Field label="输入发放率 f_in (Hz)" hint="10 个独立 Poisson 通道，同一速率">
              <input
                type="number"
                min={0}
                max={500}
                value={fIn}
                onChange={(e) => setFIn(Number(e.target.value))}
                className="num-input"
              />
            </Field>
            <Field label="查询目标 target" hint="best=fitness 最高的活体；ensemble=top-K 取均值">
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value as typeof target)}
                className="num-input"
              >
                <option value="best">best (默认)</option>
                <option value="ensemble">ensemble (top-K)</option>
                <option value="random">random</option>
              </select>
            </Field>
            <Field label="Top-K (仅 ensemble)" hint="ensemble 模式下取多少个 agent 平均">
              <input
                type="number"
                min={1}
                max={50}
                disabled={target !== "ensemble"}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
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
                onChange={(e) => setDurationMs(Number(e.target.value))}
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
                onChange={(e) => setWarmupMs(Number(e.target.value))}
                className="num-input"
              />
            </Field>
          </div>
          <div className="mt-4 flex items-center gap-3">
            <button
              onClick={query}
              disabled={busy}
              className="px-4 py-2 rounded-md bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-medium disabled:opacity-40"
            >
              发送输入 → 取输出
            </button>
            {latest && (
              <span className="text-sm text-slate-300 numeric">
                f_in <b className="text-slate-100">{latest.f_in_hz.toFixed(1)}</b> Hz →
                f_out <b className="text-emerald-300">{latest.f_out_hz.toFixed(1)}</b> Hz
                <span className="text-slate-500 ml-2">
                  · {latest.agents.length} agent · slot=[{latest.agents.map((a) => a.slot).join(",")}]
                </span>
              </span>
            )}
          </div>
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
                onChange={(e) => setCreditCorrect(Number(e.target.value))}
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
                onChange={(e) => setCreditWrong(Number(e.target.value))}
                className="num-input"
              />
            </Field>
          </div>
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
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4 text-xs text-slate-300 leading-relaxed">
          <h3 className="font-semibold text-slate-100 text-sm mb-1">这页背后做了什么</h3>
          <ol className="list-decimal list-inside space-y-1">
            <li>「发送输入」会从仿真里**复制**所选 agent 的权重，离线跑 warmup + duration ms 的 Poisson 脉冲，统计 f_out。</li>
            <li>不会暂停或干扰主仿真线程；同样的输入再问一次仍是 Poisson 抽样，结果会有微小波动，这是正常的。</li>
            <li>「打分」直接修改主仿真里那 agent 的 Credit 字段，下一窗的经济结算立刻可见，可能触发饿死或加速繁殖。</li>
          </ol>
        </div>
      </section>

      {/* 历史 */}
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
