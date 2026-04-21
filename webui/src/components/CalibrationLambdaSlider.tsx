import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { api } from "../api";

/**
 * Live slider for the fitness magnitude calibration penalty λ (SPEC v1.2 off-SPEC).
 *
 * - λ = 0  → SPEC §4.1 pure Pearson r (compressed outputs allowed).
 * - λ > 0  → fitness -= λ · |mean(f_out) - mean(f_in)| / std(f_in), pushing the
 *   swarm toward slope ≈ 1 (output magnitude matches input).
 *
 * Updates are debounced 250 ms so dragging the slider doesn't spam the backend.
 * The change takes effect from the *next* simulation window onward — no restart.
 */
export function CalibrationLambdaSlider({ initial }: { initial: number }) {
  const [value, setValue] = useState<number>(initial);
  const [busy, setBusy] = useState(false);
  const [serverValue, setServerValue] = useState<number>(initial);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<number | null>(null);
  const lastSentRef = useRef<number>(initial);

  // Sync if parent's initial changes (e.g. after restart with new config).
  useEffect(() => {
    setValue(initial);
    setServerValue(initial);
    lastSentRef.current = initial;
  }, [initial]);

  function commit(v: number) {
    if (debounceRef.current != null) {
      window.clearTimeout(debounceRef.current);
    }
    debounceRef.current = window.setTimeout(async () => {
      if (Math.abs(v - lastSentRef.current) < 1e-6) return;
      lastSentRef.current = v;
      setBusy(true);
      setError(null);
      try {
        const r = await api.setCalibrationLambda(v);
        setServerValue(r.calibration_lambda);
      } catch (e) {
        setError(String(e));
      } finally {
        setBusy(false);
      }
    }, 250);
  }

  function onChange(v: number) {
    setValue(v);
    commit(v);
  }

  const hasDrift = Math.abs(value - serverValue) > 1e-6;

  return (
    <div className="rounded-lg border border-amber-700/40 bg-amber-950/20 px-4 py-3 flex flex-wrap items-center gap-x-5 gap-y-2">
      <div className="flex flex-col">
        <span className="text-xs font-semibold text-amber-200">
          幅度校准惩罚 λ
          <span className="ml-1 text-[10px] font-normal text-amber-400/70">
            (SPEC v1.2 off-SPEC · 实时生效)
          </span>
        </span>
        <span className="text-[11px] text-amber-200/70 leading-snug max-w-[460px]">
          fitness = r − λ · |mean(f_out)−mean(f_in)| / std(f_in)。
          λ=0 是原 SPEC 的纯 Pearson r；λ ↑ 会把种群推向「斜率≈1」。
        </span>
      </div>
      <div className="flex-1 min-w-[260px] flex items-center gap-3">
        <input
          type="range"
          min={0}
          max={2}
          step={0.05}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="flex-1 accent-amber-400"
        />
        <input
          type="number"
          min={0}
          max={5}
          step={0.05}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-[70px] bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-amber-200"
        />
      </div>
      <div className="flex items-center gap-3 text-[11px] numeric">
        <span className="text-slate-400">
          server λ = <b className={clsx(hasDrift ? "text-amber-300" : "text-emerald-300")}>
            {serverValue.toFixed(3)}
          </b>
          {busy && <span className="ml-1 text-slate-500">同步中…</span>}
        </span>
        {[0, 0.3, 0.5, 1.0].map((preset) => (
          <button
            key={preset}
            onClick={() => onChange(preset)}
            className={clsx(
              "px-1.5 py-0.5 rounded text-[10px]",
              Math.abs(value - preset) < 1e-6
                ? "bg-amber-500 text-slate-950 font-semibold"
                : "bg-slate-800 hover:bg-slate-700 text-slate-300"
            )}
          >
            {preset}
          </button>
        ))}
      </div>
      {error && (
        <div className="w-full text-[11px] text-rose-300 bg-rose-500/10 rounded px-2 py-1 border border-rose-500/30">
          {error}
        </div>
      )}
    </div>
  );
}
