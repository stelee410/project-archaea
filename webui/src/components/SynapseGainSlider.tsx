import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { api } from "../api";

/**
 * Live slider for the output-layer synaptic gain g (SPEC v1.2 off-SPEC).
 *
 *   I_o[t] = I_in · g · Σ_j W_ho[j] · h_spike[t,j]
 *
 * - g = 1.0 → SPEC §1.1 bit-identical
 * - g > 1   → physically more output spikes (subject to LIF refractory limit ≈ 150 Hz/neuron)
 *
 * Takes effect from the next simulation window AND the next inference call.
 * Debounced 250 ms.
 */
export function SynapseGainSlider({ initial }: { initial: number }) {
  const [value, setValue] = useState<number>(initial);
  const [busy, setBusy] = useState(false);
  const [serverValue, setServerValue] = useState<number>(initial);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<number | null>(null);
  const lastSentRef = useRef<number>(initial);

  useEffect(() => {
    setValue(initial);
    setServerValue(initial);
    lastSentRef.current = initial;
  }, [initial]);

  function commit(v: number) {
    if (debounceRef.current != null) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(async () => {
      if (Math.abs(v - lastSentRef.current) < 1e-6) return;
      lastSentRef.current = v;
      setBusy(true);
      setError(null);
      try {
        const r = await api.setSynapseGain(v);
        setServerValue(r.synapse_gain);
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
    <div className="rounded-lg border border-fuchsia-700/40 bg-fuchsia-950/20 px-4 py-3 flex flex-wrap items-center gap-x-5 gap-y-2">
      <div className="flex flex-col">
        <span className="text-xs font-semibold text-fuchsia-200">
          输出层突触增益 g
          <span className="ml-1 text-[10px] font-normal text-fuchsia-400/70">
            (SPEC v1.2 off-SPEC · 实时生效 · 物理层音量)
          </span>
        </span>
        <span className="text-[11px] text-fuchsia-200/70 leading-snug max-w-[460px]">
          I_o = I_in · g · Σ W_ho · h_spike。g=1 是 SPEC 默认；
          g↑ 让输出神经元 <b>真的发更多 spike</b>（不是事后乘）。
          <br />
          受 LIF 不应期硬上限约束（单神经元 ≈ 150 Hz）。
        </span>
      </div>
      <div className="flex-1 min-w-[260px] flex items-center gap-3">
        <input
          type="range"
          min={1}
          max={8}
          step={0.1}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="flex-1 accent-fuchsia-400"
        />
        <input
          type="number"
          min={0.1}
          max={20}
          step={0.1}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-[70px] bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-fuchsia-200"
        />
      </div>
      <div className="flex items-center gap-3 text-[11px] numeric">
        <span className="text-slate-400">
          server g = <b className={clsx(hasDrift ? "text-fuchsia-300" : "text-emerald-300")}>
            {serverValue.toFixed(2)}
          </b>
          {busy && <span className="ml-1 text-slate-500">同步中…</span>}
        </span>
        {[1, 2, 3, 5].map((preset) => (
          <button
            key={preset}
            onClick={() => onChange(preset)}
            className={clsx(
              "px-1.5 py-0.5 rounded text-[10px]",
              Math.abs(value - preset) < 1e-6
                ? "bg-fuchsia-500 text-slate-950 font-semibold"
                : "bg-slate-800 hover:bg-slate-700 text-slate-300"
            )}
          >
            ×{preset}
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
