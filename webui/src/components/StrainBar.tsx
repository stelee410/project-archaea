import { useState } from "react";
import clsx from "clsx";
import { api } from "../api";
import { useStore } from "../store";
import type { StrainMeta, TelemetryEvent } from "../types";

/**
 * SPEC_L2_V3.0 §3.2 — observe-page bar for the admixture experiment workflow.
 *
 * Two responsibilities:
 *
 *   1. 「💾 保存为菌株」— snapshot the current living population to disk so it
 *      becomes a reusable founder for ⚗️ Mixer experiments. Modal asks for
 *      a name + free-form note.
 *   2. 杂交期 telemetry pill — when the running sim is inside its admixture
 *      window, show a pulsing badge with eff_hgt_prob / multiplier / time-left
 *      so the observer can see "yes, the genes are mixing right now".
 *
 * Mounted unconditionally on ObservePage; renders nothing useful when the sim
 * isn't running.
 */
export function StrainBar({ ev }: { ev: TelemetryEvent | null }) {
  const status = useStore((s) => s.status);
  const running = !!status?.running;
  const taskRunning = status?.config?.task ?? null;
  const [saveOpen, setSaveOpen] = useState(false);
  // SPEC_L2_V3.4 — 3-phase admixture telemetry.
  const admixPhase = (ev?.admixture_phase ?? 3) as 1 | 2 | 3;
  const admixActive = !!ev?.admixture_active;
  const commensalS = ev?.admixture_commensal_s ?? 0;
  const exchangeS = ev?.admixture_exchange_s ?? 0;
  const phase2Blend = ev?.admixture_phase2_blend ?? 0;
  const phase2Mul = ev?.admixture_phase2_prob_mul ?? 1;
  const effHgtBlend = ev?.eff_hgt_blend ?? 0;
  const effHgtProb = ev?.eff_hgt_prob ?? 0;
  const tSim = ev?.t_sim ?? 0;
  // Time remaining in the *current* phase (so the badge counts down sensibly).
  const phaseEndsAt =
    admixPhase === 1 ? commensalS : admixPhase === 2 ? commensalS + exchangeS : 0;
  const phaseTotal =
    admixPhase === 1 ? commensalS : admixPhase === 2 ? exchangeS : 0;
  const remain = Math.max(0, phaseEndsAt - tSim);

  return (
    <div className="rounded-lg border border-cyan-700/40 bg-cyan-950/10 p-3 flex flex-wrap items-center gap-3">
      <div className="flex items-baseline gap-2">
        <span className="text-base">🧪</span>
        <span className="text-sm font-semibold text-cyan-100">菌株 / 杂交皿</span>
        <span className="text-[10px] font-mono text-cyan-200/50">
          SPEC_L2_V3.0 · admixture
        </span>
      </div>

      <button
        type="button"
        onClick={() => setSaveOpen(true)}
        disabled={!running}
        className={clsx(
          "ml-auto px-3 py-1.5 rounded text-xs font-medium transition-colors",
          running
            ? "bg-cyan-500 hover:bg-cyan-400 text-slate-950"
            : "bg-slate-800 text-slate-500 cursor-not-allowed"
        )}
        title={
          running
            ? `把当前活体种群（task=${taskRunning ?? "?"}）冻存为菌株，` +
              `供 ⚗️ Mixer 当作 founder 使用`
            : "需要有正在跑的仿真"
        }
      >
        💾 保存为菌株
      </button>

      {admixActive && (
        <PhaseBadge
          phase={admixPhase}
          remain={remain}
          phaseTotal={phaseTotal}
          phase2Blend={phase2Blend}
          phase2Mul={phase2Mul}
          effHgtBlend={effHgtBlend}
          effHgtProb={effHgtProb}
        />
      )}

      {saveOpen && (
        <SaveStrainModal
          onClose={() => setSaveOpen(false)}
          taskHint={taskRunning ?? "?"}
        />
      )}
    </div>
  );
}

/**
 * SPEC_L2_V3.4 — phase-aware "杂交进行中" badge.
 *
 * Rendered only while the admixture protocol is active (phase 1 or 2).
 *   Phase 1 (commensal): cyan, "HGT 完全关闭" — observers should expect zero
 *     transfers; the two strains are getting their footing independently.
 *   Phase 2 (controlled exchange): fuchsia/pulsing, shows current eff_blend +
 *     eff_prob — observers should see HGT events fire but at low intensity.
 */
function PhaseBadge({
  phase,
  remain,
  phaseTotal,
  phase2Blend,
  phase2Mul,
  effHgtBlend,
  effHgtProb,
}: {
  phase: 1 | 2 | 3;
  remain: number;
  phaseTotal: number;
  phase2Blend: number;
  phase2Mul: number;
  effHgtBlend: number;
  effHgtProb: number;
}) {
  if (phase === 1) {
    return (
      <div className="w-full mt-1 rounded border border-cyan-500/40 bg-cyan-950/30 px-3 py-2 text-xs text-cyan-100 leading-relaxed">
        <div className="flex items-center gap-2 font-semibold">
          <span className="inline-block w-2 h-2 rounded-full bg-cyan-300" />
          ① 共栖期 (commensal · HGT off)
          <span className="font-mono numeric text-cyan-200/80 ml-2">
            剩 {remain.toFixed(1)}s / 共 {phaseTotal.toFixed(1)}s
          </span>
        </div>
        <div className="mt-1 font-mono numeric text-cyan-200/70 text-[11px]">
          eff_hgt_prob = 0 · eff_blend = 0 · 两群菌株共享培养皿但基因池隔离
        </div>
        <div className="mt-1 text-[11px] text-cyan-200/60">
          先让每个菌株各自在新环境里站稳脚跟；交换期开始前不会有任何基因转移。
        </div>
      </div>
    );
  }
  if (phase === 2) {
    return (
      <div className="w-full mt-1 rounded border border-fuchsia-500/40 bg-fuchsia-950/30 px-3 py-2 text-xs text-fuchsia-100 leading-relaxed">
        <div className="flex items-center gap-2 font-semibold">
          <span className="inline-block w-2 h-2 rounded-full bg-fuchsia-400 animate-pulse" />
          ② 受控交换期 (controlled exchange · low blend)
          <span className="font-mono numeric text-fuchsia-200/80 ml-2">
            剩 {remain.toFixed(1)}s / 共 {phaseTotal.toFixed(1)}s
          </span>
        </div>
        <div className="mt-1 font-mono numeric text-fuchsia-200/80 text-[11px]">
          eff_blend = {effHgtBlend.toFixed(3)} (cfg {phase2Blend.toFixed(2)}) ·
          eff_hgt_prob = {effHgtProb.toFixed(4)} (× {phase2Mul.toFixed(1)})
        </div>
        <div className="mt-1 text-[11px] text-fuchsia-200/70">
          基因开始小步渗透——双修个体在这段时间里诞生而不会被大幅值的一方瞬间扫荡。
        </div>
      </div>
    );
  }
  return null;
}

function SaveStrainModal({
  onClose,
  taskHint,
}: {
  onClose: () => void;
  taskHint: string;
}) {
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [saved, setSaved] = useState<StrainMeta | null>(null);

  async function submit() {
    if (!name.trim()) {
      setErr("名字不能为空");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const meta = await api.saveStrain(name.trim(), note.trim());
      setSaved(meta);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-slate-950/70 flex items-center justify-center p-6"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-xl border border-cyan-500/40 bg-slate-900 p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-baseline gap-2 mb-4">
          <span className="text-xl">🧪</span>
          <h2 className="text-lg font-semibold text-cyan-100">
            保存为菌株
          </h2>
          <span className="ml-auto text-[10px] font-mono text-slate-500">
            task={taskHint}
          </span>
        </div>

        {saved ? (
          <div className="space-y-3 text-sm">
            <div className="rounded border border-emerald-500/40 bg-emerald-950/30 p-3 text-emerald-200">
              ✅ 已保存：<b>{saved.name}</b>
              <div className="mt-1 text-[11px] font-mono text-emerald-300/80 numeric">
                id={saved.id} · n_agents={saved.n_agents} · t_sim=
                {saved.t_sim.toFixed(1)}s
                {saved.acc_and_pop_at_save != null && (
                  <>
                    {" "}· AND={(saved.acc_and_pop_at_save * 100).toFixed(0)}%
                  </>
                )}
                {saved.acc_not_pop_at_save != null && (
                  <>
                    {" "}· NOT={(saved.acc_not_pop_at_save * 100).toFixed(0)}%
                  </>
                )}
              </div>
            </div>
            <div className="text-xs text-slate-400 leading-relaxed">
              现在可以回到群落图鉴，进入 ⚗️ <b>Mixer</b> 卡片，
              把这个菌株和另一个菌株一起倒进新培养皿做杂交实验。
            </div>
            <div className="flex justify-end">
              <button
                onClick={onClose}
                className="px-3 py-1.5 rounded bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-sm font-medium"
              >
                好的
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-xs text-slate-400 leading-relaxed">
              快照「当前所有活体」的权重 + 元数据到本地磁盘
              （<code className="px-1 rounded bg-slate-800">checkpoints/strains/</code>）。
              不会停止仿真。
            </p>
            <div>
              <label className="block text-xs text-slate-300 mb-1">名字</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                maxLength={120}
                placeholder="例：AND-pure-30min · seed42"
                className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-sm font-mono"
                autoFocus
              />
            </div>
            <div>
              <label className="block text-xs text-slate-300 mb-1">
                备注 (可选)
              </label>
              <textarea
                value={note}
                onChange={(e) => setNote(e.target.value)}
                maxLength={500}
                rows={3}
                placeholder="例：balanced 难度跑 30 分钟，acc_and_pop≈0.92，acc_not_pop≈0.18 — AND 学家"
                className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-xs font-mono"
              />
            </div>
            {err && (
              <div className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/40 rounded px-2 py-1.5">
                {err}
              </div>
            )}
            <div className="flex justify-end gap-2 pt-1">
              <button
                onClick={onClose}
                className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm"
              >
                取消
              </button>
              <button
                onClick={submit}
                disabled={busy || !name.trim()}
                className="px-3 py-1.5 rounded bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 text-slate-950 text-sm font-medium"
              >
                {busy ? "保存中…" : "保存"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
