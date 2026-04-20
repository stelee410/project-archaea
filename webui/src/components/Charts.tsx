import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartPoint } from "../store";

export function FitnessChart({ data }: { data: ChartPoint[] }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3">
      <div className="text-xs text-slate-400 mb-1">
        Pearson r 趋势（蓝=r_max；青=r_mean；紫虚线=σ）
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="2 4" />
          <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#94a3b8" }} type="number" domain={["dataMin", "dataMax"]} />
          <YAxis domain={[-0.05, 1.05]} tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <Tooltip
            contentStyle={{
              background: "#0f172a",
              border: "1px solid #334155",
              fontSize: 12,
            }}
            labelFormatter={(v) => `t=${(v as number).toFixed(1)}s`}
          />
          <ReferenceLine y={0.7} stroke="#22c55e" strokeDasharray="3 3" />
          <Line dataKey="r_max" stroke="#3b82f6" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Line dataKey="r_mean" stroke="#14b8a6" dot={false} strokeWidth={1.2} isAnimationActive={false} />
          <Line dataKey="sigma" stroke="#a78bfa" dot={false} strokeWidth={1} strokeDasharray="3 3" isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PopulationChart({ data, popMax }: { data: ChartPoint[]; popMax: number }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3">
      <div className="text-xs text-slate-400 mb-1">
        种群规模 N（橙）与平均 Credit（棕）— pop_max={popMax}
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="2 4" />
          <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#94a3b8" }} type="number" domain={["dataMin", "dataMax"]} />
          <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
            labelFormatter={(v) => `t=${(v as number).toFixed(1)}s`}
          />
          <Line dataKey="pop" stroke="#f97316" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Line dataKey="credit" stroke="#b45309" dot={false} strokeWidth={1.2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function BudgetChart({ data }: { data: ChartPoint[] }) {
  const has = data.some((d) => d.budget > 0);
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3">
      <div className="text-xs text-slate-400 mb-1">
        预算紧张度 D/B（仅在 budget_mode=shared 时有意义；红线 = 1.0 是承载力上限）
      </div>
      <ResponsiveContainer width="100%" height={130}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="2 4" />
          <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#94a3b8" }} type="number" domain={["dataMin", "dataMax"]} />
          <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} domain={[0, "auto"]} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
            labelFormatter={(v) => `t=${(v as number).toFixed(1)}s`}
          />
          <ReferenceLine y={1.0} stroke="#94a3b8" strokeDasharray="3 3" />
          <Line dataKey="budget" stroke="#dc2626" dot={false} strokeWidth={1.4} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      {!has && (
        <div className="text-[11px] text-slate-500 mt-1">
          当前未启用 shared budget；曲线恒为 0。在「设置」页选 budget_mode=shared 并填 carrying_capacity。
        </div>
      )}
    </div>
  );
}

export function VitalsChart({ data }: { data: ChartPoint[] }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3">
      <div className="text-xs text-slate-400 mb-1">
        births（绿）/ deaths（红）— 进化引擎的脉搏
      </div>
      <ResponsiveContainer width="100%" height={130}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="2 4" />
          <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#94a3b8" }} type="number" domain={["dataMin", "dataMax"]} />
          <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
            labelFormatter={(v) => `t=${(v as number).toFixed(1)}s`}
          />
          <Line dataKey="births" stroke="#22c55e" dot={false} strokeWidth={1.2} isAnimationActive={false} />
          <Line dataKey="deaths" stroke="#ef4444" dot={false} strokeWidth={1.2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
