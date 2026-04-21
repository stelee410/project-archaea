import type { TelemetryEvent } from "../types";

type Row = [string, (e: TelemetryEvent) => string, string];

const BASE_ROWS: Row[] = [
  ["t_sim", (e) => `${e.t_sim.toFixed(2)} s`, "已仿真时间"],
  ["pop_size / pop_max", (e) => `${e.pop_size} / ${e.pop_max}`, "活体 / 上限"],
  ["births", (e) => String(e.births), "本窗繁殖事件"],
  ["deaths", (e) => String(e.deaths), "本窗死亡事件"],
  ["r_max", (e) => e.r_max.toFixed(4), "**核心指标**：最强个体的 Pearson r"],
  ["r_mean", (e) => e.r_mean.toFixed(4), "已定义适应度个体的平均 r"],
  ["credit_mean", (e) => e.credit_mean.toFixed(2), "种群「饱腹度」"],
  ["credit_gini", (e) => e.credit_gini.toFixed(3), "Credit 不平等 [0,1]"],
  ["weight_std", (e) => e.weight_std.toFixed(3), "基因多样性"],
  ["sigma", (e) => e.sigma.toFixed(4), "全局突变步长（隐含进度条）"],
  ["budget D/B", (e) => e.budget_pressure.toFixed(3), "off-SPEC：资源紧张度"],
];

const SLIME_ROWS: Row[] = [
  ["pheromone_max", (e) => e.pheromone_max.toFixed(3), "🍄 当前最强信息素细胞"],
  ["pheromone_mean", (e) => e.pheromone_mean.toFixed(4), "🍄 全场平均（衡量整体活跃度）"],
  ["hgt_count", (e) => String(e.hgt_count), "🍄 本窗横向基因转移事件"],
  ["migrations", (e) => String(e.migrations), "🍄 本窗趋化迁移次数"],
];

export function StatsTable({ ev }: { ev: TelemetryEvent | null }) {
  const rows = ev?.slime_enabled ? [...BASE_ROWS, ...SLIME_ROWS] : BASE_ROWS;
  return renderTable(ev, rows);
}

function renderTable(ev: TelemetryEvent | null, rows: Row[]) {

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 overflow-hidden">
      <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-800">
        实时遥测（每 500 ms 仿真窗刷新一次）
      </div>
      <table className="w-full text-sm">
        <tbody>
          {rows.map(([k, render, desc]) => (
            <tr key={k} className="border-t border-slate-800/50 first:border-t-0">
              <td className="px-3 py-1.5 text-slate-400 font-mono text-[11px] w-1/3 align-top">
                {k}
                <div className="text-[10px] text-slate-500 leading-tight mt-0.5 font-sans">
                  {desc}
                </div>
              </td>
              <td className="px-3 py-1.5 numeric text-slate-100 text-right align-top">
                {ev ? render(ev) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
