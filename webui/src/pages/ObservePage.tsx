import { useState } from "react";
import { useStore } from "../store";
import { DotGrid } from "../components/DotGrid";
import { AgentDetail } from "../components/AgentDetail";
import { StatsTable } from "../components/StatsTable";
import { CalibrationLambdaSlider } from "../components/CalibrationLambdaSlider";
import { SynapseGainSlider } from "../components/SynapseGainSlider";
import {
  BudgetChart,
  FitnessChart,
  PopulationChart,
  VitalsChart,
} from "../components/Charts";

export function ObservePage() {
  const latest = useStore((s) => s.latest);
  const history = useStore((s) => s.history);
  const status = useStore((s) => s.status);
  const popMax = latest?.pop_max ?? status?.pop_max ?? 200;
  const [selected, setSelected] = useState<number | null>(null);

  if (!status?.running && !latest) {
    return (
      <div className="max-w-[900px] mx-auto p-8 text-slate-400 text-sm">
        当前没有仿真在运行。先去「设置 / 启动」页配置参数并启动。
      </div>
    );
  }

  return (
    <div className="max-w-[1500px] mx-auto p-4 grid gap-4 grid-cols-12">
      <div className="col-span-12 grid grid-cols-1 xl:grid-cols-2 gap-3">
        <CalibrationLambdaSlider initial={status?.config?.calibration_lambda ?? 0} />
        <SynapseGainSlider initial={status?.config?.synapse_gain ?? 1} />
      </div>
      {/* 左大块：dot grid */}
      <section className="col-span-12 lg:col-span-8 space-y-4">
        <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3">
          <div className="flex items-baseline justify-between mb-2">
            <div className="text-sm font-semibold">种群点阵</div>
            <div className="text-xs text-slate-400 numeric">
              {latest
                ? `t=${latest.t_sim.toFixed(1)}s · N=${latest.pop_size}/${latest.pop_max} · b=${latest.births} d=${latest.deaths}`
                : "等待第一个数据帧…"}
            </div>
          </div>
          <DotGrid ev={latest} popMax={popMax} selectedSlot={selected} onSelect={setSelected} />
        </div>
        <FitnessChart data={history} />
        <PopulationChart data={history} popMax={popMax} />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <BudgetChart data={history} />
          <VitalsChart data={history} />
        </div>
      </section>

      {/* 右栏：stats + agent detail */}
      <aside className="col-span-12 lg:col-span-4 space-y-4">
        <StatsTable ev={latest} />
        <div className="rounded-lg border border-slate-800 bg-slate-900/50">
          <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-800 flex items-baseline justify-between">
            <span>Agent 详情（点击 dot 选中）</span>
            {selected != null && (
              <button
                onClick={() => setSelected(null)}
                className="text-[11px] text-slate-500 hover:text-slate-200"
              >
                取消选中
              </button>
            )}
          </div>
          <AgentDetail slot={selected} refreshKey={Math.floor((latest?.t_sim ?? 0) * 2)} />
        </div>
      </aside>
    </div>
  );
}
