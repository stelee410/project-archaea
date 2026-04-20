import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { AgentDetail as A } from "../types";

interface Props {
  slot: number | null;
  refreshKey: number;
}

export function AgentDetail({ slot, refreshKey }: Props) {
  const [data, setData] = useState<A | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (slot == null) return;
    let cancelled = false;
    api
      .agent(slot)
      .then((d) => !cancelled && (setData(d), setErr(null)))
      .catch((e) => !cancelled && setErr(String(e)));
    return () => {
      cancelled = true;
    };
  }, [slot, refreshKey]);

  if (slot == null) {
    return (
      <div className="text-sm text-slate-400 p-4">
        点击左侧 dot 选中某个 agent，查看其内部 10→20→1 拓扑。
      </div>
    );
  }

  return (
    <div className="p-3 flex flex-col gap-3 h-full overflow-y-auto">
      <div>
        <div className="text-xs text-slate-400">Slot</div>
        <div className="text-lg font-mono">#{slot}</div>
      </div>
      {err && <div className="text-rose-300 text-xs">{err}</div>}
      {data && (
        <>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <Cell
              k="alive"
              v={
                <span className={data.alive ? "text-emerald-300" : "text-rose-300"}>
                  {data.alive ? "yes" : "no"}
                </span>
              }
            />
            <Cell k="credit" v={<span className="numeric">{data.credit.toFixed(2)}</span>} />
            <Cell
              k="fitness r"
              v={
                <span className="numeric">
                  {data.fitness == null ? "—" : data.fitness.toFixed(4)}
                </span>
              }
            />
          </div>
          <div className="text-xs text-slate-400 -mb-1">
            内部拓扑：input(10) → hidden(20) → output(1)
          </div>
          <TopologyView detail={data} />
          <div className="text-[11px] text-slate-500 leading-snug">
            线条颜色：<span className="text-cyan-300">蓝青</span> = 正权重，
            <span className="text-rose-300">红</span> = 负权重；粗细 ∝ |w|。这就是这个个体的「DNA」表达。
          </div>
        </>
      )}
    </div>
  );
}

function Cell({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="rounded bg-slate-950/50 border border-slate-800 px-2 py-1.5">
      <div className="text-[10px] text-slate-500 font-mono">{k}</div>
      <div>{v}</div>
    </div>
  );
}

function TopologyView({ detail }: { detail: A }) {
  const { topology } = detail;
  const W = 360;
  const H = 360;
  const padX = 40;

  const inputY = useMemo(() => positions(topology.input_nodes.length, padX, H - padX), [topology.input_nodes.length, H, padX]);
  const hiddenY = useMemo(() => positions(topology.hidden_nodes.length, 8, H - 8), [topology.hidden_nodes.length, H]);
  const outputY = useMemo(() => positions(topology.output_nodes.length, 80, H - 80), [topology.output_nodes.length, H]);

  const inputX = padX;
  const hiddenX = W / 2;
  const outputX = W - padX;

  const allW = [...topology.edges_ih, ...topology.edges_ho].map((e) => Math.abs(e.w));
  const wMax = Math.max(0.01, ...allW);

  function edgeColor(w: number): string {
    return w >= 0 ? "rgba(34,211,238,0.7)" : "rgba(244,63,94,0.7)";
  }
  function edgeWidth(w: number): number {
    return 0.4 + (Math.abs(w) / wMax) * 1.6;
  }

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="bg-slate-950/40 rounded border border-slate-800">
      {topology.edges_ih.map((e, i) => {
        const si = parseInt(e.src.slice(1), 10);
        const sj = parseInt(e.dst.slice(1), 10);
        return (
          <line
            key={`ih-${i}`}
            x1={inputX}
            y1={inputY[si]}
            x2={hiddenX}
            y2={hiddenY[sj]}
            stroke={edgeColor(e.w)}
            strokeWidth={edgeWidth(e.w)}
            opacity={0.6}
          />
        );
      })}
      {topology.edges_ho.map((e, i) => {
        const sj = parseInt(e.src.slice(1), 10);
        return (
          <line
            key={`ho-${i}`}
            x1={hiddenX}
            y1={hiddenY[sj]}
            x2={outputX}
            y2={outputY[0]}
            stroke={edgeColor(e.w)}
            strokeWidth={edgeWidth(e.w)}
            opacity={0.85}
          />
        );
      })}
      {inputY.map((y, i) => (
        <circle key={`i-${i}`} cx={inputX} cy={y} r={5} fill="#10b981" />
      ))}
      {hiddenY.map((y, i) => (
        <circle key={`h-${i}`} cx={hiddenX} cy={y} r={4} fill="#a78bfa" />
      ))}
      {outputY.map((y, i) => (
        <circle key={`o-${i}`} cx={outputX} cy={y} r={7} fill="#f59e0b" />
      ))}
      <text x={inputX} y={20} fontSize={10} fill="#94a3b8" textAnchor="middle">input × 10</text>
      <text x={hiddenX} y={20} fontSize={10} fill="#94a3b8" textAnchor="middle">hidden × 20</text>
      <text x={outputX} y={20} fontSize={10} fill="#94a3b8" textAnchor="middle">output × 1</text>
    </svg>
  );
}

function positions(n: number, top: number, bottom: number): number[] {
  if (n <= 1) return [(top + bottom) / 2];
  const step = (bottom - top) / (n - 1);
  return Array.from({ length: n }, (_, i) => top + i * step);
}
