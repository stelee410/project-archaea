import { useEffect, useMemo, useRef } from "react";
import type { TelemetryEvent } from "../types";

interface Props {
  ev: TelemetryEvent | null;
  popMax: number;
  selectedSlot: number | null;
  onSelect: (slot: number) => void;
}

const C_REPRO = 200; // SPEC §4.2

function dotColor(alive: boolean, credit: number): string {
  if (!alive) return "#374151";
  if (credit < 15) return "#dc2626";
  const t = Math.min(1, credit / C_REPRO);
  if (t >= 0.55) {
    const g = 195 + Math.round(60 * ((t - 0.55) / 0.45));
    return `rgb(33, ${g}, 96)`;
  }
  if (t >= 0.25) return "#eab308";
  return "#f97316";
}

export function DotGrid({ ev, popMax, selectedSlot, onSelect }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const cols = useMemo(() => Math.ceil(Math.sqrt(popMax)), [popMax]);
  const rows = useMemo(() => Math.ceil(popMax / cols), [popMax, cols]);

  const eventChildren = useMemo(() => new Set(ev?.repro_child_slots ?? []), [ev]);
  const eventParents = useMemo(() => new Set(ev?.repro_parent_slots ?? []), [ev]);
  const eventDeads = useMemo(() => new Set(ev?.dead_slots ?? []), [ev]);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const W = cv.width;
    const H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, W, H);

    const pad = 8;
    const dxAvail = (W - pad * 2) / cols;
    const dyAvail = (H - pad * 2) / rows;
    const r = Math.max(2.5, Math.min(dxAvail, dyAvail) * 0.42);

    for (let s = 0; s < popMax; s++) {
      const cx = pad + (s % cols + 0.5) * dxAvail;
      const cy = pad + (Math.floor(s / cols) + 0.5) * dyAvail;
      const alive = ev ? !!ev.alive[s] : false;
      const credit = ev ? ev.credit[s] : 0;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = dotColor(alive, credit);
      ctx.fill();
      let stroke: string | null = null;
      let lw = 0.8;
      if (eventChildren.has(s)) {
        stroke = "#ec4899";
        lw = 2;
        ctx.strokeStyle = stroke;
        ctx.lineWidth = lw;
        ctx.stroke();
      } else if (eventParents.has(s)) {
        stroke = "#f9a8d4";
        lw = 2;
        ctx.strokeStyle = stroke;
        ctx.lineWidth = lw;
        ctx.stroke();
      } else if (eventDeads.has(s)) {
        stroke = "#94a3b8";
        lw = 1.4;
        ctx.strokeStyle = stroke;
        ctx.lineWidth = lw;
        ctx.stroke();
      }
      if (selectedSlot === s) {
        ctx.beginPath();
        ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }, [ev, popMax, cols, rows, selectedSlot, eventChildren, eventParents, eventDeads]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const cv = ref.current;
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * cv.width;
    const y = ((e.clientY - rect.top) / rect.height) * cv.height;
    const pad = 8;
    const dxAvail = (cv.width - pad * 2) / cols;
    const dyAvail = (cv.height - pad * 2) / rows;
    const c = Math.floor((x - pad) / dxAvail);
    const r = Math.floor((y - pad) / dyAvail);
    if (c < 0 || c >= cols || r < 0 || r >= rows) return;
    const slot = r * cols + c;
    if (slot >= 0 && slot < popMax) onSelect(slot);
  }

  return (
    <div className="w-full h-full flex flex-col">
      <canvas
        ref={ref}
        width={780}
        height={520}
        onClick={handleClick}
        className="w-full h-full rounded border border-slate-800 cursor-pointer"
      />
      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
        <Legend dot="#374151" label="空槽" />
        <Legend dot="rgb(33, 245, 96)" label="高 Credit" />
        <Legend dot="#eab308" label="中 Credit" />
        <Legend dot="#f97316" label="低 Credit" />
        <Legend dot="#dc2626" label="临界" />
        <Legend dot="#ec4899" label="新生 (粉描边)" />
        <Legend dot="#f9a8d4" label="亲代 (浅粉描边)" />
        <Legend dot="#94a3b8" label="饿死 (灰描边)" />
        <span className="ml-auto text-slate-500">点击 dot → 右侧查看 agent 内部 10→20→1 拓扑</span>
      </div>
    </div>
  );
}

function Legend({ dot, label }: { dot: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: dot }} />
      {label}
    </span>
  );
}
