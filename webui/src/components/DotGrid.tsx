import { useEffect, useMemo, useRef } from "react";
import type { TelemetryEvent } from "../types";

interface Props {
  ev: TelemetryEvent | null;
  popMax: number;
  selectedSlot: number | null;
  onSelect: (slot: number) => void;
}

const C_REPRO = 200; // SPEC §4.2

// SPEC_L2_V2.0 §4.2 — agents below this Credit fade red ("hunger warning").
const HUNGER_CREDIT = 20;

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

// Hunger fade: agents with low credit get drawn at reduced alpha.
// 0.0 (invisible) → 1.0 (full opacity).
function hungerAlpha(credit: number): number {
  if (credit >= HUNGER_CREDIT) return 1.0;
  return Math.max(0.25, credit / HUNGER_CREDIT);
}

export function DotGrid({ ev, popMax, selectedSlot, onSelect }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const slimeMode = !!(ev?.slime_enabled && ev?.grid_size > 0);
  const cols = useMemo(() => Math.ceil(Math.sqrt(popMax)), [popMax]);
  const rows = useMemo(() => Math.ceil(popMax / cols), [popMax, cols]);

  const eventChildren = useMemo(() => new Set(ev?.repro_child_slots ?? []), [ev]);
  const eventParents = useMemo(() => new Set(ev?.repro_parent_slots ?? []), [ev]);
  const eventDeads = useMemo(() => new Set(ev?.dead_slots ?? []), [ev]);
  // SPEC_L2_V2.0 §4.2 — slots that received non-trivial reward this window.
  // These flash gold in the next paint.
  const goldenSlots = useMemo(() => {
    const r = ev?.reward;
    if (!r) return new Set<number>();
    // Threshold: any positive reward this window counts as "earned" → flash.
    const out = new Set<number>();
    for (let i = 0; i < r.length; i++) {
      if (r[i] > 0.5) out.add(i);
    }
    return out;
  }, [ev]);
  const hgtPairs = useMemo(() => ev?.hgt_pairs ?? [], [ev]);

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

    const sets: OverlaySets = {
      children: eventChildren,
      parents: eventParents,
      deads: eventDeads,
      golden: goldenSlots,
      hgtPairs,
    };
    if (slimeMode && ev) {
      drawSpatial(ctx, W, H, ev, selectedSlot, sets);
    } else {
      drawSlotGrid(ctx, W, H, ev, popMax, cols, rows, selectedSlot, sets);
    }
  }, [ev, popMax, cols, rows, selectedSlot, eventChildren, eventParents, eventDeads, slimeMode, goldenSlots, hgtPairs]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const cv = ref.current;
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * cv.width;
    const y = ((e.clientY - rect.top) / rect.height) * cv.height;

    if (slimeMode && ev) {
      const G = ev.grid_size;
      const pad = 8;
      const cellW = (cv.width - pad * 2) / G;
      const cellH = (cv.height - pad * 2) / G;
      let bestSlot = -1;
      let bestD = Infinity;
      for (let s = 0; s < ev.alive.length; s++) {
        if (!ev.alive[s]) continue;
        const [px, py] = ev.positions[s] ?? [0, 0];
        const cx = pad + (px + 0.5) * cellW;
        const cy = pad + (py + 0.5) * cellH;
        const d = (cx - x) ** 2 + (cy - y) ** 2;
        if (d < bestD) {
          bestD = d;
          bestSlot = s;
        }
      }
      if (bestSlot >= 0 && bestD < (Math.min(cellW, cellH) * 1.5) ** 2) onSelect(bestSlot);
      return;
    }

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
      {slimeMode ? (
        <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-sm" style={{ background: "linear-gradient(90deg,#0f172a,#a855f7,#fbbf24)" }} />
            信息素 (低 → 高)
          </span>
          <Legend dot="rgb(33, 245, 96)" label="高 Credit" />
          <Legend dot="#eab308" label="中" />
          <Legend dot="#f97316" label="低" />
          <Legend dot="#dc2626" label="临界" />
          <Legend dot="#ec4899" label="新生" />
          <Legend dot="#f9a8d4" label="亲代" />
          <Legend dot="#94a3b8" label="死亡" />
          <span className="ml-auto text-slate-500">
            🍄 黏菌模式 · {ev?.grid_size}×{ev?.grid_size} · P_max={(ev?.pheromone_max ?? 0).toFixed(2)} · HGT={ev?.hgt_count ?? 0} · 移动={ev?.migrations ?? 0}
          </span>
        </div>
      ) : (
        <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
          <Legend dot="#374151" label="空槽" />
          <Legend dot="rgb(33, 245, 96)" label="高 Credit" />
          <Legend dot="#eab308" label="中 Credit" />
          <Legend dot="#f97316" label="低 Credit" />
          <Legend dot="#dc2626" label="临界" />
          <Legend dot="#fbbf24" label="✨ 获奖闪烁 (L2v2)" />
          <Legend dot="#ec4899" label="新生" />
          <Legend dot="#f9a8d4" label="亲代" />
          <Legend dot="#94a3b8" label="饿死" />
          <Legend dot="#a855f7" label="HGT 连线" />
          <span className="ml-auto text-slate-500">点击 dot → 右侧查看 agent 内部 10→20→1 拓扑</span>
        </div>
      )}
    </div>
  );
}

interface OverlaySets {
  children: Set<number>;
  parents: Set<number>;
  deads: Set<number>;
  // SPEC_L2_V2.0 §4.2 — slots that earned reward this window (flash gold)
  golden: Set<number>;
  // SPEC_L2_V2.0 §4.2 — [recipient_slot, donor_slot] pairs for HGT social lines
  hgtPairs: [number, number][];
}

function drawSlotGrid(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  ev: TelemetryEvent | null,
  popMax: number,
  cols: number,
  rows: number,
  selectedSlot: number | null,
  sets: OverlaySets
) {
  const pad = 8;
  const dxAvail = (W - pad * 2) / cols;
  const dyAvail = (H - pad * 2) / rows;
  const r = Math.max(2.5, Math.min(dxAvail, dyAvail) * 0.42);

  // Pre-compute slot centres so HGT lines can reference them.
  const centres: { x: number; y: number }[] = [];
  for (let s = 0; s < popMax; s++) {
    centres.push({
      x: pad + ((s % cols) + 0.5) * dxAvail,
      y: pad + (Math.floor(s / cols) + 0.5) * dyAvail,
    });
  }

  for (let s = 0; s < popMax; s++) {
    const { x: cx, y: cy } = centres[s];
    const alive = ev ? !!ev.alive[s] : false;
    const credit = ev ? ev.credit[s] : 0;
    const alpha = alive ? hungerAlpha(credit) : 1.0;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = dotColor(alive, credit);
    ctx.fill();
    ctx.globalAlpha = 1.0;
    drawOverlay(ctx, cx, cy, r, s, sets);
    if (selectedSlot === s) {
      ctx.beginPath();
      ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  drawHgtLines(ctx, sets.hgtPairs, centres);
}

function drawSpatial(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  ev: TelemetryEvent,
  selectedSlot: number | null,
  sets: OverlaySets
) {
  const G = ev.grid_size;
  const pad = 8;
  const cellW = (W - pad * 2) / G;
  const cellH = (H - pad * 2) / G;

  const pmax = Math.max(ev.pheromone_max, 1e-9);
  for (let i = 0; i < G; i++) {
    for (let j = 0; j < G; j++) {
      const v = (ev.pheromone[i]?.[j] ?? 0) / pmax;
      ctx.fillStyle = pheromoneColor(v);
      ctx.fillRect(pad + i * cellW, pad + j * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= G; i++) {
    ctx.beginPath();
    ctx.moveTo(pad + i * cellW, pad);
    ctx.lineTo(pad + i * cellW, pad + G * cellH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad, pad + i * cellH);
    ctx.lineTo(pad + G * cellW, pad + i * cellH);
    ctx.stroke();
  }

  // Count agents per cell to jitter overlapping dots.
  const cellCounts = new Map<number, number>();
  const cellSeen = new Map<number, number>();
  for (let s = 0; s < ev.alive.length; s++) {
    if (!ev.alive[s]) continue;
    const [px, py] = ev.positions[s] ?? [0, 0];
    const key = px * G + py;
    cellCounts.set(key, (cellCounts.get(key) ?? 0) + 1);
  }

  const r = Math.max(2.5, Math.min(cellW, cellH) * 0.28);
  // Per-agent screen positions (needed for HGT lines below).
  const slotXY = new Map<number, { x: number; y: number }>();
  for (let s = 0; s < ev.alive.length; s++) {
    if (!ev.alive[s]) continue;
    const [px, py] = ev.positions[s] ?? [0, 0];
    const key = px * G + py;
    const total = cellCounts.get(key) ?? 1;
    const idx = cellSeen.get(key) ?? 0;
    cellSeen.set(key, idx + 1);
    let cx = pad + (px + 0.5) * cellW;
    let cy = pad + (py + 0.5) * cellH;
    if (total > 1) {
      const angle = (idx / total) * Math.PI * 2;
      const radius = Math.min(cellW, cellH) * 0.22;
      cx += Math.cos(angle) * radius;
      cy += Math.sin(angle) * radius;
    }
    slotXY.set(s, { x: cx, y: cy });
    const alpha = hungerAlpha(ev.credit[s]);
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = dotColor(true, ev.credit[s]);
    ctx.fill();
    ctx.globalAlpha = 1.0;
    drawOverlay(ctx, cx, cy, r, s, sets);
    if (selectedSlot === s) {
      ctx.beginPath();
      ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // HGT social lines — only between two living agents we actually drew.
  ctx.save();
  ctx.strokeStyle = "rgba(168, 85, 247, 0.65)";
  ctx.lineWidth = 1.4;
  ctx.setLineDash([4, 3]);
  for (const [recipient, donor] of sets.hgtPairs) {
    const a = slotXY.get(recipient);
    const b = slotXY.get(donor);
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawHgtLines(
  ctx: CanvasRenderingContext2D,
  pairs: [number, number][],
  centres: { x: number; y: number }[]
) {
  if (pairs.length === 0) return;
  ctx.save();
  ctx.strokeStyle = "rgba(168, 85, 247, 0.6)";
  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 3]);
  for (const [recipient, donor] of pairs) {
    const a = centres[recipient];
    const b = centres[donor];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  s: number,
  sets: OverlaySets
) {
  // SPEC_L2_V2.0 §4.2 — gold halo for agents earning Credit this window.
  if (sets.golden.has(s)) {
    ctx.save();
    const grd = ctx.createRadialGradient(cx, cy, r * 0.6, cx, cy, r * 2.2);
    grd.addColorStop(0, "rgba(251, 191, 36, 0.85)");
    grd.addColorStop(0.5, "rgba(251, 191, 36, 0.35)");
    grd.addColorStop(1, "rgba(251, 191, 36, 0)");
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 2.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(252, 211, 77, 0.95)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(cx, cy, r + 1.2, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
  if (sets.children.has(s)) {
    ctx.strokeStyle = "#ec4899";
    ctx.lineWidth = 2;
    ctx.stroke();
  } else if (sets.parents.has(s)) {
    ctx.strokeStyle = "#f9a8d4";
    ctx.lineWidth = 2;
    ctx.stroke();
  } else if (sets.deads.has(s)) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }
}

function pheromoneColor(t: number): string {
  // Dark slate → fuchsia → amber, perceptually scaled.
  const v = Math.max(0, Math.min(1, t));
  if (v < 0.5) {
    const a = v / 0.5;
    const r = Math.round(15 + (168 - 15) * a);
    const g = Math.round(23 + (85 - 23) * a);
    const b = Math.round(42 + (247 - 42) * a);
    return `rgb(${r},${g},${b})`;
  }
  const a = (v - 0.5) / 0.5;
  const r = Math.round(168 + (251 - 168) * a);
  const g = Math.round(85 + (191 - 85) * a);
  const b = Math.round(247 + (36 - 247) * a);
  return `rgb(${r},${g},${b})`;
}

function Legend({ dot, label }: { dot: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: dot }} />
      {label}
    </span>
  );
}
