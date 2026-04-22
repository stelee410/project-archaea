import { useEffect, useMemo, useRef, useState } from "react";
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
    return `rgb(33,${g},96)`;
  }
  if (t >= 0.25) return "#eab308";
  return "#f97316";
}

// Hunger fade: agents with low credit get drawn at reduced alpha.
// Quantized to 5 bins so the per-color batching can also key on alpha.
function hungerAlphaBin(credit: number): number {
  if (credit >= HUNGER_CREDIT) return 5;
  if (credit >= 15) return 4;
  if (credit >= 10) return 3;
  if (credit >= 5) return 2;
  return 1;
}
const ALPHA_FROM_BIN = [0, 0.25, 0.4, 0.55, 0.75, 1.0];

export function DotGrid({ ev, popMax, selectedSlot, onSelect }: Props) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const slimeMode = !!(ev?.slime_enabled && ev?.grid_size > 0);
  const cols = useMemo(() => Math.ceil(Math.sqrt(popMax)), [popMax]);
  const rows = useMemo(() => Math.ceil(popMax / cols), [popMax, cols]);

  // ── Canvas physical sizing (HiDPI + container resize) ────────────────────
  // We let the canvas fill its parent and bump physical resolution by DPR.
  // pop_max scales: small grids look crisper on retina; large grids get more
  // pixels per dot which keeps clicks accurate.
  const [pxSize, setPxSize] = useState({ w: 780, h: 520 });
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        // height: keep ~2:3 ratio with width but cap by container itself.
        // legend below sits outside the canvas, so we use the container's
        // own measured height via flex.
        const h = Math.max(360, Math.min(900, Math.floor(height)));
        setPxSize({ w: Math.floor(width), h });
      }
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  const eventChildren = useMemo(() => new Set(ev?.repro_child_slots ?? []), [ev]);
  const eventParents = useMemo(() => new Set(ev?.repro_parent_slots ?? []), [ev]);
  const eventDeads = useMemo(() => new Set(ev?.dead_slots ?? []), [ev]);
  const goldenSlots = useMemo(() => {
    const r = ev?.reward;
    if (!r) return new Set<number>();
    const out = new Set<number>();
    for (let i = 0; i < r.length; i++) {
      if (r[i] > 0.5) out.add(i);
    }
    return out;
  }, [ev]);
  const hgtPairs = useMemo(() => ev?.hgt_pairs ?? [], [ev]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = pxSize.w;
    const cssH = pxSize.h;
    if (cv.width !== cssW * dpr || cv.height !== cssH * dpr) {
      cv.width = Math.max(1, Math.floor(cssW * dpr));
      cv.height = Math.max(1, Math.floor(cssH * dpr));
    }
    // Use CSS-pixel coordinates inside the draw functions.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, cssW, cssH);

    const sets: OverlaySets = {
      children: eventChildren,
      parents: eventParents,
      deads: eventDeads,
      golden: goldenSlots,
      hgtPairs,
    };
    if (slimeMode && ev) {
      drawSpatial(ctx, cssW, cssH, ev, selectedSlot, sets);
    } else {
      drawSlotGrid(ctx, cssW, cssH, ev, popMax, cols, rows, selectedSlot, sets);
    }
  }, [ev, popMax, cols, rows, selectedSlot, eventChildren, eventParents, eventDeads, slimeMode, goldenSlots, hgtPairs, pxSize]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const cv = canvasRef.current;
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    // Note: we draw in CSS pixels via setTransform(dpr,...), so map clicks
    // back to CSS pixel space too.
    const x = ((e.clientX - rect.left) / rect.width) * pxSize.w;
    const y = ((e.clientY - rect.top) / rect.height) * pxSize.h;

    if (slimeMode && ev) {
      const G = ev.grid_size;
      const pad = 8;
      const cellW = (pxSize.w - pad * 2) / G;
      const cellH = (pxSize.h - pad * 2) / G;
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
    const dxAvail = (pxSize.w - pad * 2) / cols;
    const dyAvail = (pxSize.h - pad * 2) / rows;
    const c = Math.floor((x - pad) / dxAvail);
    const r = Math.floor((y - pad) / dyAvail);
    if (c < 0 || c >= cols || r < 0 || r >= rows) return;
    const slot = r * cols + c;
    if (slot >= 0 && slot < popMax) onSelect(slot);
  }

  return (
    <div ref={wrapRef} className="w-full flex flex-col" style={{ minHeight: 380 }}>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        style={{ width: "100%", height: pxSize.h, display: "block" }}
        className="rounded border border-slate-800 cursor-pointer"
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
          <span className="ml-auto text-slate-500">
            点击 dot → 右侧查看拓扑 · {popMax} 槽 · {cols}×{rows}
          </span>
        </div>
      )}
    </div>
  );
}

interface OverlaySets {
  children: Set<number>;
  parents: Set<number>;
  deads: Set<number>;
  golden: Set<number>;
  hgtPairs: [number, number][];
}

/**
 * Slot-grid renderer (no-slime mode).
 *
 * Performance strategy (so 1000–5000 dots stay 60 fps):
 *
 *   1. Bucket dots by (color, alpha-bin) — typically 5–8 buckets.
 *   2. For r > 3 px: build one Path2D per bucket, single ctx.fill() call.
 *   3. For r ≤ 3 px: fillRect pixel squares (≈3× faster than tiny arc()s).
 *   4. Overlays (gold halo, parent/child rings, HGT lines) are drawn after,
 *      and they are always sparse (only this-window events).
 */
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
  const pad = 6;
  const dxAvail = (W - pad * 2) / cols;
  const dyAvail = (H - pad * 2) / rows;
  // dot 直径 = 格子宽 × 2 × 系数。0.32 → dot 占 64% 格子，留 36% 给间距，
  // 视觉比之前的 0.42（占 84%、几乎挤满）清爽得多；min=1.0 让超密集时
  // 像素方块路径仍能输出 2×2 px 的可见点。
  const r = Math.max(1.0, Math.min(dxAvail, dyAvail) * 0.32);

  // Bucket centres by (color, alpha-bin). Single pass over popMax.
  // Key: `${color}|${alphaBin}` keeps batching simple.
  const buckets = new Map<string, { color: string; alpha: number; pts: { x: number; y: number }[] }>();
  const pushBucket = (color: string, bin: number, x: number, y: number) => {
    const alpha = ALPHA_FROM_BIN[bin];
    const key = `${color}|${bin}`;
    let b = buckets.get(key);
    if (!b) {
      b = { color, alpha, pts: [] };
      buckets.set(key, b);
    }
    b.pts.push({ x, y });
  };

  // Centres are needed twice — once for batch draw, once for HGT lines and
  // overlay events. Storing them is O(N) memory, fine for popMax up to ~10k.
  const centres = new Float32Array(popMax * 2);
  for (let s = 0; s < popMax; s++) {
    const cx = pad + ((s % cols) + 0.5) * dxAvail;
    const cy = pad + (Math.floor(s / cols) + 0.5) * dyAvail;
    centres[s * 2] = cx;
    centres[s * 2 + 1] = cy;
    const alive = ev ? !!ev.alive[s] : false;
    const credit = ev ? ev.credit[s] : 0;
    const color = dotColor(alive, credit);
    const bin = alive ? hungerAlphaBin(credit) : 5;
    pushBucket(color, bin, cx, cy);
  }

  const usePixelMode = r <= 3.0;
  if (usePixelMode) {
    // Fast path: filled squares centred on dot. ~3× faster than arc() at
    // tiny radii and visually identical (sub-pixel circles look square anyway).
    const side = Math.max(2, Math.round(r * 2));
    const off = side / 2;
    for (const b of buckets.values()) {
      ctx.globalAlpha = b.alpha;
      ctx.fillStyle = b.color;
      for (let i = 0; i < b.pts.length; i++) {
        const p = b.pts[i];
        ctx.fillRect(p.x - off, p.y - off, side, side);
      }
    }
    ctx.globalAlpha = 1.0;
  } else {
    // Quality path: one Path2D per (color, alpha) bucket, single fill().
    for (const b of buckets.values()) {
      const path = new Path2D();
      for (let i = 0; i < b.pts.length; i++) {
        const p = b.pts[i];
        // moveTo before arc avoids "connect-the-dots" artefact between sub-paths.
        path.moveTo(p.x + r, p.y);
        path.arc(p.x, p.y, r, 0, Math.PI * 2);
      }
      ctx.globalAlpha = b.alpha;
      ctx.fillStyle = b.color;
      ctx.fill(path);
    }
    ctx.globalAlpha = 1.0;
  }

  // Overlays — always sparse (this-window events only), so per-slot loop is fine.
  drawOverlays(ctx, popMax, centres, r, sets, selectedSlot);
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

  const cellCounts = new Map<number, number>();
  const cellSeen = new Map<number, number>();
  for (let s = 0; s < ev.alive.length; s++) {
    if (!ev.alive[s]) continue;
    const [px, py] = ev.positions[s] ?? [0, 0];
    const key = px * G + py;
    cellCounts.set(key, (cellCounts.get(key) ?? 0) + 1);
  }

  const r = Math.max(1.5, Math.min(cellW, cellH) * 0.28);
  // Spatial / slime mode: draw each agent individually.
  //
  // Why no batching here: when several agents share a cell we jitter them
  // around the cell centre by ~22% of cellW. If we then build one Path2D
  // per colour bucket and fill once, neighbouring jittered circles merge
  // into a single blob (looks like a flower / clover). Per-agent stroke +
  // fill keeps each dot visually distinct so you can tell "3 agents in
  // this cell" apart from "1 agent". Slime mode is bounded by grid_size,
  // typically <1000 living agents — per-circle draw is fine.
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
    const credit = ev.credit[s];
    const alpha = ALPHA_FROM_BIN[hungerAlphaBin(credit)];
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = dotColor(true, credit);
    ctx.fill();
    // Thin dark rim helps separate adjacent / overlapping agents.
    if (total > 1) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(15,23,42,0.85)";
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
    drawSingleOverlay(ctx, cx, cy, r, s, sets);
    if (selectedSlot === s) {
      ctx.beginPath();
      ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // HGT social lines
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

/** Iterate only the slots that have an event this window. */
function drawOverlays(
  ctx: CanvasRenderingContext2D,
  popMax: number,
  centres: Float32Array,
  r: number,
  sets: OverlaySets,
  selectedSlot: number | null
) {
  // Iterate the small overlay sets directly instead of looping all popMax slots.
  const visit = (s: number) => {
    if (s < 0 || s >= popMax) return;
    const cx = centres[s * 2];
    const cy = centres[s * 2 + 1];
    drawSingleOverlay(ctx, cx, cy, r, s, sets);
  };
  for (const s of sets.golden) visit(s);
  for (const s of sets.children) visit(s);
  for (const s of sets.parents) visit(s);
  for (const s of sets.deads) visit(s);
  if (selectedSlot != null && selectedSlot >= 0 && selectedSlot < popMax) {
    const cx = centres[selectedSlot * 2];
    const cy = centres[selectedSlot * 2 + 1];
    ctx.beginPath();
    ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function drawSingleOverlay(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  s: number,
  sets: OverlaySets
) {
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
    ctx.beginPath();
    ctx.arc(cx, cy, r + 1.2, 0, Math.PI * 2);
    ctx.strokeStyle = "#ec4899";
    ctx.lineWidth = 2;
    ctx.stroke();
  } else if (sets.parents.has(s)) {
    ctx.beginPath();
    ctx.arc(cx, cy, r + 1.2, 0, Math.PI * 2);
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

function drawHgtLines(
  ctx: CanvasRenderingContext2D,
  pairs: [number, number][],
  centres: Float32Array
) {
  if (pairs.length === 0) return;
  ctx.save();
  ctx.strokeStyle = "rgba(168, 85, 247, 0.6)";
  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 3]);
  for (const [recipient, donor] of pairs) {
    const ax = centres[recipient * 2];
    const ay = centres[recipient * 2 + 1];
    const bx = centres[donor * 2];
    const by = centres[donor * 2 + 1];
    if (Number.isNaN(ax) || Number.isNaN(bx)) continue;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  ctx.restore();
}

function pheromoneColor(t: number): string {
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
