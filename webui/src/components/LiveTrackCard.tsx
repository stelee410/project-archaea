import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { api } from "../api";
import { usePersistentState } from "../hooks/usePersistentState";

/**
 * Live tracking panel.
 *
 * The user moves the mouse over an SVG canvas; the cursor's vertical position
 * is sampled as the input frequency f_in (Hz) and shipped to /api/inference
 * in a tight back-to-back loop. The returned f_out is appended to a rolling
 * buffer that **only ever holds the last `MAX_SAMPLES` points**. The time
 * axis auto-fits the actual range of those points, so the curve always fills
 * the chart and rendering cost is bounded regardless of how long the panel
 * has been open.
 *
 * Visual hierarchy:
 *   • input (sky blue, dashed)        — what the swarm "sees"
 *   • output base (faded green path)  — full retained history
 *   • output head (bright thick green) — last N=5 samples, for instant feedback
 *   • last sample dot                  — pulses to show liveness
 *
 * The loop is **request-paced**: the next /api/inference fires as soon as the
 * previous one resolves, so throughput auto-adapts to backend speed and we
 * never queue requests. When the cursor leaves the panel we idle (no API
 * traffic) but keep the chart on screen.
 */

/** Upper bound on retained sample points. Keeps render cost & memory flat. */
const MAX_SAMPLES = 10;

interface Sample {
  t: number; // ms since session start
  f_in: number;
  f_out: number;
  latency_ms: number;
}

interface LiveTrackForm {
  enabled: boolean;
  yMax: number;
  durationMs: number;
  warmupMs: number;
}

const DEFAULTS: LiveTrackForm = {
  enabled: false,
  yMax: 150,
  durationMs: 80,
  warmupMs: 20,
};

const HEAD_HIGHLIGHT = 5;

interface Props {
  // SPEC_L2_V3.5b — accepts the full InferenceTarget union (incl. colony /
  // and_expert / not_expert / dual_expert).  LiveTrackCard itself doesn't
  // care about the niche routing — it just passes target through to api.sweep.
  target: import("../types").InferenceTarget;
  topK: number;
  swarmRadius: number;
}

export function LiveTrackCard({ target, topK, swarmRadius }: Props) {
  const [form, setForm] = usePersistentState<LiveTrackForm>(
    "livetrack-form",
    DEFAULTS,
  );
  const formRef = useRef(form);
  formRef.current = form;
  const targetRef = useRef(target);
  targetRef.current = target;
  const topKRef = useRef(topK);
  topKRef.current = topK;
  const swarmRadiusRef = useRef(swarmRadius);
  swarmRadiusRef.current = swarmRadius;

  const [samples, setSamples] = useState<Sample[]>([]);
  const [insidePanel, setInsidePanel] = useState(false);
  const [currentFin, setCurrentFin] = useState(0);
  const currentFinRef = useRef(0);
  const insideRef = useRef(false);
  const sessionStart = useRef(performance.now());

  function updateFromMouse(clientY: number, svg: SVGSVGElement) {
    const rect = svg.getBoundingClientRect();
    const ratio = 1 - (clientY - rect.top) / rect.height;
    const yMax = formRef.current.yMax;
    const f = Math.max(0, Math.min(yMax, ratio * yMax));
    setCurrentFin(f);
    currentFinRef.current = f;
  }

  // Request-paced sampling loop.
  useEffect(() => {
    if (!form.enabled) return;
    let cancelled = false;
    let timer: number | null = null;
    const tick = async () => {
      if (cancelled) return;
      if (!insideRef.current) {
        timer = window.setTimeout(tick, 80);
        return;
      }
      const t0 = performance.now();
      const f_in = currentFinRef.current;
      try {
        const r = await api.inference({
          f_in_hz: f_in,
          target: targetRef.current,
          top_k: topKRef.current,
          duration_ms: formRef.current.durationMs,
          warmup_ms: formRef.current.warmupMs,
          swarm_radius: swarmRadiusRef.current,
        });
        const t1 = performance.now();
        if (cancelled) return;
        const tMid = (t0 + t1) / 2 - sessionStart.current;
        setSamples((prev) => {
          const sample: Sample = {
            t: tMid,
            f_in,
            f_out: r.f_out_hz,
            latency_ms: t1 - t0,
          };
          // Bounded ring: keep only the most recent MAX_SAMPLES points so
          // memory and SVG draw cost stay flat regardless of session length.
          if (prev.length < MAX_SAMPLES) return prev.concat(sample);
          const next = prev.slice(prev.length - (MAX_SAMPLES - 1));
          next.push(sample);
          return next;
        });
      } catch {
        // swallow transient errors; the loop must keep running
      }
      timer = window.setTimeout(tick, 8); // tiny yield for the event loop
    };
    tick();
    return () => {
      cancelled = true;
      if (timer != null) clearTimeout(timer);
    };
  }, [form.enabled]);

  function clearSamples() {
    setSamples([]);
    sessionStart.current = performance.now();
  }

  // ---- chart geometry ----
  const W = 760;
  const H = 300;
  const ML = 44;
  const MR = 14;
  const MT = 12;
  const MB = 28;
  const innerW = W - ML - MR;
  const innerH = H - MT - MB;
  const yMax = form.yMax;
  // X-axis auto-fits the actual time range of the retained samples (≤
  // MAX_SAMPLES). With no samples we synthesize a 1 s placeholder window so
  // the cursor f_in indicator still has somewhere to land.
  const tFirst = samples.length > 0 ? samples[0].t : 0;
  const tLast =
    samples.length > 0 ? samples[samples.length - 1].t : tFirst + 1000;
  const span = Math.max(500, tLast - tFirst);
  const xPad = span * 0.04;
  const xLeft = tFirst - xPad;
  const xRight = tLast + xPad;
  const xScale = (t: number) =>
    ML + innerW * ((t - xLeft) / (xRight - xLeft));
  const yScale = (v: number) =>
    MT + innerH * (1 - Math.max(0, Math.min(1, v / Math.max(1, yMax))));

  const inputPts: [number, number][] = samples.map((s) => [
    xScale(s.t),
    yScale(s.f_in),
  ]);
  const outputPts: [number, number][] = samples.map((s) => [
    xScale(s.t),
    yScale(s.f_out),
  ]);
  const headPts = outputPts.slice(-HEAD_HIGHLIGHT);
  const lastSample = samples.length > 0 ? samples[samples.length - 1] : null;
  const meanLatency =
    samples.length > 0
      ? samples.reduce((a, b) => a + b.latency_ms, 0) / samples.length
      : 0;
  const spanSec = (tLast - tFirst) / 1000;

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-5">
      <h2 className="text-base font-semibold mb-1">📡 实时跟随（鼠标驱动）</h2>
      <p className="text-xs text-slate-400 mb-3">
        在画板上<b>移动鼠标</b>：纵向位置 → 输入 <span className="text-sky-300">f_in</span>，
        群体在线推理产生输出 <span className="text-emerald-300">f_out</span>。
        画面只保留 <b>最近 {MAX_SAMPLES} 个采样点</b>（防止长时间运行越来越卡），
        最近 {HEAD_HIGHLIGHT} 个输出点用粗高亮平滑相连，最新一点呼吸闪动。
        采样节奏由后端响应速度决定（请求一返回就立刻发下一个）。
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
        <div>
          <label className="block text-xs text-slate-300 mb-1">启用</label>
          <button
            onClick={() => setForm((s) => ({ ...s, enabled: !s.enabled }))}
            className={clsx(
              "w-full px-3 py-1.5 rounded text-xs font-medium transition-colors",
              form.enabled
                ? "bg-emerald-500/30 ring-1 ring-emerald-500/40 text-emerald-200"
                : "bg-slate-800 text-slate-300 hover:bg-slate-700",
            )}
          >
            {form.enabled ? "🟢 ON · 鼠标驱动" : "⚫ OFF"}
          </button>
        </div>
        <Field label="Y 轴最大 (Hz)" hint="物理上限约 150 Hz">
          <input
            type="number"
            min={10}
            max={500}
            value={form.yMax}
            onChange={(e) =>
              setForm((s) => ({ ...s, yMax: Number(e.target.value) }))
            }
            className="num-input"
          />
        </Field>
        <Field label="duration_ms" hint="单次推理脉冲时长，越短跟随越快但越抖">
          <input
            type="number"
            min={20}
            max={500}
            step={10}
            value={form.durationMs}
            onChange={(e) =>
              setForm((s) => ({ ...s, durationMs: Number(e.target.value) }))
            }
            className="num-input"
          />
        </Field>
        <Field label="warmup_ms" hint="冷启动膜电位预热">
          <input
            type="number"
            min={0}
            max={200}
            step={10}
            value={form.warmupMs}
            onChange={(e) =>
              setForm((s) => ({ ...s, warmupMs: Number(e.target.value) }))
            }
            className="num-input"
          />
        </Field>
      </div>

      <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-[300px] cursor-crosshair touch-none select-none"
          preserveAspectRatio="none"
          onMouseMove={(e) => {
            updateFromMouse(e.clientY, e.currentTarget);
            insideRef.current = true;
            setInsidePanel(true);
          }}
          onMouseEnter={() => {
            insideRef.current = true;
            setInsidePanel(true);
          }}
          onMouseLeave={() => {
            insideRef.current = false;
            setInsidePanel(false);
          }}
        >
          <defs>
            <linearGradient id="lt-output-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#34d399" stopOpacity="0.05" />
              <stop offset="70%" stopColor="#34d399" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#34d399" stopOpacity="0.65" />
            </linearGradient>
            <linearGradient id="lt-input-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.15" />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.7" />
            </linearGradient>
          </defs>

          <rect
            x={ML}
            y={MT}
            width={innerW}
            height={innerH}
            fill="#020617"
            stroke="#1e293b"
          />
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line
                x1={ML}
                x2={ML + innerW}
                y1={MT + innerH * (1 - t)}
                y2={MT + innerH * (1 - t)}
                stroke="#1e293b"
                strokeDasharray="2 4"
              />
              <text
                x={ML - 4}
                y={MT + innerH * (1 - t) + 3}
                fontSize="9"
                fill="#64748b"
                textAnchor="end"
              >
                {Math.round(yMax * t)}
              </text>
            </g>
          ))}
          <text
            x={ML + innerW / 2}
            y={H - 8}
            fontSize="10"
            fill="#64748b"
            textAnchor="middle"
          >
            最近 {samples.length}/{MAX_SAMPLES} 个采样点 · 跨度 ≈ {spanSec.toFixed(1)}s
          </text>

          {insidePanel && (
            <>
              <line
                x1={ML}
                x2={ML + innerW}
                y1={yScale(currentFin)}
                y2={yScale(currentFin)}
                stroke="#38bdf8"
                strokeOpacity="0.35"
                strokeDasharray="3 4"
                pointerEvents="none"
              />
              <text
                x={ML + innerW - 4}
                y={Math.max(MT + 12, yScale(currentFin) - 4)}
                fontSize="10"
                fill="#7dd3fc"
                textAnchor="end"
                pointerEvents="none"
              >
                f_in = {currentFin.toFixed(0)} Hz
              </text>
            </>
          )}

          {inputPts.length >= 2 && (
            <path
              d={smoothPath(inputPts)}
              fill="none"
              stroke="url(#lt-input-grad)"
              strokeWidth="1.5"
              strokeDasharray="4 3"
              pointerEvents="none"
            />
          )}
          {outputPts.length >= 2 && (
            <path
              d={smoothPath(outputPts)}
              fill="none"
              stroke="url(#lt-output-grad)"
              strokeWidth="2"
              pointerEvents="none"
            />
          )}
          {headPts.length >= 2 && (
            <path
              d={smoothPath(headPts)}
              fill="none"
              stroke="#34d399"
              strokeWidth="3"
              strokeLinecap="round"
              pointerEvents="none"
            />
          )}
          {headPts.map(([x, y], i) => {
            const isLast = i === headPts.length - 1;
            const fade = (i + 1) / headPts.length;
            return (
              <circle
                key={i}
                cx={x}
                cy={y}
                r={isLast ? 5 : 2.8}
                fill="#34d399"
                stroke="#022c22"
                strokeWidth={isLast ? 1 : 0.5}
                opacity={0.35 + 0.65 * fade}
                pointerEvents="none"
              >
                {isLast && (
                  <animate
                    attributeName="r"
                    values="5;8;5"
                    dur="1.2s"
                    repeatCount="indefinite"
                  />
                )}
              </circle>
            );
          })}
        </svg>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-400 mt-2 px-1">
          <span>
            缓冲 <b className="text-slate-200">{samples.length}</b>/{MAX_SAMPLES}{" "}
            点
          </span>
          {lastSample && (
            <>
              <span>
                当前 f_in <b className="text-sky-300">{currentFin.toFixed(0)}</b> Hz →
                最新 f_out{" "}
                <b className="text-emerald-300">
                  {lastSample.f_out.toFixed(1)}
                </b>{" "}
                Hz
              </span>
              <span>
                推理延迟 ≈{" "}
                <b className="text-slate-200">{meanLatency.toFixed(0)}</b> ms /
                次（缓冲内平均）
              </span>
            </>
          )}
          <span className="ml-auto inline-flex items-center gap-2">
            target=<code className="px-1 bg-slate-800 rounded">{target}</code>
            {target === "ensemble" && <span>top_k={topK}</span>}
            {target === "swarm" && <span>radius={swarmRadius}</span>}
            <button
              onClick={clearSamples}
              className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
            >
              清空
            </button>
          </span>
          {samples.length === 0 && form.enabled && (
            <span className="basis-full text-slate-500">
              把鼠标移进画板，开始喂频率给种群。
            </span>
          )}
          {!form.enabled && (
            <span className="basis-full text-slate-500">
              点上方「ON」开关启动。
            </span>
          )}
        </div>
      </div>

      <div className="text-[10px] text-slate-500 mt-2 leading-snug">
        蓝虚线 = 你给的输入；绿淡线 = 缓冲内全部 {MAX_SAMPLES}{" "}
        个输出；绿粗线 = 最近 {HEAD_HIGHLIGHT}{" "}
        个输出（呼吸点是最新一点）。曲线用 Catmull-Rom 样条平滑。X 轴随采样
        自动适配，<b>缓冲只保留最近 {MAX_SAMPLES} 个点</b>——长时间运行也不会变卡。
        注意输入与输出之间会有<b>固有延迟</b>≈ warmup + duration
        毫秒，所以输出永远会"晚一拍"——这正是真正的脉冲神经元在做物理推理的标志。
      </div>
    </div>
  );
}

/** Catmull-Rom centripetal spline → cubic Bezier (closed form, fixed tension). */
function smoothPath(pts: [number, number][]): string {
  if (pts.length < 2) return "";
  if (pts.length === 2)
    return `M ${pts[0][0]} ${pts[0][1]} L ${pts[1][0]} ${pts[1][1]}`;
  let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] ?? pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] ?? p2;
    const c1x = p1[0] + (p2[0] - p0[0]) / 6;
    const c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6;
    const c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
  }
  return d;
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="block text-xs text-slate-300 mb-1">{label}</label>
      {children}
      {hint && (
        <div className="text-[10px] text-slate-500 mt-1 leading-snug">
          {hint}
        </div>
      )}
    </div>
  );
}
