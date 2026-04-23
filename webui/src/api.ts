import type {
  AgentDetail,
  FeedbackLogEntry,
  FeedbackRequest,
  FeedbackResponse,
  InferenceRequest,
  InferenceResponse,
  ServerEvent,
  SimConfig,
  SimStatus,
  StrainMeta,
  SweepRequest,
  SweepResponse,
} from "./types";

const J = { "Content-Type": "application/json" };

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

async function jdel<T>(path: string): Promise<T> {
  const r = await fetch(path, { method: "DELETE" });
  if (!r.ok) {
    const txt = await r.text().catch(() => r.statusText);
    throw new Error(`${r.status} ${r.statusText}: ${txt}`);
  }
  return r.json() as Promise<T>;
}

async function jpost<T>(
  path: string,
  body: unknown,
  opts?: { timeoutMs?: number }
): Promise<T> {
  // Default 30s — covers worst-case batched inference even with target=swarm
  // hitting hundreds of agents. /api/start /stop are short, /api/sweep can be
  // longer; callers can override via opts.timeoutMs.
  const timeoutMs = opts?.timeoutMs ?? 30000;
  const ctrl = new AbortController();
  const tid = window.setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(path, {
      method: "POST",
      headers: J,
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!r.ok) {
      const txt = await r.text().catch(() => r.statusText);
      throw new Error(`${r.status} ${r.statusText}: ${txt}`);
    }
    return r.json() as Promise<T>;
  } catch (e) {
    if ((e as DOMException)?.name === "AbortError") {
      throw new Error(
        `请求超时（${(timeoutMs / 1000).toFixed(0)}s）：后端可能在算大批 agent，` +
          `或仿真线程被卡住。可以试着减小 top_k / swarm_radius，或刷新页面。`
      );
    }
    throw e;
  } finally {
    window.clearTimeout(tid);
  }
}

export const api = {
  status: () => jget<SimStatus>("/api/status"),
  start: (cfg: SimConfig) => jpost<SimStatus>("/api/start", cfg),
  stop: () => jpost<SimStatus>("/api/stop", {}),
  inference: (req: InferenceRequest) =>
    jpost<InferenceResponse>("/api/inference", req),
  sweep: (req: SweepRequest) =>
    jpost<SweepResponse>("/api/sweep", req, { timeoutMs: 120000 }),
  setCalibrationLambda: (calibration_lambda: number) =>
    jpost<{ calibration_lambda: number }>("/api/calibration-lambda", {
      calibration_lambda,
    }),
  setSynapseGain: (synapse_gain: number) =>
    jpost<{ synapse_gain: number }>("/api/synapse-gain", { synapse_gain }),
  feedback: (req: FeedbackRequest) =>
    jpost<FeedbackResponse>("/api/feedback", req),
  agent: (slot: number) => jget<AgentDetail>(`/api/agent/${slot}`),
  feedbackLog: (limit = 100) =>
    jget<FeedbackLogEntry[]>(`/api/feedback-log?limit=${limit}`),
  // SPEC_L2_V3.0 — strains (菌株库)
  listStrains: () => jget<StrainMeta[]>("/api/strains"),
  saveStrain: (name: string, note = "") =>
    jpost<StrainMeta>("/api/strains/save", { name, note }),
  deleteStrain: (id: string) =>
    jdel<{ removed: boolean; id: string }>(`/api/strains/${encodeURIComponent(id)}`),
};

export type SubscribeOpts = {
  onMessage: (ev: ServerEvent) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
};

export function openTelemetry(opts: SubscribeOpts): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${window.location.host}/ws/telemetry`;
  const ws = new WebSocket(url);
  ws.onopen = () => opts.onOpen?.();
  ws.onclose = () => opts.onClose?.();
  ws.onerror = (e) => opts.onError?.(e);
  ws.onmessage = (m) => {
    try {
      const data = JSON.parse(m.data) as ServerEvent;
      opts.onMessage(data);
    } catch {
      /* ignore */
    }
  };
  return ws;
}
