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
} from "./types";

const J = { "Content-Type": "application/json" };

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

async function jpost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: J,
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => r.statusText);
    throw new Error(`${r.status} ${r.statusText}: ${txt}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  status: () => jget<SimStatus>("/api/status"),
  start: (cfg: SimConfig) => jpost<SimStatus>("/api/start", cfg),
  stop: () => jpost<SimStatus>("/api/stop", {}),
  inference: (req: InferenceRequest) =>
    jpost<InferenceResponse>("/api/inference", req),
  feedback: (req: FeedbackRequest) =>
    jpost<FeedbackResponse>("/api/feedback", req),
  agent: (slot: number) => jget<AgentDetail>(`/api/agent/${slot}`),
  feedbackLog: (limit = 100) =>
    jget<FeedbackLogEntry[]>(`/api/feedback-log?limit=${limit}`),
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
