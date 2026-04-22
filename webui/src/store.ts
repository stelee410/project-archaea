import { create } from "zustand";
import type { ServerEvent, SimStatus, TelemetryEvent } from "./types";

const HISTORY_LIMIT = 1200; // ~10min @ 2Hz窗 or ~60s @ 20Hz窗

export interface ChartPoint {
  t: number;
  r_max: number;
  r_mean: number;
  pop: number;
  credit: number;
  sigma: number;
  budget: number;
  births: number;
  deaths: number;
  // Aggregate "vitality" = pop * credit_mean (sum of all live Credit, ECG-style)
  vitality: number;
  // L2v2 — undefined / 0 in L1 runs
  acc_and?: number;
  acc_not?: number;
  both_pass_pct?: number;
  logic_diversity?: number;
  consensus_acc?: number;
}

interface State {
  ws: WebSocket | null;
  wsState: "idle" | "connecting" | "open" | "closed" | "error";
  status: SimStatus | null;
  latest: TelemetryEvent | null;
  history: ChartPoint[];
  // L2v2 cumulative survival counter (since current run started)
  totalBorn: number;
  totalDied: number;

  setStatus: (s: SimStatus) => void;
  ingest: (ev: ServerEvent) => void;
  setWs: (ws: WebSocket | null, state: State["wsState"]) => void;
  resetHistory: () => void;
}

export const useStore = create<State>((set) => ({
  ws: null,
  wsState: "idle",
  status: null,
  latest: null,
  history: [],
  totalBorn: 0,
  totalDied: 0,

  setStatus: (s) => set({ status: s }),
  setWs: (ws, wsState) => set({ ws, wsState }),
  resetHistory: () => set({ history: [], latest: null, totalBorn: 0, totalDied: 0 }),

  ingest: (ev) =>
    set((st) => {
      if (ev.type === "hello") {
        return { status: ev.status };
      }
      if (ev.type === "error") {
        return {};
      }
      const t = ev.t_sim;
      const vitality = ev.pop_size * ev.credit_mean;
      const point: ChartPoint = {
        t,
        r_max: ev.r_max,
        r_mean: ev.r_mean,
        pop: ev.pop_size,
        credit: ev.credit_mean,
        sigma: ev.sigma,
        budget: ev.budget_pressure,
        births: ev.births,
        deaths: ev.deaths,
        vitality,
        acc_and: ev.acc_and_pop,
        acc_not: ev.acc_not_pop,
        both_pass_pct: ev.both_pass_pct,
        logic_diversity: ev.logic_diversity,
        consensus_acc: ev.consensus_acc,
      };
      const next = st.history.length >= HISTORY_LIMIT
        ? [...st.history.slice(st.history.length - HISTORY_LIMIT + 1), point]
        : [...st.history, point];
      // status 是 HTTP /api/status 一次性快照，原本只在 mount/WS-open/start 三个时刻
      // 被刷新 → t_sim 等动态字段会一直停在启动那一刻的值。
      // 每帧 telemetry 来时顺手把这几个动态字段回写到 status，让所有读 status.t_sim
      // 的组件（顶栏、SimHealthBanner 等）自动跟上 latest 的进度。
      // last_event 也一并同步，方便那些只拿 status 的旧代码路径。
      const syncedStatus = st.status
        ? {
            ...st.status,
            running: true,
            t_sim: ev.t_sim,
            n_living: ev.pop_size,
            pop_max: ev.pop_max,
            last_event: ev,
          }
        : st.status;
      return {
        latest: ev,
        history: next,
        totalBorn: st.totalBorn + ev.births,
        totalDied: st.totalDied + ev.deaths,
        status: syncedStatus,
      };
    }),
}));
