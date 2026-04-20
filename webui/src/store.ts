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
}

interface State {
  ws: WebSocket | null;
  wsState: "idle" | "connecting" | "open" | "closed" | "error";
  status: SimStatus | null;
  latest: TelemetryEvent | null;
  history: ChartPoint[];

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

  setStatus: (s) => set({ status: s }),
  setWs: (ws, wsState) => set({ ws, wsState }),
  resetHistory: () => set({ history: [], latest: null }),

  ingest: (ev) =>
    set((st) => {
      if (ev.type === "hello") {
        return { status: ev.status };
      }
      if (ev.type === "error") {
        return {};
      }
      const t = ev.t_sim;
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
      };
      const next = st.history.length >= HISTORY_LIMIT
        ? [...st.history.slice(st.history.length - HISTORY_LIMIT + 1), point]
        : [...st.history, point];
      return { latest: ev, history: next };
    }),
}));
