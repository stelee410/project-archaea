export type BudgetMode = "none" | "shared";

export interface SimConfig {
  seed: number;
  pop_max: number;
  n_initial: number | null;
  carrying_capacity: number | null;
  budget_mode: BudgetMode;
  target_speed_hz: number;
}

export interface TelemetryEvent {
  type: "telemetry";
  t_sim: number;
  pop_size: number;
  pop_max: number;
  births: number;
  deaths: number;
  r_max: number;
  r_mean: number;
  credit_mean: number;
  credit_gini: number;
  weight_std: number;
  sigma: number;
  budget_pressure: number;
  alive: number[];
  credit: number[];
  fitness: number[];
  dead_slots: number[];
  repro_parent_slots: number[];
  repro_child_slots: number[];
}

export interface HelloEvent {
  type: "hello";
  status: SimStatus;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type ServerEvent = TelemetryEvent | HelloEvent | ErrorEvent;

export interface SimStatus {
  running: boolean;
  config: SimConfig | null;
  t_sim: number;
  n_living: number;
  pop_max: number;
  subscribers: number;
  last_event: TelemetryEvent | null;
  feedback_count: number;
}

export interface InferenceRequest {
  f_in_hz: number;
  target: "best" | "ensemble" | "random";
  top_k: number;
  duration_ms: number;
  warmup_ms: number;
}

export interface InferenceAgentResult {
  slot: number;
  fitness: number | null;
  f_out_hz: number;
}

export interface InferenceResponse {
  f_in_hz: number;
  f_out_hz: number;
  target: string;
  duration_ms: number;
  warmup_ms: number;
  agents: InferenceAgentResult[];
}

export interface FeedbackRequest {
  slots: number[];
  delta_per_slot: number;
  label: "correct" | "wrong" | "manual";
  f_in_hz?: number;
  f_out_hz?: number;
}

export interface FeedbackResultRow {
  slot: number;
  alive: boolean;
  applied: number;
  credit: number;
  killed: boolean;
}

export interface FeedbackResponse {
  results: FeedbackResultRow[];
}

export interface Edge {
  src: string;
  dst: string;
  w: number;
}

export interface Topology {
  input_nodes: string[];
  hidden_nodes: string[];
  output_nodes: string[];
  edges_ih: Edge[];
  edges_ho: Edge[];
}

export interface AgentDetail {
  slot: number;
  alive: boolean;
  credit: number;
  fitness: number | null;
  topology: Topology;
}

export interface FeedbackLogEntry {
  t_sim: number;
  label: string;
  delta_per_slot: number;
  slots: number[];
  f_in_hz: number | null;
  f_out_hz: number | null;
  wall: number;
}
