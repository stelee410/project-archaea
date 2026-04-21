export type BudgetMode = "none" | "shared";

export interface SimConfig {
  seed: number;
  pop_max: number;
  n_initial: number | null;
  carrying_capacity: number | null;
  budget_mode: BudgetMode;
  target_speed_hz: number;
  // Slime-mold extension (SPEC v1.1, off by default)
  slime_mold: boolean;
  grid_size: number;
  pheromone_decay: number;
  pheromone_diffusion: number;
  pheromone_emit: number;
  pheromone_bonus_k: number;
  hgt_enabled: boolean;
  hgt_prob: number;
  hgt_blend: number;
  migrate_enabled: boolean;
  migrate_prob: number;
  calibration_lambda: number;
  synapse_gain: number;
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
  // Slime-mold telemetry (only meaningful when slime_enabled=true)
  slime_enabled: boolean;
  grid_size: number;
  positions: number[][]; // length pop_max, each [x, y]
  pheromone: number[][]; // grid_size × grid_size
  pheromone_max: number;
  pheromone_mean: number;
  hgt_count: number;
  migrations: number;
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
  target: "best" | "ensemble" | "random" | "swarm";
  top_k: number;
  duration_ms: number;
  warmup_ms: number;
  swarm_radius?: number;
}

export interface InferenceAgentResult {
  slot: number;
  fitness: number | null;
  f_out_hz: number;
}

export interface InferenceResponseExtras {
  synapse_gain?: number;
}

export interface InferenceResponse extends InferenceResponseExtras {
  f_in_hz: number;
  f_out_hz: number;
  target: string;
  duration_ms: number;
  warmup_ms: number;
  agents: InferenceAgentResult[];
  swarm_hotspot?: [number, number] | null;
  swarm_radius_used?: number | null;
  swarm_size?: number | null;
  swarm_degraded?: string | null;
}

export interface SweepRequest {
  f_in_min: number;
  f_in_max: number;
  n_points: number;
  target: "best" | "ensemble" | "random" | "swarm";
  top_k: number;
  duration_ms: number;
  warmup_ms: number;
  swarm_radius?: number;
  repeats?: number;
  f_in_seq?: number[];
  calibrate?: boolean;
}

export interface SweepPoint {
  f_in_hz: number;
  f_out_hz_mean: number;
  f_out_hz_std: number;
  f_out_hz_per_repeat: number[];
  n_agents: number;
  f_out_hz_calibrated?: number;
}

export interface CalibrationInfo {
  applied: boolean;
  a: number | null;
  b: number | null;
  skipped_reason: string | null;
}

export interface SweepResponse {
  target: string;
  n_points: number;
  repeats: number;
  duration_ms: number;
  warmup_ms: number;
  f_in_min: number;
  f_in_max: number;
  points: SweepPoint[];
  swarm_hotspot?: [number, number] | null;
  swarm_radius_used?: number | null;
  swarm_size_first?: number | null;
  swarm_degraded?: string | null;
  calibration?: CalibrationInfo;
  synapse_gain?: number;
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
  position: [number, number] | null;
  local_pheromone: number | null;
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
