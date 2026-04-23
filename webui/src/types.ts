export type BudgetMode = "none" | "shared";

// SPEC_L2_V2.0 — evolution task selector
export type SimTask = "l1" | "l2v2_ctrl";

// SPEC_L2_V2.0 §0 — environment-shaping presets ("塑造环境而不是改变规则").
// Backend registry: archaea.oracle.TASK_DIFFICULTY_PRESETS — keep in sync.
export type TaskDifficulty =
  | "uniform"
  | "balanced"
  | "hard"
  | "extreme"
  // SPEC_L2_V3.0 §2.4 — single-gate specialist dishes for the admixture experiment
  | "and_only"
  | "not_only";

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
  task: SimTask;
  // Environment-shaping difficulty (only used for l2v2_ctrl task).
  task_difficulty: TaskDifficulty;
  // SPEC_L2_V3.0 §1.3 — admixture experiment (杂交皿).
  // When founders is non-empty, the initial slots are filled by sampling each
  // strain's living gene pool with the given fraction (sum ≤ 1; remainder is
  // random as usual). admixture_window_s > 0 boosts hgt_prob ×multiplier for
  // the first N sim-seconds to model two cultures meeting in a fresh dish.
  founders?: FounderSpec[] | null;
  admixture_window_s?: number;
  admixture_hgt_multiplier?: number;
}

// SPEC_L2_V3.0 §1.3 — one entry in the founders list.
export interface FounderSpec {
  strain_id: string;
  fraction: number; // (0, 1]
}

// SPEC_L2_V3.0 §1.1 — saved population snapshot, the unit of admixture.
export interface StrainMeta {
  id: string;
  name: string;
  task: SimTask;
  n_agents: number;
  t_sim: number;
  source_seed: number;
  source_difficulty: TaskDifficulty | null;
  acc_and_pop_at_save: number | null;
  acc_not_pop_at_save: number | null;
  fitness_mean: number;
  fitness_max: number;
  note: string;
  created_at: string;
  spec_version: string;
}

// 6-row truth-table accuracies — see archaea.population._row_bucket
export interface RowAccuracies {
  and_00: number;
  and_01: number;
  and_10: number;
  and_11: number;
  not_a0: number;
  not_a1: number;
}

// SPEC_L2_V2.0 §2.2 — one window's stimulus + ground truth
export interface OracleSnapshot {
  mode: 0 | 1;            // 0 = AND, 1 = NOT
  mode_name: "AND" | "NOT";
  bit_a: 0 | 1;
  bit_b: 0 | 1;
  f_a_hz: number;
  f_b_hz: number;
  f_s_hz: number;
  target_bit: 0 | 1;
  reward_correct: number;
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
  hgt_pairs?: [number, number][];
  migrations: number;
  // Per-slot reward this window (length pop_max). 0 for non-alive / no reward.
  reward?: number[];
  credit_delta?: number[];
  // SPEC_L2_V2.0 (only meaningful when task = "l2v2_ctrl")
  task?: SimTask;
  oracle?: OracleSnapshot | null;
  consensus_bit?: 0 | 1 | null;
  consensus_acc?: number;
  acc_and_pop?: number;
  acc_not_pop?: number;
  both_pass_pct?: number;
  logic_diversity?: number;
  // v2.3.1 — row-specific (target=1) accuracies — the "真学会" gauges.
  // Silent agents are pinned at 0 here even when混合 acc_and_pop sits at the silent ceiling.
  acc_and_11_pop?: number;
  acc_not_0_pop?: number;
  row_acc?: RowAccuracies | null;
  row_n?: RowAccuracies | null;
  task_difficulty?: TaskDifficulty | null;
  // SPEC_L2_V3.0 §1.3 — admixture telemetry.
  admixture_active?: boolean;
  admixture_window_s?: number;
  admixture_hgt_multiplier?: number;
  eff_hgt_prob?: number;
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
  // SPEC_L2_V2.0 — optional second / selector channels.
  f_b_hz?: number;
  f_s_hz?: number;
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
  f_b_hz?: number | null;
  f_s_hz?: number | null;
  task?: SimTask;
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
