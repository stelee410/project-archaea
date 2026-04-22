"""CLI entry: main simulation loop (SPEC §7–8)."""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

from . import telemetry
from .champions import save_champions
from .population import Population
from .slime import SlimeConfig
from .task import DEFAULT_TASK, VALID_TASKS
from .telemetry import (
    StdoutRollingTable,
    dump_t2h_top10_csv,
    dump_t2h_weights,
    dump_top10_csv,
    monoculture_metric,
    plot_fitness_curve,
    plot_weight_layers_hist,
    save_checkpoint,
    write_log_line,
)


WINDOW_S = 0.5
T_2H_S = 7200.0
STAGNATION_S = 7200.0
CHECKPOINT_INTERVAL_S = 600.0


def run_experiment(
    seed: int,
    duration_s: float,
    pop_max: int,
    log_path: str | None,
    n_initial: int | None = None,
    visual: bool = False,
    visual_every: int = 1,
    visual_tail: float = 600.0,
    plain_stdout: bool = False,
    console_rows: int = 100,
    carrying_capacity: int | None = None,
    budget_mode: str = "none",
    slime: SlimeConfig | None = None,
    calibration_lambda: float = 0.0,
    synapse_gain: float = 1.0,
    task: str = DEFAULT_TASK,
) -> int:
    """
    Returns process exit code: 0 on success halt, 1 on failure/pathology.
    """
    dash = None
    rng = np.random.default_rng(seed)
    pop = Population(
        pop_max,
        rng,
        n_initial=n_initial,
        carrying_capacity=carrying_capacity,
        budget_mode=budget_mode,
        slime=slime,
        calibration_lambda=calibration_lambda,
        synapse_gain=synapse_gain,
        task=task,
    )

    if visual:
        import matplotlib.pyplot as plt

        plt.ion()
        from .visualize import LiveDashboard

        dash = LiveDashboard(pop, tail_sim_s=float(visual_tail), interactive=True)

    telemetry.ensure_dirs()
    log_fp = open(log_path, "w", encoding="utf-8") if log_path else None

    stdout_table: StdoutRollingTable | None = None
    if sys.stdout.isatty() and not plain_stdout:
        stdout_table = StdoutRollingTable(sys.stdout, maxlen=int(console_rows))

    t_sim = 0.0
    t_first_success: float | None = None
    r_hist_max: deque[float] = deque(maxlen=int(STAGNATION_S / WINDOW_S) * 2 + 16)

    t_log: list[float] = []
    rmax_log: list[float] = []
    rmean_log: list[float] = []

    last_checkpoint_block = -1
    max_pop_seen = pop.n_living()
    EXTINCTION_THRESHOLD = 10

    def log_fields(row: list) -> None:
        if log_fp is not None:
            write_log_line(log_fp, row)
        if stdout_table is not None:
            stdout_table.push(row)
        else:
            write_log_line(sys.stdout, row)

    try:
        w_target = int(duration_s / WINDOW_S)
        for w in range(w_target):
            info = pop.step_window()
            births = info["births"]
            deaths = info["deaths"]
            r_max = info["r_max"]
            r_mean = info["r_mean"]
            sigma = info["sigma"]

            t_sim = (w + 1) * WINDOW_S
            r_hist_max.append(r_max)
            t_log.append(t_sim)
            rmax_log.append(r_max)
            rmean_log.append(r_mean)

            row = [
                f"{t_sim:.3f}",
                pop.n_living(),
                births,
                deaths,
                f"{r_max:.6f}",
                f"{r_mean:.6f}",
                f"{pop.credit_mean():.6f}",
                f"{pop.credit_gini():.6f}",
                f"{pop.weight_diversity_metric():.6f}",
                f"{sigma:.6f}",
                f"{float(info.get('budget_pressure', 0.0)):.6f}",
                f"{float(info.get('pheromone_max', 0.0)):.4f}",
                int(info.get('hgt_count', 0)),
                int(info.get('migrations', 0)),
            ]
            log_fields(row)

            if dash is not None:
                dash.append_point(t_sim, info, pop)
                ve = max(1, int(visual_every))
                if w % ve == 0 or (w + 1) == w_target:
                    dash.draw()

            if pop.any_success() and t_first_success is None:
                t_first_success = t_sim
                try:
                    save_champions(
                        pop,
                        "diagnostics/champions_first.npz",
                        top_k=10,
                        t_sim=t_sim,
                        seed=seed,
                    )
                except Exception as e:
                    print(f"warn: failed to save first-success champions: {e}", file=sys.stderr)

            ck_block = int(t_sim // CHECKPOINT_INTERVAL_S)
            if ck_block > last_checkpoint_block and ck_block > 0:
                last_checkpoint_block = ck_block
                p = f"checkpoints/t_{int(ck_block * CHECKPOINT_INTERVAL_S)}.npz"
                save_checkpoint(p, pop, rng)

            if (w + 1) == int(round(T_2H_S / WINDOW_S)):
                if pop.max_fitness() < 0.3:
                    telemetry.plot_fitness_curve(t_log, rmax_log, rmean_log, "diagnostics/fitness_curve.png")
                    plot_weight_layers_hist(pop, "diagnostics/weight_hist.png")
                    dump_t2h_weights(pop, "diagnostics/t2h_weights.png")
                    dump_t2h_top10_csv(pop, "diagnostics/t2h_top10.csv")
                    dump_top10_csv(pop, "diagnostics/top10.csv")
                    if log_fp:
                        log_fp.close()
                    print("HALT: failure at T+2h (max fitness < 0.3)", file=sys.stderr)
                    return 1

            n_now = pop.n_living()
            if n_now > max_pop_seen:
                max_pop_seen = n_now
            if max_pop_seen >= EXTINCTION_THRESHOLD and n_now < EXTINCTION_THRESHOLD:
                telemetry.plot_fitness_curve(t_log, rmax_log, rmean_log, "diagnostics/fitness_curve.png")
                plot_weight_layers_hist(pop, "diagnostics/weight_hist.png")
                dump_top10_csv(pop, "diagnostics/top10.csv")
                if log_fp:
                    log_fp.close()
                print("HALT: extinction", file=sys.stderr)
                return 1
            if n_now == 0:
                telemetry.plot_fitness_curve(t_log, rmax_log, rmean_log, "diagnostics/fitness_curve.png")
                if log_fp:
                    log_fp.close()
                print("HALT: extinction (n_living=0)", file=sys.stderr)
                return 1

            mono = monoculture_metric(pop)
            if mono < 0.01 and pop.n_living() >= 100:
                telemetry.plot_fitness_curve(t_log, rmax_log, rmean_log, "diagnostics/fitness_curve.png")
                plot_weight_layers_hist(pop, "diagnostics/weight_hist.png")
                dump_top10_csv(pop, "diagnostics/top10.csv")
                if log_fp:
                    log_fp.close()
                print(f"HALT: monoculture (metric={mono:.6f})", file=sys.stderr)
                return 1

            need = int(STAGNATION_S / WINDOW_S) * 2
            if len(r_hist_max) >= need:
                a = int(STAGNATION_S / WINDOW_S)
                seq = list(r_hist_max)
                recent = max(seq[-a:])
                prev = max(seq[-2 * a : -a])
                if recent - prev < 0.05 and recent < 0.7:
                    telemetry.plot_fitness_curve(t_log, rmax_log, rmean_log, "diagnostics/fitness_curve.png")
                    plot_weight_layers_hist(pop, "diagnostics/weight_hist.png")
                    dump_top10_csv(pop, "diagnostics/top10.csv")
                    if log_fp:
                        log_fp.close()
                    print("HALT: stagnation", file=sys.stderr)
                    return 1

            if not np.isfinite(pop.weights[pop.alive]).all() or not np.isfinite(pop.credit[pop.alive]).all():
                if log_fp:
                    log_fp.close()
                print("HALT: non-finite state", file=sys.stderr)
                return 1

        telemetry.plot_fitness_curve(t_log, rmax_log, rmean_log, "diagnostics/fitness_curve.png")
        plot_weight_layers_hist(pop, "diagnostics/weight_hist.png")
        dump_top10_csv(pop, "diagnostics/top10.csv")
        try:
            if pop.n_living() > 0:
                save_champions(
                    pop,
                    "diagnostics/champions_final.npz",
                    top_k=10,
                    t_sim=t_sim,
                    seed=seed,
                )
        except Exception as e:
            print(f"warn: failed to save final champions: {e}", file=sys.stderr)
        if log_fp:
            log_fp.close()
        if t_first_success is not None:
            print(
                f"SUCCESS recorded at t_first_success={t_first_success:.3f}s; "
                "champions saved to diagnostics/champions_first.npz and diagnostics/champions_final.npz"
            )
        else:
            print("Run completed without crossing r>=0.7 (not a contract failure).")
        return 0
    finally:
        if stdout_table is not None:
            try:
                stdout_table.finish()
            except Exception:
                pass
        if log_fp and not log_fp.closed:
            log_fp.close()
        if dash is not None:
            try:
                dash.close()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Project Archaea L1 runner")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration", type=float, default=86400.0, help="Simulated seconds")
    p.add_argument("--pop-max", type=int, default=1000)
    p.add_argument("--n-initial", type=int, default=None, help="Initial living agents (default pop-max)")
    p.add_argument("--log", type=str, default="run.log")
    p.add_argument("--visual", action="store_true", help="Open live matplotlib dashboard (dots + curves)")
    p.add_argument("--visual-every", type=int, default=1, help="Redraw dashboard every N windows")
    p.add_argument("--visual-tail", type=float, default=600.0, help="Sliding time window on curves (sim seconds)")
    p.add_argument(
        "--plain-stdout",
        action="store_true",
        help="Disable fixed-header rolling table on TTY; print one line per window",
    )
    p.add_argument(
        "--console-rows",
        type=int,
        default=100,
        help="Max data rows in the TTY rolling table (default 100)",
    )
    p.add_argument(
        "--carrying-capacity",
        type=int,
        default=0,
        help=(
            "[off-SPEC] Carrying capacity K for the shared-budget mode. "
            "Total per-window reward budget B = K * R_MAX (5/agent-window). "
            "0 disables (default; equivalent to SPEC §4.4)."
        ),
    )
    p.add_argument(
        "--budget-mode",
        type=str,
        default="none",
        choices=["none", "shared"],
        help=(
            "[off-SPEC] Reward distribution policy. "
            "'none' = SPEC §4.4 (uncoupled per-agent reward). "
            "'shared' = proportional haircut when total demand exceeds B = K*R_MAX, "
            "modelling finite resources / ecological carrying capacity."
        ),
    )
    # ── Slime-mold extension (SPEC v1.1, off-SPEC, opt-in) ──
    p.add_argument(
        "--slime-mold",
        action="store_true",
        help="[v1.1 off-SPEC] Enable cyber-slime-mold mode: spatial grid, pheromone field, HGT, chemotaxis.",
    )
    p.add_argument("--grid-size", type=int, default=16, help="[slime] G×G petri-dish grid size.")
    p.add_argument("--pheromone-decay", type=float, default=0.05, help="[slime] Per-window decay rate (0..1).")
    p.add_argument("--pheromone-diffusion", type=float, default=0.20, help="[slime] Per-window 4-neighbour diffusion (0..1).")
    p.add_argument("--pheromone-emit", type=float, default=0.5, help="[slime] Emission rate per fitness unit per window.")
    p.add_argument("--pheromone-bonus", type=float, default=0.5, help="[slime] Reward multiplier on a saturated trail (0..).")
    p.add_argument("--no-hgt", action="store_true", help="[slime] Disable horizontal gene transfer.")
    p.add_argument("--hgt-prob", type=float, default=0.02, help="[slime] Per-agent per-window HGT trigger probability.")
    p.add_argument("--hgt-blend", type=float, default=0.30, help="[slime] Donor blend fraction in HGT.")
    p.add_argument("--no-migrate", action="store_true", help="[slime] Disable chemotaxis migration.")
    p.add_argument("--migrate-prob", type=float, default=0.30, help="[slime] Per-agent per-window migration probability.")
    p.add_argument(
        "--calibration-lambda",
        type=float,
        default=0.0,
        help=(
            "[v1.2 off-SPEC] Fitness magnitude calibration penalty λ. "
            "0 = SPEC §4.1 pure Pearson r (allows compressed outputs). "
            ">0 (try 0.3–0.5) penalises mean(f_out) drifting from mean(f_in), "
            "pushing the swarm toward slope ≈ 1 (output magnitude matches input)."
        ),
    )
    p.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        choices=list(VALID_TASKS),
        help=(
            "Evolution task. 'l1' = SPEC §3.1 single-channel rate tracking. "
            "'l2v2_ctrl' = SPEC_L2_V2.0 logic gating with three-channel input "
            "(A=4 data, B=4 data, S=2 selector @ 20Hz=AND / 80Hz=NOT) and "
            "instruction-conditioned reward table."
        ),
    )
    p.add_argument(
        "--synapse-gain",
        type=float,
        default=1.0,
        help=(
            "[v1.2 off-SPEC] Output-layer synaptic gain g. "
            "1.0 = SPEC §1.1 bit-identical. >1 multiplies I_o, raising raw f_out by "
            "literally producing more output spikes (subject to LIF refractory limits). "
            "Try 2.0–4.0 if the population's raw output saturates well below f_in."
        ),
    )
    args = p.parse_args(argv)

    os.chdir(Path(__file__).resolve().parents[1])
    slime_cfg = SlimeConfig(
        enabled=bool(args.slime_mold),
        grid_size=int(args.grid_size),
        pheromone_decay=float(args.pheromone_decay),
        pheromone_diffusion=float(args.pheromone_diffusion),
        pheromone_emit=float(args.pheromone_emit),
        pheromone_bonus_k=float(args.pheromone_bonus),
        hgt_enabled=(not bool(args.no_hgt)),
        hgt_prob=float(args.hgt_prob),
        hgt_blend=float(args.hgt_blend),
        migrate_enabled=(not bool(args.no_migrate)),
        migrate_prob=float(args.migrate_prob),
    )
    return run_experiment(
        seed=args.seed,
        duration_s=args.duration,
        pop_max=args.pop_max,
        log_path=args.log,
        n_initial=args.n_initial,
        visual=bool(args.visual),
        visual_every=int(args.visual_every),
        visual_tail=float(args.visual_tail),
        plain_stdout=bool(args.plain_stdout),
        console_rows=int(args.console_rows),
        carrying_capacity=int(args.carrying_capacity) or None,
        budget_mode=str(args.budget_mode),
        slime=slime_cfg,
        calibration_lambda=float(args.calibration_lambda),
        synapse_gain=float(args.synapse_gain),
        task=str(args.task),
    )


if __name__ == "__main__":
    raise SystemExit(main())
