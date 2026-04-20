"""
CLI «service» for a champions archive.

Examples:

  # Show metadata + per-agent fitness
  python -m archaea.serve --champions diagnostics/champions.npz info

  # Single rate query (best agent)
  python -m archaea.serve --champions diagnostics/champions.npz rate --f-in 50 --duration 500

  # Sweep f_in 10..100 Hz, print TSV (f_in, f_out_best, f_out_mean)
  python -m archaea.serve --champions diagnostics/champions.npz sweep --start 10 --stop 100 --step 10
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .champions import ChampionEnsemble


def _cmd_info(ens: ChampionEnsemble) -> int:
    print(json.dumps(ens.info(), indent=2, ensure_ascii=False))
    return 0


def _cmd_rate(ens: ChampionEnsemble, f_in: float, duration_ms: float, seed: int | None, agent: int | None) -> int:
    rates = ens.rates_for(f_in, duration_ms, seed=seed, warmup_ms=100.0)
    if agent is None:
        idx = ens.best_index()
    else:
        idx = int(agent)
    print(f"f_in={f_in:.3f} Hz  duration={duration_ms:.1f} ms  agent={idx} (fitness={ens.fitness[idx]:.4f})")
    print(f"  f_out          = {rates[idx]:.3f} Hz")
    print(f"  f_out (mean K) = {rates.mean():.3f} Hz")
    print(f"  f_out (max K)  = {rates.max():.3f} Hz")
    return 0


def _cmd_sweep(
    ens: ChampionEnsemble,
    start: float,
    stop: float,
    step: float,
    duration_ms: float,
    seed: int | None,
    agent: int | None,
) -> int:
    grid = np.arange(start, stop + 1e-9, step, dtype=np.float64)
    print("f_in_hz\tf_out_best_hz\tf_out_mean_hz\tf_out_max_hz")
    for i, fi in enumerate(grid.tolist()):
        s = None if seed is None else seed + i
        rates = ens.rates_for(float(fi), duration_ms, seed=s, warmup_ms=100.0)
        idx = ens.best_index() if agent is None else int(agent)
        print(f"{fi:.3f}\t{rates[idx]:.3f}\t{rates.mean():.3f}\t{rates.max():.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Serve a saved Archaea champions archive")
    p.add_argument("--champions", required=True, help="path to champions .npz")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="dump metadata + per-agent fitness")

    pr = sub.add_parser("rate", help="single rate query")
    pr.add_argument("--f-in", type=float, required=True, help="input rate (Hz)")
    pr.add_argument("--duration", type=float, default=500.0, help="window length ms (default 500)")
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--agent", type=int, default=None, help="agent index (default best)")

    ps = sub.add_parser("sweep", help="sweep input rate")
    ps.add_argument("--start", type=float, default=10.0)
    ps.add_argument("--stop", type=float, default=100.0)
    ps.add_argument("--step", type=float, default=10.0)
    ps.add_argument("--duration", type=float, default=500.0)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--agent", type=int, default=None)

    args = p.parse_args(argv)
    ens = ChampionEnsemble.load(args.champions)

    if args.cmd == "info":
        return _cmd_info(ens)
    if args.cmd == "rate":
        return _cmd_rate(ens, args.f_in, args.duration, args.seed, args.agent)
    if args.cmd == "sweep":
        return _cmd_sweep(ens, args.start, args.stop, args.step, args.duration, args.seed, args.agent)
    print(f"unknown subcommand: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
