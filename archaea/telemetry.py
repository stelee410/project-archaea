"""Logging, checkpoints, halt diagnostics (SPEC §8)."""

from __future__ import annotations

import csv
import os
import pickle
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .agent import N_HISTORY
from .neuron import N_WEIGHTS, unpack_weights


def ensure_dirs() -> None:
    Path("diagnostics").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)


def write_log_line(fp, fields: list) -> None:
    line = "\t".join(str(x) for x in fields) + "\n"
    fp.write(line)
    fp.flush()


# Column order matches SPEC §8.1 / archaea.run row layout
LOG_TABLE_COLUMNS = (
    "t_sim",
    "pop_size",
    "births",
    "deaths",
    "r_max",
    "r_mean",
    "credit_mean",
    "credit_gini",
    "weight_std",
    "sigma",
    "budget_pressure",
    "phero_max",
    "hgt",
    "moves",
)


class StdoutRollingTable:
    """
    Interactive terminal: **column header pinned to the bottom row**, data above it.

    Uses absolute cursor positioning (CUP + EL). The deque holds up to ``maxlen``
    rows; only the tail that fits in ``(terminal_height - 1)`` data lines is shown,
    newest just above the footer header.

    Uses the alternate screen buffer (smcup/rmcup) when supported.
    """

    __slots__ = ("_stream", "_maxlen", "_lines", "_alt_on")

    def __init__(self, stream, maxlen: int = 100) -> None:
        if not stream.isatty():
            raise ValueError("StdoutRollingTable requires a TTY stream")
        self._stream = stream
        self._maxlen = max(1, int(maxlen))
        self._lines: deque[str] = deque(maxlen=self._maxlen)
        self._alt_on = False

    def _term_lines(self) -> int:
        try:
            return max(3, int(os.get_terminal_size(self._stream.fileno()).lines))
        except OSError:
            return 24

    def push(self, fields: list) -> None:
        line = "\t".join(str(x) for x in fields)
        if not self._alt_on:
            self._stream.write("\033[?1049h\033[2J")
            self._stream.flush()
            self._alt_on = True
        self._lines.append(line)
        self._redraw_body()

    def _redraw_body(self) -> None:
        th = self._term_lines()
        body_cap = max(1, min(self._maxlen, th - 1))
        header_row = th
        data_first = header_row - body_cap

        buf = list(self._lines)
        tail = buf[-body_cap:] if len(buf) > body_cap else buf
        n = len(tail)
        start_pad = body_cap - n

        header = "\t".join(LOG_TABLE_COLUMNS)

        for row in range(1, data_first):
            self._stream.write(f"\033[{row};1H\033[2K")

        for i in range(body_cap):
            row = data_first + i
            if row < 1 or row >= header_row:
                continue
            self._stream.write(f"\033[{row};1H\033[2K")
            if i >= start_pad:
                self._stream.write(tail[i - start_pad])

        self._stream.write(f"\033[{header_row};1H\033[2K" + header)

        self._stream.flush()

    def finish(self) -> None:
        if self._alt_on:
            self._stream.write("\033[?1049l")
            self._stream.flush()
            self._alt_on = False


def save_checkpoint(path: str, pop, rng: np.random.Generator) -> None:
    idx = pop.living_indices()
    W = pop.weights[idx]
    C = pop.credit[idx]
    hc = pop._hc[idx]
    fin = pop._fin[idx]
    fout = pop._fout[idx]
    rng_bytes = np.frombuffer(pickle.dumps(rng.bit_generator.state), dtype=np.uint8)
    np.savez_compressed(
        path,
        living_idx=idx,
        weights=W,
        credit=C,
        hist_count=hc,
        fin_hist=fin,
        fout_hist=fout,
        rng_state_bytes=rng_bytes,
    )


def plot_fitness_curve(t_hist: list[float], rmax_hist: list[float], rmean_hist: list[float], path: str) -> None:
    ensure_dirs()
    plt.figure(figsize=(10, 4))
    plt.plot(t_hist, rmax_hist, label="r_max")
    plt.plot(t_hist, rmean_hist, label="r_mean", alpha=0.7)
    plt.xlabel("t_sim (s)")
    plt.ylabel("correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_weight_hist(pop, path: str) -> None:
    ensure_dirs()
    idx = pop.living_indices()
    if idx.size == 0:
        return
    w = pop.weights[idx].ravel()
    w1, w2 = unpack_weights(pop.weights[idx])
    plt.figure(figsize=(8, 4))
    plt.hist(w1.ravel(), bins=40, alpha=0.6, label="input→hidden")
    plt.hist(w2.ravel(), bins=40, alpha=0.6, label="hidden→output")
    plt.xlabel("weight")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_weight_layers_hist(pop, path: str) -> None:
    """Alias for SPEC halt naming."""
    plot_weight_hist(pop, path)


def dump_top10_csv(pop, path: str) -> None:
    ensure_dirs()
    top = pop.top_k_slots(10, by="fitness")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot", "credit", "fitness", "weights_csv"])
        for s in top.tolist():
            fit = pop._fitness_slot(int(s)) if pop._fitness_defined(int(s)) else float("nan")
            weights_str = ",".join(f"{x:.8g}" for x in pop.weights[int(s)].tolist())
            w.writerow([int(s), pop.credit[int(s)], fit, weights_str])


def monoculture_metric(pop) -> float:
    """Mean std per weight position among top-100 by credit (SPEC §7.4)."""
    idx = pop.top_k_slots(100, by="credit")
    if idx.size <= 1:
        return 0.0
    W = pop.weights[idx]
    return float(np.mean(np.std(W, axis=0)))


def dump_t2h_weights(pop, path: str) -> None:
    ensure_dirs()
    idx = pop.living_indices()
    if idx.size == 0:
        return
    w1, w2 = unpack_weights(pop.weights[idx])
    plt.figure(figsize=(8, 4))
    plt.hist(w1.ravel(), bins=50, alpha=0.6, label="W1")
    plt.hist(w2.ravel(), bins=50, alpha=0.6, label="W2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def dump_t2h_top10_csv(pop, path: str) -> None:
    dump_top10_csv(pop, path)
