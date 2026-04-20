"""
种群「点阵」+ 底部时间序列面板（实时 / 离线均可）。

点阵语义（展示层）：
- 浅灰：空槽
- 绿→黄→橙：存活 Credit 梯度；深红：Credit 极低
- 亮粉：本窗新生儿槽；浅粉粗边：本窗亲代；灰粗边：本窗饿死槽

第二面板：
- 左轴：r_max、r_mean（Pearson，已定义个体）
- 右轴：存活数 N、平均 Credit
- 滑动时间窗（默认最近 600s 仿真时间）便于长跑

更多可视化建议见模块末尾 ``VISUALIZATION_IDEAS`` 与 README。
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import patches as mpatches

from .economy import C_REPRO
from .population import Population

WINDOW_S = 0.5

VISUALIZATION_IDEAS = """
Further visualization ideas (not all implemented here):
- Histogram: per-window distribution of Credit or of per-agent r (requires extra sampling).
- 2D embedding of weight vectors (PCA/UMAP) colored by fitness — heavy; snapshot every N minutes.
- Raster plot: rows=sample agents, x=time, dots=output spikes (memory-heavy for 1000 agents).
- Q-Q or layer-wise weight histograms over time (compare to initial Uniform(-3,3)).
- Gini(credit) time series and reproduction rate (births per minute).
- Heatmap: (slot x time) alive bitstream — see extinction waves.
- Web dashboard (Streamlit/Plotly) for remote watch — new dependency.
- Video export: imageio writer stitching PNG frames from this dashboard.
"""


def slot_positions(n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    cols = int(np.ceil(np.sqrt(n_slots)))
    rows = int(np.ceil(n_slots / cols))
    idx = np.arange(n_slots, dtype=np.int64)
    xs = (idx % cols).astype(np.float64)
    ys = (rows - 1 - (idx // cols)).astype(np.float64)
    return xs, ys


def credit_facecolor(credit: float, alive: bool) -> tuple[float, float, float, float]:
    if not alive:
        return (0.72, 0.74, 0.78, 1.0)
    c = float(np.clip(credit, 0.0, C_REPRO))
    t = c / max(C_REPRO, 1e-9)
    if credit < 15.0:
        return (0.8, 0.15, 0.12, 1.0)
    if t >= 0.55:
        g = 0.78 + 0.2 * ((t - 0.55) / 0.45)
        return (0.13, float(np.clip(g, 0.78, 0.98)), 0.28, 1.0)
    if t >= 0.25:
        u = (t - 0.25) / 0.3
        return (0.55 + 0.2 * (1 - u), 0.75 + 0.15 * u, 0.15, 1.0)
    u = t / 0.25
    return (0.92, 0.55 + 0.2 * u, 0.12, 1.0)


def build_colors(
    pop: Population,
    repro_parents: np.ndarray,
    repro_children: np.ndarray,
    dead_slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = pop.pop_max
    fc = np.zeros((n, 4), dtype=np.float64)
    ec = np.zeros((n, 4), dtype=np.float64)
    lw = np.zeros(n, dtype=np.float64)
    parents = {int(x) for x in repro_parents.tolist()}
    children = {int(x) for x in repro_children.tolist()}
    deads = {int(x) for x in dead_slots.tolist()}

    for s in range(n):
        a = bool(pop.alive[s])
        cr = float(pop.credit[s]) if a else 0.0
        fc[s] = credit_facecolor(cr, a)
        ec[s] = (0.55, 0.58, 0.62, 0.35 if a else 0.25)
        lw[s] = 0.25

        if s in children:
            fc[s] = (0.93, 0.28, 0.6, 1.0)
            ec[s] = (0.7, 0.1, 0.45, 1.0)
            lw[s] = 1.6
        elif s in parents:
            ec[s] = (0.98, 0.65, 0.85, 1.0)
            lw[s] = 2.0
        elif s in deads and not a:
            ec[s] = (0.35, 0.38, 0.45, 0.9)
            lw[s] = 1.2

    return fc, ec, lw


class LiveDashboard:
    """
    上：槽位点阵；下：滑动窗口曲线（r、种群规模、Credit 均值）。
    调用顺序：构造后每仿真窗 ``append_point``；按帧率 ``draw``。
    """

    __slots__ = (
        "_plt",
        "fig",
        "ax_dots",
        "ax_curves",
        "ax_twin",
        "ax_budget",
        "sc",
        "xs",
        "ys",
        "area",
        "pop",
        "tail_sim_s",
        "t_hist",
        "rmax_hist",
        "rmean_hist",
        "n_hist",
        "cmean_hist",
        "sigma_hist",
        "birth_hist",
        "death_hist",
        "budget_hist",
        "line_rmax",
        "line_rmean",
        "line_n",
        "line_cmean",
        "line_sigma",
        "line_budget",
        "_event_artists",
        "_last_info",
        "_interactive",
    )

    def __init__(
        self,
        pop: Population,
        *,
        tail_sim_s: float = 600.0,
        figsize: tuple[float, float] = (12.0, 10.0),
        interactive: bool = True,
    ):
        import matplotlib.pyplot as plt

        self._plt = plt
        self.pop = pop
        self.tail_sim_s = float(max(30.0, tail_sim_s))
        maxlen = int(self.tail_sim_s / WINDOW_S) + 8
        self.t_hist = deque(maxlen=maxlen)
        self.rmax_hist = deque(maxlen=maxlen)
        self.rmean_hist = deque(maxlen=maxlen)
        self.n_hist = deque(maxlen=maxlen)
        self.cmean_hist = deque(maxlen=maxlen)
        self.sigma_hist = deque(maxlen=maxlen)
        self.birth_hist = deque(maxlen=maxlen)
        self.death_hist = deque(maxlen=maxlen)
        self.budget_hist = deque(maxlen=maxlen)
        self._last_info: dict[str, Any] | None = None
        self._event_artists: list[Any] = []
        self._interactive = bool(interactive)

        self.xs, self.ys = slot_positions(pop.pop_max)
        self.area = max(8.0, 2200.0 / np.sqrt(pop.pop_max))

        self.fig, (self.ax_dots, self.ax_curves, self.ax_budget) = plt.subplots(
            3,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [2.35, 1.0, 0.55], "hspace": 0.28},
        )

        self.ax_dots.set_aspect("equal")
        self.ax_dots.set_xlim(self.xs.min() - 1, self.xs.max() + 1)
        self.ax_dots.set_ylim(self.ys.min() - 1, self.ys.max() + 1)
        self.ax_dots.axis("off")

        fc, ec, lw = build_colors(pop, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))
        self.sc = self.ax_dots.scatter(self.xs, self.ys, s=self.area, c=fc, edgecolors=ec, linewidths=lw, marker="o")

        legend_elems = [
            mpatches.Patch(facecolor=(0.72, 0.74, 0.78), edgecolor="none", label="Empty / dead"),
            mpatches.Patch(facecolor=(0.15, 0.85, 0.35), edgecolor="none", label="Alive high Credit"),
            mpatches.Patch(facecolor=(0.92, 0.72, 0.15), edgecolor="none", label="Alive mid Credit"),
            mpatches.Patch(facecolor=(0.8, 0.15, 0.12), edgecolor="none", label="Alive critical"),
            mpatches.Patch(facecolor=(0.93, 0.28, 0.6), edgecolor="none", label="Newborn slot"),
            mpatches.Patch(
                facecolor=(0.2, 0.45, 0.25),
                edgecolor=(0.98, 0.65, 0.85),
                linewidth=2,
                label="Parent (pink ring)",
            ),
            mpatches.Patch(
                facecolor=(0.72, 0.74, 0.78),
                edgecolor=(0.35, 0.38, 0.45),
                linewidth=2,
                label="Starved (gray ring)",
            ),
        ]
        self.ax_dots.legend(
            handles=legend_elems,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )

        self.ax_curves.set_ylabel("Pearson r")
        self.ax_curves.set_xlabel("sim time (s)")
        self.ax_curves.set_ylim(-0.05, 1.05)
        self.ax_curves.grid(True, alpha=0.25)
        (self.line_rmax,) = self.ax_curves.plot([], [], color="#2563EB", linewidth=1.4, label="r_max")
        (self.line_rmean,) = self.ax_curves.plot([], [], color="#0D9488", linewidth=1.1, alpha=0.85, label="r_mean")
        (self.line_sigma,) = self.ax_curves.plot([], [], color="#7C3AED", linewidth=1.0, linestyle="--", alpha=0.8, label="sigma")

        self.ax_twin = self.ax_curves.twinx()
        self.ax_twin.set_ylabel("N alive / credit mean")
        (self.line_n,) = self.ax_twin.plot([], [], color="#EA580C", linewidth=1.2, drawstyle="steps-post", label="N")
        (self.line_cmean,) = self.ax_twin.plot([], [], color="#B45309", linewidth=1.0, alpha=0.85, label="credit_mean")

        h1, l1 = self.ax_curves.get_legend_handles_labels()
        h2, l2 = self.ax_twin.get_legend_handles_labels()
        self.ax_curves.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=2)

        # Third panel: budget pressure D/B (only meaningful in shared-budget mode).
        self.ax_budget.set_ylabel("budget D/B")
        self.ax_budget.set_xlabel("sim time (s)")
        self.ax_budget.grid(True, alpha=0.25)
        self.ax_budget.axhline(1.0, color="#9CA3AF", linewidth=0.8, linestyle=":")
        (self.line_budget,) = self.ax_budget.plot(
            [], [], color="#DC2626", linewidth=1.2, label="budget pressure"
        )
        self.ax_budget.legend(loc="upper left", fontsize=8)

        self.fig.suptitle("Archaea live dashboard", fontsize=12, y=0.98)
        self.fig.subplots_adjust(right=0.82, left=0.07, top=0.93, bottom=0.06)

    def append_point(self, t_sim: float, info: dict[str, Any], pop: Population) -> None:
        self._last_info = info
        self.t_hist.append(float(t_sim))
        self.rmax_hist.append(float(info["r_max"]))
        self.rmean_hist.append(float(info["r_mean"]))
        self.n_hist.append(int(pop.n_living()))
        self.cmean_hist.append(float(pop.credit_mean()))
        self.sigma_hist.append(float(info["sigma"]))
        self.birth_hist.append(int(info["births"]))
        self.death_hist.append(int(info["deaths"]))
        self.budget_hist.append(float(info.get("budget_pressure", 0.0)))

    def draw(self) -> None:
        for art in self._event_artists:
            try:
                art.remove()
            except ValueError:
                pass
        self._event_artists.clear()

        info = self._last_info or {}
        repro_p = info.get("repro_parent_slots", np.zeros(0, dtype=np.int32))
        repro_c = info.get("repro_child_slots", np.zeros(0, dtype=np.int32))
        dead_s = info.get("dead_slots", np.zeros(0, dtype=np.int32))

        fc, ec, lw = build_colors(self.pop, repro_p, repro_c, dead_s)
        self.sc.set_facecolors(fc)
        self.sc.set_edgecolors(ec)
        self.sc.set_linewidths(lw)

        t = np.asarray(self.t_hist, dtype=np.float64)
        if t.size == 0:
            return

        self.line_rmax.set_data(t, np.asarray(self.rmax_hist, dtype=np.float64))
        self.line_rmean.set_data(t, np.asarray(self.rmean_hist, dtype=np.float64))
        self.line_sigma.set_data(t, np.asarray(self.sigma_hist, dtype=np.float64))
        self.line_n.set_data(t, np.asarray(self.n_hist, dtype=np.float64))
        self.line_cmean.set_data(t, np.asarray(self.cmean_hist, dtype=np.float64))

        y0, y1 = self.ax_curves.get_ylim()
        h = max(0.04, (y1 - y0) * 0.06)
        base = y0 + 0.02
        tb = np.asarray(self.t_hist, dtype=np.float64)
        bb = np.asarray(self.birth_hist, dtype=np.int32)
        dd = np.asarray(self.death_hist, dtype=np.int32)
        if tb.size:
            for ti, b, d in zip(tb, bb, dd):
                if b > 0:
                    p = self.ax_curves.fill_betweenx(
                        [base, base + h], ti - 0.12, ti + 0.12, color="#22C55E", alpha=0.35, linewidth=0
                    )
                    self._event_artists.append(p)
                if d > 0:
                    p = self.ax_curves.fill_betweenx(
                        [base, base + h], ti - 0.12, ti + 0.12, color="#EF4444", alpha=0.35, linewidth=0
                    )
                    self._event_artists.append(p)

        t_max = float(t[-1])
        t_min = max(0.0, t_max - self.tail_sim_s)
        self.ax_curves.set_xlim(t_min, max(t_max, t_min + 1e-6))
        self.ax_twin.relim()
        self.ax_twin.autoscale_view(scalex=False, scaley=True)
        self.ax_curves.set_ylim(-0.05, 1.05)

        budget_arr = np.asarray(self.budget_hist, dtype=np.float64)
        self.line_budget.set_data(t, budget_arr)
        self.ax_budget.set_xlim(t_min, max(t_max, t_min + 1e-6))
        if budget_arr.size and float(np.nanmax(budget_arr)) > 0.0:
            top = max(1.2, float(np.nanmax(budget_arr)) * 1.1)
            self.ax_budget.set_ylim(0.0, top)
        else:
            self.ax_budget.set_ylim(0.0, 1.2)

        n = int(self.pop.n_living())
        self.ax_dots.set_title(
            f"slots  t={t_max:.1f}s  N={n}/{self.pop.pop_max}  "
            f"b={int(info.get('births', 0))} d={int(info.get('deaths', 0))}  "
            f"r_max={float(info.get('r_max', 0)):.3f}  sigma={float(info.get('sigma', 0)):.3f}",
            fontsize=10,
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if self._interactive:
            self._plt.pause(0.001)

    def savefig(self, path: str, dpi: int = 120) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")

    def close(self) -> None:
        try:
            self._plt.close(self.fig)
        except Exception:
            pass

    def show_blocking(self) -> None:
        self._plt.ioff()
        self._plt.show()


def run_dots(
    seed: int,
    duration_s: float,
    pop_max: int,
    n_initial: int | None,
    every: int,
    save_path: str | None,
    dpi: int,
    interactive: bool,
    tail_sim_s: float,
    carrying_capacity: int | None = None,
    budget_mode: str = "none",
) -> None:
    import matplotlib

    if not interactive:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pop = Population(
        pop_max,
        rng,
        n_initial=n_initial,
        carrying_capacity=carrying_capacity,
        budget_mode=budget_mode,
    )
    dash = LiveDashboard(pop, tail_sim_s=tail_sim_s, interactive=interactive)

    n_windows = int(duration_s / 0.5)
    step = max(1, int(every))

    if interactive:
        plt.ion()

    for w in range(n_windows):
        info = pop.step_window()
        t_sim = (w + 1) * WINDOW_S
        dash.append_point(t_sim, info, pop)
        if (w + 1) % step == 0 or (w + 1) == n_windows:
            dash.draw()

    if save_path:
        dash.savefig(save_path, dpi=dpi)

    if interactive:
        dash.show_blocking()
    dash.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Archaea live dashboard (dots + curves)")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--duration", type=float, default=60.0, help="Simulated seconds")
    p.add_argument("--pop-max", type=int, default=100)
    p.add_argument("--n-initial", type=int, default=None)
    p.add_argument("--every", type=int, default=1, help="Redraw curves+dots every N windows")
    p.add_argument("--tail", type=float, default=600.0, help="Sliding window length on time axis (sim seconds)")
    p.add_argument("--save", type=str, default="", help="Save last frame PNG")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--no-show", action="store_true", help="Headless: save only")
    p.add_argument(
        "--carrying-capacity",
        type=int,
        default=0,
        help="[off-SPEC] K for shared-budget mode (0 = disabled).",
    )
    p.add_argument(
        "--budget-mode",
        type=str,
        default="none",
        choices=["none", "shared"],
        help="[off-SPEC] 'shared' enables ecological carrying-capacity model.",
    )
    args = p.parse_args(argv)

    os.chdir(Path(__file__).resolve().parents[1])
    save = args.save.strip() or None
    interactive = not args.no_show
    if args.no_show and save is None:
        save = "diagnostics/visualize_last.png"
    run_dots(
        seed=args.seed,
        duration_s=args.duration,
        pop_max=args.pop_max,
        n_initial=args.n_initial,
        every=args.every,
        save_path=save,
        dpi=args.dpi,
        interactive=interactive,
        tail_sim_s=float(args.tail),
        carrying_capacity=int(args.carrying_capacity) or None,
        budget_mode=str(args.budget_mode),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
