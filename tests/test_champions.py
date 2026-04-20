"""Save / load / serve smoke test for the champions DNA archive."""

from pathlib import Path

import numpy as np

from archaea.champions import ChampionEnsemble, save_champions
from archaea.population import Population


def test_save_and_load_champions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("diagnostics").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(33)
    pop = Population(40, rng, n_initial=40)
    for _ in range(int(60.0 / 0.5)):
        pop.step_window()

    out = save_champions(pop, "diagnostics/champions.npz", top_k=5, t_sim=60.0, seed=33)
    assert Path(out).exists()

    ens = ChampionEnsemble.load(out)
    assert ens.k <= 5
    assert ens.weights.shape[1] == 220
    assert ens.spec_version == "L1.0"

    rates = ens.rates_for(80.0, duration_ms=500.0, seed=0, warmup_ms=100.0)
    assert rates.shape == (ens.k,)
    assert np.all(np.isfinite(rates))
    assert (rates >= 0).all()


def test_sweep_monotone_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("diagnostics").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(33)
    pop = Population(40, rng, n_initial=40)
    for _ in range(int(60.0 / 0.5)):
        pop.step_window()

    save_champions(pop, "diagnostics/champions.npz", top_k=3, t_sim=60.0, seed=33)
    ens = ChampionEnsemble.load("diagnostics/champions.npz")

    fi, fo = ens.sweep([20.0, 60.0, 100.0], duration_ms=500.0, seed=1, warmup_ms=100.0)
    assert fi.shape == fo.shape == (3,)
    assert np.all(np.isfinite(fo)) and np.all(fo >= 0)
