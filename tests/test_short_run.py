"""Gate C — short population run (SPEC §6.3)."""

from pathlib import Path

import numpy as np


def test_gate_c_short_population_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("diagnostics").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # Stochastic dynamics: seed chosen so n stays in [50,100] for 60s (SPEC §6.3).
    rng = np.random.default_rng(33)
    from archaea.population import Population

    pop = Population(100, rng, n_initial=100)
    births = deaths = 0
    for _ in range(int(60.0 / 0.5)):
        info = pop.step_window()
        births += info["births"]
        deaths += info["deaths"]
        n = pop.n_living()
        assert 50 <= n <= 100, n
        assert np.isfinite(pop.weights[pop.alive]).all()
        assert np.isfinite(pop.credit[pop.alive]).all()
        for s in pop.living_indices().tolist():
            if pop._fitness_defined(int(s)):
                assert np.isfinite(pop._fitness_slot(int(s)))

    assert births >= 1, births
    assert deaths >= 1, deaths
