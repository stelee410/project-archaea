"""Unit tests for SPEC v1.2 calibration penalty (off-SPEC, opt-in)."""

from __future__ import annotations

import numpy as np
import pytest

from archaea.agent import (
    N_HISTORY,
    fitness_with_calibration_penalty,
    pearson_r,
)
from archaea.population import Population


def _make_history(scale: float, noise_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """f_in uniform 0..100; f_out = scale * f_in + small noise."""
    rng = np.random.default_rng(noise_seed)
    f_in = rng.uniform(0.0, 100.0, size=N_HISTORY)
    f_out = scale * f_in + rng.normal(0.0, 0.5, size=N_HISTORY)
    return f_in, f_out


def test_lam_zero_matches_pearson_r():
    """λ = 0  →  exact SPEC §4.1 behaviour, bit-identical."""
    fi, fo = _make_history(scale=0.5)
    assert fitness_with_calibration_penalty(fi, fo, 0.0) == pearson_r(fi, fo)


def test_perfect_match_unchanged_by_lambda():
    """When mean(f_out) == mean(f_in), the penalty is 0 regardless of λ."""
    fi, fo = _make_history(scale=1.0)
    r = pearson_r(fi, fo)
    # mean(fo) ≈ mean(fi); bias term ≈ 0
    assert abs(fitness_with_calibration_penalty(fi, fo, 1.0) - r) < 0.01


def test_compressed_output_is_penalised():
    """slope=0.5  → mean(f_out) is half of mean(f_in)  → penalty kicks in."""
    fi, fo = _make_history(scale=0.5)
    r0 = pearson_r(fi, fo)
    r_pen = fitness_with_calibration_penalty(fi, fo, 0.5)
    assert r0 > 0.95, "sanity: shape correlation is still very high"
    assert r_pen < r0, "compressed output must lose fitness"
    # the penalty magnitude should be ≈ 0.5 * (|mean diff| / std(fi))
    bias = abs(float(np.mean(fo)) - float(np.mean(fi))) / float(np.std(fi))
    expected = r0 - 0.5 * bias
    assert abs(r_pen - expected) < 1e-9


def test_population_lam_zero_matches_default_pearson():
    """Population with λ=0 must score exactly like the original SPEC §4.1 path."""
    rng_a = np.random.default_rng(7)
    pop_a = Population(20, rng_a, n_initial=20)
    rng_b = np.random.default_rng(7)
    pop_b = Population(20, rng_b, n_initial=20, calibration_lambda=0.0)

    # Inject identical synthetic histories into slot 0 of each population
    fi, fo = _make_history(scale=0.4)
    for p in (pop_a, pop_b):
        p._fin[0, :] = fi
        p._fout[0, :] = fo
        p._hc[0] = N_HISTORY  # mark history full

    assert pop_a._fitness_slot(0) == pop_b._fitness_slot(0)


def test_population_lam_positive_penalises_compressed_outputs():
    """λ > 0  →  fitness for compressed-output agent is strictly lower."""
    rng = np.random.default_rng(7)
    pop_zero = Population(20, rng, n_initial=20, calibration_lambda=0.0)
    rng2 = np.random.default_rng(7)
    pop_pen = Population(20, rng2, n_initial=20, calibration_lambda=0.5)

    fi, fo = _make_history(scale=0.5)  # compressed
    for p in (pop_zero, pop_pen):
        p._fin[0, :] = fi
        p._fout[0, :] = fo
        p._hc[0] = N_HISTORY

    assert pop_pen._fitness_slot(0) < pop_zero._fitness_slot(0)


def test_population_rejects_negative_lambda():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="calibration_lambda"):
        Population(20, rng, n_initial=20, calibration_lambda=-0.1)
