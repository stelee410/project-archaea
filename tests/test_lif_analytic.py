"""LIF membrane: Euler vs analytic subthreshold (SPEC §9 step 1)."""

import numpy as np

from archaea.neuron import (
    DT_MS,
    DT_OVER_TAU,
    R_MEM,
    V_REST,
    lif_constant_current_analytic,
)
from archaea.neuron import LIFState


def test_lif_matches_analytic_constant_current():
    I_const = 0.12
    n_steps = 500
    st = LIFState(1, 1)
    I = np.full((1, 1), I_const, dtype=np.float64)
    for _ in range(n_steps):
        st.step(I)
        assert st.v[0, 0] < 0.99
    t_end = n_steps * DT_MS
    v_an_end = float(lif_constant_current_analytic(I_const, np.array([t_end]), v0=V_REST)[0])
    assert abs(float(st.v[0, 0]) - v_an_end) < 0.02, (st.v[0, 0], v_an_end)


def test_membrane_step_matches_explicit_euler():
    st = LIFState(1, 1)
    v = V_REST
    I = 0.2
    for _ in range(50):
        dv = DT_OVER_TAU * (V_REST - v) + DT_OVER_TAU * R_MEM * I
        v_expected = v + dv
        st.step(np.array([[I]], dtype=np.float64))
        v = v_expected
        assert abs(st.v[0, 0] - v) < 1e-12
