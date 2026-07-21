"""Integration test: the full coupled mechanistic model runs without crashing
or diverging on a small case.

This does not check quantitative agreement with the paper's reported
thrombus dynamics (no reference implementation is available to compare
against) -- see coupled_solver.py's "Known limitation" comment regarding
the thrombin/fibrin generation pathway's calibration. It checks the more
basic property that a full mechanistic run (mesh -> flow -> surface ODE ->
species transport -> flow feedback, over several time steps) produces
finite, non-negative, bounded output.
"""

from __future__ import annotations

import yaml
import numpy as np
import pytest

from thrombus_bench.mechanistic.coupled_solver import run_coupled_simulation
from thrombus_bench.mechanistic.mesh import build_channel_mesh

PHYSIO_PATH = "configs/physio_params.yaml"


@pytest.fixture
def physio():
    with open(PHYSIO_PATH) as f:
        return yaml.safe_load(f)


def test_coupled_simulation_runs_and_stays_bounded(physio):
    tm = build_channel_mesh(50.0, 4.0, target_num_elements=150)
    history = run_coupled_simulation(
        tm, inlet_velocity_m_s=0.47, physio=physio,
        end_time_s=0.3, dt_s=0.1, output_every_n_steps=1, flow_resolve_every_n_steps=3,
    )
    assert len(history.states) == 3
    last = history.states[-1]
    for name, conc in last.concentrations.items():
        assert np.all(np.isfinite(conc)), f"{name} has non-finite values"
        assert np.all(conc >= 0.0), f"{name} went negative"
    assert np.all(np.isfinite(last.surface.M))
    assert np.all(last.surface.M >= 0.0)
    assert np.all(np.isfinite(last.surface.M_at))
    assert last.flow.converged


def test_coupled_simulation_platelets_stay_near_inlet_scale(physio):
    """Resting/activated platelets should stay within a modest factor of
    their inlet concentration -- a basic physical plausibility check (they
    are consumed, not created, so should not exceed inlet, and adhesion
    alone should not deplete the whole bulk in this short a window)."""

    tm = build_channel_mesh(50.0, 4.0, target_num_elements=150)
    history = run_coupled_simulation(
        tm, inlet_velocity_m_s=0.47, physio=physio,
        end_time_s=0.2, dt_s=0.1, output_every_n_steps=1, flow_resolve_every_n_steps=2,
    )
    last = history.states[-1]
    rp_inlet = physio["species"]["resting_platelets_inlet_plt_ml"]
    assert np.all(last.concentrations["RP"] < 1.5 * rp_inlet)
    assert np.all(last.concentrations["RP"] > 0.1 * rp_inlet)
