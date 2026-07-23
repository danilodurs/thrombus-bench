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

from thrombus_bench.mechanistic.coupled_solver import _thrombin_fibrin_cap_exceeded, run_coupled_simulation
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


def test_coupled_simulation_reports_thrombin_fibrin_reliable_flag(physio):
    """`CoupledSimulationHistory.thrombin_fibrin_reliable` should be a plain
    bool. Per the documented "Known limitation" in coupled_solver.py (the
    surface thrombin-generation pathway's [T]/[FI] concentration cap can
    bind within a fraction of a second once spatial transport keeps
    resupplying substrate -- see scripts/
    diagnose_thrombin_reaction_stiffness.py), even this short, otherwise
    well-behaved run is expected to trip the cap and report False; that is
    the flag correctly surfacing the known instability, not a test bug."""

    tm = build_channel_mesh(50.0, 4.0, target_num_elements=150)
    history = run_coupled_simulation(
        tm, inlet_velocity_m_s=0.47, physio=physio,
        end_time_s=0.2, dt_s=0.1, output_every_n_steps=1, flow_resolve_every_n_steps=2,
    )
    assert isinstance(history.thrombin_fibrin_reliable, bool)
    assert history.thrombin_fibrin_reliable is False


@pytest.mark.parametrize(
    "T, FI, cap_T, cap_FI, expected",
    [
        (np.array([0.5]), np.array([0.2]), 1.0, 1.0, False),
        (np.array([1.5]), np.array([0.2]), 1.0, 1.0, True),  # T over cap
        (np.array([0.5]), np.array([1.5]), 1.0, 1.0, True),  # FI over cap
        (np.array([1.0]), np.array([1.0]), 1.0, 1.0, False),  # exactly at cap: not "exceeded"
    ],
)
def test_thrombin_fibrin_cap_exceeded_helper(T, FI, cap_T, cap_FI, expected):
    new_concentrations = {"T": T, "FI": FI}
    concentration_cap = {"T": cap_T, "FI": cap_FI}
    assert _thrombin_fibrin_cap_exceeded(new_concentrations, concentration_cap) is expected
