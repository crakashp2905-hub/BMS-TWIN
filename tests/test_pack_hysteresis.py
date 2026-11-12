"""Tests: static OCV hysteresis in the terminal voltage of a series group, for n_parallel = 1, 2 and 3.

``SecondOrderECM`` is the reference cell model: its terminal voltage carries the static hysteresis offset
``-hysteresis_v * sign(I)`` of the OCV curve (8 mV for the default NMC chemistry).  A series group of
``n_parallel`` identical cells is one equivalent cell (R/n, C*n, Q*n), so it must report that same voltage
whatever ``n_parallel`` is.  Parallel groups used to report their voltage without the offset, and
``terminal_voltage(group_current=I)`` omitted it for every ``n_parallel``; both now follow the cell model.
"""

from __future__ import annotations

import numpy as np
import pytest

import bms
from bms.hysteresis import PlettHysteresis

H = 0.008                                   # default NMC hysteresis_v [V]
N_PARALLEL = (1, 2, 3)
# discharge, charge, rest, reversals either side of zero, small currents, large pulses [A, group current]
CURRENTS = (5.0, 5.0, 12.0, -3.0, -8.0, 0.0, 0.5, -0.5, 20.0, 0.0, -15.0)


def _curve(hysteresis_v: float | None = None) -> bms.OCVSOC:
    """NMC curve: ``None`` keeps the chemistry default (8 mV), ``0.0`` switches hysteresis off."""
    return bms.OCVSOC.from_chemistry("nmc", hysteresis_v=hysteresis_v)


def _pack(n_parallel: int, curve: bms.OCVSOC, *, n_cells: int = 3, scatter: bool = False) -> bms.BatteryPack:
    cfg = bms.PackConfig(
        n_cells=n_cells, n_parallel=n_parallel, nominal_capacity_Ah=2.3, seed=11,
        capacity_sigma=0.03 if scatter else 0.0, r0_sigma=0.05 if scatter else 0.0,
        initial_soc_mean=0.6, initial_soc_sigma=0.05 if scatter else 0.0)
    return bms.BatteryPack(cfg, ocv_curve=curve)


class TestGroupVoltageMatchesTheCellModel:
    @pytest.mark.parametrize("temperature_C", [25.0, 40.0, 5.0])
    @pytest.mark.parametrize("n_parallel", N_PARALLEL)
    def test_identical_cells_are_one_equivalent_cell(self, n_parallel, temperature_C):
        curve = _curve()
        assert curve.hysteresis_v == pytest.approx(H)                  # the offset under test is really there
        pack = _pack(n_parallel, curve)
        ref = bms.SecondOrderECM(params=pack.groups[0].params, ocv_curve=curve)   # R/n, C*n, Q*n
        ref.reset(float(pack.soc[0]))
        temps = np.full(pack.n_cells, temperature_C)
        for i in CURRENTS:
            out = pack.step(i, 1.0, cell_temperatures_C=temps)
            v_ref = ref.step(i, 1.0, temperature_C)
            assert out["v_cells"] == pytest.approx(np.full(pack.n_cells, v_ref), abs=1e-9)
            assert out["v_pack"] == pytest.approx(pack.n_cells * v_ref, abs=1e-9)
        assert pack.soc == pytest.approx(np.full(pack.n_cells, ref.soc), abs=1e-12)

    @pytest.mark.parametrize("n_parallel", N_PARALLEL)
    def test_offset_is_minus_h_on_discharge_plus_h_on_charge_and_zero_at_rest(self, n_parallel):
        with_h, without_h = _pack(n_parallel, _curve()), _pack(n_parallel, _curve(0.0))
        for i in CURRENTS:
            shift = with_h.step(i, 1.0)["v_cells"] - without_h.step(i, 1.0)["v_cells"]
            assert shift == pytest.approx(np.full(3, -H * np.sign(i)), abs=1e-12)

    @pytest.mark.parametrize("scatter", [False, True])
    @pytest.mark.parametrize("n_parallel", N_PARALLEL)
    def test_hysteresis_moves_only_the_reported_voltage(self, n_parallel, scatter):
        # SOC, RC states and (mismatched cells, n_parallel > 1) the circulating currents must not depend on it.
        with_h, without_h = (_pack(n_parallel, _curve(h), scatter=scatter) for h in (None, 0.0))
        for i in CURRENTS:
            with_h.step(i, 1.0)
            without_h.step(i, 1.0)
        for ga, gb in zip(with_h.groups, without_h.groups):
            for ca, cb in zip(ga.cells, gb.cells):
                assert (ca.soc, ca.v_rc1, ca.v_rc2) == (cb.soc, cb.v_rc1, cb.v_rc2)

    @pytest.mark.parametrize("scatter", [False, True])
    @pytest.mark.parametrize("n_parallel", N_PARALLEL)
    def test_step_returns_the_voltage_terminal_voltage_reports(self, n_parallel, scatter):
        curve = _curve()
        pack = _pack(n_parallel, curve, scatter=scatter)
        for i in CURRENTS:
            v_step = pack.step(i, 1.0)["v_cells"]
            v_node = np.array([g.terminal_voltage(curve, 25.0, i) for g in pack.groups])
            assert v_step == pytest.approx(v_node, abs=1e-12)

    @pytest.mark.parametrize("n_parallel", N_PARALLEL)
    def test_no_load_voltage_has_no_offset(self, n_parallel):
        pack = _pack(n_parallel, _curve())
        pack.step(8.0, 5.0)                                            # leave some polarisation behind
        flat = _curve(0.0)
        for grp, v_rest in zip(pack.groups, pack.cell_voltages()):
            emf = float(np.mean([float(flat.ocv(c.soc)) - c.v_rc1 - c.v_rc2 for c in grp.cells]))
            assert v_rest == pytest.approx(emf, abs=1e-12)


class TestMismatchedCells:
    """Cells at different SOC exchange circulating currents.  The offset is *not* resolved per cell: it follows
    the sign of the group current, and (as above) moves nothing but the reported voltage."""

    SOCS = (0.90, 0.60, 0.75)

    def _group(self, n_parallel: int, hysteresis_v: float | None) -> bms.BatteryPack:
        pack = _pack(n_parallel, _curve(hysteresis_v), n_cells=1)
        for cell, soc in zip(pack.groups[0].cells, self.SOCS):
            cell.reset(soc)
        return pack

    @pytest.mark.parametrize("group_current", [8.0, 0.05, 0.0, -0.05, -8.0])
    @pytest.mark.parametrize("n_parallel", (2, 3))
    def test_offset_follows_the_sign_of_the_group_current(self, n_parallel, group_current):
        v_with = self._group(n_parallel, None).step(group_current, 1.0)["v_cells"][0]
        v_without = self._group(n_parallel, 0.0).step(group_current, 1.0)["v_cells"][0]
        assert v_with - v_without == pytest.approx(-H * np.sign(group_current), abs=1e-12)

    @pytest.mark.parametrize("n_parallel", (2, 3))
    def test_a_small_group_current_leaves_cells_flowing_against_it(self, n_parallel):
        # The case the previous test relies on: at 0.05 A the cells circulate amps in both directions, so some
        # of them carry current against the group direction yet are given the group's offset.
        grp = self._group(n_parallel, None).groups[0]
        e, g = grp._emf_and_conductance(grp.cells[0].ocv_curve, 25.0)
        cell_i = (e - (np.dot(e, g) - 0.05) / g.sum()) * g
        assert cell_i.sum() == pytest.approx(0.05, abs=1e-9)
        assert cell_i.max() > 1.0 and cell_i.min() < -1.0


class TestDynamicHysteresisIsNotDoubleCounted:
    @pytest.mark.parametrize("n_parallel", N_PARALLEL)
    def test_a_cell_with_a_dynamic_model_gets_no_static_offset(self, n_parallel):
        # SecondOrderECM replaces the static term when a PlettHysteresis model is attached; the group must too.
        curve, flat = _curve(), _curve(0.0)
        grp = _pack(n_parallel, curve, n_cells=1).groups[0]
        for cell in grp.cells:
            cell.hysteresis = PlettHysteresis(0.02, 8.0)               # fresh: h = 0, so its own voltage is 0
        assert grp.terminal_voltage(curve, 25.0, 6.0) == pytest.approx(
            grp.terminal_voltage(flat, 25.0, 6.0), abs=1e-12)
