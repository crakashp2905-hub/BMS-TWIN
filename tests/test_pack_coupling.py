"""Tests: the vehicle pack — equivalence with the cell model, power→current, limits, SOP."""

from __future__ import annotations

import numpy as np
import pytest

import bms
from bms.pack_coupling import (
    LIM_CURRENT,
    LIM_POWER,
    LIM_SOC,
    LIM_THERMAL,
    LIM_VOLTAGE,
    PackLimits,
    PackSpec,
    VehiclePack,
)


def _small(**kw) -> PackSpec:
    base = dict(n_series=4, n_parallel=1, capacity_sigma=0.0, r0_sigma=0.0, soc_sigma=0.0)
    base.update(kw)
    return PackSpec(**base)


class TestEquivalenceWithTheCellModel:
    """The vehicle pack is the repository's ECM, vectorized: it must not drift from it."""

    def test_single_cell_groups_match_secondorderecm_exactly(self):
        spec = _small()
        vp = VehiclePack(spec, soc0=0.8)
        ecms = [bms.SecondOrderECM(params=spec.cell_params(), ocv_curve=vp.ocv_curve) for _ in range(4)]
        for e in ecms:
            e.reset(0.8)
        rng = np.random.default_rng(1)
        currents = np.concatenate([rng.uniform(-3, 8, 300), np.zeros(20), -rng.uniform(0, 5, 50)])
        worst_v = worst_soc = 0.0
        for i in currents:
            temps = vp.thermal.T.copy()                        # heating changes the temperature each step
            st = vp.step(float(i), 1.0, cooling_duty=0.5)
            v_ref = np.array([e.step(float(i), 1.0, float(temps[k])) for k, e in enumerate(ecms)])
            worst_v = max(worst_v, abs(st.v_pack_V - v_ref.sum()))
            worst_soc = max(worst_soc, float(np.max(np.abs(vp.soc - [e.soc for e in ecms]))))
        assert worst_v < 1e-9 and worst_soc < 1e-12

    def test_parallel_groups_match_batterypack(self):
        # BatteryPack reports parallel-group voltage without the static hysteresis offset
        # (its single-cell path includes it), so compare with hysteresis off.
        spec = _small(n_parallel=3, hysteresis_v=0.0)
        vp = VehiclePack(spec, soc0=0.8)
        bp = bms.BatteryPack(
            bms.PackConfig(n_cells=4, n_parallel=3, nominal_capacity_Ah=3.0, capacity_sigma=0.0, r0_sigma=0.0,
                           initial_soc_mean=0.8, initial_soc_sigma=0.0),
            base_params=spec.cell_params(), ocv_curve=vp.ocv_curve)
        rng = np.random.default_rng(2)
        worst = 0.0
        for i in rng.uniform(-10, 25, 200):
            temps = vp.thermal.T.copy()
            st = vp.step(float(i), 1.0, cooling_duty=0.3)
            out = bp.step(float(i), 1.0, cell_temperatures_C=temps)
            worst = max(worst, abs(st.v_pack_V - out["v_pack"]))
        assert worst < 1e-9
        assert np.allclose(vp.soc, bp.soc, atol=1e-12)

    def test_scatter_is_seeded_and_reproducible(self):
        a = VehiclePack(PackSpec(), seed=5)
        b = VehiclePack(PackSpec(), seed=5)
        c = VehiclePack(PackSpec(), seed=6)
        assert np.array_equal(a.soc, b.soc) and np.array_equal(a.r0_25, b.r0_25)
        assert not np.array_equal(a.soc, c.soc)

    def test_default_pack_is_a_plausible_ev_pack(self):
        spec = PackSpec()
        assert spec.n_series * spec.n_parallel == 5760
        assert 60.0 < spec.nominal_energy_kWh() < 70.0
        vp = VehiclePack(spec, soc0=0.9)
        assert 330.0 < vp.terminal_voltage(0.0, 1.0) < 400.0


class TestPowerToCurrent:
    @pytest.mark.parametrize("p", [-40_000.0, -5_000.0, 1.0, 2_000.0, 15_000.0, 60_000.0, 120_000.0])
    def test_solved_current_delivers_the_requested_power(self, p):
        vp = VehiclePack(PackSpec(), soc0=0.9)
        i = vp.solve_current(p, 0.25)
        assert vp.terminal_voltage(i, 0.25) * i == pytest.approx(p, abs=1e-3)
        assert np.sign(i) == np.sign(p)

    def test_after_the_step_the_terminal_power_is_the_requested_power(self):
        vp = VehiclePack(PackSpec(), soc0=0.7)
        for p in (30_000.0, -12_000.0, 5_000.0):
            i = vp.solve_current(p, 0.5)
            assert vp.step(i, 0.5).p_terminal_W == pytest.approx(p, abs=1e-3)

    @pytest.mark.parametrize("ambient", [-15.0, 0.0, 25.0, 42.0])
    def test_the_predicted_voltage_is_the_voltage_the_step_produces_at_any_temperature(self, ambient):
        # The solver predicts with a non-mutating copy of the model; it must agree with the real step
        # (including the temperature-dependent EMF and resistances), or the power would be wrong.
        vp = VehiclePack(PackSpec(), soc0=0.75, ambient_C=ambient)
        for p in (-20_000.0, 8_000.0, 45_000.0):
            i = vp.solve_current(p, 0.25)
            predicted = vp.terminal_voltage(i, 0.25)
            snapshot = (vp.soc.copy(), vp.v1.copy(), vp.v2.copy(), vp.T.copy())
            st = vp.step(i, 0.25, cooling_duty=0.0)
            assert st.v_pack_V == pytest.approx(predicted, abs=1e-9)
            assert st.p_terminal_W == pytest.approx(p, abs=1e-3)
            vp.soc, vp.v1, vp.v2 = snapshot[0], snapshot[1], snapshot[2]           # rewind for the next power
            vp.thermal.T[:] = snapshot[3]

    def test_zero_power_is_zero_current(self):
        assert VehiclePack(PackSpec()).solve_current(0.0, 1.0) == 0.0

    def test_power_beyond_the_circuit_maximum_is_nan_not_a_wrong_answer(self):
        vp = VehiclePack(PackSpec(n_series=4, n_parallel=1))
        assert np.isnan(vp.solve_current(vp.max_power_W(1.0) * 1.05, 1.0))
        assert np.isfinite(vp.solve_current(vp.max_power_W(1.0) * 0.95, 1.0))

    def test_current_rises_monotonically_with_power(self):
        vp = VehiclePack(PackSpec(), soc0=0.6)
        powers = np.linspace(-30_000.0, 90_000.0, 25)
        cur = [vp.solve_current(float(p), 0.25) for p in powers]
        assert np.all(np.diff(cur) > 0.0)

    def test_a_cold_pack_needs_more_current_for_the_same_power(self):
        warm = VehiclePack(PackSpec(), soc0=0.8, ambient_C=25.0)
        cold = VehiclePack(PackSpec(), soc0=0.8, ambient_C=-10.0)
        assert cold.solve_current(60_000.0, 0.25) > warm.solve_current(60_000.0, 0.25)


class TestLimits:
    def test_current_limit_binds_when_it_is_the_tightest(self):
        vp = VehiclePack(PackSpec(), PackLimits(max_discharge_C=0.5, max_charge_C=0.25), soc0=0.6)
        lim = vp.limits_now(0.25)
        assert lim.discharge_A == pytest.approx(0.5 * 180.0)
        assert lim.discharge.binding & LIM_CURRENT and not lim.discharge.binding & LIM_VOLTAGE
        assert lim.charge_A == pytest.approx(0.25 * 180.0)
        # the power limit is the power at that current, not the pack's maximum
        i = lim.discharge_A
        assert lim.discharge_W == pytest.approx(vp.terminal_voltage(i, 0.25) * i, rel=5e-3)

    def test_voltage_limit_binds_near_empty_and_is_respected(self):
        vp = VehiclePack(PackSpec(), soc0=0.04)
        lim = vp.discharge_limit(0.25)
        assert lim.binding & LIM_VOLTAGE
        assert 0.0 < lim.current_A < 3.0 * 180.0
        v = vp._group_voltages(lim.current_A, 0.25, vp._scaled(0.25))
        assert v.min() >= vp.v_min - 0.01

    def test_the_weakest_group_sets_the_limit(self):
        even = VehiclePack(PackSpec(), soc0=0.15)
        weak = VehiclePack(PackSpec(), soc0=0.15)
        weak.r0_25[7] *= 12.0                                  # one badly aged series group
        assert weak.discharge_limit(0.25).current_A < even.discharge_limit(0.25).current_A

    def test_power_cap(self):
        vp = VehiclePack(PackSpec(), PackLimits(max_discharge_power_W=30_000.0, max_charge_power_W=10_000.0),
                         soc0=0.6)
        lim = vp.limits_now(0.25)
        assert lim.discharge_W == 30_000.0 and lim.discharge.binding & LIM_POWER
        assert lim.charge_W == 10_000.0 and lim.charge.binding & LIM_POWER

    def test_soc_floor_stops_discharge_and_ceiling_stops_charge(self):
        low = VehiclePack(PackSpec(), PackLimits(soc_floor=0.05), soc0=0.03)
        assert low.discharge_limit(0.25).current_A == 0.0 and low.discharge_limit(0.25).binding & LIM_SOC
        assert low.charge_limit(0.25).current_A > 0.0
        full = VehiclePack(PackSpec(soc_sigma=0.0), soc0=1.0)
        assert full.charge_limit(0.25).current_A == 0.0 and full.charge_limit(0.25).binding & LIM_SOC
        assert full.discharge_limit(0.25).current_A > 0.0

    def test_regen_tapers_linearly_between_the_taper_start_and_the_ceiling(self):
        cur = []
        for soc in (0.85, 0.90, 0.925, 0.95, 0.975, 1.0):
            vp = VehiclePack(PackSpec(soc_sigma=0.0, n_parallel=60), soc0=soc)
            cur.append(vp.charge_limit(0.25).current_A)
        assert cur[0] == cur[1] > cur[2] > cur[3] > cur[4] > cur[5] == 0.0
        assert cur[3] == pytest.approx(cur[1] * 0.5, rel=1e-6)

    def test_hot_pack_is_derated_in_both_directions_and_cut_off(self):
        vp = VehiclePack(PackSpec(), soc0=0.7)
        base = vp.limits_now(0.25)
        vp.thermal.T[:] = 52.5                                  # halfway between 45 and 60 °C
        hot = vp.limits_now(0.25)
        assert hot.discharge_A == pytest.approx(0.5 * 540.0, rel=1e-9)
        assert hot.charge_A < base.charge_A and hot.discharge.binding & LIM_THERMAL
        vp.thermal.T[:] = 61.0
        assert vp.limits_now(0.25).discharge_A == 0.0

    def test_cold_pack_limits_charge_but_not_discharge(self):
        vp = VehiclePack(PackSpec(), soc0=0.7, ambient_C=0.0)
        lim = vp.limits_now(0.25)
        warm = VehiclePack(PackSpec(), soc0=0.7).limits_now(0.25)
        assert lim.charge.derate == pytest.approx(0.5)          # 0 °C is halfway between 10 and -10 °C
        assert lim.charge_A < 0.6 * warm.charge_A and lim.charge.binding & LIM_THERMAL
        assert lim.discharge.derate == 1.0
        vp.thermal.T[:] = -12.0
        assert vp.limits_now(0.25).charge_A == 0.0

    def test_demanding_the_limit_every_step_never_breaches_the_voltage_window(self):
        vp = VehiclePack(PackSpec(), soc0=0.12)
        worst = 10.0
        for _ in range(240):
            p = vp.discharge_limit(0.25).power_W
            i = vp.solve_current(p, 0.25)
            worst = min(worst, vp.step(i, 0.25).v_group_min_V)
        assert worst >= vp.v_min - 0.01

    def test_charging_at_the_limit_never_breaches_the_upper_voltage(self):
        vp = VehiclePack(PackSpec(), soc0=0.80)
        worst = 0.0
        for _ in range(240):
            p = vp.charge_limit(0.25).power_W
            if p <= 0.0:
                break
            worst = max(worst, vp.step(vp.solve_current(-p, 0.25), 0.25).v_group_max_V)
        assert worst <= vp.v_max + 0.01


class TestStateOfPower:
    def test_matches_the_existing_state_of_power_calculator(self):
        spec = _small(n_series=6, hysteresis_v=0.0)             # StateOfPower ignores static hysteresis
        vp = VehiclePack(spec, PackLimits(max_discharge_C=3.0, max_charge_C=1.5), soc0=0.7)
        bp = bms.BatteryPack(
            bms.PackConfig(n_cells=6, n_parallel=1, nominal_capacity_Ah=3.0, capacity_sigma=0.0, r0_sigma=0.0,
                           initial_soc_mean=0.7, initial_soc_sigma=0.0),
            base_params=spec.cell_params(), ocv_curve=vp.ocv_curve)
        ref = bms.StateOfPower(bms.SOPConfig(max_discharge_current_A=3.0 * 3.0,
                                             max_charge_current_A=1.5 * 3.0)).calculate(bp, np.full(6, 25.0))
        got = vp.sop_limits()
        assert set(got) == set(ref) == {2.0, 10.0, 30.0}
        for h in ref:
            assert got[h].traction_power_W == pytest.approx(ref[h].traction_power_W, rel=1e-9)
            assert got[h].regen_power_W == pytest.approx(ref[h].regen_power_W, rel=1e-9)
            assert got[h].traction_current_A == pytest.approx(ref[h].traction_current_A, rel=1e-9)
            assert got[h].regen_current_A == pytest.approx(ref[h].regen_current_A, rel=1e-9)

    def test_longer_horizons_allow_less_power(self):
        sop = VehiclePack(PackSpec(), soc0=0.5).sop_limits()
        assert sop[2.0].traction_power_W >= sop[10.0].traction_power_W >= sop[30.0].traction_power_W > 0.0


class TestThermalAndEnergy:
    def test_load_heats_the_pack_and_cooling_pulls_it_back(self):
        vp = VehiclePack(PackSpec(), soc0=0.9)
        t0 = float(vp.T.mean())
        for _ in range(600):
            vp.step(300.0, 0.5, cooling_duty=0.0)
        hot = float(vp.T.mean())
        assert hot > t0 + 0.5
        for _ in range(4000):
            vp.step(0.0, 0.5, cooling_duty=1.0)
        assert float(vp.T.mean()) < hot - 0.3

    def test_cooling_power_scales_with_duty_and_is_clamped(self):
        vp = VehiclePack(PackSpec())
        assert vp.cooling_power_W(0.0) == 0.0
        assert vp.cooling_power_W(0.5) == pytest.approx(300.0)
        assert vp.cooling_power_W(7.0) == pytest.approx(600.0)

    def test_chemical_energy_integral_matches_the_time_integrated_chemical_power(self):
        vp = VehiclePack(PackSpec(), soc0=0.9)
        e0, total = vp.chemical_energy_Wh(), 0.0
        for _ in range(1200):
            st = vp.step(150.0, 0.5, cooling_duty=0.0)
            total += st.p_ocv25_W * 0.5 / 3600.0
        assert e0 - vp.chemical_energy_Wh() == pytest.approx(total, rel=2e-4)

    def test_polarization_energy_is_stored_then_released(self):
        vp = VehiclePack(PackSpec(), soc0=0.9)
        for _ in range(240):
            vp.step(120.0, 0.5, cooling_duty=0.0)
        stored = vp.rc_stored_energy_Wh()
        assert stored > 0.0
        for _ in range(8000):
            vp.step(0.0, 0.5, cooling_duty=0.0)
        assert vp.rc_stored_energy_Wh() < 0.01 * stored


class TestSpecValidation:
    @pytest.mark.parametrize("kw", [dict(n_series=0), dict(n_parallel=0), dict(cell_capacity_Ah=0.0)])
    def test_invalid_topology_is_rejected(self, kw):
        with pytest.raises(ValueError):
            PackSpec(**kw)

    def test_chemistries_other_than_nmc_work(self):
        for chem in ("lfp", "lto", "nca"):
            vp = VehiclePack(PackSpec(chemistry=chem, n_series=100), soc0=0.6)
            i = vp.solve_current(20_000.0, 0.5)
            assert np.isfinite(i) and vp.terminal_voltage(i, 0.5) * i == pytest.approx(20_000.0, abs=1e-3)
