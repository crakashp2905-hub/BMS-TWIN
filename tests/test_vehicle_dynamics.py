"""Tests: time-domain vehicle, motor/inverter and drive-cycle → battery-load model."""

from __future__ import annotations

import numpy as np

import bms
from bms.vehicle_dynamics import (
    MotorInverter,
    PackModel,
    Vehicle,
    apply_motor_overrides,
    drive_to_battery_load,
    highway,
    simulate_drive,
    urban_stop_go,
)

_G = 9.80665


def _flat_cruise(kmh=100.0, n=600):
    return np.full(n, kmh / 3.6)


class TestPackModel:
    def test_pack_is_physically_consistent(self):
        pk = PackModel()
        assert 55.0 < pk.energy_kWh < 75.0             # a mid-size EV pack
        assert pk.capacity_Ah == 180.0
        # cruise at 100 km/h must be a gentle rate, not a multi-C abuse
        i = pk.current_for_power(12_700.0)
        assert 0.15 < i / pk.capacity_Ah < 0.25

    def test_power_to_current_round_trip(self):
        pk = PackModel()
        for p in (5_000.0, 20_000.0, -8_000.0):       # discharge and regeneration
            i = pk.current_for_power(p)
            assert abs(pk.bus_voltage(i) * i - p) < 1.0

    def test_unreachable_power_is_nan(self):
        pk = PackModel()
        assert np.isnan(pk.current_for_power(1e9))     # more than the pack can supply


class TestMotorInverter:
    def test_efficiency_is_realistic_across_the_map(self):
        m = MotorInverter()
        for tq, w in [(50, 200), (150, 300), (200, 800)]:
            assert 0.90 < m.efficiency(tq, w) < 0.99

    def test_torque_limit_is_constant_torque_then_constant_power(self):
        m = MotorInverter()
        base = m.peak_power_W / m.peak_torque_Nm
        assert m.max_torque_Nm(base * 0.5) == m.peak_torque_Nm   # below base speed
        assert m.max_torque_Nm(base * 2.0) < m.peak_torque_Nm    # field-weakened region


class TestLongitudinalDynamics:
    def test_cruise_power_matches_resistive_work(self):
        veh, pk = Vehicle(), PackModel()
        r = simulate_drive(_flat_cruise(), 1.0, veh, pk)
        v = 100.0 / 3.6
        rho = 1.225
        f_res = 0.5 * rho * veh.drag_coefficient * veh.frontal_area_m2 * v ** 2 \
            + veh.rolling_resistance * veh.mass_kg * _G
        overhead = r.p_electrical_W[300] / (f_res * v)
        # ≈ 1/(η_gear · η_motor · η_inverter): a few percent over the wheel power
        assert 1.02 < overhead < 1.15

    def test_hill_climb_raises_power_and_descent_regenerates(self):
        veh, pk = Vehicle(), PackModel()
        v = _flat_cruise()
        flat = simulate_drive(v, 1.0, veh, pk).p_electrical_W[300]
        up = simulate_drive(v, 1.0, veh, pk, grade_rad=np.arctan(0.05)).p_electrical_W[300]
        down = simulate_drive(v, 1.0, veh, pk, grade_rad=-np.arctan(0.05)).p_electrical_W[300]
        assert up > flat + 15_000.0                    # ≈ m·g·sinθ·v of extra power
        assert down < 0.0                              # descent returns energy

    def test_stop_go_recovers_energy_but_not_all(self):
        veh, pk = Vehicle(), PackModel()
        r = simulate_drive(urban_stop_go(), 1.0, veh, pk)
        assert r.all_feasible
        assert 0.0 < r.energy_regenerated_Wh < r.energy_consumed_Wh
        assert r.net_energy_Wh > 0.0

    def test_hard_acceleration_is_flagged_infeasible(self):
        veh, pk = Vehicle(), PackModel()
        dt = 0.1
        t = np.arange(0.0, 40.0, dt)
        v = np.minimum(t * (27.8 / 2.0), 27.8)         # 0–100 km/h in 2 s
        r = simulate_drive(v, dt, veh, pk)
        assert not r.all_feasible                      # the motor cannot follow this
        assert np.max(np.abs(r.motor_torque_Nm)) > veh.motor.peak_torque_Nm

    def test_motor_power_limit_binds_at_high_speed(self):
        # 0–100 km/h in 8 s: within the tyre-road limit, so the motor is what binds.
        dt = 0.1
        t = np.arange(0.0, 30.0, dt)
        v = np.minimum(t * (27.8 / 8.0), 27.8)
        weak = Vehicle()                                           # 60 kW motor
        strong = apply_motor_overrides(weak, peak_torque_Nm=400.0, peak_power_W=200_000.0)
        assert not simulate_drive(v, dt, weak, PackModel()).all_feasible
        assert simulate_drive(v, dt, strong, PackModel()).all_feasible

    def test_traction_limit_binds_even_for_a_huge_motor(self):
        # 0–100 km/h in 2 s needs more force than the tyres can transmit (μ = 0.8):
        # no motor can deliver it, so the infeasibility must persist.
        dt = 0.1
        t = np.arange(0.0, 40.0, dt)
        v = np.minimum(t * (27.8 / 2.0), 27.8)
        monster = apply_motor_overrides(Vehicle(), peak_torque_Nm=5_000.0,
                                        peak_power_W=2_000_000.0)
        assert not simulate_drive(v, dt, monster, PackModel()).all_feasible


class TestDriveCycleToBatteryLoad:
    def test_drive_produces_a_battery_current_trace(self):
        r = simulate_drive(highway(), 1.0, Vehicle(), PackModel())
        i = drive_to_battery_load(r)
        assert i.shape == r.t_s.shape
        assert np.all(np.isfinite(i))
        assert i.max() > 0.0 and i.min() < 0.0         # discharge and regeneration

    def test_consumption_scales_with_mass(self):
        pk = PackModel()
        light = simulate_drive(highway(), 1.0, Vehicle(mass_kg=1100.0), pk)
        heavy = simulate_drive(highway(), 1.0, Vehicle(mass_kg=1900.0), pk)
        assert heavy.consumption_Wh_per_km > light.consumption_Wh_per_km

    def test_twin_tracks_the_drive_energy(self):
        # The drive cycle's battery current, applied to a cell twin, should
        # discharge the cell by about the coulomb-counted amount.
        pk = PackModel()
        r = simulate_drive(urban_stop_go(), 1.0, Vehicle(), pk)
        i_cell = drive_to_battery_load(r) / pk.n_parallel
        P = bms.ECMParameters(R0=0.03, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=3.0)
        sim = bms.SecondOrderECM(params=P, ocv_curve=bms.OCVSOC.from_chemistry("nmc")
                                 ).simulate(i_cell, 1.0, soc0=0.9)
        tw = bms.UnifiedTwin(params=P, chemistry="nmc")
        tw.reset(0.9)
        end = tw.run(i_cell, sim["v_terminal"], 1.0)[-1]
        coulomb_end = 0.9 - np.sum(i_cell) / 3600.0 / 3.0
        assert abs(sim["soc"][-1] - coulomb_end) < 1e-3
        assert abs(end.soc - coulomb_end) < 0.02       # twin within 2 % SoC on this trace
