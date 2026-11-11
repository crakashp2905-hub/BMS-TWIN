"""Tests: forward-dynamics virtual EV — physics, limits, energy balance, reproducibility, faults.

These tests check the simulator's *internal consistency* (energy closes, limits are obeyed,
results are reproducible, faults are handled) and its *physical plausibility*.  They do not
validate the model against a physical vehicle, and nothing here claims they do.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import bms
from bms.drive_cycles import DriveCycle, DriveCycleError, constant_speed, get_cycle
from bms.ev_drive import (
    FLAG_MOTOR,
    FLAG_PACK_DISCHARGE,
    FLAG_REGEN_PACK,
    FLAG_SOC_EMPTY,
    FLAG_THERMAL,
    FLAG_TRACTION,
    AuxiliaryLoads,
    EVConfig,
    EVSimulator,
    PackPowerError,
    SensorModel,
    describe_flags,
    generic_compact_ev,
    loss_breakdown,
    max_motoring_torque_for_power,
    max_regen_torque_for_power,
    simulate_cycle,
)
from bms.pack_coupling import PackLimits, PackSpec
from bms.sensor_fdi import SensorMonitor
from bms.telemetry import TelemetryRecord, check_sequence
from bms.vehicle_dynamics import PackModel, simulate_drive

_ROOT = Path(__file__).resolve().parents[1]
_G = 9.80665


def _cfg(**kw) -> EVConfig:
    """Coarse internal step (0.5 s) keeps the suite fast; accuracy is checked separately."""
    kw.setdefault("dt_sim_s", 0.5)
    return EVConfig(**kw)


def _run(cycle, cfg: EVConfig | None = None) -> bms.DriveRun:
    return EVSimulator(cfg or _cfg()).run(cycle)


def _road_load_N(veh, v, grade=0.0, rho=1.225):
    return (0.5 * rho * veh.drag_coefficient * veh.frontal_area_m2 * v * v
            + veh.rolling_resistance * veh.mass_kg * _G * math.cos(grade) + veh.mass_kg * _G * math.sin(grade))


# ── Motor helpers ──────────────────────────────────────────────────────
class TestMotorHelpers:
    def test_loss_breakdown_sums_to_the_existing_loss_function(self):
        mot = generic_compact_ev().motor
        rng = np.random.default_rng(0)
        for _ in range(50):
            tau, w = rng.uniform(-300, 300), rng.uniform(0, 1200)
            assert sum(loss_breakdown(mot, tau, w).values()) == pytest.approx(mot.loss_W(tau, w), rel=1e-12)

    @pytest.mark.parametrize("omega", [5.0, 100.0, 400.0, 800.0, 1200.0])
    @pytest.mark.parametrize("cap_W", [2_000.0, 20_000.0, 60_000.0])
    def test_motoring_cap_inverts_the_loss_model(self, omega, cap_W):
        mot = generic_compact_ev().motor
        tau = max_motoring_torque_for_power(mot, omega, cap_W)
        assert tau >= 0.0
        assert mot.electrical_power_W(tau, omega) == pytest.approx(cap_W, rel=1e-9)
        assert mot.electrical_power_W(tau * 1.01, omega) > cap_W            # a little more breaks the cap

    def test_motoring_cap_is_zero_when_it_is_below_the_idle_loss(self):
        mot = generic_compact_ev().motor
        assert max_motoring_torque_for_power(mot, 300.0, mot.p_idle_W * 0.5) == 0.0

    @pytest.mark.parametrize("omega", [60.0, 300.0, 900.0])
    @pytest.mark.parametrize("cap_W", [500.0, 5_000.0, 20_000.0])
    def test_regen_cap_inverts_the_recovered_power(self, omega, cap_W):
        mot = generic_compact_ev().motor
        m = max_regen_torque_for_power(mot, omega, cap_W)
        if not math.isfinite(m):
            return                                                          # cap above anything the motor can recover
        assert -mot.electrical_power_W(-m, omega) == pytest.approx(cap_W, rel=1e-9)
        assert -mot.electrical_power_W(-m * 0.99, omega) < cap_W            # slightly less torque recovers less

    def test_regen_cap_is_unlimited_when_unreachable(self):
        mot = generic_compact_ev().motor
        assert max_regen_torque_for_power(mot, 100.0, 1e9) == math.inf

    def test_efficiency_map_is_realistic(self):
        mot = generic_compact_ev().motor
        points = [(t, mot.efficiency(t, rpm * math.pi / 30)) for t in (10, 40, 100, 200, 300)
                  for rpm in (1000, 3000, 6000, 9000) if t * rpm * math.pi / 30 <= mot.peak_power_W]
        effs = [e for _, e in points]
        assert 0.93 <= max(effs) <= 0.985
        assert min(effs) > 0.80                                              # nowhere on the map is it absurd
        assert all(e > 0.92 for t, e in points if t in (40, 100))            # mid-load region is efficient
        assert mot.efficiency(10.0, 100.0) < 0.90                            # very light load is worse


# ── Physics checks against closed-form expectations ────────────────────
class TestLongitudinalPhysics:
    def test_steady_cruise_matches_road_load_through_the_drivetrain(self):
        cfg = _cfg()
        veh = cfg.vehicle
        v = 100.0 / 3.6
        run = _run(constant_speed(100.0, 60.0), cfg)
        p = run["p_dc_inverter_W"][5]
        f = _road_load_N(veh, v)
        tau = f * veh.wheel_radius_m / (veh.gear_ratio * veh.gear_efficiency)
        w = v * veh.gear_ratio / veh.wheel_radius_m
        assert p == pytest.approx(veh.motor.electrical_power_W(tau, w), rel=1e-9)
        assert run["p_wheel_W"][5] == pytest.approx(f * v, rel=1e-9)
        assert np.allclose(run["speed_mps"], v) and np.all(run["limit_flags"] == 0)

    def test_steady_cruise_agrees_with_the_legacy_inverse_model(self):
        cfg = _cfg()
        run = _run(constant_speed(100.0, 60.0), cfg)
        old = simulate_drive(constant_speed(100.0, 60.0).speed_mps, 1.0, cfg.vehicle, PackModel())
        assert run["p_dc_inverter_W"][10] == pytest.approx(old.p_electrical_W[10], rel=1e-9)

    @pytest.mark.parametrize("name", ["udds", "hwfet", "us06", "ece15"])
    def test_whole_cycle_energy_agrees_with_the_legacy_inverse_model(self, name):
        # The legacy model assumes the car always keeps up and uses 1 Hz central differences; the
        # forward model sub-steps and applies fade and limits.  Net motor-side energy must still agree.
        cfg = EVConfig()
        cyc = get_cycle(name)
        old = simulate_drive(cyc.speed_mps, 1.0, cfg.vehicle, PackModel())
        run = EVSimulator(cfg).run(cyc)
        p = run["p_dc_inverter_W"][1:]
        net = float(np.sum(p) * run.dt_s / 3600.0)
        assert old.all_feasible
        assert net == pytest.approx(old.net_energy_Wh, rel=0.01)

    def test_uphill_costs_exactly_the_gravity_work(self):
        cfg = _cfg()
        th = math.atan(0.05)
        flat = _run(constant_speed(60.0, 40.0), cfg)
        up = _run(constant_speed(60.0, 40.0).with_grade(th), cfg)
        v = 60.0 / 3.6
        veh = cfg.vehicle
        extra = veh.mass_kg * _G * v * (math.sin(th) + veh.rolling_resistance * (math.cos(th) - 1.0))
        assert up["p_wheel_W"][5] - flat["p_wheel_W"][5] == pytest.approx(extra, rel=1e-9)
        assert up.ledger.potential > 0.0

    def test_steep_descent_regenerates_and_charges_the_pack(self):
        down = _run(constant_speed(60.0, 60.0).with_grade(-math.atan(0.05)))
        assert np.all(down["p_wheel_W"][1:] < 0.0)
        assert down["power_W"][10] < 0.0 and down["current_A"][10] < 0.0
        assert down["soc_mean"][-1] > down["soc_mean"][0]
        assert down.ledger.potential < 0.0

    def test_a_standing_car_draws_only_auxiliaries_and_controller_losses(self):
        cfg = _cfg()
        run = _run(DriveCycle("stand", np.array([0.0, 120.0]), np.array([0.0, 0.0])), cfg)
        base, loss, hvac = cfg.aux.split_W(cfg.ambient_C)
        expected = base + loss + hvac + cfg.vehicle.motor.p_idle_W
        assert np.allclose(run["power_W"][1:], expected, rtol=1e-9)
        assert np.all(run["speed_mps"] == 0.0) and run.distance_km == 0.0
        assert run.ledger.vehicle_residual == pytest.approx(0.0, abs=1e-9)

    def test_kinetic_energy_gained_while_accelerating(self):
        cfg = _cfg()
        t = np.array([0.0, 20.0, 40.0])
        run = _run(DriveCycle("up", t, np.array([0.0, 20.0, 20.0])), cfg)
        m_eff = cfg.vehicle.effective_mass_kg
        assert run.ledger.kinetic == pytest.approx(0.5 * m_eff * 400.0 / 3600.0, rel=1e-12)

    def test_the_car_stops_exactly_and_never_reverses(self):
        cyc = get_cycle("ece15")
        run = _run(cyc)
        assert np.min(run["speed_mps"]) >= 0.0
        assert run["speed_mps"][-1] == 0.0
        # an unconstrained car decelerates exactly as the trace does, never harder
        assert np.min(run["accel_mps2"][1:]) >= -cyc.stats().max_decel_mps2 - 1e-9

    def test_driving_to_a_stop_on_a_steep_downhill_holds_the_car(self):
        # Gravity would roll it forward; the held-at-rest branch must keep it stationary.
        t = np.array([0.0, 20.0, 40.0, 90.0])
        run = _run(DriveCycle("down_stop", t, np.array([0.0, 10.0, 0.0, 0.0]), np.full(4, -0.2)))
        assert np.all(run["speed_mps"][run["t_s"] >= 40.5] == 0.0)
        assert run.ledger.relative_residuals()["vehicle"] < 1e-8

    def test_a_car_with_no_power_coasts_to_rest_and_the_energy_still_balances(self):
        # With the pack empty the only forces are resistances, which would carry the car backwards
        # in the last step; the stop must be exact and the ledger must still close.
        run = _run(constant_speed(36.0, 150.0), _cfg(soc0=0.0, limits=PackLimits(soc_floor=0.0)))
        v = run["speed_mps"]
        assert v[0] == 10.0 and v[-1] == 0.0 and np.all(v >= 0.0)
        assert np.all(np.diff(v) <= 1e-12)                                       # only ever slows down
        assert run.ledger.relative_residuals()["vehicle"] < 1e-9
        assert run.ledger.kinetic == pytest.approx(-0.5 * run.config.vehicle.effective_mass_kg * 100.0 / 3600.0,
                                                    rel=1e-12)

    def test_a_correctly_commanded_stop_keeps_its_motor_force(self):
        # The deadbeat command already lands exactly on zero speed; rounding must not make the stop
        # logic discard the motor force and re-scale the resistances instead.
        run = _run(DriveCycle("slow_stop", np.array([0.0, 40.0, 41.0, 60.0]), np.array([0.2, 0.2, 0.0, 0.0])))
        assert run["speed_mps"][-1] == 0.0
        assert np.all(np.isfinite(run["f_wheel_N"]))
        assert run.ledger.relative_residuals()["vehicle"] < 1e-9

    def test_hard_acceleration_is_limited_and_flagged_not_followed(self):
        t = np.arange(0.0, 30.0, 0.5)
        run = _run(DriveCycle("launch", t, np.minimum(t * (27.8 / 2.0), 27.8)))      # 0-100 km/h in 2 s
        m = run.metrics()
        assert m["trace_max_error_kmh"] > 20.0 and m["time_power_limited_pct"] > 5.0
        assert np.any(run["limit_flags"].astype(int) & (FLAG_MOTOR | FLAG_TRACTION))
        assert np.max(run["f_wheel_N"]) <= 0.8 * 1500.0 * _G + 1e-6            # never beyond tyre grip
        assert run["speed_mps"].max() <= run["speed_target_mps"].max() + 1e-9


# ── Pack, motor and tyre limits are respected ──────────────────────────
class TestLimitsAreRespected:
    US06 = get_cycle("us06").slice(0.0, 330.0)

    def test_unconstrained_car_follows_the_trace_exactly(self):
        m = _run(self.US06).metrics()
        assert m["trace_rms_error_kmh"] < 1e-6 and m["time_power_limited_pct"] == 0.0

    @pytest.mark.parametrize("cap_W", [25_000.0, 40_000.0, 60_000.0])
    def test_discharge_power_cap(self, cap_W):
        run = _run(self.US06, _cfg(limits=PackLimits(max_discharge_power_W=cap_W)))
        assert run["power_W"].max() <= cap_W * (1.0 + 1e-4)
        assert run.metrics()["trace_rms_error_kmh"] > 0.0
        assert np.any(run["limit_flags"].astype(int) & FLAG_PACK_DISCHARGE)
        assert run.ledger.relative_residuals()["vehicle"] < 1e-8              # energy still closes

    def test_discharge_current_cap(self):
        run = _run(self.US06, _cfg(limits=PackLimits(max_discharge_C=0.5)))
        assert run["current_A"].max() <= 0.5 * 180.0 * (1.0 + 1e-3)
        assert np.any(run["limit_flags"].astype(int) & FLAG_PACK_DISCHARGE)

    def test_tighter_caps_cost_more_trace_error(self):
        errs = [_run(self.US06, _cfg(limits=PackLimits(max_discharge_power_W=cap))).metrics()["trace_rms_error_kmh"]
                for cap in (60_000.0, 40_000.0, 25_000.0)]
        assert errs[0] < errs[1] < errs[2]

    def test_regen_power_cap_shifts_energy_to_the_friction_brakes(self):
        free = _run("urban_stop_go")
        capped = _run("urban_stop_go", _cfg(limits=PackLimits(max_charge_power_W=3_000.0)))
        assert capped["power_W"].min() >= -3_000.0 * (1.0 + 1e-4)
        # braking asks for far more than 3 kW, so the cap must be fully used (net of the auxiliaries)
        assert capped["power_W"].min() == pytest.approx(-3_000.0, rel=1e-3)
        assert capped.ledger.regen_out < free.ledger.regen_out
        assert capped.ledger.friction_brakes > free.ledger.friction_brakes
        assert np.any(capped["limit_flags"].astype(int) & FLAG_REGEN_PACK)
        assert capped.ledger.relative_residuals()["vehicle"] < 1e-8

    def test_regeneration_fades_near_standstill_and_friction_takes_over(self):
        # 3 m/s² from 1.5 m/s: at the step's mean speed (0.75 m/s) the regen allowance is halved,
        # so the motor takes exactly half of its regen torque limit and the friction brakes the rest.
        t = np.array([0.0, 10.0, 10.5, 20.0])
        run = _run(DriveCycle("hard_stop", t, np.array([1.5, 1.5, 0.0, 0.0])))
        k = int(np.argmax(run["f_brake_N"]))
        mot = run.config.vehicle.motor
        assert run["f_brake_N"][k] > 0.0
        assert run["motor_torque_Nm"][k] == pytest.approx(-0.5 * mot.regen_torque_fraction * mot.peak_torque_Nm,
                                                          rel=1e-9)
        assert run.ledger.relative_residuals()["vehicle"] < 1e-9

    def test_motor_torque_and_power_limits(self):
        weak = replace(generic_compact_ev(), motor=replace(generic_compact_ev().motor, peak_power_W=45_000.0,
                                                           peak_torque_Nm=180.0))
        run = _run(self.US06, _cfg(vehicle=weak))
        assert np.max(run["motor_torque_Nm"]) <= 180.0 + 1e-6
        assert np.max(run["p_shaft_W"]) <= 45_000.0 * 1.04                     # mean-speed accounting allowance
        assert np.any(run["limit_flags"].astype(int) & FLAG_MOTOR)

    def test_tyre_grip_limits_traction_on_a_slippery_road(self):
        icy = replace(generic_compact_ev(), friction_coefficient=0.15)
        run = _run(self.US06, _cfg(vehicle=icy))
        assert np.max(np.abs(run["f_wheel_N"])) <= 0.15 * 1500.0 * _G + 1e-6
        assert np.any(run["limit_flags"].astype(int) & FLAG_TRACTION)

    def test_cell_voltage_window_is_never_breached(self):
        run = _run(self.US06, _cfg(soc0=0.08))
        v_min = bms.get_chemistry_props("nmc")["v_min"]
        assert run["cell_v_min_V"][1:].min() >= v_min - 0.01
        assert run["soc_min"].min() >= 0.0

    def test_empty_pack_stops_the_car_without_overdrawing_it(self):
        run = _run(self.US06, _cfg(soc0=0.01, limits=PackLimits(soc_floor=0.005)))
        assert run["soc_min"].min() >= 0.0
        assert np.any(run["limit_flags"].astype(int) & FLAG_SOC_EMPTY)
        assert run.metrics()["trace_within_tolerance_pct"] < 60.0

    def test_cold_pack_blocks_regeneration_entirely(self):
        run = _run("urban_stop_go", _cfg(ambient_C=-25.0, initial_temperature_C=-25.0, aux=AuxiliaryLoads()))
        assert run.ledger.regen_out == pytest.approx(0.0, abs=1e-9)
        assert np.all(run["power_W"][1:] >= -1e-6)                              # nothing flows back in
        assert run.ledger.friction_brakes > 0.0
        assert np.any(run["limit_flags"].astype(int) & FLAG_THERMAL)

    def test_full_pack_tapers_regeneration(self):
        full = _run("udds", _cfg(soc0=0.995))
        mid = _run("udds", _cfg(soc0=0.5))
        # SOC falls a couple of percent over the cycle, so the taper relaxes as it runs: a clear but
        # partial loss of regeneration, not a cut-off.
        assert full.ledger.regen_out < 0.85 * mid.ledger.regen_out
        assert full.ledger.friction_brakes > mid.ledger.friction_brakes
        assert np.any(full["limit_flags"].astype(int) & FLAG_REGEN_PACK)
        assert _run("udds", _cfg(soc0=1.0)).ledger.regen_out < full.ledger.regen_out

    def test_hot_pack_is_derated_and_flagged(self):
        run = _run(self.US06, _cfg(ambient_C=55.0, initial_temperature_C=58.0))
        flags = run["limit_flags"].astype(int)
        assert np.mean(flags & FLAG_THERMAL != 0) > 0.9
        # the derate (60 − T)/15 of the current cap is what limits power, and every step obeys it
        lim = run["limit_power_W"]
        driving = np.isfinite(lim) & (lim > 0.0)
        assert np.all(run["power_W"][driving] <= lim[driving] * (1.0 + 1e-4))
        assert run["power_W"].max() < 0.5 * _run(self.US06).metrics()["peak_power_kW"] * 1000.0


# ── Energy accounting ──────────────────────────────────────────────────
_SCENARIOS = {
    "urban_stop_go": (get_cycle("urban_stop_go"), {}),
    "ece15": (get_cycle("ece15"), {}),
    "eudc_slice": (get_cycle("eudc").slice(0.0, 250.0), {}),
    "hilly": (get_cycle("arterial").with_grade(np.sin(np.linspace(0, 6 * np.pi, 361)) * 0.06), {}),
    "us06_power_limited": (get_cycle("us06").slice(0.0, 250.0), {"limits": PackLimits(max_discharge_power_W=35_000.0)}),
    "regen_limited": (get_cycle("urban_stop_go"), {"limits": PackLimits(max_charge_power_W=2_000.0)}),
    "cold_start": (get_cycle("ece15"), {"ambient_C": -10.0}),
    "hot_day": (get_cycle("ece15"), {"ambient_C": 38.0}),
    "high_soc": (get_cycle("udds").slice(0.0, 300.0), {"soc0": 0.99}),
    "lfp_pack": (get_cycle("ece15"), {"pack": PackSpec(chemistry="lfp", n_series=110)}),
}


class TestEnergyBalance:
    @pytest.mark.parametrize("name", sorted(_SCENARIOS))
    def test_vehicle_side_energy_closes_to_numerical_precision(self, name):
        cycle, kw = _SCENARIOS[name]
        lg = _run(cycle, _cfg(**kw)).ledger
        # terminal energy = Δ kinetic + Δ potential + resistances + brakes + all drivetrain losses + auxiliaries
        assert abs(lg.vehicle_residual) < 1e-8 * max(lg.traction_in, 1.0)

    @pytest.mark.parametrize("name", sorted(_SCENARIOS))
    def test_pack_side_energy_closes_to_the_time_stepping_error(self, name):
        cycle, kw = _SCENARIOS[name]
        lg = _run(cycle, _cfg(**kw)).ledger
        # chemical energy = terminal + ohmic + polarization + hysteresis − EMF shift (O(dt) in the RC states)
        assert abs(lg.pack_residual) < 1e-3 * abs(lg.chemical_ocv) + 0.05

    @pytest.mark.parametrize("name", ["urban_stop_go", "ece15", "eudc_slice", "hilly", "us06_power_limited"])
    def test_chemical_energy_matches_the_exact_ocv_integral(self, name):
        # Independent of the time stepping: the pack's drop in stored energy per the OCV curve.
        cycle, kw = _SCENARIOS[name]
        lg = _run(cycle, _cfg(**kw)).ledger
        assert abs(lg.spline_residual) < 1e-3 * abs(lg.chemical_spline) + 0.05

    def test_pack_residual_shrinks_as_the_step_shrinks(self):
        res = [abs(_run("ece15", EVConfig(dt_sim_s=dt)).ledger.pack_residual) for dt in (1.0, 0.5, 0.25, 0.125)]
        assert res[0] > res[1] > res[2] > res[3]
        assert res[0] / res[3] > 4.0

    def test_every_dissipation_term_is_non_negative_where_it_must_be(self):
        lg = _run("urban_stop_go").ledger
        for name in ("aerodynamic", "rolling", "friction_brakes", "gearbox", "motor_copper", "motor_iron",
                     "inverter", "controller", "aux_low_voltage", "aux_converter", "ohmic", "hysteresis"):
            assert getattr(lg, name) >= 0.0, name

    def test_a_round_trip_over_a_closed_cycle_returns_less_than_it_took(self):
        lg = _run("urban_stop_go").ledger
        assert 0.0 < lg.regen_out < lg.traction_in
        assert lg.kinetic == pytest.approx(0.0, abs=1e-12) and lg.potential == pytest.approx(0.0, abs=1e-9)

    def test_energy_channel_integrates_the_terminal_power(self):
        run = _run("ece15")
        assert run["energy_Wh"][-1] == pytest.approx(run.ledger.terminal_net, rel=1e-12)

    def test_soc_drop_is_consistent_with_the_energy_drawn(self):
        run = _run("udds")
        drop_Wh = (run["soc_mean"][0] - run["soc_mean"][-1]) * 65_000.0
        assert drop_Wh == pytest.approx(run.ledger.chemical_spline, rel=0.08)


# ── Reproducibility ────────────────────────────────────────────────────
class TestReproducibility:
    def test_same_configuration_gives_bit_identical_results(self):
        a, b = _run("urban_stop_go"), _run("urban_stop_go")
        assert a.digest() == b.digest()
        assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.channels)

    def test_a_fresh_interpreter_reproduces_the_same_digest(self):
        code = ("import bms;"
                "r=bms.EVSimulator(bms.EVConfig(dt_sim_s=0.5)).run('urban_stop_go');print(r.digest())")
        out = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == _run("urban_stop_go").digest()

    def test_the_seed_changes_cell_scatter_but_not_the_physics(self):
        a = _run("urban_stop_go", _cfg(seed=1))
        b = _run("urban_stop_go", _cfg(seed=2))
        assert a.digest() != b.digest()
        assert a.metrics()["consumption_Wh_per_km"] == pytest.approx(b.metrics()["consumption_Wh_per_km"], rel=0.01)

    def test_manifest_pins_what_is_needed_to_reproduce(self):
        run = _run("ece15")
        man = run.manifest()
        assert man["bms_version"] == bms.__version__ and man["cycle"] == "ece15"
        assert man["cycle_fingerprint"] == get_cycle("ece15").fingerprint()
        assert man["config_hash"] == _cfg().config_hash() and man["result_digest"] == run.digest()
        assert man == _run("ece15").manifest()

    def test_config_hash_tracks_every_parameter(self):
        base = _cfg().config_hash()
        assert base == _cfg().config_hash()
        for kw in (dict(soc0=0.8), dict(ambient_C=20.0), dict(seed=3), dict(dt_sim_s=0.25),
                   dict(vehicle=replace(generic_compact_ev(), mass_kg=1501.0)),
                   dict(pack=PackSpec(n_parallel=59)), dict(limits=PackLimits(max_discharge_C=2.9)),
                   dict(aux=AuxiliaryLoads(base_W=351.0))):
            assert _cfg(**kw).config_hash() != base, kw

    def test_replaying_a_run_from_its_manifest_reproduces_it(self):
        cfg = _cfg(soc0=0.7, ambient_C=10.0, seed=9)
        first = _run("ece15", cfg).manifest()
        again = _run("ece15", cfg).manifest()
        assert first == again

    def test_telemetry_is_reproducible_and_identical_to_the_streamed_records(self):
        cyc = get_cycle("ece15")
        sim = EVSimulator(_cfg())
        streamed = list(sim.stream(cyc, telemetry_dt_s=1.0))
        recorded = _run(cyc).telemetry(1.0)
        assert streamed == recorded
        assert recorded == _run(cyc).telemetry(1.0)


# ── Plausibility ───────────────────────────────────────────────────────
class TestPlausibility:
    def test_consumption_cannot_beat_the_resistive_energy(self):
        for name in ("urban_stop_go", "ece15", "udds"):
            run = _run(name)
            resistive = run.ledger.aerodynamic + run.ledger.rolling
            assert run.ledger.terminal_net >= resistive               # losses and auxiliaries only add
            assert run.ledger.terminal_net < 3.0 * resistive          # and are not absurdly large

    def test_standard_cycle_consumption_is_in_a_sane_band(self):
        # Order-of-magnitude sanity for a 1.5 t compact EV at the battery terminals, not a validation.
        for name, lo, hi in (("udds", 70, 150), ("hwfet", 90, 170), ("us06", 110, 220), ("nedc", 80, 150)):
            c = _run(name).metrics()["consumption_Wh_per_km"]
            assert lo < c < hi, (name, c)

    def test_highway_costs_more_per_km_than_a_gentle_arterial(self):
        assert _run("highway").metrics()["consumption_Wh_per_km"] > _run("arterial").metrics()["consumption_Wh_per_km"]

    def test_drivetrain_efficiency_is_in_the_realistic_band(self):
        for name in ("udds", "hwfet", "us06"):
            assert 0.82 < _run(name).metrics()["drivetrain_efficiency"] < 0.96

    @pytest.mark.parametrize("field, low, high", [
        ("mass_kg", 1300.0, 1900.0), ("drag_coefficient", 0.24, 0.40), ("rolling_resistance", 0.008, 0.015),
    ])
    def test_consumption_rises_with_mass_drag_and_rolling_resistance(self, field, low, high):
        v = generic_compact_ev()
        a = _run("udds", _cfg(vehicle=replace(v, **{field: low}))).metrics()["consumption_Wh_per_km"]
        b = _run("udds", _cfg(vehicle=replace(v, **{field: high}))).metrics()["consumption_Wh_per_km"]
        assert b > a

    def test_auxiliary_load_costs_energy_in_proportion_to_time(self):
        base = _run("udds", _cfg(aux=AuxiliaryLoads(base_W=0.0, hvac_cooling_W_per_K=0.0, hvac_heating_W_per_K=0.0)))
        loaded = _run("udds", _cfg(aux=AuxiliaryLoads(base_W=1000.0 * 0.93, dcdc_efficiency=0.93,
                                                      hvac_cooling_W_per_K=0.0, hvac_heating_W_per_K=0.0)))
        extra = loaded.ledger.terminal_net - base.ledger.terminal_net
        assert extra == pytest.approx(1000.0 * 1369.0 / 3600.0, rel=0.01)

    def test_cold_and_hot_days_both_cost_more_than_a_mild_one(self):
        mild = _run("udds", _cfg(ambient_C=22.0)).metrics()["consumption_Wh_per_km"]
        cold = _run("udds", _cfg(ambient_C=-10.0)).metrics()["consumption_Wh_per_km"]
        hot = _run("udds", _cfg(ambient_C=38.0)).metrics()["consumption_Wh_per_km"]
        assert cold > 1.1 * mild and hot > mild

    def test_electrical_quantities_stay_inside_physical_windows(self):
        run = _run("us06")
        spec, props = PackSpec(), bms.get_chemistry_props("nmc")
        assert np.all(run["v_pack_V"] > spec.n_series * props["v_min"])
        assert np.all(run["v_pack_V"] < spec.n_series * props["v_max"])
        assert np.all(np.abs(run["current_A"]) < 3.0 * spec.capacity_Ah)
        assert 0.0 <= run["soc_min"].min() and run["soc_max"].max() <= 1.0
        assert run["soc_mean"][-1] < run["soc_mean"][0]
        assert run["temp_min_C"].min() >= 25.0 - 1e-9 and run["temp_max_C"].max() < 40.0

    def test_regen_share_is_substantial_in_town_and_negligible_at_a_steady_cruise(self):
        assert 0.15 < _run("udds").metrics()["regen_share_of_traction"] < 0.45
        assert _run(constant_speed(100.0, 120.0)).ledger.regen_out == pytest.approx(0.0, abs=1e-9)

    def test_speed_and_accel_stay_physical(self):
        run = _run("us06")
        assert np.max(np.abs(run["accel_mps2"])) < 6.0 and run["speed_mps"].max() < 40.0


# ── Fault injection ────────────────────────────────────────────────────
class TestFaultInjection:
    US06 = get_cycle("us06").slice(0.0, 300.0)

    def test_unknown_cycle_and_bad_configuration_fail_clearly(self):
        with pytest.raises(DriveCycleError):
            EVSimulator(_cfg()).run("no_such_cycle")
        with pytest.raises(ValueError):
            EVConfig(dt_sim_s=0.0)
        with pytest.raises(ValueError):
            EVConfig(soc0=1.5)
        with pytest.raises(ValueError):
            EVConfig(vehicle=replace(generic_compact_ev(), mass_kg=0.0))

    def test_a_pack_too_small_to_carry_the_auxiliaries_raises_instead_of_lying(self):
        with pytest.raises(PackPowerError, match="exceeds what the pack"):
            _run("ece15", _cfg(pack=PackSpec(n_series=2, n_parallel=1)))

    def test_a_weak_cell_appearing_mid_drive_lowers_the_limit_and_is_survived(self):
        def weaken(k, t, pack):
            if t >= 60.0 and not getattr(pack, "_weakened", False):
                pack.r0_25[11] *= 15.0
                pack._weakened = True
        cfg = _cfg(soc0=0.12)
        clean = _run(self.US06, cfg)
        faulty = EVSimulator(cfg, on_step=weaken).run(self.US06)
        assert faulty.metrics()["trace_rms_error_kmh"] >= clean.metrics()["trace_rms_error_kmh"]
        assert faulty["cell_v_min_V"][1:].min() >= bms.get_chemistry_props("nmc")["v_min"] - 0.01
        assert faulty.ledger.relative_residuals()["vehicle"] < 1e-8

    def test_a_sudden_cell_imbalance_is_survived_and_the_weakest_group_governs(self):
        def drop_one(k, t, pack):
            if k == 10:
                pack.soc[3] -= 0.55
        run = EVSimulator(_cfg(soc0=0.6), on_step=drop_one).run("ece15")
        assert run["soc_min"][-1] < run["soc_mean"][-1] - 0.4
        assert run["cell_v_min_V"][-1] < run["cell_v_max_V"][-1] - 0.3
        assert run["soc_min"].min() >= 0.0

    def test_a_thermal_event_derates_the_pack_instead_of_overheating_it(self):
        def heat(k, t, pack):
            if k == 20:
                pack.thermal.T[:] = 59.5
        clean = _run(self.US06)
        run = EVSimulator(_cfg(), on_step=heat).run(self.US06)
        after = run["limit_flags"][25:].astype(int)
        assert np.mean(after & FLAG_THERMAL != 0) > 0.9
        window = slice(21, 70)                       # right after the event the pack is almost cut off
        assert run["power_W"][window].max() < 0.5 * clean["power_W"][window].max()
        assert run["temp_max_C"].max() < 61.0
        assert run.ledger.relative_residuals()["vehicle"] < 1e-8

    def test_failed_cooling_in_a_hot_climate_is_survived(self):
        spec = PackSpec(h_cooling_max_W_per_K=20.0, cooling_power_max_W=0.0)
        run = _run(self.US06, _cfg(pack=spec, ambient_C=52.0, initial_temperature_C=52.0))
        assert run["temp_max_C"].max() < 60.0 + 1.0
        assert run.ledger.relative_residuals()["vehicle"] < 1e-8

    def test_extreme_cold_soak_is_survived_with_regeneration_off_and_energy_still_balanced(self):
        run = _run("ece15", _cfg(ambient_C=-30.0, initial_temperature_C=-30.0))
        assert np.all(run["power_W"][1:] >= -1e-6)
        assert abs(run.ledger.vehicle_residual) < 1e-8 * run.ledger.traction_in

    def test_an_impossible_hill_is_flagged_not_climbed(self):
        run = _run(constant_speed(60.0, 40.0).with_grade(0.45), _cfg())
        assert np.any(run["limit_flags"].astype(int) & (FLAG_MOTOR | FLAG_TRACTION | FLAG_PACK_DISCHARGE))
        assert run["speed_mps"][-1] < run["speed_target_mps"][-1]
        assert abs(run.ledger.vehicle_residual) < 1e-8 * max(run.ledger.traction_in, 1.0)

    def test_a_dead_battery_start_does_not_crash_or_overdraw(self):
        run = _run(self.US06, _cfg(soc0=0.0, limits=PackLimits(soc_floor=0.0)))
        assert run["soc_min"].min() >= 0.0 and np.all(np.isfinite(run["current_A"]))

    # Sensor faults on the telemetry stream, caught by the repository's own sensor monitors.
    @staticmethod
    def _monitor() -> SensorMonitor:
        return SensorMonitor(lo=200.0, hi=420.0, max_rate=40.0, stuck_window=15)

    def _labels(self, values) -> list[str]:
        mon = self._monitor()
        return [mon.check(float(v)) for v in values]

    def test_clean_telemetry_raises_no_sensor_alarms(self):
        volts = [r.voltage_V for r in _run("udds").telemetry(1.0)]
        assert set(self._labels(volts)) == {"ok"}

    def test_a_stuck_voltage_sensor_is_detected(self):
        volts = [r.voltage_V for r in _run("udds").telemetry(1.0)]
        volts[200:] = [volts[200]] * (len(volts) - 200)
        labels = self._labels(volts)
        assert "stuck" in labels[200:] and set(labels[:200]) == {"ok"}

    def test_a_dropped_out_voltage_sensor_is_detected(self):
        volts = [r.voltage_V for r in _run("udds").telemetry(1.0)]
        volts[150] = float("nan")
        assert self._labels(volts)[150] == "dropout"

    def test_a_voltage_spike_is_detected(self):
        volts = [r.voltage_V for r in _run("udds").telemetry(1.0)]
        volts[300] += 150.0
        assert self._labels(volts)[300] in ("rate", "out_of_range")

    def test_a_corrupted_record_fails_schema_validation(self):
        recs = _run("ece15").telemetry(1.0)
        bad = replace(recs[10], voltage_V=float("nan"))
        assert bad.problems() and not recs[10].problems()
        assert replace(recs[11], soc=1.4).problems()

    def test_sensor_noise_changes_the_telemetry_but_never_the_physics(self):
        run = _run("ece15")
        noisy = run.telemetry(1.0, sensor=SensorModel(voltage_sigma_V=0.5, current_sigma_A=1.0,
                                                      temperature_sigma_C=0.2, seed=4))
        clean = run.telemetry(1.0)
        assert noisy != clean and noisy == run.telemetry(1.0, sensor=SensorModel(
            voltage_sigma_V=0.5, current_sigma_A=1.0, temperature_sigma_C=0.2, seed=4))
        assert float(np.std([n.voltage_V - c.voltage_V for n, c in zip(noisy, clean)])) == pytest.approx(0.5, rel=0.3)
        assert run.digest() == _run("ece15").digest()


# ── Telemetry produced by runs ─────────────────────────────────────────
class TestRunTelemetry:
    def test_records_are_valid_contiguous_and_timestamped(self):
        run = _run("udds")
        recs = run.telemetry(1.0)
        assert len(recs) == 1370 and all(isinstance(r, TelemetryRecord) for r in recs)
        assert all(not r.problems() for r in recs)
        rep = check_sequence(recs)
        assert rep.ok and rep.max_dt_s == pytest.approx(1.0)
        assert recs[0].timestamp == "2026-01-01T00:00:00.000Z" and recs[-1].timestamp == "2026-01-01T00:22:49.000Z"

    def test_core_channels_agree_with_the_run(self):
        run = _run("ece15")
        recs = run.telemetry(1.0)
        idx = run.sample_indices(1.0)
        for r, i in list(zip(recs, idx))[::40]:
            assert r.speed_mps == run["speed_mps"][i] and r.power_W == run["power_W"][i]
            assert r.current_A == run["current_A"][i] and r.voltage_V == run["v_pack_V"][i]
            assert r.power_W == pytest.approx(r.voltage_V * r.current_A, rel=1e-9, abs=1e-6) or i == 0

    def test_state_and_limit_flags_are_reported(self):
        run = _run(get_cycle("us06").slice(0.0, 200.0), _cfg(limits=PackLimits(max_discharge_power_W=30_000.0)))
        recs = run.telemetry(1.0)
        flagged = [r for r in recs if r.limit_flags]
        assert flagged and "pack_discharge_limit" in describe_flags(flagged[0].limit_flags)
        assert {r.state for r in recs} <= {"idle", "driving"}

    def test_a_faster_telemetry_rate_gives_proportionally_more_records(self):
        run = _run("ece15")
        assert len(run.telemetry(0.5)) == 2 * (len(run.telemetry(1.0)) - 1) + 1

    def test_simulate_cycle_one_liner(self):
        run = simulate_cycle("ece15", soc0=0.5, dt_sim_s=0.5)
        assert run["soc_mean"][0] == pytest.approx(0.5, abs=0.02)

    def test_cycles_not_starting_at_rest_or_time_zero_still_run(self):
        t = np.array([100.0, 130.0, 160.0])
        run = _run(DriveCycle("offset", t, np.array([10.0, 10.0, 0.0])))
        assert run["speed_mps"][0] == 10.0 and run["speed_mps"][-1] == 0.0 and run["t_s"][0] == 0.0
        assert hashlib.sha256(run["t_s"].tobytes()).hexdigest()
