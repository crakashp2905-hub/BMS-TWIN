"""Forward-dynamics virtual EV: speed profile → driver → drivetrain → battery → telemetry.

:mod:`bms.vehicle_dynamics` runs the *inverse* problem (given the speed, what power?)
and assumes the vehicle always keeps up.  This module runs the **forward** problem: a
driver tries to follow the target speed, the powertrain and the battery limit what the
car can actually do, and the speed that results is what gets reported.  When the pack,
the motor or the tyres cannot deliver, the car falls behind the trace and says why.

    target speed v*(t)
        │  driver: feed-forward + PI on speed error → wheel force command
        ▼
    powertrain limits: motor torque/power, tyre grip, pack discharge/charge power
        │  regeneration blended with friction brakes (fades near standstill)
        ▼
    wheel force → gear → motor + inverter loss model → DC bus  (+ auxiliary loads)
        │  bus power P
        ▼
    VehiclePack: P = V(I)·I → current, SOC, polarization, temperature  → terminal voltage

Energy accounting
-----------------
Every step computes each power flow from the *same* force, speed and current, then the
:class:`EnergyLedger` sums them independently:

    pack terminal energy  =  Δ kinetic + Δ potential + aerodynamic + rolling
                             + friction brakes + gearbox + motor copper + motor iron
                             + inverter + controller + auxiliaries

and, on the battery side, the chemical energy the pack gave up (an exact integral of its
OCV curve, independent of the time stepping) equals terminal energy plus ohmic,
polarization and hysteresis losses.  The residuals are reported, not hidden.

Honest scope
------------
This is a longitudinal, quasi-3-D simulation with a lumped motor loss model and generic
parameters.  It has **not** been validated against a physical vehicle.  Its energy
accounting is checked for internal consistency, and its outputs for physical plausibility;
neither is validation against measured vehicle data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Iterator

import numpy as np

from .drive_cycles import DriveCycle, get_cycle
from .pack_coupling import LIM_SOC, PackLimits, PackSpec, VehiclePack
from .sop import SOPLimit
from .telemetry import DEFAULT_EPOCH_UNIX_S, TelemetryRecord, format_timestamp
from .vehicle_dynamics import MotorInverter, Vehicle

_G = 9.80665
_V_REST_MPS = 1e-9      # speeds below this count as at rest (snaps float dust so a stop is exactly 0)

# Active-limit flags, reported in ``limit_flags`` of each telemetry record.
FLAG_PACK_DISCHARGE = 1      # pack discharge limit reduced traction
FLAG_MOTOR = 2               # motor torque / power / speed limit reduced traction
FLAG_TRACTION = 4            # tyre-road friction limited the wheel force
FLAG_REGEN_PACK = 8          # pack charge limit, SOC or temperature reduced regeneration
FLAG_REGEN_MOTOR = 16        # motor capability or low-speed fade reduced regeneration
FLAG_SOC_EMPTY = 32          # pack at its SOC floor: no discharge available
FLAG_THERMAL = 64            # temperature derating active
SOP_HORIZONS_S = (2.0, 10.0, 30.0)     # the horizons the CAN SOP frames carry
_FLAG_NAMES = {FLAG_PACK_DISCHARGE: "pack_discharge_limit", FLAG_MOTOR: "motor_limit",
               FLAG_TRACTION: "traction_limit", FLAG_REGEN_PACK: "regen_pack_limit",
               FLAG_REGEN_MOTOR: "regen_motor_limit", FLAG_SOC_EMPTY: "soc_empty",
               FLAG_THERMAL: "thermal_derate"}
_TRACTION_LIMIT_FLAGS = FLAG_PACK_DISCHARGE | FLAG_MOTOR | FLAG_TRACTION


def describe_flags(flags: int) -> list[str]:
    """Names of the active limits in a ``limit_flags`` bitmask."""
    return [name for bit, name in _FLAG_NAMES.items() if int(flags) & bit]


class PackPowerError(RuntimeError):
    """The demanded bus power exceeds what the pack's circuit can deliver at all."""


# ── Configuration ──────────────────────────────────────────────────────
def generic_compact_ev() -> Vehicle:
    """A generic compact EV: 1500 kg, 300 N·m / 110 kW motor, mid-range loss coefficients.

    The legacy ``Vehicle()`` default has a 60 kW motor and very small loss coefficients
    (about 97 % efficiency across the map).  These are more conventional: roughly 95 %
    at mid load, falling to the low 80s at very light load.  Generic, not measured.
    """
    return Vehicle(mass_kg=1500.0, drag_coefficient=0.28, frontal_area_m2=2.2,
                   rolling_resistance=0.010,
                   motor=MotorInverter(peak_torque_Nm=300.0, peak_power_W=110_000.0, max_speed_rpm=12_000.0,
                                       k_cu_W_per_Nm2=0.045, k_fe_W_per_radps2=1.0e-3, p_idle_W=150.0,
                                       k_inv_fraction=0.020, regen_torque_fraction=0.70))


@dataclass(frozen=True)
class AuxiliaryLoads:
    """Electrical loads other than traction.

    ``base_W`` is the low-voltage load (computers, lights, pumps), supplied through a DC-DC
    converter of efficiency ``dcdc_efficiency``.  HVAC draws from the high-voltage bus in
    proportion to how far ambient is from the cabin setpoint, outside a dead band.
    The coefficients are illustrative, not measured.
    """

    base_W: float = 350.0
    dcdc_efficiency: float = 0.93
    hvac_cooling_W_per_K: float = 45.0
    hvac_heating_W_per_K: float = 110.0
    cabin_setpoint_C: float = 22.0
    hvac_deadband_K: float = 2.0
    hvac_max_W: float = 5000.0

    def split_W(self, ambient_C: float) -> tuple[float, float, float]:
        """(low-voltage load, converter loss, HVAC) in watts at the HV bus."""
        loss = self.base_W * (1.0 / self.dcdc_efficiency - 1.0)
        d = ambient_C - self.cabin_setpoint_C
        if d > self.hvac_deadband_K:
            hvac = self.hvac_cooling_W_per_K * (d - self.hvac_deadband_K)
        elif d < -self.hvac_deadband_K:
            hvac = self.hvac_heating_W_per_K * (-d - self.hvac_deadband_K)
        else:
            hvac = 0.0
        return float(self.base_W), float(loss), float(min(hvac, self.hvac_max_W))

    def total_W(self, ambient_C: float) -> float:
        return float(sum(self.split_W(ambient_C)))


@dataclass(frozen=True)
class EVConfig:
    """Everything that defines a virtual EV run (the cycle is passed separately)."""

    vehicle: Vehicle = field(default_factory=generic_compact_ev)
    pack: PackSpec = field(default_factory=PackSpec)
    limits: PackLimits = field(default_factory=PackLimits)
    aux: AuxiliaryLoads = field(default_factory=AuxiliaryLoads)
    ambient_C: float = 25.0
    initial_temperature_C: float | None = None      # None → ambient
    soc0: float = 0.90
    seed: int = 0
    dt_sim_s: float = 0.25                          # largest internal step
    air_density_kg_m3: float = 1.225
    regen_fade_speed_mps: float = 1.5               # regeneration fades linearly to 0 below this speed
    # Driver: feed-forward on the trace's acceleration plus a P term on speed error.  ``None`` is
    # deadbeat (gain 1/dt): ask for exactly the acceleration that reaches the next target sample,
    # so a feasible trace is followed exactly and a limited car approaches the trace without overshoot.
    # Give a fixed gain to model a slower, lagging driver.
    driver_kp_per_s: float | None = None
    driver_ki_per_s2: float = 0.0                   # optional integral action (anti-windup while limited)
    driver_integral_limit_m: float = 2.0
    speed_tolerance_kmh: float = 2.0

    def __post_init__(self) -> None:
        if not self.dt_sim_s > 0.0:
            raise ValueError("dt_sim_s must be positive")
        if not 0.0 <= self.soc0 <= 1.0:
            raise ValueError("soc0 must be in [0, 1]")
        if self.vehicle.mass_kg <= 0.0:
            raise ValueError("vehicle mass must be positive")

    def config_hash(self) -> str:
        """SHA-256 of the full configuration (for run manifests)."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


# ── Motor helpers: invert the loss model under an electrical power cap ──
def loss_breakdown(motor: MotorInverter, torque_Nm: float, omega_rad_s: float) -> dict[str, float]:
    """The motor + inverter loss terms [W] at an operating point (they sum to ``loss_W``)."""
    p_mech = abs(torque_Nm * omega_rad_s)
    return {"copper": motor.k_cu_W_per_Nm2 * torque_Nm ** 2,
            "iron": motor.k_fe_W_per_radps2 * omega_rad_s ** 2,
            "controller": motor.p_idle_W,
            "inverter": motor.k_inv_fraction * p_mech}


def max_motoring_torque_for_power(motor: MotorInverter, omega_rad_s: float, p_elec_W: float) -> float:
    """Largest torque τ ≥ 0 whose electrical input stays within ``p_elec_W``.

    Electrical input is ``τω(1+k_inv) + k_cu τ² + k_fe ω² + P_idle``, a quadratic in τ.
    """
    a = motor.k_cu_W_per_Nm2
    b = (1.0 + motor.k_inv_fraction) * omega_rad_s
    c = motor.k_fe_W_per_radps2 * omega_rad_s ** 2 + motor.p_idle_W - p_elec_W
    if c >= 0.0:
        return 0.0
    if a <= 0.0:
        return -c / b if b > 0.0 else float("inf")
    return (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def max_regen_torque_for_power(motor: MotorInverter, omega_rad_s: float, p_recovered_W: float) -> float:
    """Largest braking torque magnitude whose *recovered* electrical power stays within ``p_recovered_W``.

    Recovered power is ``mω(1−k_inv) − k_cu m² − k_fe ω² − P_idle``, a concave quadratic in m;
    the cap is its smaller root.  ``inf`` if the motor can never recover that much.
    """
    a = motor.k_cu_W_per_Nm2
    g = omega_rad_s * (1.0 - motor.k_inv_fraction)
    c = motor.k_fe_W_per_radps2 * omega_rad_s ** 2 + motor.p_idle_W + p_recovered_W
    if a <= 0.0:
        return c / g if g > 0.0 else float("inf")
    disc = g * g - 4.0 * a * c
    if disc <= 0.0:
        return float("inf")
    return max(0.0, (g - math.sqrt(disc)) / (2.0 * a))


# ── Results ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EnergyLedger:
    """Energy flows over a run [Wh].  Positive = drawn from the pack / dissipated."""

    terminal_net: float
    traction_in: float            # Σ positive terminal power
    regen_out: float              # −Σ negative terminal power
    kinetic: float                # change in kinetic energy (incl. rotating inertia)
    potential: float              # work against gravity (negative when descending)
    aerodynamic: float
    rolling: float
    friction_brakes: float
    gearbox: float
    motor_copper: float
    motor_iron: float
    inverter: float
    controller: float
    aux_low_voltage: float
    aux_converter: float
    aux_hvac: float
    aux_cooling: float
    # Pack side
    chemical_ocv: float           # Σ OCV·I·dt, the time-integrated chemical power
    chemical_spline: float        # exact OCV-curve integral of the SOC drop (independent of stepping)
    emf_shift: float              # reversible EMF offset with temperature
    hysteresis: float
    ohmic: float
    polarization: float           # energy through the RC branches (includes what remains stored)
    rc_stored_change: float

    @property
    def vehicle_sinks(self) -> float:
        return (self.kinetic + self.potential + self.aerodynamic + self.rolling + self.friction_brakes
                + self.gearbox + self.motor_copper + self.motor_iron + self.inverter + self.controller
                + self.aux_low_voltage + self.aux_converter + self.aux_hvac + self.aux_cooling)

    @property
    def vehicle_residual(self) -> float:
        """Terminal energy minus everything the vehicle did with it (should be ~0)."""
        return self.terminal_net - self.vehicle_sinks

    @property
    def pack_residual(self) -> float:
        """Chemical energy minus terminal energy and pack losses (the time-stepping error)."""
        return self.chemical_ocv - (self.terminal_net + self.hysteresis + self.ohmic + self.polarization
                                    - self.emf_shift)

    @property
    def spline_residual(self) -> float:
        """Time-integrated chemical energy minus the exact OCV-curve integral."""
        return self.chemical_ocv - self.chemical_spline

    def relative_residuals(self) -> dict[str, float]:
        scale_v = max(self.traction_in + self.regen_out, 1e-12)
        scale_c = max(abs(self.chemical_ocv), 1e-12)
        return {"vehicle": abs(self.vehicle_residual) / scale_v,
                "pack": abs(self.pack_residual) / scale_c,
                "spline": abs(self.spline_residual) / scale_c}

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(vehicle_residual=self.vehicle_residual, pack_residual=self.pack_residual,
                 spline_residual=self.spline_residual)
        return d


@dataclass(frozen=True)
class SensorModel:
    """Additive Gaussian measurement noise for telemetry (the physics run is always noise-free)."""

    voltage_sigma_V: float = 0.0
    current_sigma_A: float = 0.0
    temperature_sigma_C: float = 0.0
    seed: int = 0


@dataclass
class DriveRun:
    """Time-resolved result of a drive-cycle run."""

    config: EVConfig
    cycle_name: str
    cycle_fingerprint: str
    dt_s: float
    channels: dict[str, np.ndarray]
    ledger: EnergyLedger
    sample_stride: int

    # ── Convenience views ──────────────────────────────────────────────
    def __getitem__(self, name: str) -> np.ndarray:
        return self.channels[name]

    @property
    def t_s(self) -> np.ndarray:
        return self.channels["t_s"]

    @property
    def n_steps(self) -> int:
        return int(self.t_s.size - 1)

    @property
    def distance_km(self) -> float:
        return float(self.channels["odometer_m"][-1] / 1000.0)

    # ── Metrics ────────────────────────────────────────────────────────
    def metrics(self) -> dict[str, float]:
        c, lg = self.channels, self.ledger
        err = c["speed_mps"] - c["speed_target_mps"]
        tol = self.config.speed_tolerance_kmh / 3.6
        dist = max(self.distance_km, 1e-9)
        cap_Ah = self.config.pack.capacity_Ah
        flags = c["limit_flags"][1:].astype(int)
        limited = float(np.mean((flags & _TRACTION_LIMIT_FLAGS) != 0)) if flags.size else 0.0
        return {
            "duration_s": float(self.t_s[-1]),
            "distance_km": self.distance_km,
            "net_energy_Wh": lg.terminal_net,
            "consumption_Wh_per_km": lg.terminal_net / dist,
            "traction_energy_Wh": lg.traction_in,
            "regen_recovered_Wh": lg.regen_out,
            "regen_share_of_traction": lg.regen_out / max(lg.traction_in, 1e-12),
            "drivetrain_efficiency": self._drivetrain_efficiency(),
            "soc_start": float(c["soc_mean"][0]),
            "soc_end": float(c["soc_mean"][-1]),
            "soc_used_pct": float(100.0 * (c["soc_mean"][0] - c["soc_mean"][-1])),
            "peak_current_A": float(np.max(c["current_A"])),
            "peak_regen_current_A": float(-np.min(c["current_A"])),
            "peak_discharge_C": float(np.max(c["current_A"]) / cap_Ah),
            "peak_power_kW": float(np.max(c["power_W"]) / 1000.0),
            "peak_regen_kW": float(-np.min(c["power_W"]) / 1000.0),
            "min_cell_voltage_V": float(np.min(c["cell_v_min_V"][1:])) if self.n_steps else float("nan"),
            "max_temperature_C": float(np.max(c["temp_max_C"])),
            "trace_rms_error_kmh": float(np.sqrt(np.mean(err ** 2)) * 3.6),
            "trace_max_error_kmh": float(np.max(np.abs(err)) * 3.6),
            "trace_within_tolerance_pct": float(100.0 * np.mean(np.abs(err) <= tol)),
            "time_power_limited_pct": 100.0 * limited,
        }

    def wheel_energy_Wh(self) -> float:
        """Positive mechanical energy delivered at the wheels [Wh]."""
        return float(np.sum(np.maximum(self.channels["p_wheel_W"], 0.0)) * self.dt_s / 3600.0)

    def _drivetrain_efficiency(self) -> float:
        """Wheel energy out / DC-bus energy into the drivetrain, over motoring intervals."""
        c = self.channels
        motoring = c["p_dc_inverter_W"] > 0.0
        e_in = float(np.sum(c["p_dc_inverter_W"][motoring]))
        e_out = float(np.sum(np.maximum(c["p_wheel_W"], 0.0)))
        return e_out / e_in if e_in > 0.0 else float("nan")

    def sop_at(self, i: int) -> dict[float, SOPLimit]:
        """State-of-power limits at channel index ``i`` (the nearest earlier telemetry instant).

        ``temperature_derate`` is not recorded and is reported as 1.0: the CAN SOP frames, which
        are the consumers of these limits, do not carry it.
        """
        c = self.channels
        j = int(i)
        while j > 0 and math.isnan(c[f"sop{int(SOP_HORIZONS_S[0])}_traction_W"][j]):
            j -= 1
        return {h: SOPLimit(horizon_s=h, traction_power_W=float(c[f"sop{int(h)}_traction_W"][j]),
                            regen_power_W=float(c[f"sop{int(h)}_regen_W"][j]),
                            traction_current_A=float(c[f"sop{int(h)}_traction_A"][j]),
                            regen_current_A=float(c[f"sop{int(h)}_regen_A"][j]),
                            temperature_derate=1.0) for h in SOP_HORIZONS_S}

    # ── Telemetry ──────────────────────────────────────────────────────
    def sample_indices(self, telemetry_dt_s: float | None = None) -> np.ndarray:
        """Indices into the channel arrays at telemetry instants (always includes the last)."""
        stride = self.sample_stride if telemetry_dt_s is None else max(1, round(telemetry_dt_s / self.dt_s))
        idx = np.arange(0, self.n_steps + 1, stride)
        if idx[-1] != self.n_steps:
            idx = np.append(idx, self.n_steps)
        return idx

    def telemetry(self, telemetry_dt_s: float | None = None, *, epoch_unix_s: float = DEFAULT_EPOCH_UNIX_S,
                  sensor: SensorModel | None = None) -> list[TelemetryRecord]:
        """Timestamped records at the telemetry rate (default: the cycle's native sample rate).

        Sample ``j`` reports the state at the end of internal step ``j·stride``: speed, SOC,
        voltage and temperature at that instant, and the current and power of the step that
        ended there.  Record 0 is the initial state at rest electrically (current 0).
        """
        rng = np.random.default_rng(sensor.seed) if sensor else None
        out: list[TelemetryRecord] = []
        for seq, i in enumerate(self.sample_indices(telemetry_dt_s)):
            noise = (0.0, 0.0, 0.0)
            if sensor is not None:
                noise = (float(rng.normal(0.0, sensor.voltage_sigma_V)) if sensor.voltage_sigma_V else 0.0,
                         float(rng.normal(0.0, sensor.current_sigma_A)) if sensor.current_sigma_A else 0.0,
                         float(rng.normal(0.0, sensor.temperature_sigma_C)) if sensor.temperature_sigma_C else 0.0)
            out.append(_make_record(self.channels, int(i), seq, epoch_unix_s, noise))
        return out

    # ── Identity ───────────────────────────────────────────────────────
    def digest(self) -> str:
        """SHA-256 over the raw bytes of every channel: equal only for bit-identical runs."""
        h = hashlib.sha256()
        for name in sorted(self.channels):
            h.update(name.encode())
            h.update(np.ascontiguousarray(self.channels[name]).tobytes())
        return h.hexdigest()

    def manifest(self) -> dict:
        """What is needed to reproduce this run, plus a digest of what it produced."""
        import numpy
        import scipy

        from . import __version__
        return {"bms_version": __version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
                "cycle": self.cycle_name, "cycle_fingerprint": self.cycle_fingerprint,
                "config_hash": self.config.config_hash(), "seed": self.config.seed,
                "dt_sim_s": self.dt_s, "n_steps": self.n_steps, "result_digest": self.digest()}


# ── Simulator ──────────────────────────────────────────────────────────
class EVSimulator:
    """Replays a :class:`DriveCycle` on a configured virtual EV.

    ``on_step(k, t_s, pack)``, if given, is called before every internal step and may mutate
    the :class:`VehiclePack` (raise a group's resistance, drop its SOC, heat it).  It is the
    fault-injection hook: faults enter through the plant, so the rest of the chain (limits,
    flags, ledger) responds exactly as it would to a real degradation.
    """

    def __init__(self, config: EVConfig | None = None, *, on_step=None) -> None:
        self.config = config or EVConfig()
        self.on_step = on_step
        self.last_run: DriveRun | None = None

    def run(self, cycle: DriveCycle | str, *, telemetry_dt_s: float | None = None) -> DriveRun:
        """Simulate the whole cycle and return the :class:`DriveRun`.

        ``telemetry_dt_s`` only sets the default sampling of :meth:`DriveRun.telemetry`
        (None → the cycle's own sample interval).
        """
        for _ in self._loop(cycle, telemetry_dt_s, emit=False):
            pass
        assert self.last_run is not None
        return self.last_run

    def stream(self, cycle: DriveCycle | str, *, telemetry_dt_s: float = 1.0,
               epoch_unix_s: float = DEFAULT_EPOCH_UNIX_S) -> Iterator[TelemetryRecord]:
        """Yield telemetry records as the simulation advances (the run is kept in ``last_run``)."""
        yield from self._loop(cycle, telemetry_dt_s, emit=True, epoch_unix_s=epoch_unix_s)

    # The single time-stepping loop; yields a record at each telemetry instant when ``emit``.
    def _loop(self, cycle, telemetry_dt_s, *, emit: bool, epoch_unix_s: float = DEFAULT_EPOCH_UNIX_S):
        cfg = self.config
        cycle = get_cycle(cycle) if isinstance(cycle, str) else cycle
        veh, mot, aux = cfg.vehicle, cfg.vehicle.motor, cfg.aux

        duration = cycle.duration_s
        n_steps = max(1, math.ceil(duration / cfg.dt_sim_s - 1e-9))
        dt = duration / n_steps
        stride = max(1, round((telemetry_dt_s if telemetry_dt_s else max(cycle.dt_s, dt)) / dt))
        t_grid = np.arange(n_steps + 1) * dt
        v_ref = np.interp(t_grid + cycle.t_s[0], cycle.t_s, cycle.speed_mps)
        a_ref = np.diff(v_ref) / dt
        if cycle.grade_rad is None:
            grade = np.zeros(n_steps)
        else:
            grade = np.interp(t_grid[:-1] + 0.5 * dt + cycle.t_s[0], cycle.t_s, cycle.grade_rad)

        pack = VehiclePack(cfg.pack, cfg.limits, ambient_C=cfg.ambient_C, soc0=cfg.soc0, seed=cfg.seed)
        if cfg.initial_temperature_C is not None:
            pack.reset(cfg.soc0, cfg.initial_temperature_C)

        names = ("t_s", "speed_mps", "speed_target_mps", "accel_mps2", "odometer_m", "grade_rad",
                 "f_cmd_N", "f_wheel_N", "f_brake_N", "f_aero_N", "f_roll_N", "f_grade_N",
                 "motor_torque_Nm", "motor_rpm", "p_wheel_W", "p_shaft_W", "p_dc_inverter_W", "p_aux_W",
                 "power_W", "current_A", "v_pack_V", "cell_v_min_V", "cell_v_max_V",
                 "soc_mean", "soc_min", "soc_max", "temp_mean_C", "temp_min_C", "temp_max_C",
                 "cooling_duty", "limit_flags", "limit_power_W", "limit_current_A", "energy_Wh",
                 "p_aero_W", "p_roll_W", "p_grade_W", "p_brake_W", "p_gear_W", "p_cu_W", "p_fe_W",
                 "p_inv_W", "p_idle_W", "p_aux_lv_W", "p_aux_dcdc_W", "p_aux_hvac_W", "p_aux_cool_W",
                 "p_ocv25_W", "p_tshift_W", "p_hyst_W", "p_r0_W", "p_pol_in_W", "p_rc_store_W", "heat_W")
        ch = {n: np.zeros(n_steps + 1) for n in names}
        # State of power at telemetry instants only (NaN between), for the CAN SOP frames.
        for h in SOP_HORIZONS_S:
            for q in ("traction_W", "regen_W", "traction_A", "regen_A"):
                ch[f"sop{int(h)}_{q}"] = np.full(n_steps + 1, np.nan)

        def put_sop(i):
            for h, lim in pack.sop_limits(SOP_HORIZONS_S).items():
                ch[f"sop{int(h)}_traction_W"][i] = lim.traction_power_W
                ch[f"sop{int(h)}_regen_W"][i] = lim.regen_power_W
                ch[f"sop{int(h)}_traction_A"][i] = lim.traction_current_A
                ch[f"sop{int(h)}_regen_A"][i] = lim.regen_current_A

        m, m_eff = veh.mass_kg, veh.effective_mass_kg
        r, gear, eta = veh.wheel_radius_m, veh.gear_ratio, veh.gear_efficiency
        cda_half = 0.5 * cfg.air_density_kg_m3 * veh.drag_coefficient * veh.frontal_area_m2
        base_lv, dcdc_loss, hvac = aux.split_W(cfg.ambient_C)
        omega_max = mot.max_speed_rpm * 2.0 * math.pi / 60.0
        kp = 1.0 / dt if cfg.driver_kp_per_s is None else cfg.driver_kp_per_s
        ki, z_max = cfg.driver_ki_per_s2, cfg.driver_integral_limit_m
        fade_v = max(cfg.regen_fade_speed_mps, 1e-9)

        v = float(v_ref[0])
        x = 0.0
        z = 0.0
        energy_Wh = 0.0
        e_chem0 = pack.chemical_energy_Wh()
        e_rc0 = pack.rc_stored_energy_Wh()

        def put_state(i, tstep):
            snap = pack
            ch["t_s"][i] = tstep
            ch["speed_mps"][i] = v
            ch["odometer_m"][i] = x
            ch["energy_Wh"][i] = energy_Wh
            ch["soc_mean"][i] = float(np.mean(snap.soc))
            ch["soc_min"][i] = float(snap.soc.min())
            ch["soc_max"][i] = float(snap.soc.max())
            ch["temp_mean_C"][i] = float(np.mean(snap.thermal.T))
            ch["temp_min_C"][i] = float(snap.thermal.T.min())
            ch["temp_max_C"][i] = float(snap.thermal.T.max())

        put_state(0, 0.0)
        ch["speed_target_mps"][0] = v_ref[0]
        ch["v_pack_V"][0] = pack.terminal_voltage(0.0, dt)
        snap0 = pack.snapshot(dt)
        ch["cell_v_min_V"][0] = float(snap0["v_group_V"].min())
        ch["cell_v_max_V"][0] = float(snap0["v_group_V"].max())
        ch["grade_rad"][0] = grade[0]
        put_sop(0)
        seq = 0
        if emit:
            yield _make_record(ch, 0, seq, epoch_unix_s)

        for k in range(n_steps):
            if self.on_step is not None:
                self.on_step(k, float(t_grid[k]), pack)
            th = float(grade[k])
            cos_th, sin_th = math.cos(th), math.sin(th)
            f_aero = cda_half * v * v
            f_roll = veh.rolling_resistance * m * _G * cos_th      # applies whenever the car is not held
            f_grade = m * _G * sin_th
            f_res = f_aero + f_roll + f_grade

            duty = pack.cooling_duty_request(dt)
            p_cool = pack.cooling_power_W(duty)
            p_aux = base_lv + dcdc_loss + hvac + p_cool
            ctx = pack.begin_step(dt)

            e = float(v_ref[k]) - v
            f_cmd = m_eff * (float(a_ref[k]) + kp * e + ki * z) + f_res
            f_trac = veh.friction_coefficient * m * _G * cos_th * veh.driven_axle_fraction
            omega_k = v * gear / r
            flags = 0
            f_wheel = f_fb = 0.0
            lim_w = lim_a = float("nan")
            # At rest the car stays put (brakes or static friction hold it, doing no work) unless
            # the trace is about to move or the command beats grade plus rolling resistance.
            held = v == 0.0 and (v_ref[k + 1] <= 1e-9 or f_cmd <= f_grade + f_roll)

            if held:
                pass
            elif f_cmd >= 0.0:
                dis = pack.discharge_limit(dt, ctx)
                lim_w, lim_a = dis.power_W, dis.current_A
                # Power is accounted at the step's mean speed, so evaluate the caps there: start from
                # the present speed and refine once with the acceleration the capped force would give.
                omega_e = omega_k
                for _ in range(2):
                    tau_max_e = 0.0 if omega_e > omega_max else mot.max_torque_Nm(omega_e)
                    f_lim_motor = tau_max_e * gear * eta / r
                    f_lim_pack = max_motoring_torque_for_power(mot, omega_e, dis.power_W - p_aux) * gear * eta / r
                    f_wheel = min(f_cmd, f_lim_motor, f_lim_pack, f_trac)
                    omega_e = 0.5 * (omega_k + max(0.0, omega_k + dt * gear / r * (f_wheel - f_res) / m_eff))
                if f_wheel < f_cmd - 1e-9:
                    if f_wheel == f_lim_pack:
                        flags |= FLAG_PACK_DISCHARGE
                    elif f_wheel == f_lim_motor:
                        flags |= FLAG_MOTOR
                    else:
                        flags |= FLAG_TRACTION
                if dis.binding & LIM_SOC:
                    flags |= FLAG_SOC_EMPTY
                if dis.derate < 1.0:
                    flags |= FLAG_THERMAL
            else:
                f_req = -f_cmd
                chg = pack.charge_limit(dt, ctx)
                lim_w, lim_a = -chg.power_W, -chg.current_A
                # Friction brakes make up whatever the motor does not take, so the car decelerates at
                # exactly the commanded rate; evaluate the caps at the step's mean speed (not the start).
                v_mid_pred = 0.5 * (v + max(0.0, v + dt * (f_cmd - f_res) / m_eff))
                omega_e = v_mid_pred * gear / r
                tau_max_e = 0.0 if omega_e > omega_max else mot.max_torque_Nm(omega_e)
                tau_cap_motor = mot.regen_torque_fraction * tau_max_e * min(1.0, v_mid_pred / fade_v)
                tau_cap_pack = max_regen_torque_for_power(mot, omega_e, chg.power_W + p_aux)
                f_cap_motor = tau_cap_motor * gear / (eta * r)
                f_cap_pack = tau_cap_pack * gear / (eta * r) if math.isfinite(tau_cap_pack) else float("inf")
                f_reg = min(f_req, f_cap_motor, f_cap_pack, f_trac)
                if f_reg > 0.0:                                     # regenerate only if it recovers net energy
                    tau_m = f_reg * r * eta / gear
                    if tau_m * omega_e <= sum(loss_breakdown(mot, tau_m, omega_e).values()):
                        f_reg = 0.0
                f_wheel, f_fb = -f_reg, f_req - f_reg
                if f_reg < f_req - 1e-9:
                    if f_reg == f_cap_pack:
                        flags |= FLAG_REGEN_PACK
                    elif f_reg == f_trac:
                        flags |= FLAG_TRACTION
                    else:
                        flags |= FLAG_REGEN_MOTOR
                if chg.derate < 1.0:
                    flags |= FLAG_THERMAL

            # Integrate speed.  If the car would reverse within the step, apply exactly the force
            # that brings it to rest, so energy still balances (see the ledger test).
            lam = 1.0
            a = (f_wheel - f_fb - f_res) / m_eff
            v_next = v + a * dt
            if held:
                a, v_next, lam = 0.0, 0.0, 0.0           # no motion, so no resistive work either
            elif v_next < -_V_REST_MPS:                  # would reverse within the step: stop exactly instead
                need = -m_eff * v / dt
                x_target = need + f_res
                if x_target <= 0.0:
                    excess = x_target - (f_wheel - f_fb)
                    d = min(f_fb, excess)
                    f_fb -= d
                    if excess - d > 0.0:
                        f_wheel += excess - d
                else:
                    f_wheel = f_fb = 0.0
                    lam = -need / f_res
                v_next, a = 0.0, -v / dt
            elif v_next < _V_REST_MPS:                   # float dust around zero: snap, keep the forces
                v_next, a = 0.0, -v / dt
            v_mid = 0.5 * (v + v_next)

            # Powertrain chain at the mean speed of the step.
            tau_m = f_wheel * r / (gear * eta) if f_wheel >= 0.0 else f_wheel * r * eta / gear
            omega_m = v_mid * gear / r
            p_wheel = f_wheel * v_mid
            p_shaft = tau_m * omega_m
            p_cu = mot.k_cu_W_per_Nm2 * tau_m * tau_m
            p_fe = mot.k_fe_W_per_radps2 * omega_m * omega_m
            p_inv = mot.k_inv_fraction * abs(p_shaft)
            p_dc = p_shaft + p_cu + p_fe + mot.p_idle_W + p_inv
            p_bus = p_dc + p_aux

            current = pack.solve_current(p_bus, dt, ctx)
            if not math.isfinite(current):
                raise PackPowerError(f"bus demand {p_bus:.0f} W at t={t_grid[k]:.2f} s exceeds what the pack "
                                     f"circuit can deliver ({pack.max_power_W(dt):.0f} W)")
            ps = pack.step(current, dt, duty, ctx)

            v, x = v_next, x + v_mid * dt
            energy_Wh += ps.p_terminal_W * dt / 3600.0
            if not flags & _TRACTION_LIMIT_FLAGS:
                z = max(-z_max, min(z_max, z + e * dt))

            i = k + 1
            put_state(i, t_grid[i])
            ch["speed_target_mps"][i] = v_ref[i]
            ch["accel_mps2"][i] = a
            ch["grade_rad"][i] = th
            ch["f_cmd_N"][i], ch["f_wheel_N"][i], ch["f_brake_N"][i] = f_cmd, f_wheel, f_fb
            ch["f_aero_N"][i], ch["f_roll_N"][i], ch["f_grade_N"][i] = lam * f_aero, lam * f_roll, lam * f_grade
            ch["motor_torque_Nm"][i] = tau_m
            ch["motor_rpm"][i] = omega_m * 60.0 / (2.0 * math.pi)
            ch["p_wheel_W"][i], ch["p_shaft_W"][i], ch["p_dc_inverter_W"][i] = p_wheel, p_shaft, p_dc
            ch["p_aux_W"][i], ch["power_W"][i], ch["current_A"][i] = p_aux, ps.p_terminal_W, current
            ch["v_pack_V"][i], ch["cell_v_min_V"][i], ch["cell_v_max_V"][i] = (ps.v_pack_V, ps.v_group_min_V,
                                                                              ps.v_group_max_V)
            ch["cooling_duty"][i], ch["limit_flags"][i] = duty, flags
            ch["limit_power_W"][i], ch["limit_current_A"][i] = lim_w, lim_a
            ch["p_aero_W"][i], ch["p_roll_W"][i], ch["p_grade_W"][i] = (lam * f_aero * v_mid, lam * f_roll * v_mid,
                                                                       lam * f_grade * v_mid)
            ch["p_brake_W"][i], ch["p_gear_W"][i] = f_fb * v_mid, p_shaft - p_wheel
            ch["p_cu_W"][i], ch["p_fe_W"][i], ch["p_inv_W"][i], ch["p_idle_W"][i] = p_cu, p_fe, p_inv, mot.p_idle_W
            ch["p_aux_lv_W"][i], ch["p_aux_dcdc_W"][i], ch["p_aux_hvac_W"][i], ch["p_aux_cool_W"][i] = (
                base_lv, dcdc_loss, hvac, p_cool)
            ch["p_ocv25_W"][i], ch["p_tshift_W"][i], ch["p_hyst_W"][i] = ps.p_ocv25_W, ps.p_tshift_W, ps.p_hyst_W
            ch["p_r0_W"][i], ch["p_pol_in_W"][i], ch["p_rc_store_W"][i] = ps.p_r0_W, ps.p_pol_in_W, ps.p_pol_stored_dW
            ch["heat_W"][i] = ps.heat_W

            if i % stride == 0 or i == n_steps:
                put_sop(i)
                if emit:
                    seq += 1
                    yield _make_record(ch, i, seq, epoch_unix_s)

        wh = dt / 3600.0

        def tot(name: str) -> float:
            return float(np.sum(ch[name][1:]) * wh)

        p_bus_all = ch["power_W"][1:]
        v_arr = ch["speed_mps"]
        ledger = EnergyLedger(
            terminal_net=float(np.sum(p_bus_all) * wh),
            traction_in=float(np.sum(np.maximum(p_bus_all, 0.0)) * wh),
            regen_out=float(-np.sum(np.minimum(p_bus_all, 0.0)) * wh),
            kinetic=0.5 * m_eff * float(v_arr[-1] ** 2 - v_arr[0] ** 2) / 3600.0,
            potential=tot("p_grade_W"), aerodynamic=tot("p_aero_W"), rolling=tot("p_roll_W"),
            friction_brakes=tot("p_brake_W"), gearbox=tot("p_gear_W"), motor_copper=tot("p_cu_W"),
            motor_iron=tot("p_fe_W"), inverter=tot("p_inv_W"), controller=tot("p_idle_W"),
            aux_low_voltage=tot("p_aux_lv_W"), aux_converter=tot("p_aux_dcdc_W"), aux_hvac=tot("p_aux_hvac_W"),
            aux_cooling=tot("p_aux_cool_W"),
            chemical_ocv=tot("p_ocv25_W"), chemical_spline=e_chem0 - pack.chemical_energy_Wh(),
            emf_shift=tot("p_tshift_W"), hysteresis=tot("p_hyst_W"), ohmic=tot("p_r0_W"),
            polarization=tot("p_pol_in_W"), rc_stored_change=pack.rc_stored_energy_Wh() - e_rc0)
        self.last_run = DriveRun(config=cfg, cycle_name=cycle.name, cycle_fingerprint=cycle.fingerprint(),
                                 dt_s=dt, channels=ch, ledger=ledger, sample_stride=stride)

def _make_record(ch: dict[str, np.ndarray], i: int, seq: int, epoch_unix_s: float,
                 noise: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> TelemetryRecord:
    """The telemetry record for channel index ``i`` (``noise`` = voltage, current, temperature offsets)."""
    t = float(ch["t_s"][i])
    return TelemetryRecord(
        seq=seq, t_s=t, timestamp=format_timestamp(epoch_unix_s + t),
        speed_mps=float(ch["speed_mps"][i]), soc=float(ch["soc_mean"][i]),
        voltage_V=float(ch["v_pack_V"][i]) + noise[0], current_A=float(ch["current_A"][i]) + noise[1],
        temperature_C=float(ch["temp_mean_C"][i]) + noise[2], power_W=float(ch["power_W"][i]),
        accel_mps2=float(ch["accel_mps2"][i]), speed_target_mps=float(ch["speed_target_mps"][i]),
        odometer_m=float(ch["odometer_m"][i]), energy_Wh=float(ch["energy_Wh"][i]),
        motor_torque_Nm=float(ch["motor_torque_Nm"][i]),
        cell_v_min_V=float(ch["cell_v_min_V"][i]), cell_v_max_V=float(ch["cell_v_max_V"][i]),
        soc_min=float(ch["soc_min"][i]), soc_max=float(ch["soc_max"][i]),
        temp_min_C=float(ch["temp_min_C"][i]), temp_max_C=float(ch["temp_max_C"][i]),
        limit_flags=int(ch["limit_flags"][i]),
        state="idle" if (ch["speed_mps"][i] <= 0.0 and ch["speed_target_mps"][i] <= 0.0) else "driving")


def simulate_cycle(cycle: DriveCycle | str, config: EVConfig | None = None, **overrides) -> DriveRun:
    """One-call convenience: ``simulate_cycle("udds", soc0=0.8, ambient_C=0.0)``."""
    cfg = replace(config or EVConfig(), **overrides) if overrides else (config or EVConfig())
    return EVSimulator(cfg).run(cycle)
