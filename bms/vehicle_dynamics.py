"""Time-domain vehicle, motor/inverter and drive-cycle → battery-load model.

The repository already had a *quasi-static* route energy model
(:mod:`bms.range_predictor`): per-segment energy from average speed.  This module
adds the **time-domain** physics needed to turn a speed profile into the actual
current the battery sees, sample by sample:

    speed profile v(t)
        │  longitudinal dynamics  F = m_eff·a + ½ρCdA v² + Crm·g·cosθ + m·g·sinθ
        ▼
    wheel force → wheel torque → motor torque & speed (gear, wheel radius)
        │  motor/inverter loss map  (copper, iron, switching, auxiliary)
        ▼
    electrical power P_batt (positive = drawn, negative = regeneration)
        │  pack equivalent circuit  V = n·(OCV − I·R)  →  solve  P = V·I  for I
        ▼
    battery current I(t)  →  fed to a cell twin / UnifiedTwin

Assumptions (stated, not hidden)
--------------------------------
* The driver *follows* the speed profile; the model reports when the motor or
  the tyre-road friction limit makes that infeasible (``feasible`` flag), rather
  than silently pretending the car kept up.
* Regeneration is limited by a motor-torque fraction; any braking force beyond
  that is friction braking and returns no energy to the battery.
* Motor losses are a lumped copper + iron + inverter model with published-style
  coefficients for a generic permanent-magnet traction motor.  They are
  *parameters* (``MotorInverter``) so a real motor map can replace them.
* Speed profiles are synthetic (urban stop-go, arterial, highway, hilly).  No
  standard test-cycle files are bundled; :func:`load_speed_profile_csv` accepts
  any measured trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

_G = 9.80665
_RHO_SEA = 1.225


@dataclass(frozen=True)
class MotorInverter:
    """Traction motor + inverter with a lumped loss model.

    Losses (motoring): ``P_loss = k_cu·τ² + k_fe·ω² + P_idle + k_inv·|P_mech|``.
    Regeneration removes the same loss from the recovered mechanical power.
    """

    peak_torque_Nm: float = 220.0
    peak_power_W: float = 60_000.0
    max_speed_rpm: float = 12_000.0
    k_cu_W_per_Nm2: float = 0.0009     # copper loss ∝ torque²
    k_fe_W_per_radps2: float = 0.0000025  # iron loss ∝ speed²
    p_idle_W: float = 120.0            # controller / bearings / cooling pump
    k_inv_fraction: float = 0.025      # inverter switching + conduction (fraction of P)
    regen_torque_fraction: float = 0.70  # share of braking torque the motor may recover

    def max_torque_Nm(self, omega_rad_s: float) -> float:
        """Constant-torque up to base speed, constant-power beyond it."""
        if omega_rad_s <= 1e-6:
            return self.peak_torque_Nm
        return float(min(self.peak_torque_Nm, self.peak_power_W / omega_rad_s))

    def loss_W(self, torque_Nm: float, omega_rad_s: float) -> float:
        p_mech = abs(torque_Nm * omega_rad_s)
        return (self.k_cu_W_per_Nm2 * torque_Nm ** 2
                + self.k_fe_W_per_radps2 * omega_rad_s ** 2
                + self.p_idle_W + self.k_inv_fraction * p_mech)

    def electrical_power_W(self, torque_Nm: float, omega_rad_s: float) -> float:
        """Battery-side power [W]; positive = drawn from the battery."""
        p_mech = torque_Nm * omega_rad_s
        loss = self.loss_W(torque_Nm, omega_rad_s)
        if p_mech >= 0.0:
            return p_mech + loss                 # motoring: battery supplies mech + loss
        return p_mech + loss if (-p_mech) > loss else 0.0   # generating: loss eats recovery

    def efficiency(self, torque_Nm: float, omega_rad_s: float) -> float:
        """Mechanical-to-electrical efficiency at an operating point (motoring)."""
        p_mech = torque_Nm * omega_rad_s
        if p_mech <= 1e-9:
            return 0.0
        return float(p_mech / (p_mech + self.loss_W(torque_Nm, omega_rad_s)))


@dataclass(frozen=True)
class Vehicle:
    """Longitudinal vehicle parameters (kept separate from the quasi-static model)."""

    mass_kg: float = 1400.0
    drag_coefficient: float = 0.28
    frontal_area_m2: float = 2.2
    rolling_resistance: float = 0.010
    wheel_radius_m: float = 0.32
    gear_ratio: float = 9.0
    gear_efficiency: float = 0.97
    rotational_inertia_fraction: float = 0.04   # rotating parts add to effective mass
    friction_coefficient: float = 0.80          # tyre-road limit on traction force
    driven_axle_fraction: float = 1.0           # share of weight on the driven wheels
    motor: MotorInverter = field(default_factory=MotorInverter)

    @property
    def effective_mass_kg(self) -> float:
        return self.mass_kg * (1.0 + self.rotational_inertia_fraction)


@dataclass(frozen=True)
class PackModel:
    """Equivalent-circuit pack seen from the vehicle's DC bus.

    Defaults describe a 96s60p pack of 3 Ah cells (≈ 64 kWh, ≈ 350 V nominal):
    the pack DC resistance is derived from the cell resistance, so the C-rate at
    cruise stays realistic (≈ 0.2 C at 100 km/h) rather than being set by hand.
    """

    n_series: int = 96
    n_parallel: int = 60
    cell_capacity_Ah: float = 3.0
    cell_r_ohm: float = 0.030         # cell DC resistance at the operating point
    ocv_V_per_cell: float = 3.7       # operating-point OCV per cell (pack-level approximation)

    @property
    def r_pack_ohm(self) -> float:
        return self.n_series * self.cell_r_ohm / self.n_parallel

    @property
    def capacity_Ah(self) -> float:
        return self.n_parallel * self.cell_capacity_Ah

    @property
    def energy_kWh(self) -> float:
        return self.n_series * self.n_parallel * self.cell_capacity_Ah * self.ocv_V_per_cell / 1000.0

    def cell_current(self, pack_current_A: float) -> float:
        """Current seen by each parallel cell group member [A] (equal sharing)."""
        return pack_current_A / self.n_parallel

    def current_for_power(self, p_batt_W: float) -> float:
        """Battery current [A] that delivers ``p_batt_W`` (positive = discharge).

        Solves P = V·I with V = E − I·R for the physical root.
        """
        e = self.n_series * self.ocv_V_per_cell
        r = self.r_pack_ohm
        disc = e * e - 4.0 * r * p_batt_W
        if disc < 0.0:                    # demanded power exceeds what the pack can deliver
            return float("nan")
        return float((e - np.sqrt(disc)) / (2.0 * r))

    def bus_voltage(self, current_A: float) -> float:
        return self.n_series * self.ocv_V_per_cell - self.r_pack_ohm * current_A


@dataclass
class DriveResult:
    """Time-resolved outputs of a drive cycle."""

    t_s: np.ndarray
    speed_mps: np.ndarray
    accel_mps2: np.ndarray
    wheel_force_N: np.ndarray
    motor_torque_Nm: np.ndarray
    motor_speed_rpm: np.ndarray
    p_electrical_W: np.ndarray
    pack_current_A: np.ndarray
    feasible: np.ndarray
    energy_consumed_Wh: float
    energy_regenerated_Wh: float
    distance_km: float

    @property
    def net_energy_Wh(self) -> float:
        return self.energy_consumed_Wh - self.energy_regenerated_Wh

    @property
    def consumption_Wh_per_km(self) -> float:
        return self.net_energy_Wh / max(self.distance_km, 1e-9)

    @property
    def all_feasible(self) -> bool:
        return bool(np.all(self.feasible))

    def to_dict(self) -> dict:
        return {"distance_km": self.distance_km,
                "net_energy_Wh": self.net_energy_Wh,
                "consumption_Wh_per_km": self.consumption_Wh_per_km,
                "energy_regenerated_Wh": self.energy_regenerated_Wh,
                "all_feasible": self.all_feasible}


def _resistive_force_N(v: float, grade_rad: float, veh: Vehicle, rho: float) -> float:
    aero = 0.5 * rho * veh.drag_coefficient * veh.frontal_area_m2 * v * abs(v)
    roll = veh.rolling_resistance * veh.mass_kg * _G * np.cos(grade_rad)
    grade = veh.mass_kg * _G * np.sin(grade_rad)
    return float(aero + roll + grade)


def simulate_drive(speed_mps: np.ndarray, dt: float, vehicle: Vehicle, pack: PackModel,
                   grade_rad: np.ndarray | float = 0.0, rho: float = _RHO_SEA) -> DriveResult:
    """Follow a speed profile and return the battery current it demands.

    ``feasible`` is False where the required motor torque exceeds the motor limit or
    the required traction force exceeds the tyre-road friction limit.
    """
    v = np.asarray(speed_mps, float)
    n = v.size
    grade = np.full(n, float(grade_rad)) if np.isscalar(grade_rad) else np.asarray(grade_rad, float)
    a = np.gradient(v, dt)
    omega_w = v / vehicle.wheel_radius_m
    omega_m = omega_w * vehicle.gear_ratio
    omega_m_rpm = omega_m * 60.0 / (2.0 * np.pi)

    f_res = np.array([_resistive_force_N(v[k], grade[k], vehicle, rho) for k in range(n)])
    f_drive = vehicle.effective_mass_kg * a + f_res          # force the wheels must produce

    # Friction limit on traction (driven axle only).
    f_max = vehicle.friction_coefficient * vehicle.mass_kg * _G * vehicle.driven_axle_fraction
    traction_ok = np.abs(f_drive) <= f_max + 1e-9

    tau_wheel = f_drive * vehicle.wheel_radius_m
    # Motoring: the drivetrain loses torque (÷η). Braking: less torque reaches the motor (×η).
    tau_motor = np.where(f_drive >= 0.0,
                         tau_wheel / (vehicle.gear_ratio * vehicle.gear_efficiency),
                         tau_wheel * vehicle.gear_efficiency / vehicle.gear_ratio)

    p_elec = np.zeros(n)
    feasible = traction_ok.copy()
    motor = vehicle.motor
    for k in range(n):
        w = float(omega_m[k])
        tau = float(tau_motor[k])
        if tau >= 0.0:
            limit = motor.max_torque_Nm(abs(w))
            if abs(tau) > limit + 1e-9:
                feasible[k] = False
                tau = np.sign(tau) * limit
            p_elec[k] = motor.electrical_power_W(tau, w)
        else:
            # Braking: the motor recovers only a fraction of the demanded braking torque.
            allowed = motor.regen_torque_fraction * motor.max_torque_Nm(abs(w))
            tau_regen = max(tau, -allowed)            # negative torque, clipped to regen limit
            p_elec[k] = motor.electrical_power_W(tau_regen, w) if abs(w) > 1e-6 else 0.0

    current = np.array([pack.current_for_power(float(p)) for p in p_elec])
    feasible &= np.isfinite(current)                  # pack cannot supply it → infeasible
    current = np.where(np.isfinite(current), current, np.nan)

    dt_h = dt / 3600.0
    e_pos = float(np.sum(np.clip(p_elec, 0.0, None)) * dt_h)
    e_neg = float(-np.sum(np.clip(p_elec, None, 0.0)) * dt_h)
    distance_km = float(np.sum(v) * dt / 1000.0)

    return DriveResult(
        t_s=np.arange(n) * dt, speed_mps=v, accel_mps2=a, wheel_force_N=f_drive,
        motor_torque_Nm=tau_motor, motor_speed_rpm=omega_m_rpm, p_electrical_W=p_elec,
        pack_current_A=current, feasible=feasible,
        energy_consumed_Wh=e_pos, energy_regenerated_Wh=e_neg, distance_km=distance_km,
    )


# ── Drive-cycle generators (synthetic, documented shapes) ──────────────────

def _piecewise(points: list[tuple[float, float]], dt: float) -> np.ndarray:
    t_end = points[-1][0]
    t = np.arange(0.0, t_end + dt, dt)
    ts, vs = zip(*points)
    return np.interp(t, ts, vs) / 3.6         # km/h → m/s


def urban_stop_go(dt: float = 1.0, cycles: int = 3) -> np.ndarray:
    """Repeated accelerate / cruise / brake / idle blocks (~40 km/h peak)."""
    block = [(0, 0), (20, 35), (40, 35), (55, 0), (75, 0)]
    pts = [(0.0, 0.0)]
    t0 = 0.0
    for _ in range(cycles):
        for t, v in block[1:]:
            pts.append((t0 + t, v))
        t0 += block[-1][0]
    return _piecewise(pts, dt)


def arterial(dt: float = 1.0) -> np.ndarray:
    """Moderate-speed road with a few speed changes (~60 km/h)."""
    return _piecewise([(0, 0), (30, 55), (150, 60), (210, 40), (280, 60), (360, 0)], dt)


def highway(dt: float = 1.0) -> np.ndarray:
    """Sustained high-speed cruise (~100 km/h) with a gentle merge."""
    return _piecewise([(0, 0), (30, 100), (600, 100), (660, 115), (720, 100), (780, 0)], dt)


def load_speed_profile_csv(path: str, time_col: str = "time_s",
                           speed_col: str = "speed_kmh") -> np.ndarray:
    """Load a measured speed trace (km/h) and resample to uniform 1 s steps."""
    import pandas as pd
    df = pd.read_csv(path, comment="#")
    t = df[time_col].to_numpy(float)
    v = df[speed_col].to_numpy(float) / 3.6
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0)
    return np.interp(grid, t, v)


def drive_to_battery_load(result: DriveResult) -> np.ndarray:
    """The battery-current trace (A, discharge positive) for a simulated drive."""
    return np.nan_to_num(result.pack_current_A, nan=0.0)


def apply_motor_overrides(vehicle: Vehicle, **motor_kwargs) -> Vehicle:
    """Return a copy of ``vehicle`` with its motor parameters overridden."""
    return replace(vehicle, motor=replace(vehicle.motor, **motor_kwargs))
