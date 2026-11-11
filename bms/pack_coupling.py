"""Vehicle-side battery pack: power demand in, pack current out, limits respected.

The vehicle asks the pack for *power*; a battery works in *current*.  This module is
the coupling between the two:

    bus power P  ──►  limits (current, voltage, power, SOC, temperature)
                 ──►  current I solving  P = V_pack(I)·I  on the pack's own equivalent circuit
                 ──►  state update (SOC, polarization, temperature)  ──►  terminal voltage

The plant is the repository's second-order ECM (:class:`bms.ecm.SecondOrderECM`),
vectorized over series groups so a 96s60p pack (5,760 cells) runs in milliseconds:
each series group is *one* equivalent cell whose resistances are ``R / n_parallel`` and
whose capacity and capacitances are ``× n_parallel``.  Tests check it against
``SecondOrderECM`` (exactly) and against ``BatteryPack``, so the vehicle pack and the
cell model cannot drift apart.

Limits (``PackLimits``)
-----------------------
Discharge and charge are each limited by the tightest of: a C-rate current limit, the
weakest series group reaching its minimum (or maximum) voltage, an optional power cap,
the SOC window (no discharge at the floor, regeneration tapered toward the ceiling), and
a temperature envelope (hot derating, cold charge derating).  The limit objects say which
constraint binds, so a simulator can report why the car lost power.

Honest scope: parameters are generic and illustrative.  Nothing here has been validated
against a physical vehicle pack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from .ecm import ECMParameters
from .ocv_soc import OCVSOC
from .sop import SOPLimit
from .thermal import PredictiveCoolingController, ThermalModel, ThermalParameters

_T_REF_K = 298.15

# Which limit is binding (bit flags; 0 = none).
LIM_NONE = 0
LIM_CURRENT = 1
LIM_VOLTAGE = 2
LIM_POWER = 4
LIM_SOC = 8
LIM_THERMAL = 16


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, 0.0), 1.0)


@dataclass(frozen=True)
class PackLimits:
    """Operating envelope of the pack."""

    max_discharge_C: float = 3.0           # current limit, multiples of pack capacity
    max_charge_C: float = 1.5
    max_discharge_power_W: float | None = None
    max_charge_power_W: float | None = None
    cell_v_min_V: float | None = None      # None → the chemistry's minimum
    cell_v_max_V: float | None = None      # None → the chemistry's maximum
    soc_floor: float = 0.0                 # no discharge at or below this
    soc_ceiling: float = 1.0               # no charge at or above this
    regen_taper_start: float = 0.90        # charge allowance falls linearly to 0 at soc_ceiling
    hot_derate_start_C: float = 45.0       # both directions fall linearly to 0 at hot_cutoff_C
    hot_cutoff_C: float = 60.0
    cold_charge_full_C: float = 10.0       # charge allowance falls linearly to 0 at cold_charge_zero_C
    cold_charge_zero_C: float = -10.0


@dataclass(frozen=True)
class PackSpec:
    """Topology, cell model, scatter and thermal design of the vehicle pack.

    Defaults describe a 96s60p pack of 3 Ah NMC cells: about 65 kWh and 355 V nominal.
    All values are generic assumptions, not measurements of a particular product.
    """

    n_series: int = 96
    n_parallel: int = 60
    chemistry: str = "nmc"
    cell_capacity_Ah: float = 3.0
    ecm: ECMParameters | None = None       # per-cell RC parameters at 25 °C; None → chemistry default
    capacity_sigma: float = 0.01           # series-group capacity scatter (already averaged over n_parallel)
    r0_sigma: float = 0.02
    soc_sigma: float = 0.005               # spread of initial SOC between groups
    hysteresis_v: float | None = None      # None → chemistry default; 0.0 disables
    # Thermal design (whole-pack values; split evenly across series groups).
    cell_mass_g: float = 46.0
    cp_J_per_gK: float = 0.9
    structure_mass_factor: float = 1.3     # casing, busbars and cold plates add thermal mass
    h_passive_W_per_K: float = 20.0        # conductance to ambient without active cooling
    h_cooling_max_W_per_K: float = 200.0   # conductance at full cooling duty
    kappa_W_per_K: float = 5.0             # between adjacent series groups
    cooling_setpoint_C: float = 30.0
    cooling_power_max_W: float = 600.0     # pump and fan power at full duty

    def __post_init__(self) -> None:
        if self.n_series < 1 or self.n_parallel < 1:
            raise ValueError("n_series and n_parallel must be at least 1")
        if not self.cell_capacity_Ah > 0.0:
            raise ValueError("cell_capacity_Ah must be positive")

    @property
    def capacity_Ah(self) -> float:
        return self.n_parallel * self.cell_capacity_Ah

    def cell_params(self) -> ECMParameters:
        if self.ecm is not None:
            return self.ecm
        from .chemistry import get_chemistry_props
        d = get_chemistry_props(self.chemistry)["default_ecm"]
        return ECMParameters(R0=d["R0"], R1=d["R1"], C1=d["C1"], R2=d["R2"], C2=d["C2"],
                             Q_nom_Ah=self.cell_capacity_Ah, chemistry=self.chemistry)

    def nominal_energy_kWh(self) -> float:
        ocv = OCVSOC.from_chemistry(self.chemistry, hysteresis_v=0.0)
        s = np.linspace(0.0, 1.0, 201)
        return self.n_series * self.capacity_Ah * float(np.mean(ocv.ocv(s))) / 1000.0


@dataclass(frozen=True)
class Limit:
    """One direction's instantaneous limit (positive magnitudes) and what sets it."""

    current_A: float
    power_W: float
    binding: int
    derate: float


@dataclass(frozen=True)
class CurrentLimits:
    """Both directions' limits."""

    discharge: Limit
    charge: Limit

    @property
    def discharge_A(self) -> float:
        return self.discharge.current_A

    @property
    def charge_A(self) -> float:
        return self.charge.current_A

    @property
    def discharge_W(self) -> float:
        return self.discharge.power_W

    @property
    def charge_W(self) -> float:
        return self.charge.power_W


@dataclass
class PackStep:
    """Everything one pack step produced (scalars; arrays via :meth:`VehiclePack.snapshot`)."""

    current_A: float
    v_pack_V: float
    p_terminal_W: float
    v_group_min_V: float
    v_group_max_V: float
    p_ocv25_W: float          # Σ OCV(25 °C)·I
    p_tshift_W: float         # Σ (EMF shift with temperature)·I
    p_hyst_W: float           # hysteresis loss
    p_r0_W: float             # ohmic loss
    p_pol_in_W: float         # mean power flowing into the two RC branches over the step
    p_pol_stored_dW: float    # d/dt of energy stored in the RC capacitances
    heat_W: float
    cooling_duty: float


class StepContext:
    """Per-step cache: temperature-scaled parameters and the linearized terminal model.

    Built once per step from the state at that moment and discarded, so it can never go
    stale.  Pass it to :meth:`VehiclePack.solve_current`, the limit methods and
    :meth:`VehiclePack.step` to avoid recomputing the same arrays.
    """

    __slots__ = ("pack", "dt", "prm", "_lin")

    def __init__(self, pack: "VehiclePack", dt: float) -> None:
        self.pack = pack
        self.dt = float(dt)
        self.prm = pack._scaled(self.dt)
        self._lin: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def linear(self, direction: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-group open-circuit terminal voltage ``e`` and slope magnitude ``rho`` for one direction."""
        hit = self._lin.get(direction)
        if hit is None:
            hit = self._lin[direction] = self.pack._linear_model(direction, self.dt, self.prm)
        return hit


class VehiclePack:
    """Series groups of an equivalent-circuit pack with limits, thermal state and a power solver."""

    def __init__(self, spec: PackSpec | None = None, limits: PackLimits | None = None, *,
                 ambient_C: float = 25.0, soc0: float = 0.90, seed: int = 0,
                 ocv_curve: OCVSOC | None = None) -> None:
        self.spec = spec or PackSpec()
        self.limits = limits or PackLimits()
        s = self.spec
        self.ocv_curve = ocv_curve or OCVSOC.from_chemistry(s.chemistry, hysteresis_v=s.hysteresis_v)
        base = s.cell_params()
        from .chemistry import get_chemistry_props
        props = get_chemistry_props(s.chemistry)
        self._arrhenius_K = float(get_chemistry_props(base.chemistry)["arrhenius_K"])
        self.v_min = float(self.limits.cell_v_min_V if self.limits.cell_v_min_V is not None else props["v_min"])
        self.v_max = float(self.limits.cell_v_max_V if self.limits.cell_v_max_V is not None else props["v_max"])
        self.ambient_C = float(ambient_C)

        n = s.n_series
        rng = np.random.default_rng(seed)
        cap_factor = np.clip(rng.normal(1.0, s.capacity_sigma, n), 0.8, 1.2)
        r_factor = np.clip(rng.normal(1.0, s.r0_sigma, n), 0.5, 1.5)
        self._soc_offsets = rng.normal(0.0, s.soc_sigma, n)
        npar = float(s.n_parallel)
        # Per series group: one equivalent cell (R/n_p, C·n_p, Q·n_p).
        self.q0_Ah = base.Q_nom_Ah * npar * cap_factor
        self.r0_25 = base.R0 * r_factor / npar
        self.r1_25 = np.full(n, base.R1 / npar)
        self.r2_25 = np.full(n, base.R2 / npar)
        self.c1 = np.full(n, base.C1 * npar)
        self.c2 = np.full(n, base.C2 * npar)

        # Own spline of the OCV table (same PCHIP data as OCVSOC): slope and energy integral.
        tab = self.ocv_curve.table
        spline = PchipInterpolator(tab[:, 0], tab[:, 1], extrapolate=True)
        self._spline = spline
        self._docv = spline.derivative()
        self._ocv_antiderivative = spline.antiderivative()
        # OCVSOC.ocv = spline(soc) − hysteresis·sign(I) + coeff·(T − 25).  The step loop evaluates
        # that directly (one spline call instead of several wrapper calls); the coefficient is
        # probed from the public API, and tests pin the result to SecondOrderECM.
        self._hyst = float(self.ocv_curve.hysteresis_v)
        hot = float(self.ocv_curve.ocv(0.5, 0.0, T_C=125.0))
        self._temp_coeff = (hot - float(self.ocv_curve.ocv(0.5, 0.0, T_C=25.0))) / 100.0

        thermal_mass_g = s.cell_mass_g * s.n_parallel * s.structure_mass_factor     # per series group
        self.thermal = ThermalModel(n_cells=n, params=ThermalParameters(
            mass_g=thermal_mass_g, cp_J_per_gK=s.cp_J_per_gK, kappa_W_per_K=s.kappa_W_per_K,
            h_min_W_per_K=s.h_passive_W_per_K / n, h_max_W_per_K=s.h_cooling_max_W_per_K / n,
            T_amb_C=self.ambient_C))
        self.cooler = PredictiveCoolingController(setpoint=s.cooling_setpoint_C, ff_gain=0.0)
        self.reset(soc0)

    # ── State ──────────────────────────────────────────────────────────
    @property
    def n_series(self) -> int:
        return self.spec.n_series

    def reset(self, soc0: float = 0.90, temperature_C: float | None = None) -> None:
        self.soc = _clip01(float(soc0) + self._soc_offsets)
        self.v1 = np.zeros(self.n_series)
        self.v2 = np.zeros(self.n_series)
        self.thermal.reset(self.ambient_C if temperature_C is None else temperature_C)
        self.cooler.reset()
        self.last_current_A = 0.0
        self.last_cooling_duty = 0.0

    @property
    def T(self) -> np.ndarray:
        return self.thermal.T

    def begin_step(self, dt: float) -> StepContext:
        """Snapshot the temperature-scaled parameters for one step of ``dt`` seconds."""
        return StepContext(self, dt)

    # ── Temperature-dependent parameters (same Arrhenius law as ECMParameters.at_temperature) ──
    def _scaled(self, dt: float):
        T = self.thermal.T
        factor = np.exp(self._arrhenius_K * (1.0 / (T + 273.15) - 1.0 / _T_REF_K))
        q_scale = 1.0 - 0.002 * np.maximum(0.0, 25.0 - T)
        r0, r1, r2 = self.r0_25 * factor, self.r1_25 * factor, self.r2_25 * factor
        tau1 = np.maximum(r1 * self.c1, 1e-9)
        tau2 = np.maximum(r2 * self.c2, 1e-9)
        return r0, r1, r2, tau1, tau2, np.exp(-dt / tau1), np.exp(-dt / tau2), self.q0_Ah * q_scale

    def _ocv(self, soc: np.ndarray, sign: float, temperature_C: np.ndarray) -> np.ndarray:
        """Group OCV with the same hysteresis and temperature rule as :meth:`OCVSOC.ocv`."""
        return self._spline(_clip01(soc)) - self._hyst * sign + self._temp_coeff * (temperature_C - 25.0)

    # ── Non-mutating terminal model (same arithmetic as SecondOrderECM.step) ──
    def _group_voltages(self, current: float, dt: float, prm) -> np.ndarray:
        r0, r1, r2, _, _, a1, a2, q = prm
        soc_new = _clip01(self.soc - current * dt / (q * 3600.0))
        v1 = a1 * self.v1 + (1.0 - a1) * r1 * current
        v2 = a2 * self.v2 + (1.0 - a2) * r2 * current
        ocv = self._ocv(soc_new, np.sign(current), self.thermal.T)
        return ocv - v1 - v2 - r0 * current

    def terminal_voltage(self, current: float, dt: float) -> float:
        """Pack terminal voltage [V] if ``current`` flowed for one step of ``dt`` (no state change)."""
        return float(np.sum(self._group_voltages(current, dt, self._scaled(dt))))

    def _linear_model(self, direction: float, dt: float, prm):
        """Per-group terminal voltage at I→0 in ``direction``, and its slope magnitude dV/dI."""
        r0, r1, r2, _, _, a1, a2, q = prm
        e0 = self._ocv(self.soc, direction, self.thermal.T) - a1 * self.v1 - a2 * self.v2
        slope = self._docv(_clip01(self.soc))
        rho = r0 + (1.0 - a1) * r1 + (1.0 - a2) * r2 + slope * dt / (3600.0 * q)
        return e0, rho

    # ── Power → current ────────────────────────────────────────────────
    def solve_current(self, p_W: float, dt: float, ctx: StepContext | None = None) -> float:
        """Pack current [A] (positive = discharge) at which the terminal delivers ``p_W``.

        Solves ``V_pack(I)·I = p_W`` with the model's own post-update terminal voltage
        (the exact nonlinear function, refined by Newton from the linearized quadratic), so
        after ``step(I)`` the terminal power equals ``p_W`` to numerical precision.
        Returns NaN if the pack cannot deliver ``p_W`` at all.
        """
        if p_W == 0.0:
            return 0.0
        ctx = ctx or self.begin_step(dt)
        e0, rho = ctx.linear(1.0 if p_W > 0.0 else -1.0)
        e, r = float(e0.sum()), float(rho.sum())
        disc = e * e - 4.0 * r * p_W
        if disc < 0.0:
            return float("nan")
        i = (e - np.sqrt(disc)) / (2.0 * r)
        for _ in range(4):
            v = float(np.sum(self._group_voltages(i, dt, ctx.prm)))
            f = v * i - p_W
            if abs(f) <= 1e-9 * max(1.0, abs(p_W)):
                break
            i -= f / (v - r * i)
        return float(i)

    def max_power_W(self, dt: float) -> float:
        """Largest discharge power the circuit can deliver at all (the quadratic's vertex)."""
        e0, rho = self.begin_step(dt).linear(1.0)
        return float(e0.sum() ** 2 / (4.0 * rho.sum()))

    # ── Limits ─────────────────────────────────────────────────────────
    def _hot_derate(self) -> float:
        lim = self.limits
        t_max = float(self.thermal.T.max())
        if t_max <= lim.hot_derate_start_C:
            return 1.0
        if t_max >= lim.hot_cutoff_C:
            return 0.0
        return (lim.hot_cutoff_C - t_max) / (lim.hot_cutoff_C - lim.hot_derate_start_C)

    def _cold_charge_derate(self) -> float:
        lim = self.limits
        t_min = float(self.thermal.T.min())
        if t_min >= lim.cold_charge_full_C:
            return 1.0
        if t_min <= lim.cold_charge_zero_C:
            return 0.0
        return (t_min - lim.cold_charge_zero_C) / (lim.cold_charge_full_C - lim.cold_charge_zero_C)

    def discharge_limit(self, dt: float, ctx: StepContext | None = None) -> Limit:
        """Maximum discharge current/power now, and the constraint that sets it."""
        lim = self.limits
        ctx = ctx or self.begin_step(dt)
        hot = self._hot_derate()
        e_d, rho_d = ctx.linear(1.0)
        i_volt = float(np.min((e_d - self.v_min) / rho_d))
        i_cur = lim.max_discharge_C * self.spec.capacity_Ah * hot
        i_dis = max(0.0, min(i_cur, i_volt))
        binding = LIM_VOLTAGE if i_volt < i_cur else LIM_CURRENT     # the tighter of the two sets the cap
        if hot < 1.0:
            binding |= LIM_THERMAL
        if float(self.soc.min()) <= lim.soc_floor:
            i_dis, binding = 0.0, binding | LIM_SOC
        p_dis = max(0.0, (float(e_d.sum()) - float(rho_d.sum()) * i_dis) * i_dis)
        if lim.max_discharge_power_W is not None and p_dis > lim.max_discharge_power_W:
            p_dis, binding = float(lim.max_discharge_power_W), binding | LIM_POWER
        return Limit(i_dis, p_dis, binding, hot)

    def charge_limit(self, dt: float, ctx: StepContext | None = None) -> Limit:
        """Maximum charge (regeneration) current/power now, and the constraint that sets it."""
        lim = self.limits
        ctx = ctx or self.begin_step(dt)
        hot, cold = self._hot_derate(), self._cold_charge_derate()
        e_c, rho_c = ctx.linear(-1.0)
        i_volt = float(np.min((self.v_max - e_c) / rho_c))
        taper = 1.0
        s_max = float(self.soc.max())
        if s_max >= lim.soc_ceiling:
            taper = 0.0
        elif s_max > lim.regen_taper_start:
            taper = (lim.soc_ceiling - s_max) / (lim.soc_ceiling - lim.regen_taper_start)
        i_cur = lim.max_charge_C * self.spec.capacity_Ah * hot * cold * taper
        i_chg = max(0.0, min(i_cur, i_volt))
        binding = LIM_VOLTAGE if i_volt < i_cur else LIM_CURRENT
        if hot < 1.0 or cold < 1.0:
            binding |= LIM_THERMAL
        if taper < 1.0:
            binding |= LIM_SOC
        p_chg = max(0.0, (float(e_c.sum()) + float(rho_c.sum()) * i_chg) * i_chg)
        if lim.max_charge_power_W is not None and p_chg > lim.max_charge_power_W:
            p_chg, binding = float(lim.max_charge_power_W), binding | LIM_POWER
        return Limit(i_chg, p_chg, binding, hot * cold)

    def limits_now(self, dt: float) -> CurrentLimits:
        ctx = self.begin_step(dt)
        return CurrentLimits(self.discharge_limit(dt, ctx), self.charge_limit(dt, ctx))

    def sop_limits(self, horizons_s=(2.0, 10.0, 30.0)) -> dict[float, SOPLimit]:
        """State-of-power by horizon, in the same convention as :class:`bms.sop.StateOfPower`.

        Resistance is the step response of the temperature-adjusted 2-RC branch over the
        horizon; the weakest series group sets the current limit.  Current limits and the
        hot derate come from :class:`PackLimits`.
        """
        lim = self.limits
        r0, r1, r2, tau1, tau2, _, _, _ = self._scaled(1.0)
        i_now = self.last_current_A
        v_now = self._rest_group_voltages(i_now, r0)
        hot = self._hot_derate()
        cap = self.spec.capacity_Ah
        out: dict[float, SOPLimit] = {}
        for h in horizons_s:
            r_h = r0 + r1 * (1.0 - np.exp(-h / tau1)) + r2 * (1.0 - np.exp(-h / tau2))
            i_dis_v = float(np.min(i_now + (v_now - self.v_min) / r_h))
            i_chg_v = float(np.min((self.v_max - v_now) / r_h - i_now))
            i_dis = max(0.0, min(i_dis_v, lim.max_discharge_C * cap * hot))
            i_chg = max(0.0, min(i_chg_v, lim.max_charge_C * cap * hot))
            v_dis = float(np.sum(v_now - r_h * (i_dis - i_now)))
            v_chg = float(np.sum(v_now + r_h * (i_chg + i_now)))
            out[float(h)] = SOPLimit(horizon_s=float(h), traction_power_W=max(0.0, v_dis * i_dis),
                                     regen_power_W=max(0.0, v_chg * i_chg), traction_current_A=i_dis,
                                     regen_current_A=i_chg, temperature_derate=hot)
        return out

    def _rest_group_voltages(self, current: float, r0: np.ndarray) -> np.ndarray:
        """Present group terminal voltages ``OCV − V1 − V2 − R0·I`` from the stored state."""
        return self.ocv_curve.ocv(self.soc, current, T_C=self.thermal.T) - self.v1 - self.v2 - r0 * current

    def cooling_duty_request(self, dt: float) -> float:
        """Cooling duty [0, 1] the thermal controller asks for, from the hottest group."""
        return float(self.cooler.step(float(self.thermal.T.max()), dt))

    # ── Time step ──────────────────────────────────────────────────────
    def step(self, current: float, dt: float, cooling_duty: float | None = None,
             ctx: StepContext | None = None) -> PackStep:
        """Advance the pack by ``dt`` at ``current`` [A] (positive = discharge)."""
        r0, r1, r2, tau1, tau2, a1, a2, q = (ctx or self.begin_step(dt)).prm
        T = self.thermal.T.copy()

        v1_old, v2_old = self.v1, self.v2
        self.v1 = a1 * v1_old + (1.0 - a1) * r1 * current
        self.v2 = a2 * v2_old + (1.0 - a2) * r2 * current
        self.soc = _clip01(self.soc - current * dt / (q * 3600.0))
        ocv25 = self._spline(self.soc)                                 # EMF at 25 °C, no hysteresis
        ocv_nohyst = ocv25 + self._temp_coeff * (T - 25.0)             # EMF at temperature, no hysteresis
        h = self._hyst
        ocv_eff = ocv_nohyst - h * np.sign(current)                    # OCVSOC.ocv's hysteresis rule
        vg = ocv_eff - self.v1 - self.v2 - r0 * current
        heat_g = np.maximum(current * (ocv_nohyst - vg), r0 * current * current)

        if cooling_duty is None:
            cooling_duty = self.cooling_duty_request(dt)
        self.thermal.step(heat_g, cooling_duty, dt)

        # Exact mean power into each RC branch over the step, and the change in stored energy.
        in1 = r1 * current ** 2 + (v1_old - r1 * current) * current * tau1 * (1.0 - a1) / dt
        in2 = r2 * current ** 2 + (v2_old - r2 * current) * current * tau2 * (1.0 - a2) / dt
        d_store = (0.5 * self.c1 * (self.v1 ** 2 - v1_old ** 2) + 0.5 * self.c2 * (self.v2 ** 2 - v2_old ** 2)) / dt

        self.last_current_A = float(current)
        self.last_cooling_duty = float(cooling_duty)
        return PackStep(
            current_A=float(current), v_pack_V=float(vg.sum()), p_terminal_W=float(vg.sum() * current),
            v_group_min_V=float(vg.min()), v_group_max_V=float(vg.max()),
            p_ocv25_W=float(ocv25.sum() * current),
            p_tshift_W=float(np.sum(ocv_nohyst - ocv25) * current),
            p_hyst_W=float(h * self.n_series * abs(current)),
            p_r0_W=float(r0.sum() * current ** 2),
            p_pol_in_W=float(in1.sum() + in2.sum()), p_pol_stored_dW=float(d_store.sum()),
            heat_W=float(heat_g.sum()), cooling_duty=float(cooling_duty))

    # ── Snapshots and energy ───────────────────────────────────────────
    def snapshot(self, dt: float = 1.0) -> dict:
        """Per-group arrays at the present state (copies)."""
        r0 = self._scaled(dt)[0]
        return {"soc": self.soc.copy(), "T_C": self.thermal.T.copy(),
                "v_group_V": self._rest_group_voltages(self.last_current_A, r0),
                "v1": self.v1.copy(), "v2": self.v2.copy()}

    def chemical_energy_Wh(self, soc: np.ndarray | None = None) -> float:
        """Energy stored between SOC 0 and ``soc`` per the 25 °C OCV curve, summed over groups [Wh].

        Uses the exact antiderivative of the OCV spline, so it is independent of any time
        integration: the drop in this quantity over a run is the chemical energy the pack gave up.
        """
        s = self.soc if soc is None else np.asarray(soc, float)
        return float(np.sum(self.q0_Ah * (self._ocv_antiderivative(s) - self._ocv_antiderivative(0.0))))

    def rc_stored_energy_Wh(self) -> float:
        return float(np.sum(0.5 * self.c1 * self.v1 ** 2 + 0.5 * self.c2 * self.v2 ** 2) / 3600.0)

    def cooling_power_W(self, duty: float) -> float:
        return float(min(max(duty, 0.0), 1.0) * self.spec.cooling_power_max_W)
