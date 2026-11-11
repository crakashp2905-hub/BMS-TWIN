"""Drive cycles: standardized and custom speed-versus-time profiles.

A :class:`DriveCycle` is a validated speed trace (plus optional road grade) that the
EV simulator (:mod:`bms.ev_drive`) replays.  Three sources feed it:

* **Standardized cycles** bundled in ``bms/cycles/`` (EPA UDDS, HWFET, US06, and the
  UN/ECE ECE-15 and EUDC parts, with an ``nedc`` composite).  The files are the
  EPA's own, unchanged; see ``bms/cycles/README.md`` for provenance and checksums.
* **Synthetic cycles** (``urban_stop_go``, ``arterial``, ``highway``, ``cruise_100kmh``).
* **Custom profiles**: :func:`load_cycle_csv` for any measured or hand-made trace,
  or :meth:`DriveCycle.from_kmh` for arrays.

Every constructor validates its input.  A bad trace (NaN, negative speed, time going
backwards, a unit mistake that implies a 50 m/s² launch) raises
:class:`DriveCycleError` with the reason, instead of producing a plausible-looking
but wrong simulation.

    cycle = get_cycle("udds")                 # standardized
    cycle = load_cycle_csv("trip.csv", speed_unit="kmh")   # custom
    run = EVSimulator(config).run(cycle)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

_MPH_TO_MPS = 0.44704
_KMH_TO_MPS = 1.0 / 3.6
_CYCLE_DIR = Path(__file__).resolve().parent / "cycles"

_MOVING_MPS = 0.1          # below this the vehicle counts as stopped
_MAX_SPEED_MPS = 100.0     # 360 km/h: above this a profile is almost certainly a unit error
_MAX_ACCEL_MPS2 = 8.0      # above this a profile is almost certainly a unit error
_MAX_GRADE_RAD = 0.5       # about ±53 %


class DriveCycleError(ValueError):
    """A drive-cycle profile is malformed or physically implausible."""


@dataclass(frozen=True)
class CycleStats:
    """Summary of a speed trace."""

    duration_s: float
    distance_km: float
    mean_speed_kmh: float            # including stops
    mean_running_speed_kmh: float    # excluding stops
    max_speed_kmh: float
    max_accel_mps2: float
    max_decel_mps2: float            # reported as a positive magnitude
    idle_fraction: float
    n_stops: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class DriveCycle:
    """A validated speed (and optional grade) trace on a strictly increasing time axis."""

    name: str
    t_s: np.ndarray
    speed_mps: np.ndarray
    grade_rad: np.ndarray | None = None
    source: str = "custom"
    description: str = ""

    def __post_init__(self) -> None:
        t = np.array(self.t_s, dtype=float)
        v = np.array(self.speed_mps, dtype=float)
        if t.ndim != 1 or v.ndim != 1 or t.size != v.size:
            raise DriveCycleError("time and speed must be 1-D arrays of the same length")
        if t.size < 2:
            raise DriveCycleError("a drive cycle needs at least two samples")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v))):
            raise DriveCycleError("time and speed must be finite (found NaN or inf)")
        if np.any(np.diff(t) <= 0.0):
            raise DriveCycleError("time must be strictly increasing")
        if np.any(v < 0.0):
            raise DriveCycleError("speed must be non-negative (no reverse driving)")
        g = None
        if self.grade_rad is not None:
            g = np.array(self.grade_rad, dtype=float)
            if g.shape != t.shape:
                raise DriveCycleError("grade must have the same length as time")
            if not np.all(np.isfinite(g)):
                raise DriveCycleError("grade must be finite")
            if np.any(np.abs(g) > _MAX_GRADE_RAD):
                raise DriveCycleError(f"|grade| exceeds {_MAX_GRADE_RAD} rad; is it in percent?")
            g.setflags(write=False)
        t.setflags(write=False)
        v.setflags(write=False)
        object.__setattr__(self, "t_s", t)
        object.__setattr__(self, "speed_mps", v)
        object.__setattr__(self, "grade_rad", g)

    # ── Constructors ───────────────────────────────────────────────────
    @classmethod
    def from_kmh(cls, name: str, t_s, speed_kmh, *, grade_rad=None, source: str = "custom",
                 description: str = "") -> "DriveCycle":
        """Build a cycle from km/h samples."""
        return cls(name, np.asarray(t_s, float), np.asarray(speed_kmh, float) * _KMH_TO_MPS,
                   None if grade_rad is None else np.asarray(grade_rad, float), source, description)

    # ── Basic properties ───────────────────────────────────────────────
    @property
    def speed_kmh(self) -> np.ndarray:
        return self.speed_mps * 3.6

    @property
    def duration_s(self) -> float:
        return float(self.t_s[-1] - self.t_s[0])

    @property
    def dt_s(self) -> float:
        """Median sample interval."""
        return float(np.median(np.diff(self.t_s)))

    @property
    def is_uniform(self) -> bool:
        d = np.diff(self.t_s)
        return bool(np.allclose(d, d[0], rtol=1e-9, atol=1e-12))

    @property
    def distance_m(self) -> float:
        return float(np.sum(0.5 * (self.speed_mps[1:] + self.speed_mps[:-1]) * np.diff(self.t_s)))

    def __len__(self) -> int:
        return int(self.t_s.size)

    def __repr__(self) -> str:
        return (f"DriveCycle({self.name!r}, {len(self)} samples, {self.duration_s:.0f} s, "
                f"{self.distance_m / 1000.0:.2f} km, source={self.source!r})")

    # ── Analysis ───────────────────────────────────────────────────────
    def stats(self) -> CycleStats:
        dt = np.diff(self.t_s)
        acc = np.diff(self.speed_mps) / dt
        moving = self.speed_mps > _MOVING_MPS
        # A segment counts as moving time when either of its ends is moving.
        t_moving = float(np.sum(dt[moving[1:] | moving[:-1]]))
        dist_km = self.distance_m / 1000.0
        n_stops = int(np.sum(moving[:-1] & ~moving[1:]))      # moving → stopped transitions
        return CycleStats(
            duration_s=self.duration_s,
            distance_km=dist_km,
            mean_speed_kmh=dist_km / (self.duration_s / 3600.0),
            mean_running_speed_kmh=dist_km / (t_moving / 3600.0) if t_moving > 0 else 0.0,
            max_speed_kmh=float(self.speed_mps.max() * 3.6),
            max_accel_mps2=float(max(0.0, acc.max())),
            max_decel_mps2=float(max(0.0, -acc.min())),
            idle_fraction=float(1.0 - t_moving / self.duration_s),
            n_stops=n_stops,
        )

    def check_plausibility(self, max_accel_mps2: float = _MAX_ACCEL_MPS2,
                           max_speed_mps: float = _MAX_SPEED_MPS) -> "DriveCycle":
        """Raise :class:`DriveCycleError` if the trace implies impossible driving."""
        if float(self.speed_mps.max()) > max_speed_mps:
            raise DriveCycleError(
                f"peak speed {self.speed_mps.max():.1f} m/s exceeds {max_speed_mps:.0f} m/s; "
                "check the speed unit")
        acc = np.abs(np.diff(self.speed_mps) / np.diff(self.t_s))
        if float(acc.max()) > max_accel_mps2:
            raise DriveCycleError(
                f"peak acceleration {acc.max():.1f} m/s² exceeds {max_accel_mps2:.0f} m/s²; "
                "check the speed and time units")
        return self

    # ── Transformations (each returns a new cycle) ─────────────────────
    def resample(self, dt_s: float) -> "DriveCycle":
        """Linear resampling to a uniform grid starting at the first sample."""
        if not dt_s > 0.0:
            raise DriveCycleError("dt_s must be positive")
        grid = np.arange(self.t_s[0], self.t_s[-1] + 1e-9, dt_s)
        v = np.interp(grid, self.t_s, self.speed_mps)
        g = None if self.grade_rad is None else np.interp(grid, self.t_s, self.grade_rad)
        return replace(self, t_s=grid - grid[0], speed_mps=v, grade_rad=g)

    def with_grade(self, grade_rad) -> "DriveCycle":
        """Return a copy with a constant or per-sample road grade [rad]."""
        g = np.full(len(self), float(grade_rad)) if np.isscalar(grade_rad) else np.asarray(grade_rad, float)
        return replace(self, grade_rad=g)

    def repeat(self, n: int) -> "DriveCycle":
        if int(n) < 1:
            raise DriveCycleError("repeat count must be at least 1")
        return concatenate([self] * int(n), name=f"{self.name}_x{int(n)}")

    def slice(self, t0_s: float, t1_s: float) -> "DriveCycle":
        """The part of the cycle between two times, re-zeroed at ``t0_s``."""
        keep = (self.t_s >= t0_s) & (self.t_s <= t1_s)
        if keep.sum() < 2:
            raise DriveCycleError("slice contains fewer than two samples")
        g = None if self.grade_rad is None else self.grade_rad[keep]
        return replace(self, t_s=self.t_s[keep] - self.t_s[keep][0], speed_mps=self.speed_mps[keep],
                       grade_rad=g)

    # ── Identity ───────────────────────────────────────────────────────
    def fingerprint(self) -> str:
        """SHA-256 over the trace (rounded to 1 µm/s), for run manifests."""
        h = hashlib.sha256()
        h.update(np.round(self.t_s, 6).tobytes())
        h.update(np.round(self.speed_mps, 6).tobytes())
        if self.grade_rad is not None:
            h.update(np.round(self.grade_rad, 8).tobytes())
        return h.hexdigest()

    def to_csv(self, path: str | Path) -> None:
        """Write ``time_s,speed_kmh[,grade_rad]`` (readable by :func:`load_cycle_csv`)."""
        import pandas as pd
        cols = {"time_s": self.t_s, "speed_kmh": self.speed_kmh}
        if self.grade_rad is not None:
            cols["grade_rad"] = self.grade_rad
        pd.DataFrame(cols).to_csv(path, index=False)


def concatenate(cycles: list[DriveCycle], *, name: str = "concatenated", gap_s: float = 0.0) -> DriveCycle:
    """Join cycles end to end.

    With ``gap_s == 0`` the joint sample is shared, so the end speed of one cycle must
    equal the start speed of the next (true for the bundled cycles, which all start and
    end at rest).  Otherwise insert a positive ``gap_s``.
    """
    if not cycles:
        raise DriveCycleError("nothing to concatenate")
    ts = [np.asarray(cycles[0].t_s - cycles[0].t_s[0])]
    vs = [np.asarray(cycles[0].speed_mps)]
    gs = [cycles[0].grade_rad]
    have_grade = any(c.grade_rad is not None for c in cycles)
    if have_grade:
        gs[0] = cycles[0].grade_rad if cycles[0].grade_rad is not None else np.zeros(len(cycles[0]))
    offset = float(ts[0][-1])
    for c in cycles[1:]:
        t = np.asarray(c.t_s - c.t_s[0]) + offset + gap_s
        v = np.asarray(c.speed_mps)
        g = (c.grade_rad if c.grade_rad is not None else np.zeros(len(c))) if have_grade else None
        if gap_s == 0.0:
            if abs(float(v[0]) - float(vs[-1][-1])) > 1e-9:
                raise DriveCycleError("speed discontinuity at the joint; use gap_s > 0")
            t, v = t[1:], v[1:]
            g = None if g is None else g[1:]
        ts.append(t)
        vs.append(v)
        if have_grade:
            gs.append(g)
        offset = float(t[-1])
    return DriveCycle(name, np.concatenate(ts), np.concatenate(vs),
                      np.concatenate(gs) if have_grade else None,
                      source="composite", description=f"concatenation of {len(cycles)} cycle(s)")


# ── Loaders ────────────────────────────────────────────────────────────
def load_cycle_csv(path: str | Path, *, speed_unit: str = "kmh", time_col: str = "time_s",
                   speed_col: str | None = None, grade_col: str | None = None,
                   grade_unit: str = "rad", name: str | None = None,
                   resample_dt_s: float | None = None) -> DriveCycle:
    """Load a custom speed profile from a CSV with a header row.

    ``speed_unit`` is ``"kmh"``, ``"mps"`` or ``"mph"``; ``speed_col`` defaults to
    ``speed_<unit>``.  ``grade_unit`` is ``"rad"``, ``"deg"`` or ``"percent"``.  Lines
    starting with ``#`` are comments.
    """
    import pandas as pd
    factors = {"kmh": _KMH_TO_MPS, "mps": 1.0, "mph": _MPH_TO_MPS}
    if speed_unit not in factors:
        raise DriveCycleError(f"speed_unit must be one of {sorted(factors)}")
    speed_col = speed_col or f"speed_{speed_unit}"
    try:
        df = pd.read_csv(path, comment="#")
    except Exception as exc:                       # unreadable file → one clear error type
        raise DriveCycleError(f"cannot read {path}: {exc}") from exc
    for col in (time_col, speed_col):
        if col not in df.columns:
            raise DriveCycleError(f"{path}: missing column {col!r} (found {list(df.columns)})")
    grade = None
    if grade_col is not None:
        if grade_col not in df.columns:
            raise DriveCycleError(f"{path}: missing grade column {grade_col!r}")
        raw = df[grade_col].to_numpy(float)
        grade = {"rad": raw, "deg": np.deg2rad(raw), "percent": np.arctan(raw / 100.0)}.get(grade_unit)
        if grade is None:
            raise DriveCycleError("grade_unit must be 'rad', 'deg' or 'percent'")
    cycle = DriveCycle(name or Path(path).stem, df[time_col].to_numpy(float),
                       df[speed_col].to_numpy(float) * factors[speed_unit], grade,
                       source=str(path), description="custom profile")
    cycle.check_plausibility()
    return cycle.resample(resample_dt_s) if resample_dt_s else cycle


def load_epa_cycle(path: str | Path, *, name: str | None = None) -> DriveCycle:
    """Load an EPA dynamometer schedule: two header lines, then ``time  speed`` rows.

    The unit (``mph`` or ``kph``) is read from the second header line.
    """
    p = Path(path)
    lines = p.read_text(encoding="latin-1").splitlines()
    if len(lines) < 4:
        raise DriveCycleError(f"{p.name}: too short to be a drive schedule")
    header = lines[1].lower()
    if "mph" in header:
        factor = _MPH_TO_MPS
    elif "kph" in header or "km/h" in header:
        factor = _KMH_TO_MPS
    else:
        raise DriveCycleError(f"{p.name}: second header line names no speed unit (mph/kph)")
    t, v = [], []
    for number, line in enumerate(lines[2:], start=3):
        parts = line.split()
        if not parts:
            continue
        try:
            t.append(float(parts[0]))
            v.append(float(parts[1]) * factor)
        except (ValueError, IndexError) as exc:
            raise DriveCycleError(f"{p.name}:{number}: not a 'time speed' row: {line!r}") from exc
    title = lines[0].split("\t")[0].strip() or p.stem
    cycle = DriveCycle(name or p.stem, np.array(t), np.array(v), source=f"EPA {title}",
                       description=lines[0].split("\t", 1)[-1].strip())
    return cycle.check_plausibility()


# ── Registry ───────────────────────────────────────────────────────────
_BUNDLED = {
    "udds": ("uddscol.txt", "EPA Urban Dynamometer Driving Schedule (city)"),
    "hwfet": ("hwycol.txt", "EPA Highway Fuel Economy Test"),
    "us06": ("us06col.txt", "EPA US06 supplemental schedule (aggressive, high speed)"),
    "ece15": ("ececol.txt", "UN/ECE Reg 83 elementary urban cycle (one part)"),
    "eudc": ("eudccol.txt", "UN/ECE Reg 83 extra-urban driving cycle"),
}
_BUILTIN_NAMES = frozenset({"nedc", "urban_stop_go", "arterial", "highway", "cruise_100kmh"})
_custom: dict[str, DriveCycle] = {}


def constant_speed(speed_kmh: float, duration_s: float, dt_s: float = 1.0,
                   name: str | None = None) -> DriveCycle:
    """Steady cruise from rest-free start: the speed is held for the whole duration."""
    t = np.arange(0.0, duration_s + 1e-9, dt_s)
    return DriveCycle.from_kmh(name or f"cruise_{speed_kmh:g}kmh", t, np.full(t.size, float(speed_kmh)),
                               source="synthetic", description=f"constant {speed_kmh:g} km/h")


def _synthetic(name: str) -> DriveCycle:
    from . import vehicle_dynamics as vd
    gen = {"urban_stop_go": (vd.urban_stop_go, "repeated accelerate / cruise / brake / idle, 35 km/h peak"),
           "arterial": (vd.arterial, "moderate-speed arterial road, about 60 km/h"),
           "highway": (vd.highway, "sustained 100 km/h cruise with a merge")}[name]
    v = gen[0](1.0)
    return DriveCycle(name, np.arange(v.size, dtype=float), v, source="synthetic", description=gen[1])


@lru_cache(maxsize=None)
def _bundled(name: str) -> DriveCycle:
    if name == "nedc":
        ece, eudc = _bundled("ece15"), _bundled("eudc")
        c = concatenate([ece] * 4 + [eudc], name="nedc")
        return replace(c, source="EPA ECE-15 x4 + EUDC",
                       description="composite: four ECE-15 parts then one EUDC part (not a certified test)")
    fname, desc = _BUNDLED[name]
    c = load_epa_cycle(_CYCLE_DIR / fname, name=name)
    return replace(c, description=desc)


def available_cycles() -> dict[str, str]:
    """Registered cycle names with a one-line description."""
    out = {n: d for n, (_, d) in _BUNDLED.items()}
    out["nedc"] = "composite: four ECE-15 parts then one EUDC part"
    out.update({"urban_stop_go": "synthetic urban stop-go", "arterial": "synthetic arterial road",
                "highway": "synthetic highway cruise", "cruise_100kmh": "10 minutes at a steady 100 km/h"})
    out.update({n: c.description for n, c in _custom.items()})
    return out


def register_cycle(name: str, cycle: DriveCycle, *, overwrite: bool = False) -> None:
    """Make a custom cycle available through :func:`get_cycle`."""
    key = name.lower()
    if key in _BUNDLED or key in _BUILTIN_NAMES:
        raise DriveCycleError(f"{name!r} is a built-in cycle name")
    if key in _custom and not overwrite:
        raise DriveCycleError(f"a cycle named {name!r} is already registered")
    _custom[key] = replace(cycle, name=key)


def get_cycle(name: str, *, dt_s: float | None = None) -> DriveCycle:
    """Look up a cycle by name (case-insensitive), optionally resampled to ``dt_s``."""
    key = name.lower()
    if key in _custom:
        cycle = _custom[key]
    elif key in _BUNDLED or key == "nedc":
        cycle = _bundled(key)
    elif key in {"urban_stop_go", "arterial", "highway"}:
        cycle = _synthetic(key)
    elif key == "cruise_100kmh":
        cycle = constant_speed(100.0, 600.0, name="cruise_100kmh")
    else:
        raise DriveCycleError(f"unknown cycle {name!r}; available: {sorted(available_cycles())}")
    return cycle.resample(dt_s) if dt_s else cycle
