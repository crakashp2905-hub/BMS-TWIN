"""Fit capacity-fade models to measured cycle-life data, and test them honestly.

The repository's :class:`bms.aging.AgingModel` ships *literature-plausible*
coefficients, not coefficients fitted to data.  This module provides the missing
step: fit a fade model to measured capacity-vs-cycle curves, then validate it
**leave-one-cell-out** — train on all cells but one, predict the held-out cell.
A fit that only reproduces its own training data is not evidence of anything.

Models
------
``PowerLawFade``   SoH(n) = 1 − exp(α + β·ln C)·n^b
    A stress-scaled power law: fade grows as n^b (the capacity-fade knee shape),
    with the C-rate C entering through β.  Fitted by nonlinear least squares.

``LiteratureFade`` the repository's shipped :class:`AgingModel` per-cycle fade,
    applied unchanged.  This is the baseline a fitted model must beat.

``LinearPooledFade`` SoH(n) = 1 − k·n fitted to all cells, ignoring stress.
    The simplest honest baseline.

All capacities are normalised to each cell's first-cycle capacity, so the models
compete on the *shape* of fade rather than on cell-to-cell offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

RATED_AH = 2.0          # NASA PCoE 18650 rated capacity
EOL_SOH = 0.80          # end-of-life convention: 80 % of first-cycle capacity


@dataclass(frozen=True)
class FadeCurve:
    """A cell's measured capacity-fade trajectory and its test conditions."""

    cell: str
    c_rate: float
    temperature_C: float
    cycles: np.ndarray
    capacity_Ah: np.ndarray
    n_repaired: int = 0          # capacity dips replaced by the despiker

    @property
    def soh(self) -> np.ndarray:
        """Capacity relative to the first measured cycle (the fade-shape target)."""
        return self.capacity_Ah / self.capacity_Ah[0]

    def cycle_to_eol(self) -> float:
        """Cycles, relative to the first measured cycle, until SoH reaches EOL (NaN if never)."""
        below = np.where(self.soh <= EOL_SOH)[0]
        return float(self.cycles[below[0]] - self.cycles[0]) if below.size else float("nan")


def load_fade_curve(mat_path: str, cell: str) -> FadeCurve:
    """Build a :class:`FadeCurve` from a raw NASA ``.mat`` file.

    The discharge rate is taken from the 90th percentile of the measured
    discharge current (the median can land in a rest period), then expressed as a
    C-rate against the 2 Ah rating.
    """
    import scipy.io as sio

    from .datasets import nasa_mat_to_capacity

    m = sio.loadmat(mat_path, simplify_cells=True)
    cyc = m[cell]["cycle"]
    disch = [x for x in cyc if x["type"] == "discharge"]
    currents = np.array([np.percentile(np.abs(np.asarray(x["data"]["Current_measured"], float)), 90)
                         for x in disch])
    temps = np.array([float(x["ambient_temperature"]) for x in disch])
    from .signal import hampel
    n, q = nasa_mat_to_capacity(mat_path)
    # Measurement dips (aborted or mis-measured discharges) are isolated low outliers;
    # a robust median/MAD despike repairs them without touching the fade trend.
    q_clean, mask = hampel(np.asarray(q, float), window=7, n_sigma=3.0)
    return FadeCurve(cell=cell,
                     c_rate=float(np.median(currents) / RATED_AH),
                     temperature_C=float(np.median(temps)),
                     cycles=np.asarray(n, float), capacity_Ah=q_clean,
                     n_repaired=int(mask.sum()))


def save_fade_csv(curves: list[FadeCurve], path: str) -> None:
    """Write curves to a committed, attributed CSV (derived from public-domain data)."""
    import pandas as pd
    rows = []
    for c in curves:
        for n, q in zip(c.cycles, c.capacity_Ah):
            rows.append({"cell": c.cell, "c_rate": round(c.c_rate, 3),
                         "temperature_C": c.temperature_C, "cycle": int(n),
                         "capacity_Ah": round(float(q), 5)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# NASA Ames PCoE Li-ion battery aging dataset (public domain, US Government).\n")
        f.write("# Discharge capacity per cycle, derived by bms.aging_fit.load_fade_curve.\n")
        pd.DataFrame(rows).to_csv(f, index=False)


def load_fade_csv(path: str) -> list[FadeCurve]:
    import pandas as pd
    df = pd.read_csv(path, comment="#")
    curves = []
    for cell, g in df.groupby("cell", sort=False):
        g = g.sort_values("cycle")
        curves.append(FadeCurve(cell=str(cell), c_rate=float(g["c_rate"].iloc[0]),
                                temperature_C=float(g["temperature_C"].iloc[0]),
                                cycles=g["cycle"].to_numpy(float),
                                capacity_Ah=g["capacity_Ah"].to_numpy(float)))
    return curves


# ── Models ──────────────────────────────────────────────────────────────────

@dataclass
class PowerLawFade:
    """SoH(n, C) = 1 − exp(α + β·ln C)·n^b, fitted by least squares."""

    alpha: float = float("nan")
    beta: float = float("nan")
    b: float = float("nan")
    residual_sigma: float = float("nan")
    _jac_cov: np.ndarray | None = field(default=None, repr=False)

    def _fade(self, n, c, alpha, beta, b):
        return np.exp(alpha + beta * np.log(c)) * np.power(np.maximum(n, 1e-9), b)

    def fit(self, curves: list[FadeCurve]) -> "PowerLawFade":
        n = np.concatenate([c.cycles for c in curves]) - curves[0].cycles[0]
        n = np.concatenate([c.cycles - c.cycles[0] for c in curves])
        cr = np.concatenate([np.full(c.cycles.size, c.c_rate) for c in curves])
        y = np.concatenate([1.0 - c.soh for c in curves])
        mask = n > 0
        n, cr, y = n[mask], cr[mask], y[mask]

        def resid(theta):
            return self._fade(n, cr, *theta) - y

        res = least_squares(resid, x0=[np.log(0.01), 0.0, 0.7],
                            bounds=([-20.0, -5.0, 0.1], [5.0, 5.0, 2.0]))
        self.alpha, self.beta, self.b = (float(v) for v in res.x)
        dof = max(1, y.size - 3)
        self.residual_sigma = float(np.sqrt(np.sum(res.fun ** 2) / dof))
        return self

    def predict_soh(self, cycles: np.ndarray, c_rate: float) -> np.ndarray:
        n = np.asarray(cycles, float)
        return 1.0 - self._fade(n, c_rate, self.alpha, self.beta, self.b)


@dataclass
class LinearPooledFade:
    """SoH(n) = 1 − k·n, one slope for every cell (no stress information)."""

    k: float = float("nan")
    residual_sigma: float = float("nan")

    def fit(self, curves: list[FadeCurve]) -> "LinearPooledFade":
        n = np.concatenate([c.cycles - c.cycles[0] for c in curves])
        y = np.concatenate([1.0 - c.soh for c in curves])
        self.k = float(np.sum(n * y) / np.sum(n * n))
        self.residual_sigma = float(np.std(y - self.k * n))
        return self

    def predict_soh(self, cycles: np.ndarray, c_rate: float) -> np.ndarray:
        return 1.0 - self.k * (np.asarray(cycles, float))


@dataclass
class LiteratureFade:
    """The repository's shipped AgingModel per-cycle fade, applied unchanged."""

    temperature_C: float = 24.0

    def fit(self, curves: list[FadeCurve]) -> "LiteratureFade":
        return self                          # nothing to fit: this is the literature prior

    def predict_soh(self, cycles: np.ndarray, c_rate: float) -> np.ndarray:
        from .aging import AgingModel
        per_cycle, _ = AgingModel().charge_fade(
            c_rate=c_rate, temperature_C=self.temperature_C, dod=1.0,
            soc_avg=0.5, throughput_efc=1.0)
        return 1.0 - per_cycle * np.asarray(cycles, float)


# ── Validation ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CellScore:
    cell: str
    c_rate: float
    soh_rmse: float
    eol_error_cycles: float        # NaN when the held-out cell never reaches EOL


def leave_one_out(curves: list[FadeCurve], model_factory) -> list[CellScore]:
    """Train on all cells but one; score the prediction on the held-out cell."""
    scores = []
    for i, held in enumerate(curves):
        train = [c for j, c in enumerate(curves) if j != i]
        model = model_factory().fit(train)
        pred = model.predict_soh(held.cycles - held.cycles[0], held.c_rate)
        rmse = float(np.sqrt(np.mean((pred - held.soh) ** 2)))
        eol_true = held.cycle_to_eol()
        eol_pred = _predicted_eol(model, held)
        eol_err = (eol_pred - eol_true) if np.isfinite(eol_true) else float("nan")
        scores.append(CellScore(held.cell, held.c_rate, rmse, eol_err))
    return scores


def _predicted_eol(model, curve: FadeCurve) -> float:
    grid = np.arange(0.0, 5000.0)
    soh = model.predict_soh(grid, curve.c_rate)
    below = np.where(soh <= EOL_SOH)[0]
    return float(grid[below[0]]) if below.size else float("nan")


# ── Honest interval calibration ─────────────────────────────────────────────

@dataclass(frozen=True)
class NestedCalibration:
    """Out-of-sample check of a prediction-interval scale.

    For each held-out cell, the interval scale is learned only from inner
    leave-one-cell-out predictions on the *other* cells, then applied to the held-out
    cell. Coverage is therefore never measured on data that set the scale.
    """

    model: str
    level: float
    scales: dict            # held-out cell -> scale learned without it
    raw_coverage: float     # held-out coverage with unscaled sigma
    nested_coverage: float  # held-out coverage with the nested scale
    per_cell_coverage: dict


def nested_interval_calibration(curves: list[FadeCurve], model_factory,
                                level: float = 0.95) -> NestedCalibration:
    """Run the nested calibration check described in :class:`NestedCalibration`."""
    scales, raw_hits, nest_hits, per_cell = {}, [], [], {}
    all_raw, all_nested = [], []
    for i, held in enumerate(curves):
        others = [c for j, c in enumerate(curves) if j != i]
        yi, mi, si = [], [], []
        for inner in others:                       # inner LOCO on the other cells only
            train = [c for c in others if c is not inner]
            m = model_factory().fit(train)
            n = inner.cycles - inner.cycles[0]
            yi.append(inner.soh)
            mi.append(m.predict_soh(n, inner.c_rate))
            si.append(np.full(n.size, m.residual_sigma))
        yi, mi, si = (np.concatenate(v) for v in (yi, mi, si))
        scale = calibration_scale_for(yi, mi, si, level)
        scales[held.cell] = scale

        m = model_factory().fit(others)
        n = held.cycles - held.cycles[0]
        y, mu, s = held.soh, m.predict_soh(n, held.c_rate), np.full(n.size, m.residual_sigma)
        raw = coverage_for(y, mu, s, level)
        nest = coverage_for(y, mu, scale * s, level)
        raw_hits.append(raw)
        nest_hits.append(nest)
        per_cell[held.cell] = nest
        all_raw.append((y, mu, s))
        all_nested.append((y, mu, scale * s))
    name = getattr(model_factory, "__name__", str(model_factory))
    return NestedCalibration(
        model=name, level=level, scales=scales,
        raw_coverage=float(np.mean(raw_hits)), nested_coverage=float(np.mean(nest_hits)),
        per_cell_coverage=per_cell)


def calibration_scale_for(y, mu, sigma, level):
    from .uncertainty import calibration_scale
    return calibration_scale(y, mu, sigma, level)


def coverage_for(y, mu, sigma, level):
    from .uncertainty import coverage
    return coverage(y, mu, sigma, level)
