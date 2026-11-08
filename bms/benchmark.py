"""Synthetic ground-truth benchmark — the thing real datasets can't give you.

Real data tells you voltage; it rarely tells you the *true* SoC, R0 or capacity.
The simulator can: generate a cell with **known hidden state**, show the twin only
telemetry, and measure how well it reconstructs what it was never told.  Running
this across many randomly-drawn cells (different chemistries, resistances,
capacities, starting SoC) is a **cross-cell generalization** test, and comparing
methods on the *same* cells is a clean **ablation**.

    run_synthetic_benchmark(n=20, method="autocal")   # vs "ekf_generic", "coulomb"

`BenchmarkResult` reports SoC-reconstruction RMSE (mean / P90) plus, for the
calibrating method, chemistry-identification accuracy and capacity error — so the
value of auto-calibration over a generic model is measured, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ecm import ECMParameters, SecondOrderECM
from .ocv_soc import OCVSOC

# Three maximally OCV-distinct chemistries (sloped / flat-plateau / low-voltage)
# keep the benchmark honest and fast to identify — NMC and NCA are near-identical
# and deliberately not both used here.
_CHEMS = ("nmc", "lfp", "lto")
_CAP = {"nmc": 2.3, "lfp": 3.2, "lto": 2.0}


@dataclass
class GroundTruthCell:
    """A cell whose full state is known (and hidden from the twin)."""

    chemistry: str
    params: ECMParameters
    soc0: float
    noise_V: float = 0.003
    current_bias_frac: float = 0.03      # a realistic current-sensor scale error

    def telemetry(self, dt: float = 1.0, n: int = 900, seed: int = 0):
        """Return (current_measured, voltage, temperature, soc_true).

        The plant runs on the *true* current; the estimator is handed a **biased**
        current (a scale error, as a real current sensor has), so coulomb-counting
        drifts and voltage-anchored estimators must earn their keep. Only
        ``current_measured``/``voltage``/``temperature`` are 'seen'; ``soc_true``
        comes from the true current.
        """
        q = self.params.Q_nom_Ah / 2.3
        i_true = np.zeros(n)
        i_true[60:340] = 1.4 * q
        i_true[400:600] = -1.0 * q
        i_true[660:880] = 1.2 * q
        sim = SecondOrderECM(params=self.params, ocv_curve=OCVSOC.from_chemistry(self.chemistry)
                             ).simulate(i_true, dt, soc0=self.soc0)
        v = sim["v_terminal"] + np.random.default_rng(seed).normal(0, self.noise_V, n)
        i_meas = i_true * (1.0 + self.current_bias_frac)      # sensor scale error
        return i_meas, v, np.full(n, 25.0), sim["soc"]


def _random_cell(rng: np.random.Generator) -> GroundTruthCell:
    chem = str(rng.choice(_CHEMS))
    cap = _CAP[chem] * float(rng.uniform(0.85, 1.0))     # some aging
    p = ECMParameters(R0=float(rng.uniform(0.015, 0.06)),
                      R1=float(rng.uniform(0.01, 0.03)), C1=2500,
                      R2=float(rng.uniform(0.02, 0.05)), C2=9000,
                      Q_nom_Ah=cap, chemistry=chem)
    return GroundTruthCell(chemistry=chem, params=p, soc0=float(rng.uniform(0.75, 0.95)))


@dataclass(frozen=True)
class BenchmarkResult:
    """Reconstruction accuracy of a method across a synthetic cell population."""

    method: str
    n_cells: int
    soc_rmse_mean: float
    soc_rmse_p90: float
    chemistry_id_accuracy: float        # NaN for non-calibrating methods
    r0_error_mean: float                # NaN for non-calibrating methods
    per_cell: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"method": self.method, "n_cells": self.n_cells,
                "soc_rmse_mean": self.soc_rmse_mean, "soc_rmse_p90": self.soc_rmse_p90,
                "chemistry_id_accuracy": self.chemistry_id_accuracy,
                "r0_error_mean": self.r0_error_mean}


def _soc_rmse(estimator, soc0, i, v, dt, soc_true) -> float:
    estimator.reset(float(soc0))
    soc = np.asarray(estimator.run(i, v, dt), float)
    return float(np.sqrt(np.mean((soc - soc_true) ** 2)))


def run_synthetic_benchmark(n: int = 20, *, method: str = "autocal", dt: float = 1.0,
                            seed: int = 0) -> BenchmarkResult:
    """Reconstruct hidden state for ``n`` random cells and score it.

    Capacity is treated as **nameplate-known** (it is, in practice); the
    reconstructed unknowns scored here are SoC, chemistry and ``R0``.

    ``method``:
      * ``"coulomb"`` — coulomb counter (nameplate capacity);
      * ``"ekf_generic"`` — EKF with a fixed default (NMC) model (no calibration);
      * ``"autocal"`` — :func:`bms.auto_calibrate` (identify the cell) then EKF.
    """
    from .autocal import auto_calibrate
    from .estimation import make_soc_estimator

    rng = np.random.default_rng(seed)
    rmses, chem_hits, r0_errs, per_cell = [], [], [], []
    for c in range(n):
        cell = _random_cell(rng)
        cap = cell.params.Q_nom_Ah                       # nameplate capacity
        i, v, T, soc_true = cell.telemetry(dt=dt, seed=c)

        if method == "autocal":
            cal = auto_calibrate(i, v, dt, capacity_Ah=cap, chemistries=_CHEMS)
            est = cal.make_estimator("ekf")
            rmse = _soc_rmse(est, cal.soc0, i, v, dt, soc_true)
            chem_hits.append(1.0 if cal.chemistry == cell.chemistry else 0.0)
            r0_errs.append(abs(cal.params.R0 - cell.params.R0) / cell.params.R0)
        elif method == "coulomb":
            est = make_soc_estimator("coulomb", params=ECMParameters(Q_nom_Ah=cap),
                                     ocv_curve=OCVSOC(), capacity_Ah=cap)
            rmse = _soc_rmse(est, soc_true[0], i, v, dt, soc_true)
        else:  # ekf_generic
            est = make_soc_estimator("ekf", params=ECMParameters(Q_nom_Ah=cap),
                                     ocv_curve=OCVSOC.from_chemistry("nmc"), capacity_Ah=cap)
            rmse = _soc_rmse(est, soc_true[0], i, v, dt, soc_true)

        rmses.append(rmse)
        per_cell.append({"chemistry": cell.chemistry, "soc_rmse": rmse})

    rmses = np.asarray(rmses, float)
    return BenchmarkResult(
        method=method, n_cells=n,
        soc_rmse_mean=float(np.mean(rmses)), soc_rmse_p90=float(np.percentile(rmses, 90)),
        chemistry_id_accuracy=float(np.mean(chem_hits)) if chem_hits else float("nan"),
        r0_error_mean=float(np.mean(r0_errs)) if r0_errs else float("nan"),
        per_cell=per_cell,
    )
