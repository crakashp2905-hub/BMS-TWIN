"""Does an aging model transfer across datasets once the cycle axis is throughput?

The cross-dataset failure in docs/aging_validation.md (NASA → Oxford, 678–838 %)
has one obvious suspect: the x-axis. NASA cells cycle to 2 Ah, Oxford cells to
0.74 Ah, and Oxford characterisation points are sparse (every 100 cycles). Counting
*cycles* therefore compares different amounts of charge passed. This script re-indexes
each curve by equivalent full cycles (EFC):

    EFC(n) = ∫₀ⁿ Q(n') dn' / Q_rated

where Q is the measured discharge capacity, integrated by the trapezoid rule over the
measured points, so sparse sampling is handled, and Q_rated is each dataset's nominal
capacity. Models are then fitted on NASA and scored on Oxford in EFC, exactly as in
scripts/validate_oxford.py test C.

    python scripts/validate_throughput_transfer.py path/to/Oxford_Battery_Degradation_Dataset_1.mat

Requires the NASA cache (.nasa_cache/B00*.mat), which validate_aging.py downloads.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from bms.aging_fit import (  # noqa: E402
    RATED_AH,
    FadeCurve,
    LinearPooledFade,
    PowerLawFade,
    load_fade_curve,
    load_oxford_curves,
)

_ROOT = Path(__file__).resolve().parents[1]
NASA = ("B0005", "B0006", "B0007", "B0018", "B0025", "B0026", "B0027", "B0028")
OXFORD_RATED_AH = 0.74   # Kokam 740 mAh pouch


def to_efc(curve: FadeCurve, rated_Ah: float) -> FadeCurve:
    """Re-index a curve's cycle axis by cumulative equivalent full cycles."""
    n = np.asarray(curve.cycles, float)
    q = np.asarray(curve.capacity_Ah, float)
    throughput = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(n))])
    return replace(curve, cycles=throughput / rated_Ah)


def _transfer(nasa, oxford, label: str) -> None:
    """Fit on NASA; score on all Oxford points, and on the subset inside NASA's range.

    The in-range score separates two failures: a wrong fade law, versus a model
    extrapolated far beyond the x-range it was fitted on (NASA stops at ~137 EFC).
    """
    x_max = max(float(c.cycles[-1] - c.cycles[0]) for c in nasa)
    print(f"\n== {label}: fit on NASA (x ≤ {x_max:.0f}), predict Oxford (no Oxford data) ==")
    for name, cls in (("linear pooled", LinearPooledFade), ("stress power law", PowerLawFade)):
        model = cls().fit(nasa)
        all_errs, in_errs = [], []
        for c in oxford:
            x = c.cycles - c.cycles[0]
            pred = model.predict_soh(x, c.c_rate)
            all_errs.append(np.sqrt(np.mean((pred - c.soh) ** 2)))
            inside = x <= x_max
            if inside.sum() >= 1:
                in_errs.append(np.sqrt(np.mean((pred[inside] - c.soh[inside]) ** 2)))
        print(f"  {name:18s} all points  {np.mean(all_errs) * 100:8.1f}% RMSE")
        if in_errs:
            print(f"  {'':18s} in-range    {np.mean(in_errs) * 100:8.1f}% RMSE "
                  f"({len(in_errs)} cells with points inside the NASA range)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ox = load_oxford_curves(sys.argv[1])
    nasa = [load_fade_curve(str(_ROOT / ".nasa_cache" / f"{c}.mat"), c) for c in NASA]

    print("== Equivalent full cycles at end of recorded life ==")
    for c in nasa:
        e = to_efc(c, RATED_AH)
        print(f"  NASA   {c.cell}: {c.cycles[-1] - c.cycles[0]:6.0f} cycles = {e.cycles[-1]:7.1f} EFC")
    for c in ox:
        e = to_efc(c, OXFORD_RATED_AH)
        print(f"  Oxford {c.cell}: {c.cycles[-1] - c.cycles[0]:6.0f} cycles = {e.cycles[-1]:7.1f} EFC")

    _transfer(nasa, ox, "cycle axis (baseline, as in validate_oxford.py)")
    nasa_efc = [to_efc(c, RATED_AH) for c in nasa]
    ox_efc = [to_efc(c, OXFORD_RATED_AH) for c in ox]
    _transfer(nasa_efc, ox_efc, "throughput axis (EFC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
