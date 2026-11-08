"""Cross-dataset aging validation on the Oxford Battery Degradation Dataset 1.

Oxford cells (Kokam 740 mAh pouch, aged at 40 °C under an urban drive cycle) are a
different cell, lab and stress from the NASA 18650 cells the models were fitted on.
Three tests, all on held-out data:

  A. the shipped literature defaults, applied at the Oxford temperature;
  B. leave-one-cell-out fitting within Oxford;
  C. transfer: fit on all NASA cells, predict the Oxford cells with no Oxford data.

    python scripts/validate_oxford.py path/to/Oxford_Battery_Degradation_Dataset_1.mat

The data file is not committed (ODbL licence). Download it from the Oxford Research
Archive. The derived numbers are reported in docs/aging_validation.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from bms.aging_fit import (  # noqa: E402
    LinearPooledFade,
    LiteratureFade,
    PowerLawFade,
    leave_one_out,
    load_fade_curve,
    load_oxford_curves,
)

_ROOT = Path(__file__).resolve().parents[1]
NASA = ("B0005", "B0006", "B0007", "B0018", "B0025", "B0026", "B0027", "B0028")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ox = load_oxford_curves(sys.argv[1])
    nasa_dir = _ROOT / ".nasa_cache"
    nasa = [load_fade_curve(str(nasa_dir / f"{c}.mat"), c) for c in NASA]

    print("== Oxford cells (SoH end of recorded life) ==")
    for c in ox:
        print(f"  {c.cell}: {len(c.cycles)} points to cycle {int(c.cycles[-1])}, "
              f"SoH end {c.soh[-1]:.3f}, dips repaired {c.n_repaired}")

    print("\n== A. shipped literature defaults at 40 °C (no Oxford data used) ==")
    lit = [_eval_literature(ox)]
    _print(lit)

    print("\n== B. leave-one-cell-out fitted within Oxford ==")
    _print([_eval_loo(ox, LinearPooledFade, "linear pooled"),
            _eval_loo(ox, PowerLawFade, "stress power law")])

    print("\n== C. transfer: fit on all NASA cells, predict Oxford (no Oxford data) ==")
    for name, cls in (("linear pooled", LinearPooledFade), ("stress power law", PowerLawFade)):
        model = cls().fit(nasa)
        errs = []
        for c in ox:
            pred = model.predict_soh(c.cycles - c.cycles[0], c.c_rate)
            errs.append(np.sqrt(np.mean((pred - c.soh) ** 2)))
        print(f"  {name:18s} mean SoH RMSE {np.mean(errs) * 100:6.2f}%  "
              f"(per cell: {', '.join(f'{e * 100:.1f}' for e in errs)})")
    return 0


def _eval_literature(ox):
    errs = []
    for c in ox:
        m = LiteratureFade(temperature_C=c.temperature_C)
        pred = m.predict_soh(c.cycles - c.cycles[0], c.c_rate)
        errs.append(np.sqrt(np.mean((pred - c.soh) ** 2)))
    return ("literature (40 °C)", float(np.mean(errs)), errs)


def _eval_loo(ox, cls, name):
    scores = leave_one_out(ox, cls)
    errs = [s.soh_rmse for s in scores]
    return (name, float(np.mean(errs)), errs)


def _print(rows):
    for name, mean, errs in rows:
        print(f"  {name:22s} mean SoH RMSE {mean * 100:6.2f}%  "
              f"(per cell: {', '.join(f'{e * 100:.1f}' for e in errs)})")


if __name__ == "__main__":
    raise SystemExit(main())
