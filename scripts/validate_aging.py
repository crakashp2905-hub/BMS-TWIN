"""Reproduce the multi-cell aging validation against the NASA PCoE dataset.

Eight cells (seven at ~1 C or ~2 C per the measured discharge current; B0028 is
~1 C despite its batch name) are
loaded from the public-domain NASA ``.mat`` files, despiked, and scored with
**leave-one-cell-out**: each cell is predicted by models fitted to the other seven.

    python scripts/validate_aging.py              # download (cached) + print tables
    python scripts/validate_aging.py path/to/dir  # use already-downloaded .mat files

Also rewrites ``data/samples/nasa_capacity_fade.csv`` (the committed derived dataset
that CI uses) when run with ``--write-csv``.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from bms.aging_fit import (  # noqa: E402
    LinearPooledFade,
    LiteratureFade,
    PowerLawFade,
    leave_one_out,
    load_fade_curve,
    save_fade_csv,
)

_ROOT = Path(__file__).resolve().parents[1]
_CACHE = _ROOT / ".nasa_cache"
_MIRROR = "https://raw.githubusercontent.com/natskiu/Nasa-Battery/main/Data/{}.mat"
CELLS = ("B0005", "B0006", "B0007", "B0018", "B0025", "B0026", "B0027", "B0028")


def fetch(cell: str, directory: Path) -> Path:
    directory.mkdir(exist_ok=True)
    dest = directory / f"{cell}.mat"
    if not dest.exists():
        print(f"  downloading {cell} …", flush=True)
        urllib.request.urlretrieve(_MIRROR.format(cell), dest)
    return dest


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    directory = Path(args[0]) if args else _CACHE
    paths = {c: fetch(c, directory) for c in CELLS}
    curves = [load_fade_curve(str(paths[c]), c) for c in CELLS]

    print("== Cells (C-rate from the 90th-percentile discharge current) ==")
    for c in curves:
        print(f"  {c.cell}: {c.c_rate:.2f} C, {len(c.cycles)} cycles, "
              f"SoH end {c.soh[-1]:.3f}, EOL at cycle {c.cycle_to_eol():.0f}, "
              f"dips repaired {c.n_repaired}")

    if "--write-csv" in sys.argv:
        save_fade_csv(curves, str(_ROOT / "data" / "samples" / "nasa_capacity_fade.csv"))
        print("  wrote data/samples/nasa_capacity_fade.csv")

    print("\n== Leave-one-cell-out SoH error ==")
    results = {}
    for name, factory in (("literature (AgingModel)", LiteratureFade),
                          ("linear pooled", LinearPooledFade),
                          ("stress power law", PowerLawFade)):
        scores = leave_one_out(curves, factory)
        results[name] = scores
        print(f"  {name:24s} mean SoH RMSE {np.mean([s.soh_rmse for s in scores]) * 100:5.2f}%")
        for s in scores:
            eol = "n/a" if not np.isfinite(s.eol_error_cycles) else f"{s.eol_error_cycles:+.0f}"
            print(f"      {s.cell} ({s.c_rate:.1f} C): RMSE {s.soh_rmse * 100:5.2f}%   EOL error {eol} cycles")

    pl = PowerLawFade().fit(curves)
    print(f"\n  power-law fit on all cells: alpha={pl.alpha:.2f} beta={pl.beta:.2f} b={pl.b:.2f}")
    print("  (beta < 0 means the 2 C batch fades slower — a batch confound, not a stress law)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
