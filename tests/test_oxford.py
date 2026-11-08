"""Tests: the Oxford cross-dataset aging validation.

The Oxford data file is licensed under the ODbL and is not committed. These tests
run only when it is present (set BMS_OXFORD_MAT, or place it under .nasa_cache/oxford).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from bms.aging_fit import (
    LinearPooledFade,
    PowerLawFade,
    leave_one_out,
    load_oxford_curves,
)

_ROOT = Path(__file__).resolve().parents[1]
_MAT = os.environ.get("BMS_OXFORD_MAT") or str(
    _ROOT / ".nasa_cache" / "oxford" / "Oxford_Battery_Degradation_Dataset_1.mat")

pytestmark = pytest.mark.skipif(not Path(_MAT).exists(),
                                reason="Oxford data file not present (ODbL, not committed)")


@pytest.fixture(scope="module")
def oxford():
    return load_oxford_curves(_MAT)


def test_eight_cells_loaded_in_ah(oxford):
    assert len(oxford) == 8
    assert all(0.6 < c.capacity_Ah[0] < 0.8 for c in oxford)     # a 740 mAh pouch cell
    assert all(c.temperature_C == 40.0 for c in oxford)


def test_fade_is_monotone_on_average(oxford):
    for c in oxford:
        assert c.soh[-1] < 0.9                                  # every cell fades substantially


def test_within_dataset_fits_generalise_across_cells(oxford):
    linear = np.mean([s.soh_rmse for s in leave_one_out(oxford, LinearPooledFade)])
    power = np.mean([s.soh_rmse for s in leave_one_out(oxford, PowerLawFade)])
    assert linear < 0.05 and power < 0.05                       # held-out cells within 5 %
    assert power < linear                                       # on Oxford the power law wins


def test_transfer_from_nasa_fails_without_covariates(oxford):
    # Documents a negative result: a model fitted on NASA cells cannot predict Oxford
    # cells (50x the cycle count, 16 degC hotter). Covariates are needed.
    from bms.aging_fit import load_fade_curve
    nasa_dir = _ROOT / ".nasa_cache"
    if not all((nasa_dir / f"{c}.mat").exists() for c in ("B0005", "B0006")):
        pytest.skip("NASA .mat files not cached for the transfer comparison")
    nasa = [load_fade_curve(str(nasa_dir / f"{c}.mat"), c)
            for c in ("B0005", "B0006", "B0007", "B0018", "B0025", "B0026", "B0027", "B0028")]
    model = LinearPooledFade().fit(nasa)
    err = np.mean([np.sqrt(np.mean((model.predict_soh(c.cycles - c.cycles[0], c.c_rate)
                                    - c.soh) ** 2)) for c in oxford])
    assert err > 0.5
