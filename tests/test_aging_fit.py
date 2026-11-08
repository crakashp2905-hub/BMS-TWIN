"""Tests: capacity-fade fitting, validated leave-one-cell-out on committed NASA data.

These run in CI on ``data/samples/nasa_capacity_fade.csv`` (a derived, attributed
slice of the public-domain NASA PCoE dataset), so no multi-MB download is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import bms
from bms.aging_fit import (
    FadeCurve,
    LinearPooledFade,
    LiteratureFade,
    PowerLawFade,
    leave_one_out,
    load_fade_csv,
)

_CSV = Path(__file__).resolve().parents[1] / "data" / "samples" / "nasa_capacity_fade.csv"


def _curves():
    return load_fade_csv(str(_CSV))


class TestCommittedDataset:
    def test_eight_cells_with_data_derived_stress_rates(self):
        curves = _curves()
        assert [c.cell for c in curves] == ["B0005", "B0006", "B0007", "B0018",
                                            "B0025", "B0026", "B0027", "B0028"]
        rates = {c.cell: c.c_rate for c in curves}
        # Rates come from the measured data, not from batch names: B0028 is a
        # 1 C test even though it sits in the 2 C-named batch.
        assert all(0.9 < rates[c] < 1.1 for c in ("B0005", "B0006", "B0007", "B0018", "B0028"))
        assert all(1.8 < rates[c] < 2.2 for c in ("B0025", "B0026", "B0027"))

    def test_capacity_dips_were_despiked(self):
        # B0026 has isolated measurement dips (≈1.39 Ah) that must not set its EOL.
        b26 = next(c for c in _curves() if c.cell == "B0026")
        assert b26.capacity_Ah.min() > 1.7
        assert np.isnan(b26.cycle_to_eol())            # 28 cycles: it never reaches EOL

    def test_eol_is_measured_relative_to_first_cycle(self):
        b05 = next(c for c in _curves() if c.cell == "B0005")
        assert 90.0 < b05.cycle_to_eol() < 110.0       # NASA B0005 reaches 80 % near cycle 100


class TestFadeModels:
    def test_power_law_is_monotone_in_cycles(self):
        m = PowerLawFade().fit(_curves())
        soh = m.predict_soh(np.arange(0, 200, 10.0), 1.0)
        assert np.all(np.diff(soh) <= 1e-9)
        assert np.all(np.isfinite(soh))

    def test_fitted_models_beat_the_literature_prior(self):
        curves = _curves()
        lit = np.mean([s.soh_rmse for s in leave_one_out(curves, LiteratureFade)])
        lin = np.mean([s.soh_rmse for s in leave_one_out(curves, LinearPooledFade)])
        # The shipped AgingModel defaults are far too gentle for this data.
        assert lit > 2.0 * lin

    def test_linear_model_predicts_one_c_end_of_life_closely(self):
        scores = {s.cell: s for s in leave_one_out(_curves(), LinearPooledFade)}
        for cell in ("B0005", "B0006", "B0007", "B0018"):
            assert abs(scores[cell].eol_error_cycles) < 60.0

    def test_literature_prior_predicts_end_of_life_far_too_late(self):
        scores = {s.cell: s for s in leave_one_out(_curves(), LiteratureFade)}
        assert scores["B0005"].eol_error_cycles > 500.0

    def test_prediction_intervals_are_overconfident_on_held_out_cells(self):
        # The honest real-data finding: a nominal 95 % interval from the residual
        # spread covers well under 95 % of held-out cells, so it needs widening.
        curves = _curves()
        y, mu, s = [], [], []
        for i, held in enumerate(curves):
            model = LinearPooledFade().fit([c for j, c in enumerate(curves) if j != i])
            n = held.cycles - held.cycles[0]
            y.append(held.soh)
            mu.append(model.predict_soh(n, held.c_rate))
            s.append(np.full(n.size, model.residual_sigma))
        y, mu, s = map(np.concatenate, (y, mu, s))
        assert bms.coverage(y, mu, s, 0.95) < 0.90
        assert bms.calibration_scale(y, mu, s, 0.95) > 1.3

    def test_fade_curve_soh_is_normalised(self):
        c = FadeCurve(cell="x", c_rate=1.0, temperature_C=24.0,
                      cycles=np.array([1.0, 2.0, 3.0]), capacity_Ah=np.array([2.0, 1.9, 1.8]))
        assert c.soh[0] == 1.0
        assert abs(c.soh[-1] - 0.9) < 1e-12


class TestNestedCalibration:
    def test_scale_learned_without_the_held_out_cell_restores_coverage(self):
        from bms.aging_fit import nested_interval_calibration
        res = nested_interval_calibration(_curves(), LinearPooledFade)
        assert res.raw_coverage < 0.90                    # the uncorrected interval is too tight
        assert 0.85 <= res.nested_coverage <= 1.0         # out-of-sample coverage recovers

    def test_one_scale_per_held_out_cell(self):
        from bms.aging_fit import nested_interval_calibration
        res = nested_interval_calibration(_curves(), LinearPooledFade)
        assert set(res.scales) == {c.cell for c in _curves()}
        assert all(1.0 < s < 4.0 for s in res.scales.values())
