"""Tests: synthetic ground-truth benchmark (reconstruct known hidden state)."""

from __future__ import annotations

import numpy as np

import bms


class TestGroundTruthCell:
    def test_telemetry_shapes_and_ranges(self):
        rng = np.random.default_rng(0)
        from bms.benchmark import _random_cell
        cell = _random_cell(rng)
        i, v, T, soc = cell.telemetry(n=600, seed=0)
        assert len(i) == len(v) == len(T) == len(soc) == 600
        assert np.all((soc >= 0) & (soc <= 1))
        assert v.min() > 1.0 and v.max() < 5.0


class TestSyntheticBenchmark:
    def test_current_bias_makes_coulomb_drift(self):
        # With a biased current sensor, coulomb-counting is no longer exact on
        # synthetic data — the benchmark is a real test, not truth-by-construction.
        res = bms.run_synthetic_benchmark(n=6, method="coulomb", seed=2)
        assert res.n_cells == 6
        # Net throughput on this profile is small, so the drift is modest but
        # measurable (~0.1 %) and bounded — asserted as a band, not a magic number.
        assert 0.0005 < res.soc_rmse_mean < 0.02
        assert res.soc_rmse_p90 >= res.soc_rmse_mean - 1e-9

    def test_generic_ekf_reconstructs_soc(self):
        res = bms.run_synthetic_benchmark(n=6, method="ekf_generic", seed=2)
        assert np.isfinite(res.soc_rmse_mean) and res.soc_rmse_mean >= 0

    def test_autocal_identifies_chemistry_above_chance(self):
        res = bms.run_synthetic_benchmark(n=4, method="autocal", seed=3)
        # 3 candidate chemistries → chance is 0.33; auto-cal should clearly beat it.
        assert res.chemistry_id_accuracy > 0.5
        assert np.isfinite(res.r0_error_mean) and res.r0_error_mean < 0.3
        assert res.soc_rmse_mean < 0.2

    def test_result_serialises(self):
        res = bms.run_synthetic_benchmark(n=4, method="coulomb", seed=0)
        d = res.to_dict()
        for k in ("method", "soc_rmse_mean", "soc_rmse_p90", "chemistry_id_accuracy"):
            assert k in d
