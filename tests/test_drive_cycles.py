"""Tests: standardized and custom drive cycles (loading, validation, statistics)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import bms
from bms.drive_cycles import (
    DriveCycle,
    DriveCycleError,
    available_cycles,
    concatenate,
    constant_speed,
    get_cycle,
    load_cycle_csv,
    load_epa_cycle,
    register_cycle,
)

_CYCLE_DIR = Path(bms.__file__).resolve().parent / "cycles"

# SHA-256 of the bundled files exactly as the EPA publishes them (also listed in cycles/README.md).
_SHA256 = {
    "uddscol.txt": "cc9f3edf2f0626976a6552d9f5c88867e1805ed3f321df9ec943a3e0353687b6",
    "hwycol.txt": "e48e278a40312dfe84d31b5f448b601e7046c72487a6a20d3db1d4f85fb3e10a",
    "us06col.txt": "84e9d8cc16fcd4ee3c30c3880f333eb25b61352e56f24e453411a7245c725ad2",
    "ececol.txt": "4c04507a838e4774694daa7f03811a861ab361d2267acee3c31677f4668a2e8f",
    "eudccol.txt": "dc557fbfcebeb33889822864bd2937377ffea2295270c5485de5729488bec43a",
}


class TestBundledCycles:
    @pytest.mark.parametrize("fname", sorted(_SHA256))
    def test_files_are_byte_identical_to_the_published_ones(self, fname):
        digest = hashlib.sha256((_CYCLE_DIR / fname).read_bytes()).hexdigest()
        assert digest == _SHA256[fname]

    def test_readme_lists_the_same_checksums(self):
        readme = (_CYCLE_DIR / "README.md").read_text(encoding="utf-8")
        for fname, digest in _SHA256.items():
            assert fname in readme and digest in readme

    # Duration, distance and peak speed published for these schedules.
    @pytest.mark.parametrize("name, duration_s, distance_km, vmax_kmh, stops", [
        ("udds", 1369, 11.99, 91.25, 17),
        ("hwfet", 765, 16.51, 96.40, None),
        ("us06", 600, 12.89, 129.23, None),
        ("ece15", 195, 0.994, 50.0, None),
        ("eudc", 400, 6.955, 120.0, None),
    ])
    def test_statistics_match_the_published_values(self, name, duration_s, distance_km, vmax_kmh, stops):
        c = get_cycle(name)
        s = c.stats()
        assert c.duration_s == duration_s
        assert s.distance_km == pytest.approx(distance_km, abs=0.01)
        assert s.max_speed_kmh == pytest.approx(vmax_kmh, abs=0.05)
        if stops is not None:
            assert s.n_stops == stops
        assert c.is_uniform and c.dt_s == 1.0

    def test_nedc_is_four_ece_parts_then_one_eudc(self):
        nedc, ece, eudc = get_cycle("nedc"), get_cycle("ece15"), get_cycle("eudc")
        assert nedc.duration_s == 4 * ece.duration_s + eudc.duration_s
        assert nedc.distance_m == pytest.approx(4 * ece.distance_m + eudc.distance_m, rel=1e-9)
        assert nedc.speed_mps[0] == 0.0 and nedc.speed_mps[-1] == 0.0

    def test_epa_loader_reads_the_unit_from_the_header(self):
        mph = load_epa_cycle(_CYCLE_DIR / "uddscol.txt")       # "mph" header
        kph = load_epa_cycle(_CYCLE_DIR / "ececol.txt")        # "kph" header
        assert mph.speed_kmh.max() == pytest.approx(56.7 * 1.609344, rel=1e-9)
        assert kph.speed_kmh.max() == pytest.approx(50.0, rel=1e-12)

    def test_epa_loader_rejects_a_file_without_a_unit(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("TITLE\tx\nTest Time, secs Target Speed\n0\t0\n1\t1\n2\t2\n")
        with pytest.raises(DriveCycleError, match="unit"):
            load_epa_cycle(p)

    def test_epa_loader_names_the_bad_line(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("TITLE\tx\nTest Time, secs Target Speed, mph\n0\t0\n1\tabc\n2\t2\n")
        with pytest.raises(DriveCycleError, match=r":4:"):
            load_epa_cycle(p)


class TestSyntheticAndRegistry:
    def test_every_registered_cycle_loads_and_is_valid(self):
        for name in available_cycles():
            c = get_cycle(name)
            assert len(c) >= 2 and c.duration_s > 0 and np.all(c.speed_mps >= 0)

    def test_lookup_is_case_insensitive_and_unknown_names_fail_clearly(self):
        assert get_cycle("UDDS").name == "udds"
        with pytest.raises(DriveCycleError, match="unknown cycle"):
            get_cycle("no_such_cycle")

    def test_get_cycle_can_resample(self):
        c = get_cycle("udds", dt_s=0.5)
        assert c.dt_s == 0.5 and c.duration_s == 1369

    def test_register_custom_cycle(self):
        c = constant_speed(50.0, 30.0, name="mine")
        register_cycle("my_test_cycle_xyz", c)
        assert get_cycle("my_test_cycle_xyz").duration_s == 30.0
        with pytest.raises(DriveCycleError, match="already registered"):
            register_cycle("my_test_cycle_xyz", c)
        register_cycle("my_test_cycle_xyz", constant_speed(60.0, 10.0), overwrite=True)
        assert get_cycle("my_test_cycle_xyz").duration_s == 10.0

    def test_builtin_names_cannot_be_shadowed(self):
        with pytest.raises(DriveCycleError, match="built-in"):
            register_cycle("udds", constant_speed(50.0, 10.0))

    def test_cached_cycles_are_immutable(self):
        c = get_cycle("udds")
        with pytest.raises(ValueError):
            c.speed_mps[0] = 99.0
        with pytest.raises(ValueError):
            c.t_s[0] = 1.0


class TestCustomProfiles:
    def test_csv_round_trip_in_every_speed_unit(self, tmp_path):
        t = np.arange(0.0, 61.0)
        v_kmh = np.concatenate([np.linspace(0, 54, 31), np.linspace(54, 0, 30)])
        import pandas as pd
        for unit, factor in (("kmh", 1.0), ("mps", 1 / 3.6), ("mph", 1 / 1.609344)):
            p = tmp_path / f"{unit}.csv"
            pd.DataFrame({"time_s": t, f"speed_{unit}": v_kmh * factor}).to_csv(p, index=False)
            c = load_cycle_csv(p, speed_unit=unit)
            assert np.allclose(c.speed_kmh, v_kmh, atol=1e-9)

    def test_to_csv_then_load_is_lossless(self, tmp_path):
        c = get_cycle("urban_stop_go").with_grade(0.02)
        c.to_csv(tmp_path / "c.csv")
        back = load_cycle_csv(tmp_path / "c.csv", grade_col="grade_rad")
        assert np.allclose(back.speed_mps, c.speed_mps) and np.allclose(back.grade_rad, 0.02)
        assert back.fingerprint() == c.fingerprint()

    @pytest.mark.parametrize("unit, raw, expected", [("rad", 0.05, 0.05), ("deg", 3.0, np.deg2rad(3.0)),
                                                    ("percent", 5.0, np.arctan(0.05))])
    def test_grade_units(self, tmp_path, unit, raw, expected):
        import pandas as pd
        pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "speed_kmh": [0.0, 10.0, 0.0],
                      "g": [raw] * 3}).to_csv(tmp_path / "g.csv", index=False)
        c = load_cycle_csv(tmp_path / "g.csv", grade_col="g", grade_unit=unit)
        assert c.grade_rad[0] == pytest.approx(expected)

    def test_missing_column_names_what_was_found(self, tmp_path):
        (tmp_path / "x.csv").write_text("t,v\n0,0\n1,1\n")
        with pytest.raises(DriveCycleError, match="missing column"):
            load_cycle_csv(tmp_path / "x.csv")

    def test_unreadable_file_is_a_drive_cycle_error(self, tmp_path):
        with pytest.raises(DriveCycleError, match="cannot read"):
            load_cycle_csv(tmp_path / "does_not_exist.csv")

    @pytest.mark.parametrize("t, v, message", [
        ([0.0, 1.0, 2.0], [0.0, float("nan"), 1.0], "finite"),
        ([0.0, 1.0, 2.0], [0.0, float("inf"), 1.0], "finite"),
        ([0.0, 2.0, 1.0], [0.0, 1.0, 2.0], "strictly increasing"),
        ([0.0, 1.0, 1.0], [0.0, 1.0, 2.0], "strictly increasing"),
        ([0.0, 1.0, 2.0], [0.0, -1.0, 2.0], "non-negative"),
        ([0.0], [0.0], "at least two"),
        ([0.0, 1.0], [0.0], "same length"),
    ])
    def test_malformed_traces_are_rejected(self, t, v, message):
        with pytest.raises(DriveCycleError, match=message):
            DriveCycle("bad", np.array(t), np.array(v))

    def test_a_unit_mistake_is_caught_by_the_plausibility_check(self, tmp_path):
        import pandas as pd
        # km/h values read as m/s imply 100 m/s → rejected; so does a 50 m/s² launch.
        pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "speed_mps": [0.0, 120.0, 0.0]}).to_csv(tmp_path / "a.csv", index=False)
        with pytest.raises(DriveCycleError, match="peak speed|acceleration"):
            load_cycle_csv(tmp_path / "a.csv", speed_unit="mps")
        pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "speed_mps": [0.0, 30.0, 0.0]}).to_csv(tmp_path / "b.csv", index=False)
        with pytest.raises(DriveCycleError, match="acceleration"):
            load_cycle_csv(tmp_path / "b.csv", speed_unit="mps")

    def test_grade_that_looks_like_percent_is_rejected(self):
        with pytest.raises(DriveCycleError, match="percent"):
            DriveCycle("g", np.array([0.0, 1.0]), np.array([0.0, 1.0]), grade_rad=np.array([5.0, 5.0]))


class TestTransformations:
    def test_resample_preserves_distance_and_duration(self):
        c = get_cycle("udds")
        r = c.resample(0.25)
        assert r.duration_s == c.duration_s and r.dt_s == 0.25
        assert r.distance_m == pytest.approx(c.distance_m, rel=1e-6)

    def test_repeat_and_concatenate_share_the_joint_sample(self):
        c = get_cycle("ece15")
        r = c.repeat(3)
        assert r.duration_s == 3 * c.duration_s
        assert r.distance_m == pytest.approx(3 * c.distance_m, rel=1e-9)

    def test_concatenating_across_a_speed_jump_needs_an_explicit_gap(self):
        a, b = constant_speed(50.0, 10.0), constant_speed(80.0, 10.0)
        with pytest.raises(DriveCycleError, match="discontinuity"):
            concatenate([a, b])
        joined = concatenate([a, b], gap_s=5.0)
        assert joined.duration_s == pytest.approx(25.0)

    def test_slice_rezeroes_time(self):
        s = get_cycle("udds").slice(100.0, 200.0)
        assert s.t_s[0] == 0.0 and s.duration_s == 100.0

    def test_with_grade_scalar_and_array(self):
        c = get_cycle("highway")
        assert np.all(c.with_grade(0.03).grade_rad == 0.03)
        g = np.linspace(0.0, 0.05, len(c))
        assert np.allclose(c.with_grade(g).grade_rad, g)

    def test_grade_follows_resampling_and_concatenation(self):
        c = get_cycle("ece15").with_grade(0.01)
        assert np.allclose(c.resample(0.5).grade_rad, 0.01)
        assert np.allclose(concatenate([c, get_cycle("ece15")]).grade_rad[:len(c)], 0.01)

    def test_fingerprint_is_stable_and_sensitive(self):
        a = get_cycle("udds")
        assert a.fingerprint() == get_cycle("udds").fingerprint()
        assert a.fingerprint() != get_cycle("hwfet").fingerprint()
        assert a.fingerprint() != a.with_grade(0.001).fingerprint()

    def test_statistics_of_a_known_trace(self):
        # 0 → 36 km/h in 10 s, cruise 10 s, back to rest in 10 s, idle 10 s.
        t = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        c = DriveCycle("known", t, np.array([0.0, 10.0, 10.0, 0.0, 0.0]))
        s = c.stats()
        assert s.distance_km == pytest.approx((50.0 + 100.0 + 50.0) / 1000.0)
        assert s.max_accel_mps2 == pytest.approx(1.0) and s.max_decel_mps2 == pytest.approx(1.0)
        assert s.idle_fraction == pytest.approx(0.25) and s.n_stops == 1
