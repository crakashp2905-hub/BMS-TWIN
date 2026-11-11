"""Tests: the telemetry schema — validation, serialization, streams and injected faults."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from bms.telemetry import (
    DEFAULT_EPOCH_UNIX_S,
    FIELD_NAMES,
    FIELDS,
    SCHEMA_VERSION,
    TelemetryRecord,
    TelemetryValidationError,
    check_sequence,
    format_timestamp,
    json_schema,
    parse_timestamp,
    read_csv,
    read_jsonl,
    write_csv,
    write_jsonl,
)


def _rec(seq=0, t=0.0, **kw) -> TelemetryRecord:
    base = dict(seq=seq, t_s=t, timestamp=format_timestamp(DEFAULT_EPOCH_UNIX_S + t), speed_mps=12.5, soc=0.8,
                voltage_V=352.4, current_A=41.2, temperature_C=27.5, power_W=14_500.0)
    base.update(kw)
    return TelemetryRecord(**base)


def _stream(n=20):
    return [_rec(seq=i, t=float(i)) for i in range(n)]


class TestSchemaTable:
    def test_core_channels_are_exactly_the_requested_ones(self):
        required = {f.name for f in FIELDS if f.required}
        assert required == {"seq", "t_s", "timestamp", "speed_mps", "soc", "voltage_V", "current_A",
                            "temperature_C", "power_W"}

    def test_dataclass_fields_and_table_agree(self):
        from dataclasses import fields
        assert tuple(f.name for f in fields(TelemetryRecord)) == FIELD_NAMES

    def test_json_schema_is_generated_from_the_table(self):
        s = json_schema()
        assert set(s["properties"]) == set(FIELD_NAMES) | {"schema_version"}
        assert set(s["required"]) == {f.name for f in FIELDS if f.required}
        assert s["additionalProperties"] is False
        assert s["properties"]["soc"]["minimum"] == 0.0 and s["properties"]["soc"]["maximum"] == 1.0
        assert s["properties"]["seq"]["type"] == "integer"
        assert s["properties"]["state"]["enum"]
        json.dumps(s)                                   # serializable


class TestTimestamps:
    def test_fixed_epoch_gives_reproducible_timestamps(self):
        assert format_timestamp(DEFAULT_EPOCH_UNIX_S) == "2026-01-01T00:00:00.000Z"
        assert format_timestamp(DEFAULT_EPOCH_UNIX_S + 1.5) == "2026-01-01T00:00:01.500Z"

    def test_round_trip_to_the_millisecond(self):
        for t in (0.0, 0.001, 59.999, 3600.5, 86_400.25):
            assert parse_timestamp(format_timestamp(DEFAULT_EPOCH_UNIX_S + t)) == pytest.approx(
                DEFAULT_EPOCH_UNIX_S + t, abs=5e-4)

    @pytest.mark.parametrize("bad", ["2026-01-01T00:00:00", "yesterday", "", "2026-13-01T00:00:00.000Z"])
    def test_bad_timestamps_are_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_timestamp(bad)
        assert any("timestamp" in p for p in _rec(timestamp=bad or "x").problems())


class TestValidation:
    def test_a_good_record_is_valid(self):
        assert _rec().problems() == []
        assert _rec().validate().speed_kmh == pytest.approx(45.0)

    @pytest.mark.parametrize("field, value", [
        ("speed_mps", -0.1), ("speed_mps", 500.0), ("soc", 1.01), ("soc", -0.01), ("voltage_V", -5.0),
        ("voltage_V", 5000.0), ("current_A", 9000.0), ("temperature_C", -300.0), ("temperature_C", 999.0),
        ("power_W", 1e9), ("t_s", -1.0), ("seq", -1),
    ])
    def test_out_of_range_values_are_named(self, field, value):
        problems = _rec(**{field: value}).problems()
        assert problems and field in problems[0]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_numbers_are_rejected_for_every_float_channel(self, value):
        for spec in FIELDS:
            if spec.kind == "float" and spec.required and spec.name != "t_s":
                assert any("not finite" in p for p in _rec(**{spec.name: value}).problems()), spec.name

    def test_missing_required_values_and_wrong_types(self):
        assert any("required" in p for p in _rec(soc=None).problems())
        assert any("number" in p for p in _rec(soc="0.5").problems())
        assert any("number" in p for p in _rec(soc=True).problems())          # bool is not a number here
        assert any("integer" in p for p in _rec(seq=1.5).problems())
        assert any("state" in p for p in _rec(state="exploding").problems())

    def test_validate_raises_with_every_problem(self):
        with pytest.raises(TelemetryValidationError) as err:
            _rec(soc=2.0, speed_mps=-1.0).validate()
        assert "soc" in str(err.value) and "speed_mps" in str(err.value)


class TestSerialization:
    def test_json_round_trip_is_lossless(self):
        r = _rec(accel_mps2=0.4, limit_flags=3, state="driving", odometer_m=1234.5)
        assert TelemetryRecord.from_json(r.to_json()) == r

    def test_json_is_compact_sorted_and_deterministic(self):
        r = _rec()
        text = r.to_json()
        assert text == r.to_json() and " " not in text
        keys = list(json.loads(text))
        assert keys == sorted(keys)
        assert json.loads(text)["schema_version"] == SCHEMA_VERSION

    def test_optional_channels_are_omitted_when_absent(self):
        assert set(json.loads(_rec().to_json())) == {f.name for f in FIELDS if f.required} | {"schema_version"}

    def test_non_finite_values_never_reach_the_wire(self):
        with pytest.raises(TelemetryValidationError):
            _rec(voltage_V=float("nan")).to_json()

    @pytest.mark.parametrize("payload, message", [
        ("not json", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        ('{"seq": 0}', "missing required"),
        (json.dumps({**json.loads(_rec().to_json()), "extra": 1}), "unknown field"),
        (json.dumps({**json.loads(_rec().to_json()), "schema_version": 99}), "schema_version"),
        (json.dumps({**json.loads(_rec().to_json()), "soc": 7}), "soc"),
    ])
    def test_hostile_or_broken_payloads_are_rejected(self, payload, message):
        with pytest.raises(TelemetryValidationError, match=message):
            TelemetryRecord.from_json(payload)

    def test_non_strict_parsing_ignores_unknown_keys(self):
        d = {**json.loads(_rec().to_json()), "extra": 1}
        assert TelemetryRecord.from_dict(d, strict=False) == _rec()

    def test_files_round_trip(self, tmp_path):
        recs = [_rec(seq=i, t=float(i), limit_flags=(i % 4) or None, state="driving" if i % 2 else None)
                for i in range(10)]
        assert write_jsonl(recs, tmp_path / "a.jsonl") == 10
        assert read_jsonl(tmp_path / "a.jsonl") == recs
        assert write_csv(recs, tmp_path / "a.csv") == 10
        assert read_csv(tmp_path / "a.csv") == recs

    def test_a_corrupt_line_is_reported_with_its_number(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text(_rec().to_json() + "\n" + '{"seq": 1}\n')
        with pytest.raises(TelemetryValidationError, match=r"bad.jsonl:2"):
            read_jsonl(p)


class TestFaultInjectedStreams:
    """A consumer must be able to tell that a stream was damaged, and how."""

    def test_clean_stream_is_ok(self):
        rep = check_sequence(_stream())
        assert rep.ok and rep.n_records == 20 and rep.max_dt_s == 1.0

    def test_dropped_records_show_up_as_gaps(self):
        s = [r for r in _stream() if r.seq not in (5, 6, 12)]
        rep = check_sequence(s)
        assert rep.gaps == ((4, 7), (11, 13)) and not rep.ok
        assert rep.max_dt_s == 3.0

    def test_duplicates_are_reported(self):
        s = _stream()
        s.insert(8, s[7])
        rep = check_sequence(s)
        assert rep.duplicates == (7,) and not rep.ok

    def test_reordering_is_reported(self):
        s = _stream()
        s[4], s[5] = s[5], s[4]
        rep = check_sequence(s)
        assert rep.out_of_order == 1 and rep.time_reversals == 1

    def test_clock_going_backwards_is_reported_even_with_good_sequence_numbers(self):
        s = _stream()
        s[10] = replace(s[10], t_s=3.0, timestamp=format_timestamp(DEFAULT_EPOCH_UNIX_S + 3.0))
        rep = check_sequence(s)
        assert rep.time_reversals >= 1 and not rep.ok

    def test_a_stream_with_a_corrupted_value_is_caught_by_validation(self):
        s = _stream()
        s[9] = replace(s[9], current_A=math.nan)
        bad = [r.seq for r in s if r.problems()]
        assert bad == [9]
        s[9] = replace(s[9], current_A=1e7)
        assert [r.seq for r in s if r.problems()] == [9]

    def test_empty_stream_is_trivially_ok(self):
        assert check_sequence([]).ok
