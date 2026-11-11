"""Telemetry schema: one timestamped record of vehicle and pack state.

A single field table (:data:`FIELDS`) defines every channel's name, unit, range and
whether it is required.  Validation, the JSON Schema, the CSV header and the
documentation all come from that table, so they cannot disagree.

Core channels (required): ``seq``, ``t_s``, ``timestamp``, ``speed_mps``, ``soc``,
``voltage_V``, ``current_A``, ``temperature_C``, ``power_W``.  Extended channels are
optional and omitted from the JSON when absent.

Conventions
-----------
* ``current_A`` and ``power_W`` are **positive on discharge**, negative on regeneration.
* ``soc`` is a fraction in [0, 1] (pack mean); ``speed_mps`` is never negative.
* ``timestamp`` is ISO-8601 UTC with millisecond precision, ``epoch + t_s``.  The epoch
  is explicit, so a run is reproducible; pass a real clock time to anchor it.
* Records are immutable.  Non-finite numbers are rejected rather than serialized, because
  JSON has no NaN and a silently dropped value would look like a healthy sensor.

Scope: this is a *schema for simulated or logged data*.  Nothing here asserts that any
record came from a physical vehicle.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = 1
DEFAULT_EPOCH_UNIX_S = 1_767_225_600.0        # 2026-01-01T00:00:00Z, a fixed epoch for reproducible runs
_UTC = timezone.utc
_EPOCH0 = datetime(1970, 1, 1, tzinfo=_UTC)


class TelemetryValidationError(ValueError):
    """A telemetry record or stream is malformed or out of range."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    unit: str
    description: str
    kind: str                 # "int" | "float" | "str"
    required: bool = False
    lo: float | None = None
    hi: float | None = None


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("seq", "1", "Record number, contiguous from 0", "int", True, 0, 2 ** 53),
    FieldSpec("t_s", "s", "Seconds since the start of the run", "float", True, 0.0, 1e9),
    FieldSpec("timestamp", "ISO-8601 UTC", "Wall-clock time of the sample (epoch + t_s)", "str", True),
    FieldSpec("speed_mps", "m/s", "Vehicle speed", "float", True, 0.0, 120.0),
    FieldSpec("soc", "1", "Pack state of charge (mean), 0..1", "float", True, 0.0, 1.0),
    FieldSpec("voltage_V", "V", "Pack terminal voltage", "float", True, 0.0, 1500.0),
    FieldSpec("current_A", "A", "Pack current, positive = discharge", "float", True, -5000.0, 5000.0),
    FieldSpec("temperature_C", "degC", "Pack temperature (mean)", "float", True, -60.0, 200.0),
    FieldSpec("power_W", "W", "Pack terminal power, positive = discharge", "float", True, -2e6, 2e6),
    FieldSpec("accel_mps2", "m/s^2", "Vehicle acceleration", "float", False, -30.0, 30.0),
    FieldSpec("speed_target_mps", "m/s", "Drive-cycle target speed", "float", False, 0.0, 120.0),
    FieldSpec("odometer_m", "m", "Distance travelled since the start", "float", False, 0.0, 1e9),
    FieldSpec("energy_Wh", "Wh", "Net energy drawn from the pack since the start", "float", False, -1e7, 1e7),
    FieldSpec("motor_torque_Nm", "N*m", "Motor shaft torque, negative = regeneration", "float", False, -5000.0, 5000.0),
    FieldSpec("cell_v_min_V", "V", "Lowest series-group voltage", "float", False, 0.0, 10.0),
    FieldSpec("cell_v_max_V", "V", "Highest series-group voltage", "float", False, 0.0, 10.0),
    FieldSpec("soc_min", "1", "Lowest series-group SOC", "float", False, 0.0, 1.0),
    FieldSpec("soc_max", "1", "Highest series-group SOC", "float", False, 0.0, 1.0),
    FieldSpec("temp_min_C", "degC", "Coldest series group", "float", False, -60.0, 200.0),
    FieldSpec("temp_max_C", "degC", "Hottest series group", "float", False, -60.0, 200.0),
    FieldSpec("limit_flags", "bitmask", "Active power limits (see bms.ev_drive.describe_flags)", "int",
              False, 0, 65535),
    FieldSpec("state", "enum", "Vehicle state", "str", False),
)
FIELD_NAMES = tuple(f.name for f in FIELDS)
_SPEC = {f.name: f for f in FIELDS}
STATES = ("idle", "driving", "fault")


@dataclass(frozen=True)
class TelemetryRecord:
    """One timestamped sample.  Optional channels are ``None`` when not measured."""

    seq: int
    t_s: float
    timestamp: str
    speed_mps: float
    soc: float
    voltage_V: float
    current_A: float
    temperature_C: float
    power_W: float
    accel_mps2: float | None = None
    speed_target_mps: float | None = None
    odometer_m: float | None = None
    energy_Wh: float | None = None
    motor_torque_Nm: float | None = None
    cell_v_min_V: float | None = None
    cell_v_max_V: float | None = None
    soc_min: float | None = None
    soc_max: float | None = None
    temp_min_C: float | None = None
    temp_max_C: float | None = None
    limit_flags: int | None = None
    state: str | None = None

    schema_version = SCHEMA_VERSION      # class constant, not a dataclass field

    # ── Validation ─────────────────────────────────────────────────────
    def problems(self) -> list[str]:
        """Every way this record violates the schema (empty if valid)."""
        out: list[str] = []
        for spec in FIELDS:
            value = getattr(self, spec.name)
            if value is None:
                if spec.required:
                    out.append(f"{spec.name}: required")
                continue
            if spec.kind == "str":
                if not isinstance(value, str) or not value:
                    out.append(f"{spec.name}: must be a non-empty string")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                out.append(f"{spec.name}: must be a number, got {type(value).__name__}")
                continue
            if spec.kind == "int" and (not isinstance(value, int) or isinstance(value, bool)):
                out.append(f"{spec.name}: must be an integer")
                continue
            if not math.isfinite(value):
                out.append(f"{spec.name}: not finite ({value})")
                continue
            if spec.lo is not None and value < spec.lo:
                out.append(f"{spec.name}: {value} below minimum {spec.lo} {spec.unit}")
            if spec.hi is not None and value > spec.hi:
                out.append(f"{spec.name}: {value} above maximum {spec.hi} {spec.unit}")
        if isinstance(self.timestamp, str) and self.timestamp:
            try:
                parse_timestamp(self.timestamp)
            except ValueError:
                out.append("timestamp: not an ISO-8601 UTC time")
        if self.state is not None and self.state not in STATES:
            out.append(f"state: {self.state!r} not one of {STATES}")
        return out

    def validate(self) -> "TelemetryRecord":
        issues = self.problems()
        if issues:
            raise TelemetryValidationError("; ".join(issues))
        return self

    # ── (De)serialization ──────────────────────────────────────────────
    def to_dict(self, *, include_none: bool = False) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        return d if include_none else {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict, *, strict: bool = True) -> "TelemetryRecord":
        """Build and validate a record.  ``strict`` rejects unknown keys."""
        if not isinstance(data, dict):
            raise TelemetryValidationError("a record must be a JSON object")
        unknown = set(data) - set(FIELD_NAMES) - {"schema_version"}
        if unknown and strict:
            raise TelemetryValidationError(f"unknown field(s): {sorted(unknown)}")
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise TelemetryValidationError(f"unsupported schema_version {version!r} (expected {SCHEMA_VERSION})")
        kwargs = {k: v for k, v in data.items() if k in FIELD_NAMES}
        missing = [s.name for s in FIELDS if s.required and s.name not in kwargs]
        if missing:
            raise TelemetryValidationError(f"missing required field(s): {missing}")
        return cls(**kwargs).validate()

    def to_json(self) -> str:
        """Compact, key-sorted JSON.  Raises on non-finite numbers instead of emitting invalid JSON."""
        self.validate()
        return json.dumps({"schema_version": SCHEMA_VERSION, **self.to_dict()}, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, text: str | bytes) -> "TelemetryRecord":
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise TelemetryValidationError(f"not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    @property
    def speed_kmh(self) -> float:
        return self.speed_mps * 3.6


# ── Timestamps ─────────────────────────────────────────────────────────
def format_timestamp(unix_s: float) -> str:
    """ISO-8601 UTC with millisecond precision, e.g. ``2026-01-01T00:00:01.500Z``."""
    dt = _EPOCH0 + timedelta(milliseconds=round(float(unix_s) * 1000.0))
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_timestamp(text: str) -> float:
    """Inverse of :func:`format_timestamp`; returns Unix seconds."""
    if not isinstance(text, str) or not text.endswith("Z"):
        raise ValueError("timestamp must be ISO-8601 UTC ending in 'Z'")
    dt = datetime.fromisoformat(text[:-1] + "+00:00")
    return (dt - _EPOCH0).total_seconds()


# ── Streams ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SequenceReport:
    """Continuity of a record stream."""

    n_records: int
    gaps: tuple[tuple[int, int], ...]          # (last seq seen, next seq) where records are missing
    duplicates: tuple[int, ...]
    out_of_order: int
    time_reversals: int
    max_dt_s: float

    @property
    def ok(self) -> bool:
        return not (self.gaps or self.duplicates or self.out_of_order or self.time_reversals)


def check_sequence(records: Iterable[TelemetryRecord]) -> SequenceReport:
    """Report gaps, duplicates, reordering and time reversals in a record stream."""
    gaps: list[tuple[int, int]] = []
    dups: list[int] = []
    seen: set[int] = set()
    n = out_of_order = reversals = 0
    last_seq: int | None = None
    last_t: float | None = None
    max_dt = 0.0
    for r in records:
        n += 1
        if r.seq in seen:
            dups.append(r.seq)
        seen.add(r.seq)
        if last_seq is not None:
            if r.seq < last_seq:
                out_of_order += 1
            elif r.seq > last_seq + 1:
                gaps.append((last_seq, r.seq))
        if last_t is not None:
            if r.t_s < last_t:
                reversals += 1
            else:
                max_dt = max(max_dt, r.t_s - last_t)
        last_seq = r.seq if last_seq is None else max(last_seq, r.seq)
        last_t = r.t_s
    return SequenceReport(n, tuple(gaps), tuple(dups), out_of_order, reversals, max_dt)


# ── Files ──────────────────────────────────────────────────────────────
def write_jsonl(records: Iterable[TelemetryRecord], path: str | Path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(r.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[TelemetryRecord]:
    out: list[TelemetryRecord] = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    out.append(TelemetryRecord.from_json(line))
                except TelemetryValidationError as exc:
                    raise TelemetryValidationError(f"{path}:{number}: {exc}") from exc
    return out


def _csv_cell(value) -> str:
    if value is None:
        return ""
    return repr(value) if isinstance(value, float) else str(value)     # repr round-trips floats exactly


def write_csv(records: Iterable[TelemetryRecord], path: str | Path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(FIELD_NAMES)
        for r in records:
            r.validate()
            w.writerow([_csv_cell(getattr(r, c)) for c in FIELD_NAMES])
            n += 1
    return n


def read_csv(path: str | Path) -> list[TelemetryRecord]:
    out: list[TelemetryRecord] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for number, row in enumerate(csv.DictReader(fh), start=2):
            data: dict = {}
            for name, text in row.items():
                if name not in _SPEC or text in ("", None):
                    continue
                kind = _SPEC[name].kind
                try:
                    data[name] = int(text) if kind == "int" else float(text) if kind == "float" else text
                except ValueError as exc:
                    raise TelemetryValidationError(f"{path}:{number}: {name}: {exc}") from exc
            try:
                out.append(TelemetryRecord.from_dict(data))
            except TelemetryValidationError as exc:
                raise TelemetryValidationError(f"{path}:{number}: {exc}") from exc
    return out


def iter_records(source: Iterable[TelemetryRecord]) -> Iterator[TelemetryRecord]:
    """Validate records as they pass through (raises on the first bad one)."""
    for r in source:
        yield r.validate()


# ── Schema export ──────────────────────────────────────────────────────
def json_schema() -> dict:
    """JSON Schema (draft 2020-12) of one record, generated from :data:`FIELDS`."""
    props: dict = {"schema_version": {"const": SCHEMA_VERSION}}
    for f in FIELDS:
        p: dict = {"type": {"int": "integer", "float": "number", "str": "string"}[f.kind],
                   "description": f"{f.description} [{f.unit}]"}
        if f.lo is not None:
            p["minimum"] = f.lo
        if f.hi is not None:
            p["maximum"] = f.hi
        if f.name == "state":
            p["enum"] = list(STATES)
        props[f.name] = p
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "BMS digital-twin telemetry record", "type": "object",
            "properties": props, "required": [f.name for f in FIELDS if f.required],
            "additionalProperties": False}
