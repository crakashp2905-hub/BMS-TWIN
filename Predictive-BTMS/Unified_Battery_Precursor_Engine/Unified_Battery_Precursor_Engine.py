#!/usr/bin/env python3
"""
Unified Battery Precursor Evaluation Engine
Version: 0.1.0-foundation

Source-independent Python reference implementation.
Standard-library only. No PyBaMM dependency.

IMPORTANT:
- This is an input/validation/orchestration foundation, not a validated diagnostic product.
- It does not infer micro-shorts, plating, SEI growth, electrolyte decomposition,
  self-discharge, or contact resistance from ordinary V/I/T/SOC/SOH alone.
- A model returns INSUFFICIENT_DATA unless its required, physically meaningful
  evidence is supplied.
- Heat is reported only when the relevant physical power is identifiable.
- No cooling actuation or safety-critical control is implemented.

Input:
  python Unified_Battery_Precursor_Engine.py --input battery_packet.json
  python Unified_Battery_Precursor_Engine.py --example
  python Unified_Battery_Precursor_Engine.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import unittest
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VERSION = "0.1.0-foundation"


class Status(str, Enum):
    EVALUATED = "EVALUATED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"


class Severity(str, Enum):
    NOT_ASSESSED = "NOT_ASSESSED"
    INFORMATIONAL = "INFORMATIONAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"


@dataclass
class Signal:
    value: float
    unit: str
    timestamp_s: float
    source: str
    quality: str = "GOOD"
    sigma: Optional[float] = None

    @staticmethod
    def from_dict(raw: Dict[str, Any], name: str) -> "Signal":
        required = ("value", "unit", "timestamp_s", "source")
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f"{name}: missing fields {missing}")
        value = float(raw["value"])
        timestamp = float(raw["timestamp_s"])
        sigma = None if raw.get("sigma") is None else float(raw["sigma"])
        if not math.isfinite(value) or not math.isfinite(timestamp):
            raise ValueError(f"{name}: value and timestamp must be finite")
        if sigma is not None and (not math.isfinite(sigma) or sigma < 0):
            raise ValueError(f"{name}: sigma must be finite and >= 0")
        quality = str(raw.get("quality", "GOOD")).upper()
        if quality not in {"GOOD", "SUSPECT", "BAD", "STALE"}:
            raise ValueError(f"{name}: unsupported quality '{quality}'")
        return Signal(value, str(raw["unit"]), timestamp, str(raw["source"]),
                      quality, sigma)


@dataclass
class HeatContribution:
    mechanism: str
    owner: str
    power_w: Optional[float]
    energy_j: Optional[float]
    basis: str
    status: str


class HeatLedger:
    """Prevents duplicate physical ownership; does not infer missing heat."""

    def __init__(self) -> None:
        self._items: List[HeatContribution] = []
        self._owners: set[str] = set()

    def add(self, item: HeatContribution) -> None:
        if item.owner in self._owners:
            raise ValueError(f"Duplicate heat owner '{item.owner}'")
        if item.power_w is not None and (
            not math.isfinite(item.power_w) or item.power_w < 0
        ):
            raise ValueError("Heat power must be finite and non-negative")
        if item.energy_j is not None and (
            not math.isfinite(item.energy_j) or item.energy_j < 0
        ):
            raise ValueError("Heat energy must be finite and non-negative")
        self._owners.add(item.owner)
        self._items.append(item)

    def report(self) -> Dict[str, Any]:
        known_power = [x.power_w for x in self._items if x.power_w is not None]
        known_energy = [x.energy_j for x in self._items if x.energy_j is not None]
        return {
            "contributions": [asdict(x) for x in self._items],
            "sum_of_identified_power_w": sum(known_power),
            "sum_of_identified_energy_j": sum(known_energy),
            "warning": (
                "Sums include only identified contributions; they are not total "
                "cell heat unless all physical heat owners are represented."
            ),
        }


@dataclass
class ModelResult:
    model_id: str
    status: Status
    severity: Severity
    identifiability: str
    findings: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    calculated: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)


class BatteryPacket:
    """Canonical, source-independent measurement/state packet."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        self.raw = raw
        self.cell_id = str(raw.get("cell_id", "UNSPECIFIED"))
        self.chemistry = str(raw.get("chemistry", "UNSPECIFIED")).upper()
        self.signals: Dict[str, Signal] = {}
        for name, item in raw.get("signals", {}).items():
            if not isinstance(item, dict):
                raise ValueError(f"signals.{name} must be an object")
            self.signals[name] = Signal.from_dict(item, name)
        self.parameters = raw.get("parameters", {})
        self.context = raw.get("context", {})
        self._validate_common()

    def _validate_common(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("cell_id cannot be empty")
        if not isinstance(self.parameters, dict) or not isinstance(self.context, dict):
            raise ValueError("parameters and context must be JSON objects")
        timestamps = [s.timestamp_s for s in self.signals.values()]
        if timestamps and max(timestamps) - min(timestamps) > float(
            self.context.get("max_signal_time_skew_s", 5.0)
        ):
            raise ValueError("Signal timestamps exceed configured time-skew limit")

    def get(self, name: str, unit: str, max_age_s: Optional[float] = None) -> Optional[Signal]:
        s = self.signals.get(name)
        if s is None or s.quality != "GOOD" or s.unit != unit:
            return None
        now = self.context.get("evaluation_timestamp_s")
        if max_age_s is not None and now is not None:
            age = float(now) - s.timestamp_s
            if age < 0 or age > max_age_s:
                return None
        return s


class PrecursorModel:
    model_id = "BASE"

    def evaluate(self, packet: BatteryPacket) -> ModelResult:
        raise NotImplementedError


class InternalMicroShortModel(PrecursorModel):
    """
    Direct-path calculation only.

    Required evidence:
      isc_path_voltage_v: voltage measured across the identified path
      isc_path_current_a: current through that same path
    Optional:
      isc_observation_duration_s
      isc_effective_volume_m3
    These measurements identify path electrical behaviour, not necessarily
    the physical origin as an internal battery micro-short.
    """
    model_id = "INTERNAL_MICRO_SHORT"

    def evaluate(self, p: BatteryPacket) -> ModelResult:
        v = p.get("isc_path_voltage_v", "V")
        i = p.get("isc_path_current_a", "A")
        if v is None or i is None:
            return ModelResult(
                self.model_id, Status.INSUFFICIENT_DATA, Severity.NOT_ASSESSED,
                "NOT_IDENTIFIABLE",
                required_evidence=[
                    "isc_path_voltage_v: measured across the identified conductive path",
                    "isc_path_current_a: measured through the same path",
                ],
                limitations=[
                    "Ordinary cell terminal voltage/current/SOC/SOH are not sufficient "
                    "to uniquely identify a local micro-short."
                ],
            )
        if v.value <= 0 or i.value <= 0:
            return ModelResult(
                self.model_id, Status.INVALID_INPUT, Severity.NOT_ASSESSED,
                "NOT_IDENTIFIABLE",
                findings=["Direct path voltage and current must both be positive."],
            )
        resistance = v.value / i.value
        power = v.value * i.value
        duration = p.get("isc_observation_duration_s", "s")
        volume = p.get("isc_effective_volume_m3", "m^3")
        energy = None
        heat_density = None
        if duration is not None:
            if duration.value <= 0:
                return ModelResult(self.model_id, Status.INVALID_INPUT,
                                   Severity.NOT_ASSESSED, "IDENTIFIABLE",
                                   findings=["Observation duration must be positive."])
            energy = power * duration.value
        if volume is not None:
            if volume.value <= 0:
                return ModelResult(self.model_id, Status.INVALID_INPUT,
                                   Severity.NOT_ASSESSED, "IDENTIFIABLE",
                                   findings=["Effective path volume must be positive."])
            heat_density = power / volume.value
        return ModelResult(
            self.model_id, Status.EVALUATED, Severity.INFORMATIONAL,
            "PATH_ELECTRICAL_BEHAVIOUR_IDENTIFIABLE; PHYSICAL_ORIGIN_UNCONFIRMED",
            findings=["A dissipative conductive path was directly characterised."],
            calculated={
                "path_resistance_ohm": resistance,
                "path_power_w": power,
                "path_energy_j": energy,
                "path_heat_density_w_m3": heat_density,
            },
            limitations=[
                "This does not prove that the path is an internal micro-short.",
                "Heat density is not cell-average heat or local hotspot temperature.",
                "No short resistance/current/heat is inferred from ordinary BMS signals.",
            ],
        )


class EvidenceGateModel(PrecursorModel):
    """Honest scaffold for mechanisms that need specialised evidence/models."""

    def __init__(self, model_id: str, evidence: List[str], limitations: List[str]):
        self.model_id = model_id
        self.evidence = evidence
        self.limitations = limitations

    def evaluate(self, packet: BatteryPacket) -> ModelResult:
        return ModelResult(
            self.model_id, Status.INSUFFICIENT_DATA, Severity.NOT_ASSESSED,
            "NOT_IDENTIFIABLE",
            findings=[
                "Mechanism-specific physics implementation is not yet enabled. "
                "The engine will not substitute generic voltage or temperature "
                "thresholds for a validated mechanism model."
            ],
            required_evidence=self.evidence,
            limitations=self.limitations,
        )


def build_models() -> List[PrecursorModel]:
    return [
        InternalMicroShortModel(),
        EvidenceGateModel(
            "LITHIUM_PLATING",
            [
                "Negative-electrode potential versus Li/Li+ or a validated estimator",
                "Cell temperature and charge current history",
                "Chemistry-, SOC-, temperature- and rate-valid plating model parameters",
                "Independent validation evidence for plating onset/magnitude",
            ],
            ["Terminal voltage alone cannot uniquely quantify plated lithium."],
        ),
        EvidenceGateModel(
            "SEI_GROWTH",
            [
                "Validated degradation/side-reaction model states or diagnostic data",
                "Time, temperature, SOC and cycling history",
                "Chemistry-specific kinetic and transport parameters",
                "Capacity-loss/impedance evidence with uncertainty",
            ],
            ["SEI growth cannot be uniquely inferred from one V/I/T packet."],
        ),
        EvidenceGateModel(
            "ELECTROLYTE_DECOMPOSITION",
            [
                "Mechanism-specific electrochemical/chemical diagnostic evidence",
                "Chemistry-specific reaction kinetics and parameterisation",
                "Temperature, SOC and operating history",
            ],
            ["Generic voltage/temperature excursions do not identify decomposition."],
        ),
        EvidenceGateModel(
            "ABNORMAL_SELF_DISCHARGE",
            [
                "Isolated rest-period current or independently measured charge loss",
                "Voltage relaxation/hysteresis reference",
                "Independent SOC reference or capacity/charge-balance evidence",
                "Temperature stability and sufficient observation duration",
            ],
            [
                "Must distinguish true self-discharge from voltage relaxation, "
                "SOC estimator drift, internal leakage and redox shuttle effects."
            ],
        ),
        EvidenceGateModel(
            "ABNORMAL_CONTACT_RESISTANCE",
            [
                "Four-wire/Kelvin voltage drop across the identified contact",
                "Current through that same contact",
                "Contact temperature and validated healthy baseline",
            ],
            ["Cell terminal voltage alone cannot isolate contact resistance."],
        ),
    ]


def evaluate_packet(raw: Dict[str, Any]) -> Dict[str, Any]:
    packet = BatteryPacket(raw)
    results: List[ModelResult] = []
    ledger = HeatLedger()

    for model in build_models():
        result = model.evaluate(packet)
        results.append(result)
        if result.model_id == "INTERNAL_MICRO_SHORT":
            power = result.calculated.get("path_power_w")
            energy = result.calculated.get("path_energy_j")
            ledger.add(HeatContribution(
                mechanism="INTERNAL_MICRO_SHORT",
                owner="directly_observed_conductive_path",
                power_w=power,
                energy_j=energy,
                basis="V_path * I_path; only if both refer to the same identified path",
                status="IDENTIFIED_PATH_DISSIPATION" if power is not None else "UNAVAILABLE",
            ))

    return {
        "engine": "UNIFIED_BATTERY_PRECURSOR_EVALUATION_ENGINE",
        "version": VERSION,
        "cell_id": packet.cell_id,
        "chemistry": packet.chemistry,
        "evaluation_status": "COMPLETED_WITH_EVIDENCE_GATES",
        "models": [
            {
                **asdict(r),
                "status": r.status.value,
                "severity": r.severity.value,
            }
            for r in results
        ],
        "heat_ledger": ledger.report(),
        "system_limitations": [
            "Research/reference software only; not safety-certified firmware.",
            "No model claims experimental validation from software unit tests.",
            "No missing measurement is silently replaced with zero.",
            "Unimplemented mechanism models explicitly return INSUFFICIENT_DATA.",
            "Use chemistry- and cell-specific validated parameters before quantitative deployment.",
        ],
    }


def example_packet() -> Dict[str, Any]:
    # Deliberately ordinary BMS-like signals: these do NOT identify a micro-short.
    t = 1794250000.0
    def sig(value: float, unit: str) -> Dict[str, Any]:
        return {
            "value": value, "unit": unit, "timestamp_s": t,
            "source": "illustrative_example", "quality": "GOOD"
        }
    return {
        "cell_id": "LFP_EXAMPLE_001",
        "chemistry": "LFP_GRAPHITE",
        "context": {"evaluation_timestamp_s": t},
        "parameters": {},
        "signals": {
            "cell_voltage_v": sig(3.31, "V"),
            "cell_current_a": sig(0.0, "A"),
            "cell_temperature_k": sig(298.15, "K"),
            "soc_percent": sig(65.0, "%"),
            "soh_percent": sig(98.5, "%"),
        },
    }


class EngineTests(unittest.TestCase):
    def test_standard_packet_does_not_claim_microshort(self):
        out = evaluate_packet(example_packet())
        m = next(x for x in out["models"] if x["model_id"] == "INTERNAL_MICRO_SHORT")
        self.assertEqual(m["status"], Status.INSUFFICIENT_DATA.value)

    def test_direct_path_calculation(self):
        p = example_packet()
        t = 1794250000.0
        for name, value, unit in [
            ("isc_path_voltage_v", 0.2, "V"),
            ("isc_path_current_a", 0.01, "A"),
            ("isc_observation_duration_s", 10.0, "s"),
        ]:
            p["signals"][name] = {
                "value": value, "unit": unit, "timestamp_s": t,
                "source": "calibrated_path_instrument", "quality": "GOOD"
            }
        out = evaluate_packet(p)
        m = next(x for x in out["models"] if x["model_id"] == "INTERNAL_MICRO_SHORT")
        self.assertAlmostEqual(m["calculated"]["path_resistance_ohm"], 20.0)
        self.assertAlmostEqual(m["calculated"]["path_power_w"], 0.002)
        self.assertAlmostEqual(m["calculated"]["path_energy_j"], 0.02)

    def test_bad_signal_quality_is_not_used(self):
        p = example_packet()
        t = 1794250000.0
        p["signals"]["isc_path_voltage_v"] = {
            "value": 0.2, "unit": "V", "timestamp_s": t,
            "source": "instrument", "quality": "BAD"
        }
        p["signals"]["isc_path_current_a"] = {
            "value": 0.01, "unit": "A", "timestamp_s": t,
            "source": "instrument", "quality": "GOOD"
        }
        out = evaluate_packet(p)
        m = next(x for x in out["models"] if x["model_id"] == "INTERNAL_MICRO_SHORT")
        self.assertEqual(m["status"], Status.INSUFFICIENT_DATA.value)

    def test_unit_mismatch_rejected_as_missing_evidence(self):
        p = example_packet()
        t = 1794250000.0
        p["signals"]["isc_path_voltage_v"] = {
            "value": 0.2, "unit": "mV", "timestamp_s": t,
            "source": "instrument", "quality": "GOOD"
        }
        p["signals"]["isc_path_current_a"] = {
            "value": 0.01, "unit": "A", "timestamp_s": t,
            "source": "instrument", "quality": "GOOD"
        }
        out = evaluate_packet(p)
        m = next(x for x in out["models"] if x["model_id"] == "INTERNAL_MICRO_SHORT")
        self.assertEqual(m["status"], Status.INSUFFICIENT_DATA.value)

    def test_nonfinite_signal_rejected(self):
        p = example_packet()
        p["signals"]["cell_voltage_v"]["value"] = float("nan")
        with self.assertRaises(ValueError):
            evaluate_packet(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Battery Precursor Evaluation Engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Path to canonical battery packet JSON")
    group.add_argument("--example", action="store_true", help="Run illustrative BMS-like packet")
    group.add_argument("--self-test", action="store_true", help="Run built-in unit tests")
    args = parser.parse_args()

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(EngineTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    try:
        if args.example:
            raw = example_packet()
        else:
            with args.input.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        output = evaluate_packet(raw)
        print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
