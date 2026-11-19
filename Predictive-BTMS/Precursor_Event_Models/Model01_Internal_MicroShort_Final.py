#!/usr/bin/env python3
"""
MODEL 01 — INTERNAL MICRO-SHORT / ABNORMAL INTERNAL-LEAKAGE ASSESSMENT
Version: 1.0.0 (consolidated single-file reference implementation)

Purpose
-------
1) Direct path characterisation when voltage across and current through the
   SAME identified conductive path are independently measured.
2) Indirect rest-period assessment of excess charge loss using independently
   obtained start/end SOC and measured usable capacity, corrected by a
   matched healthy-cell baseline self-discharge rate.
3) Conditional equivalent electrical dissipation and energy accounting.
4) Strict input validation, uncertainty propagation, evidence status and JSON
   reporting.

Scientific boundary
-------------------
The indirect branch can quantify excess self-discharge / internal-leakage
equivalent current only if its independent SOC/capacity/baseline assumptions
are satisfied. It cannot uniquely prove that the cause is a local micro-short.
Voltage relaxation is reported as corroborating data only; it is NOT converted
into leakage current. A local hotspot cannot be predicted by this lumped model.

This is a research/reference model, not safety-certified firmware and not an
experimentally validated diagnostic for any particular cell.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_ID = "INTERNAL_MICRO_SHORT"
MODEL_VERSION = "1.0.0-final-reference"


class Status(str, Enum):
    EVALUATED = "EVALUATED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"


def finite_number(obj: Dict[str, Any], key: str, where: str,
                  required: bool = True) -> Optional[float]:
    if key not in obj or obj[key] is None:
        if required:
            raise ValueError(f"{where}.{key} is required")
        return None
    try:
        value = float(obj[key])
    except (TypeError, ValueError):
        raise ValueError(f"{where}.{key} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{where}.{key} must be finite")
    return value


def positive(value: float, name: str, allow_zero: bool = False) -> None:
    if (value < 0 if allow_zero else value <= 0):
        op = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {op}")


def uncertainty_pair(raw: Dict[str, Any], value_key: str,
                     sigma_key: str, label: str) -> Tuple[float, float]:
    value = finite_number(raw, value_key, label)
    sigma = finite_number(raw, sigma_key, label, required=False)
    if sigma is None:
        raise ValueError(f"{label}.{sigma_key} is required for uncertainty-aware result")
    positive(sigma, f"{label}.{sigma_key}", allow_zero=True)
    return value, sigma


def normalised_status(value: str, label: str) -> None:
    if str(value).upper() not in {"GOOD", "SUSPECT", "BAD"}:
        raise ValueError(f"{label} quality must be GOOD, SUSPECT or BAD")
    if str(value).upper() != "GOOD":
        raise ValueError(f"{label} quality is {value}; only GOOD data are accepted")


@dataclass
class Assessment:
    model_id: str
    model_version: str
    status: str
    conclusion: str
    identifiability: str
    findings: List[str]
    calculated: Dict[str, Any]
    uncertainty: Dict[str, Any]
    heat: Dict[str, Any]
    required_evidence: List[str]
    limitations: List[str]


def unavailable(reason: str, required: List[str]) -> Assessment:
    return Assessment(
        MODEL_ID, MODEL_VERSION, Status.INSUFFICIENT_DATA.value,
        "NO_MICRO_SHORT_DIAGNOSIS", "NOT_IDENTIFIABLE",
        [reason], {}, {}, {"status": "UNAVAILABLE"},
        required,
        [
            "Ordinary terminal voltage/current/temperature/SOC/SOH alone do not "
            "uniquely identify a local internal micro-short.",
            "No missing quantity is silently replaced with zero.",
        ],
    )


def direct_path_assessment(raw: Dict[str, Any]) -> Assessment:
    """Characterise a directly instrumented path; physical origin remains open."""
    required = ["path_voltage_v", "path_current_a"]
    missing = [k for k in required if raw.get(k) is None]
    if missing:
        return unavailable(
            "Direct conductive-path measurements were not supplied.",
            [
                "path_voltage_v: voltage across the identified path",
                "path_current_a: current through that same path",
            ],
        )

    normalised_status(raw.get("quality", "GOOD"), "direct_path")
    v, sv = uncertainty_pair(raw, "path_voltage_v", "path_voltage_sigma_v",
                             "direct_path")
    i, si = uncertainty_pair(raw, "path_current_a", "path_current_sigma_a",
                             "direct_path")
    positive(v, "path_voltage_v")
    positive(i, "path_current_a")
    positive(sv, "path_voltage_sigma_v", allow_zero=True)
    positive(si, "path_current_sigma_a", allow_zero=True)

    r = v / i
    power = v * i
    # First-order independent uncertainty propagation.
    sr = r * math.sqrt((sv / v) ** 2 + (si / i) ** 2)
    sp = power * math.sqrt((sv / v) ** 2 + (si / i) ** 2)

    duration = finite_number(raw, "duration_s", "direct_path", required=False)
    energy = None
    senergy = None
    if duration is not None:
        positive(duration, "duration_s")
        energy = power * duration
        senergy = sp * duration

    volume = finite_number(raw, "effective_path_volume_m3",
                           "direct_path", required=False)
    heat_density = None
    if volume is not None:
        positive(volume, "effective_path_volume_m3")
        heat_density = power / volume

    return Assessment(
        MODEL_ID, MODEL_VERSION, Status.EVALUATED.value,
        "DIRECTLY_OBSERVED_DISSIPATIVE_PATH; INTERNAL_MICRO_SHORT_ORIGIN_NOT_CONFIRMED",
        "PATH_VOLTAGE_CURRENT_AND_EQUIVALENT_DISSIPATION_IDENTIFIABLE",
        [
            "Voltage and current were supplied for the same identified physical path.",
            "The measured path is dissipative; its physical origin is not established.",
        ],
        {
            "path_resistance_ohm": r,
            "path_power_w": power,
            "observation_duration_s": duration,
            "path_energy_j": energy,
            "effective_path_volume_m3": volume,
            "path_heat_density_w_m3": heat_density,
        },
        {
            "path_resistance_standard_uncertainty_ohm": sr,
            "path_power_standard_uncertainty_w": sp,
            "path_energy_standard_uncertainty_j": senergy,
            "propagation": "First-order, independent voltage/current uncertainties.",
        },
        {
            "status": "IDENTIFIED_PATH_DISSIPATION",
            "power_w": power,
            "energy_j": energy,
            "heat_density_w_m3": heat_density,
            "interpretation": (
                "Path dissipation only. Heat density uses supplied effective path "
                "volume and is not cell-average heat or a hotspot prediction."
            ),
        },
        [],
        [
            "A measured conductive path is not, by itself, proof of an internal battery micro-short.",
            "The path voltage must be measured across, and path current through, the same physical path.",
            "Local hotspot temperature requires a spatial electrical-thermal model and geometry.",
        ],
    )


def indirect_rest_assessment(raw: Dict[str, Any]) -> Assessment:
    """
    Estimate excess charge-loss current over a rest window.

    SOC endpoints must be independent of the current integration used for this
    assessment (e.g. independent reference measurement). Capacity must be a
    measured/validated usable capacity. A matched healthy baseline leakage rate
    is mandatory; otherwise only total observed charge loss can be reported,
    not abnormal excess leakage.
    """
    required = [
        "samples[].timestamp_s",
        "samples[].soc_independent_percent and soc_sigma_percent",
        "capacity_ah and capacity_sigma_ah",
        "healthy_baseline_self_discharge_a and its standard uncertainty",
        "minimum_rest_duration_s",
        "maximum_abs_external_current_a",
        "maximum_temperature_span_k",
    ]
    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        return unavailable("At least two rest-window samples are required.", required)

    # Validate configuration explicitly; no hidden defaults.
    min_duration = finite_number(raw, "minimum_rest_duration_s", "rest_window")
    max_current = finite_number(raw, "maximum_abs_external_current_a", "rest_window")
    max_temp_span = finite_number(raw, "maximum_temperature_span_k", "rest_window")
    positive(min_duration, "minimum_rest_duration_s")
    positive(max_current, "maximum_abs_external_current_a", allow_zero=True)
    positive(max_temp_span, "maximum_temperature_span_k", allow_zero=True)

    cap, scap = uncertainty_pair(raw, "capacity_ah", "capacity_sigma_ah",
                                 "rest_window")
    baseline, sbaseline = uncertainty_pair(
        raw, "healthy_baseline_self_discharge_a",
        "healthy_baseline_self_discharge_sigma_a", "rest_window"
    )
    positive(cap, "capacity_ah")
    positive(scap, "capacity_sigma_ah", allow_zero=True)
    positive(baseline, "healthy_baseline_self_discharge_a", allow_zero=True)

    parsed = []
    for idx, row in enumerate(samples):
        if not isinstance(row, dict):
            raise ValueError(f"samples[{idx}] must be an object")
        t = finite_number(row, "timestamp_s", f"samples[{idx}]")
        soc, ssoc = uncertainty_pair(row, "soc_independent_percent",
                                     "soc_sigma_percent", f"samples[{idx}]")
        current = finite_number(row, "external_current_a", f"samples[{idx}]")
        temp = finite_number(row, "temperature_k", f"samples[{idx}]")
        voltage = finite_number(row, "cell_voltage_v", f"samples[{idx}]",
                                required=False)
        normalised_status(row.get("quality", "GOOD"), f"samples[{idx}]")
        if not 0 <= soc <= 100:
            raise ValueError(f"samples[{idx}].soc_independent_percent must be 0..100")
        positive(ssoc, f"samples[{idx}].soc_sigma_percent", allow_zero=True)
        positive(temp, f"samples[{idx}].temperature_k")
        parsed.append((t, soc, ssoc, current, temp, voltage))

    parsed.sort(key=lambda x: x[0])
    times = [x[0] for x in parsed]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Rest-window timestamps must be unique and increasing")
    duration = times[-1] - times[0]
    if duration < min_duration:
        return Assessment(
            MODEL_ID, MODEL_VERSION, Status.OUT_OF_DOMAIN.value,
            "REST_WINDOW_TOO_SHORT_FOR_CONFIGURED_ASSESSMENT",
            "NOT_IDENTIFIABLE", [f"Observed rest duration {duration:.3f} s is below "
                                 f"configured minimum {min_duration:.3f} s."],
            {"duration_s": duration}, {},
            {"status": "UNAVAILABLE"}, required,
            ["The minimum duration must be selected from the test protocol and expected signal-to-noise ratio."],
        )

    max_abs_i = max(abs(x[3]) for x in parsed)
    if max_abs_i > max_current:
        return Assessment(
            MODEL_ID, MODEL_VERSION, Status.OUT_OF_DOMAIN.value,
            "REST_CONDITION_NOT_MET", "NOT_IDENTIFIABLE",
            [f"External current {max_abs_i:.6g} A exceeds configured rest limit "
             f"{max_current:.6g} A."],
            {"maximum_abs_external_current_a": max_abs_i}, {},
            {"status": "UNAVAILABLE"}, required,
            ["A rest-window leakage assessment requires a verified isolation condition."],
        )

    temps = [x[4] for x in parsed]
    temp_span = max(temps) - min(temps)
    if temp_span > max_temp_span:
        return Assessment(
            MODEL_ID, MODEL_VERSION, Status.OUT_OF_DOMAIN.value,
            "TEMPERATURE_STABILITY_REQUIREMENT_NOT_MET", "NOT_IDENTIFIABLE",
            [f"Temperature span {temp_span:.6g} K exceeds configured limit "
             f"{max_temp_span:.6g} K."],
            {"temperature_span_k": temp_span}, {},
            {"status": "UNAVAILABLE"}, required,
            ["Temperature dependence of self-discharge must be controlled or explicitly modelled."],
        )

    t0, soc0, ssoc0 = parsed[0][0], parsed[0][1], parsed[0][2]
    t1, soc1, ssoc1 = parsed[-1][0], parsed[-1][1], parsed[-1][2]
    delta_soc = soc0 - soc1
    if delta_soc < 0:
        return Assessment(
            MODEL_ID, MODEL_VERSION, Status.NOT_IDENTIFIABLE.value,
            "SOC_INCREASE_DURING_REST_REQUIRES_INVESTIGATION", "NOT_IDENTIFIABLE",
            ["Independent SOC increased over the rest window; simple charge-loss model is not applicable."],
            {"soc_change_percentage_points": -delta_soc}, {},
            {"status": "UNAVAILABLE"}, required,
            ["Voltage relaxation, estimator bias, temperature effects or measurement errors may explain the apparent SOC change."],
        )

    hours = duration / 3600.0
    observed_loss_ah = cap * delta_soc / 100.0
    # Propagate independent uncertainty in capacity and the two SOC endpoints.
    sdelta_soc = math.sqrt(ssoc0 ** 2 + ssoc1 ** 2)
    sloss_ah = math.sqrt(
        ((delta_soc / 100.0) * scap) ** 2 +
        ((cap / 100.0) * sdelta_soc) ** 2
    )
    observed_current = observed_loss_ah / hours
    sobserved_current = sloss_ah / hours

    baseline_loss_ah = baseline * hours
    excess_current = observed_current - baseline
    sexcess = math.sqrt(sobserved_current ** 2 + sbaseline ** 2)
    excess_loss_ah = observed_loss_ah - baseline_loss_ah

    # Negative estimate is not clipped into a positive fault current.
    if excess_current <= 0:
        status = Status.EVALUATED.value
        conclusion = "NO_EXCESS_CHARGE_LOSS_RESOLVED_ABOVE_HEALTHY_BASELINE"
        ident = "EXCESS_LEAKAGE_NOT_RESOLVED"
        severity = "NOT_ELEVATED"
        heat_power = 0.0
        findings = [
            "Observed charge-loss rate does not exceed the supplied healthy baseline.",
            "This does not prove the cell is free of a micro-short; the test may lack sensitivity.",
        ]
    elif excess_current <= 1.96 * sexcess:
        status = Status.NOT_IDENTIFIABLE.value
        conclusion = "EXCESS_CHARGE_LOSS_NOT_RESOLVED_AT_APPROX_95_PERCENT"
        ident = "EXCESS_LEAKAGE_MAGNITUDE_NOT_RESOLVED"
        severity = "NOT_ASSESSED"
        heat_power = None
        findings = [
            "Point estimate exceeds baseline, but its approximate 95% uncertainty interval includes zero.",
            "Do not report a positive abnormal leakage current or heat estimate.",
        ]
    else:
        status = Status.EVALUATED.value
        conclusion = "EXCESS_SELF_DISCHARGE_OR_INTERNAL_LEAKAGE_CANDIDATE"
        ident = "EXCESS_CHARGE_LOSS_RATE_IDENTIFIABLE; PHYSICAL_MECHANISM_NOT_UNIQUE"
        severity = "ELEVATED_CANDIDATE"
        heat_power = None  # Set only if cell voltage supplied and terminal-shunt assumption accepted.
        findings = [
            "Excess charge-loss rate is resolved above the supplied healthy baseline under the stated assumptions.",
            "This is not unique evidence of a local micro-short; other parasitic reactions or leakage paths remain possible.",
        ]

    # Voltage relaxation is descriptive only; never used to derive leakage current.
    voltage_values = [x[5] for x in parsed if x[5] is not None]
    voltage_delta = (
        voltage_values[-1] - voltage_values[0] if len(voltage_values) >= 2 else None
    )
    mean_voltage = statistics.mean(voltage_values) if voltage_values else None
    equivalent_power = None
    equivalent_energy = None
    if status == Status.EVALUATED.value and excess_current > 0 and mean_voltage is not None:
        # This is an equivalent terminal-shunt power only, not local defect heat.
        equivalent_power = mean_voltage * excess_current
        equivalent_energy = equivalent_power * duration
        heat_power = equivalent_power

    return Assessment(
        MODEL_ID, MODEL_VERSION, status, conclusion, ident, findings,
        {
            "rest_duration_s": duration,
            "start_soc_percent": soc0,
            "end_soc_percent": soc1,
            "soc_drop_percentage_points": delta_soc,
            "observed_charge_loss_ah": observed_loss_ah,
            "healthy_baseline_charge_loss_ah": baseline_loss_ah,
            "excess_charge_loss_ah": excess_loss_ah,
            "observed_mean_loss_current_a": observed_current,
            "healthy_baseline_self_discharge_a": baseline,
            "excess_loss_current_point_estimate_a": excess_current,
            "maximum_abs_external_current_a": max_abs_i,
            "temperature_span_k": temp_span,
            "mean_cell_voltage_v": mean_voltage,
            "voltage_change_v_descriptive_only": voltage_delta,
        },
        {
            "observed_charge_loss_standard_uncertainty_ah": sloss_ah,
            "observed_mean_loss_current_standard_uncertainty_a": sobserved_current,
            "excess_loss_current_standard_uncertainty_a": sexcess,
            "approx_95_percent_excess_current_interval_a": [
                excess_current - 1.96 * sexcess,
                excess_current + 1.96 * sexcess,
            ],
            "propagation": "First-order independent uncertainty propagation; approximate 95% interval.",
        },
        {
            "status": (
                "EQUIVALENT_TERMINAL_SHUNT_DISSIPATION_ONLY"
                if equivalent_power is not None else "UNAVAILABLE_OR_NOT_RESOLVED"
            ),
            "equivalent_power_w": equivalent_power,
            "equivalent_energy_j": equivalent_energy,
            "assumption": (
                "P_eq = mean terminal voltage * excess leakage-equivalent current. "
                "This is a lumped equivalent energy estimate, not proof that all "
                "energy is deposited at a local micro-short site."
            ),
        },
        required,
        [
            "SOC endpoints must be independent of the current integration being assessed; otherwise the result is circular.",
            "Capacity must represent validated usable capacity at the test condition.",
            "Healthy baseline self-discharge must be measured on a matched cell/protocol and corrected for temperature and age.",
            "The excess charge-loss rate is not uniquely attributable to a micro-short.",
            "Voltage change is descriptive only; relaxation is not converted into leakage current.",
            "Equivalent terminal-shunt power is not local hotspot heat or cell temperature rise.",
            "Approximate uncertainty assumes independent input errors and does not include unmodelled systematic bias.",
        ],
    )


def evaluate(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Input root must be a JSON object")
    results: Dict[str, Any] = {}
    direct = raw.get("direct_path")
    rest = raw.get("rest_window")
    if direct is not None:
        results["direct_path_assessment"] = asdict(direct_path_assessment(direct))
    else:
        results["direct_path_assessment"] = asdict(unavailable(
            "No direct-path measurement block supplied.",
            ["direct_path.path_voltage_v", "direct_path.path_current_a",
             "direct_path.path_voltage_sigma_v", "direct_path.path_current_sigma_a"],
        ))
    if rest is not None:
        results["indirect_rest_assessment"] = asdict(indirect_rest_assessment(rest))
    else:
        results["indirect_rest_assessment"] = asdict(unavailable(
            "No indirect rest-window block supplied.",
            ["rest_window.samples", "independent SOC endpoints", "validated capacity",
             "matched healthy baseline self-discharge rate", "rest protocol limits"],
        ))
    return {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "overall_status": "COMPLETED",
        "assessments": results,
        "global_limitations": [
            "This model assesses directly observed path dissipation and/or excess rest-period charge loss.",
            "Neither route alone uniquely confirms a local internal micro-short.",
            "No diagnosis is made from ordinary BMS voltage/current/temperature/SOC/SOH alone.",
            "No local hotspot temperature is predicted.",
            "Research/reference implementation; not safety-certified or experimentally validated for a specific cell.",
        ],
    }


def example_input() -> Dict[str, Any]:
    # Synthetic example ONLY to demonstrate the data contract and uncertainty logic.
    t0 = 0.0
    dt = 7 * 24 * 3600.0
    samples = []
    for k in range(8):
        frac = k / 7
        samples.append({
            "timestamp_s": t0 + frac * dt,
            "soc_independent_percent": 70.0 - frac * 0.12,
            "soc_sigma_percent": 0.01,
            "external_current_a": 0.0,
            "temperature_k": 298.15,
            "cell_voltage_v": 3.30 - frac * 0.004,
            "quality": "GOOD",
        })
    return {
        "direct_path": None,
        "rest_window": {
            "samples": samples,
            "capacity_ah": 2.3,
            "capacity_sigma_ah": 0.02,
            "healthy_baseline_self_discharge_a": 0.000005,
            "healthy_baseline_self_discharge_sigma_a": 0.000001,
            "minimum_rest_duration_s": 604800,
            "maximum_abs_external_current_a": 0.001,
            "maximum_temperature_span_k": 0.5,
        },
    }


class Model01Tests(unittest.TestCase):
    def test_direct_path(self):
        x = {
            "path_voltage_v": 0.2, "path_voltage_sigma_v": 0.001,
            "path_current_a": 0.01, "path_current_sigma_a": 0.0001,
            "duration_s": 10.0, "quality": "GOOD",
        }
        r = direct_path_assessment(x)
        self.assertEqual(r.status, Status.EVALUATED.value)
        self.assertAlmostEqual(r.calculated["path_resistance_ohm"], 20.0)
        self.assertAlmostEqual(r.calculated["path_power_w"], 0.002)
        self.assertAlmostEqual(r.calculated["path_energy_j"], 0.02)

    def test_rest_candidate(self):
        x = example_input()["rest_window"]
        r = indirect_rest_assessment(x)
        self.assertEqual(r.status, Status.EVALUATED.value)
        self.assertIn("CANDIDATE", r.conclusion)

    def test_missing_direct_path_is_not_diagnosed(self):
        r = direct_path_assessment({})
        self.assertEqual(r.status, Status.INSUFFICIENT_DATA.value)

    def test_reject_bad_quality(self):
        x = {
            "path_voltage_v": 0.2, "path_voltage_sigma_v": 0.001,
            "path_current_a": 0.01, "path_current_sigma_a": 0.0001,
            "quality": "BAD",
        }
        with self.assertRaises(ValueError):
            direct_path_assessment(x)

    def test_reject_zero_path_current(self):
        x = {
            "path_voltage_v": 0.2, "path_voltage_sigma_v": 0.001,
            "path_current_a": 0.0, "path_current_sigma_a": 0.0001,
        }
        with self.assertRaises(ValueError):
            direct_path_assessment(x)

    def test_reject_duplicate_times(self):
        x = example_input()["rest_window"]
        x["samples"][1]["timestamp_s"] = x["samples"][0]["timestamp_s"]
        with self.assertRaises(ValueError):
            indirect_rest_assessment(x)

    def test_reject_non_rest_current(self):
        x = example_input()["rest_window"]
        x["samples"][3]["external_current_a"] = 0.1
        r = indirect_rest_assessment(x)
        self.assertEqual(r.status, Status.OUT_OF_DOMAIN.value)

    def test_reject_temperature_span(self):
        x = example_input()["rest_window"]
        x["samples"][-1]["temperature_k"] = 300.0
        r = indirect_rest_assessment(x)
        self.assertEqual(r.status, Status.OUT_OF_DOMAIN.value)

    def test_short_duration_out_of_domain(self):
        x = example_input()["rest_window"]
        x["minimum_rest_duration_s"] = 999999
        r = indirect_rest_assessment(x)
        self.assertEqual(r.status, Status.OUT_OF_DOMAIN.value)

    def test_soc_increase_not_misread_as_leakage(self):
        x = example_input()["rest_window"]
        for row in x["samples"]:
            row["soc_independent_percent"] = 70.0 + row["timestamp_s"] / 1e7
        r = indirect_rest_assessment(x)
        self.assertEqual(r.status, Status.NOT_IDENTIFIABLE.value)

    def test_no_voltage_does_not_block_charge_loss_estimate(self):
        x = example_input()["rest_window"]
        for row in x["samples"]:
            row.pop("cell_voltage_v", None)
        r = indirect_rest_assessment(x)
        self.assertIsNone(r.heat["equivalent_power_w"])

    def test_missing_baseline_rejected(self):
        x = example_input()["rest_window"]
        x.pop("healthy_baseline_self_discharge_a")
        with self.assertRaises(ValueError):
            indirect_rest_assessment(x)


def main() -> int:
    parser = argparse.ArgumentParser(description="Model 01 — Internal Micro-Short Assessment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="JSON input file")
    group.add_argument("--example", action="store_true", help="Run synthetic contract example")
    group.add_argument("--self-test", action="store_true", help="Run unit tests")
    args = parser.parse_args()

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Model01Tests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    try:
        raw = example_input() if args.example else json.loads(
            args.input.read_text(encoding="utf-8")
        )
        print(json.dumps(evaluate(raw), indent=2, allow_nan=False))
        return 0
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
