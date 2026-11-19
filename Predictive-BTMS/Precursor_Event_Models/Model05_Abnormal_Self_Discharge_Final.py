#!/usr/bin/env python3
"""
Model 05 — Abnormal Self-Discharge Assessment
Version: 1.0.0-reference

Purpose
-------
Quantifies excess rest-period charge loss relative to a matched healthy-cell
self-discharge baseline. It explicitly separates:
  * independently observed SOC/charge loss,
  * external-current contamination,
  * healthy baseline self-discharge,
  * excess unexplained charge loss,
  * terminal-voltage relaxation (descriptive only).

The SOC endpoints must be independent of the same current integration used
to establish charge loss. Suitable examples may include a validated
independent SOC observer or a controlled reference measurement. If SOC is
derived from the same coulomb counter being audited, the result is circular.

For each rest interval:
    Q_observed_loss = (SOC_start - SOC_end) * Q_usable
    I_observed_loss = Q_observed_loss / duration
    Q_external = integral(I_external dt)  [signed, positive into cell]
    Q_baseline = I_healthy_baseline * duration
    Q_excess = Q_observed_loss - Q_external - Q_baseline

Positive Q_excess is an abnormal self-discharge / parasitic-loss candidate.
It is not unique proof of internal micro-short. Candidate causes include
parasitic reactions, internal leakage, redox shuttles and other side reactions.

Uncertainty is propagated assuming independent SOC endpoint, capacity,
external-current and baseline-current uncertainties. The approximate 95%
interval uses 1.96*sigma. Correlated uncertainties are not represented.

Units:
  SOC: percent; capacity: Ah; current: A; time: s; temperature: degC;
  voltage: V; energy: J.

Research/reference implementation only; not experimentally validated,
cell-calibrated, or safety-certified.
"""

from __future__ import annotations
import argparse
import json
import math
import statistics
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

VERSION = "1.0.0-reference"


def num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def need(d: Dict[str, Any], key: str, *, minimum=None, positive=False) -> float:
    x = d.get(key)
    if not num(x):
        raise ValueError(f"'{key}' must be a finite number.")
    x = float(x)
    if positive and x <= 0:
        raise ValueError(f"'{key}' must be > 0.")
    if minimum is not None and x < minimum:
        raise ValueError(f"'{key}' must be >= {minimum}.")
    return x


def evaluate(packet: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(packet, dict):
        raise ValueError("Input root must be a JSON object.")
    rest = packet.get("rest_window")
    baseline = packet.get("healthy_baseline")
    if not isinstance(rest, dict) or not isinstance(baseline, dict):
        raise ValueError("Provide 'rest_window' and 'healthy_baseline' objects.")

    t0 = need(rest, "start_time_s", minimum=0)
    t1 = need(rest, "end_time_s", minimum=0)
    duration = t1 - t0
    if duration <= 0:
        raise ValueError("Rest-window end_time_s must exceed start_time_s.")

    soc0 = need(rest, "soc_start_pct", minimum=0)
    soc1 = need(rest, "soc_end_pct", minimum=0)
    if soc0 > 100 or soc1 > 100:
        raise ValueError("SOC values must be within 0–100%.")
    if soc1 > soc0:
        raise ValueError("SOC increased over the rest window; this model does not interpret that as self-discharge.")

    sigma_soc0 = need(rest, "soc_start_sigma_pct", minimum=0)
    sigma_soc1 = need(rest, "soc_end_sigma_pct", minimum=0)
    capacity = need(rest, "usable_capacity_Ah", positive=True)
    sigma_capacity = need(rest, "usable_capacity_sigma_Ah", minimum=0)

    temp0 = need(rest, "temperature_start_C")
    temp1 = need(rest, "temperature_end_C")
    max_temp_span = need(rest, "maximum_temperature_span_C", minimum=0)
    temp_span = abs(temp1 - temp0)
    if temp_span > max_temp_span:
        raise ValueError("Temperature span exceeds configured limit; compare only temperature-stable rest windows.")

    max_external_current = need(rest, "maximum_absolute_external_current_A", minimum=0)
    external = rest.get("external_current_samples")
    if not isinstance(external, list) or len(external) < 2:
        raise ValueError("At least two external_current_samples are required to audit rest current.")

    clean = []
    for i, row in enumerate(external):
        if not isinstance(row, dict):
            raise ValueError(f"External-current sample {i} must be an object.")
        if str(row.get("quality", "GOOD")).upper() not in {"GOOD", "VALID", "OK"}:
            continue
        clean.append((need(row, "time_s", minimum=0), need(row, "current_A"),
                      need(row, "current_sigma_A", minimum=0)))
    clean.sort(key=lambda x: x[0])
    if len(clean) < 2:
        raise ValueError("Insufficient valid external-current samples.")
    if clean[0][0] > t0 or clean[-1][0] < t1:
        raise ValueError("External-current samples must cover the complete rest window.")
    if any(clean[i+1][0] <= clean[i][0] for i in range(len(clean)-1)):
        raise ValueError("External-current timestamps must be strictly increasing.")

    # Clip/interpolate current trace to exact rest-window boundaries.
    def interp(t: float):
        for i in range(len(clean)-1):
            a, b = clean[i], clean[i+1]
            if a[0] <= t <= b[0]:
                f = (t-a[0])/(b[0]-a[0])
                return (a[1] + f*(b[1]-a[1]), a[2] + f*(b[2]-a[2]))
        if t == clean[-1][0]:
            return clean[-1][1], clean[-1][2]
        raise ValueError("Cannot interpolate external current at rest-window boundary.")

    clipped = [(t0, *interp(t0))]
    clipped.extend(x for x in clean if t0 < x[0] < t1)
    clipped.append((t1, *interp(t1)))

    max_abs_i = max(abs(x[1]) for x in clipped)
    if max_abs_i > max_external_current:
        raise ValueError("External current exceeds configured rest-current limit; this is not a valid rest window.")

    q_external = 0.0
    var_q_external = 0.0
    for a, b in zip(clipped[:-1], clipped[1:]):
        dt_h = (b[0]-a[0])/3600.0
        q_external += 0.5*(a[1]+b[1])*dt_h
        # Approximate independent endpoint-current uncertainty for trapezoid.
        var_q_external += (0.5*dt_h*a[2])**2 + (0.5*dt_h*b[2])**2

    baseline_i = need(baseline, "healthy_self_discharge_current_A", minimum=0)
    baseline_sigma_i = need(baseline, "healthy_self_discharge_current_sigma_A", minimum=0)
    baseline_temp = need(baseline, "reference_temperature_C")
    baseline_temp_tolerance = need(baseline, "maximum_baseline_temperature_difference_C", minimum=0)
    mean_temp = 0.5*(temp0+temp1)
    if abs(mean_temp-baseline_temp) > baseline_temp_tolerance:
        raise ValueError("Healthy baseline temperature is not sufficiently matched to the rest window.")

    delta_soc = soc0-soc1
    q_observed = (delta_soc/100.0)*capacity
    sigma_delta_soc = math.sqrt(sigma_soc0**2 + sigma_soc1**2)
    sigma_q_observed = math.sqrt(
        (capacity*sigma_delta_soc/100.0)**2 +
        (delta_soc*sigma_capacity/100.0)**2
    )

    q_baseline = baseline_i*duration/3600.0
    sigma_q_baseline = baseline_sigma_i*duration/3600.0
    q_excess = q_observed - q_external - q_baseline
    sigma_q_excess = math.sqrt(sigma_q_observed**2 + var_q_external + sigma_q_baseline**2)
    lower = q_excess - 1.96*sigma_q_excess
    upper = q_excess + 1.96*sigma_q_excess

    observed_i = q_observed/(duration/3600.0)
    external_i = q_external/(duration/3600.0)
    excess_i = q_excess/(duration/3600.0)
    sigma_excess_i = sigma_q_excess/(duration/3600.0)

    # Voltage change is intentionally descriptive and never converted to leakage.
    voltage = rest.get("terminal_voltage_start_V")
    voltage_end = rest.get("terminal_voltage_end_V")
    voltage_change = None
    if num(voltage) and num(voltage_end):
        voltage_change = float(voltage_end)-float(voltage)

    if lower > 0:
        status = "EXCESS_SELF_DISCHARGE_CANDIDATE_RESOLVED"
        interpretation = "Excess rest-period charge loss is positive beyond the approximate 95% uncertainty interval."
    elif upper < 0:
        status = "OBSERVED_LOSS_BELOW_HEALTHY_BASELINE"
        interpretation = "Observed rest-period loss is below the supplied healthy baseline; review baseline and estimator consistency."
    else:
        status = "NO_RESOLVED_EXCESS_WITHIN_UNCERTAINTY"
        interpretation = "Excess loss is not resolved from zero at the approximate 95% level."

    return {
        "model": "MODEL_05_ABNORMAL_SELF_DISCHARGE",
        "version": VERSION,
        "status": status,
        "interpretation": interpretation,
        "rest_duration_s": duration,
        "mean_cell_temperature_C": mean_temp,
        "temperature_span_C": temp_span,
        "maximum_external_current_magnitude_A": max_abs_i,
        "observed_soc_loss_pct_points": delta_soc,
        "observed_charge_loss_Ah": q_observed,
        "observed_charge_loss_sigma_Ah": sigma_q_observed,
        "integrated_external_charge_Ah": q_external,
        "integrated_external_charge_sigma_Ah": math.sqrt(var_q_external),
        "healthy_baseline_charge_loss_Ah": q_baseline,
        "healthy_baseline_charge_loss_sigma_Ah": sigma_q_baseline,
        "excess_charge_loss_Ah": q_excess,
        "excess_charge_loss_95pct_interval_Ah": [lower, upper],
        "observed_loss_current_A": observed_i,
        "external_current_average_A": external_i,
        "healthy_baseline_current_A": baseline_i,
        "excess_loss_current_A": excess_i,
        "excess_loss_current_sigma_A": sigma_excess_i,
        "terminal_voltage_change_V_descriptive_only": voltage_change,
        "identifiability": "Excess charge loss is not mechanism-specific; it cannot uniquely distinguish micro-short, parasitic reactions, internal leakage or redox shuttle.",
        "input_integrity_warnings": [
            "SOC endpoints must be independent of the current integration used for this audit.",
            "Healthy baseline must be matched for chemistry, cell design, age/SOH, SOC, temperature and rest protocol.",
            "External current must include all connected loads and measurement offsets."
        ],
        "limitations": [
            "Voltage relaxation is not converted into leakage current or charge loss.",
            "No unique physical mechanism is inferred from excess rest-period loss.",
            "SOC bias, capacity uncertainty and baseline mismatch can dominate small self-discharge signals.",
            "Uncertainty propagation assumes independent inputs and does not represent systematic/correlated errors.",
            "No local heat or hotspot estimate is made because the responsible mechanism and location are unidentified."
        ]
    }


def example_packet():
    # Synthetic example only. Seven-day stable rest window.
    start, end = 0.0, 604800.0
    return {
        "rest_window": {
            "start_time_s": start,
            "end_time_s": end,
            "soc_start_pct": 70.0,
            "soc_end_pct": 69.88,
            "soc_start_sigma_pct": 0.005,
            "soc_end_sigma_pct": 0.005,
            "usable_capacity_Ah": 2.3,
            "usable_capacity_sigma_Ah": 0.02,
            "temperature_start_C": 25.0,
            "temperature_end_C": 25.2,
            "maximum_temperature_span_C": 1.0,
            "maximum_absolute_external_current_A": 1e-6,
            "terminal_voltage_start_V": 3.302,
            "terminal_voltage_end_V": 3.298,
            "external_current_samples": [
                {"time_s": start, "current_A": 0.0, "current_sigma_A": 1e-7, "quality": "GOOD"},
                {"time_s": end/2, "current_A": 0.0, "current_sigma_A": 1e-7, "quality": "GOOD"},
                {"time_s": end, "current_A": 0.0, "current_sigma_A": 1e-7, "quality": "GOOD"}
            ]
        },
        "healthy_baseline": {
            "healthy_self_discharge_current_A": 5e-6,
            "healthy_self_discharge_current_sigma_A": 1e-6,
            "reference_temperature_C": 25.0,
            "maximum_baseline_temperature_difference_C": 1.0
        }
    }


class Model05Tests(unittest.TestCase):
    def test_excess_loss_candidate(self):
        r = evaluate(example_packet())
        self.assertEqual(r["status"], "EXCESS_SELF_DISCHARGE_CANDIDATE_RESOLVED")
        self.assertGreater(r["excess_charge_loss_Ah"], 0)

    def test_voltage_is_descriptive_only(self):
        r = evaluate(example_packet())
        self.assertIn("descriptive_only", "terminal_voltage_change_V_descriptive_only")

    def test_external_current_violation_rejected(self):
        p = example_packet()
        p["rest_window"]["external_current_samples"][1]["current_A"] = 1e-3
        with self.assertRaises(ValueError):
            evaluate(p)

    def test_temperature_mismatch_rejected(self):
        p = example_packet()
        p["healthy_baseline"]["reference_temperature_C"] = 10.0
        with self.assertRaises(ValueError):
            evaluate(p)

    def test_soc_increase_rejected(self):
        p = example_packet()
        p["rest_window"]["soc_end_pct"] = 70.1
        with self.assertRaises(ValueError):
            evaluate(p)

    def test_independent_current_integration(self):
        p = example_packet()
        r = evaluate(p)
        self.assertAlmostEqual(r["integrated_external_charge_Ah"], 0.0, places=10)


def main():
    parser = argparse.ArgumentParser(description="Model 05 — Abnormal Self-Discharge")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(Model05Tests))
        return 0 if result.wasSuccessful() else 1
    if args.example:
        packet = example_packet()
    elif args.input:
        try:
            packet = json.loads(args.input.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Input error: {exc}", file=sys.stderr)
            return 2
    else:
        parser.error("Specify --input, --example or --test.")

    try:
        result = evaluate(packet)
    except Exception as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
