#!/usr/bin/env python3
"""
Model 06 — Abnormal Contact Resistance Assessment
Version: 1.0.0-reference

Purpose
-------
Evaluates an identified electrical contact/joint using voltage drop measured
directly across that contact and current through the same path.

For each valid sample:
    R_contact = V_drop / I
    P_contact = I * V_drop = I^2 * R_contact

If a matched healthy reference resistance is supplied:
    Delta_R = R_measured - R_healthy
    Delta_P = I^2 * max(Delta_R, 0)

The model propagates voltage/current uncertainty for R and power, and
reference-resistance uncertainty for excess resistance. It also integrates
measured contact power and positive incremental power over time.

Important:
- Voltage must be measured across the actual contact using a suitable
  four-wire/Kelvin measurement arrangement where practical.
- Current must be measured in the same electrical path and time-aligned.
- Do not use terminal cell voltage as contact voltage drop.
- A resistance estimate based on V/I is invalid when current is near zero.
- Healthy reference must match contact design, current direction, temperature,
  ageing, clamping force and measurement setup.
- The incremental heat estimate is at the identified contact. It is not cell
  bulk heat and must not be added to cell heat without a thermal boundary model.
- This model does not localise an unknown contact fault from pack-level V/I/T.

Research/reference implementation only; not safety-certified.
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


def is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def need(d: Dict[str, Any], key: str, *, minimum=None, positive=False) -> float:
    x = d.get(key)
    if not is_num(x):
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
    contact = packet.get("contact")
    samples = packet.get("samples")
    if not isinstance(contact, dict) or not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("Provide 'contact' metadata and at least two measurement samples.")

    name = str(contact.get("contact_id", "UNSPECIFIED_CONTACT"))
    min_current = need(contact, "minimum_current_for_resistance_A", positive=True)
    max_temp_span = need(contact, "maximum_temperature_span_C", minimum=0)

    reference = contact.get("healthy_reference")
    if not isinstance(reference, dict):
        raise ValueError("Provide 'healthy_reference' with matched resistance and uncertainty.")
    r_ref = need(reference, "resistance_ohm", minimum=0)
    sigma_r_ref = need(reference, "resistance_sigma_ohm", minimum=0)
    ref_temp = need(reference, "temperature_C")

    clean = []
    rejected = 0
    for i, row in enumerate(samples):
        if not isinstance(row, dict):
            raise ValueError(f"Sample {i} must be an object.")
        if str(row.get("quality", "GOOD")).upper() not in {"GOOD", "VALID", "OK"}:
            rejected += 1
            continue
        t = need(row, "time_s", minimum=0)
        current = need(row, "current_A")
        voltage = need(row, "contact_voltage_drop_V")
        sigma_i = need(row, "current_sigma_A", minimum=0)
        sigma_v = need(row, "voltage_sigma_V", minimum=0)
        temp = need(row, "contact_temperature_C")
        if abs(current) < min_current:
            rejected += 1
            continue
        clean.append({
            "time_s": t, "current_A": current, "voltage_drop_V": voltage,
            "current_sigma_A": sigma_i, "voltage_sigma_V": sigma_v,
            "temperature_C": temp
        })

    clean.sort(key=lambda x: x["time_s"])
    if len(clean) < 2:
        raise ValueError("Fewer than two valid samples remain above minimum current.")
    if any(clean[i+1]["time_s"] <= clean[i]["time_s"] for i in range(len(clean)-1)):
        raise ValueError("Valid sample timestamps must be strictly increasing.")

    temps = [x["temperature_C"] for x in clean]
    temp_span = max(temps) - min(temps)
    if temp_span > max_temp_span:
        raise ValueError("Contact temperature span exceeds configured limit; use a temperature-matched assessment.")
    mean_temp = statistics.fmean(temps)
    if abs(mean_temp - ref_temp) > max_temp_span:
        raise ValueError("Healthy reference temperature is not sufficiently matched.")

    rows = []
    for x in clean:
        i = x["current_A"]
        v = x["voltage_drop_V"]
        si = x["current_sigma_A"]
        sv = x["voltage_sigma_V"]
        r = v / i
        # First-order propagation; independent voltage/current uncertainties.
        sigma_r = math.sqrt((sv / i) ** 2 + ((v * si) / (i * i)) ** 2)
        p = i * v
        sigma_p = math.sqrt((i * sv) ** 2 + (v * si) ** 2)
        delta_r = r - r_ref
        sigma_delta_r = math.sqrt(sigma_r ** 2 + sigma_r_ref ** 2)
        delta_p = (i * i) * max(delta_r, 0.0)
        rows.append({
            "time_s": x["time_s"],
            "current_A": i,
            "contact_voltage_drop_V": v,
            "contact_resistance_ohm": r,
            "contact_resistance_sigma_ohm": sigma_r,
            "contact_power_W": p,
            "contact_power_sigma_W": sigma_p,
            "excess_resistance_vs_reference_ohm": delta_r,
            "excess_resistance_sigma_ohm": sigma_delta_r,
            "incremental_contact_heat_power_W": delta_p
        })

    # Trapezoidal integration on actual time intervals; preserve sign of
    # measured electrical power, while incremental excess heat is nonnegative.
    energy_j = 0.0
    incremental_energy_j = 0.0
    for a, b in zip(rows[:-1], rows[1:]):
        dt = b["time_s"] - a["time_s"]
        energy_j += 0.5 * (a["contact_power_W"] + b["contact_power_W"]) * dt
        incremental_energy_j += 0.5 * (
            a["incremental_contact_heat_power_W"] +
            b["incremental_contact_heat_power_W"]
        ) * dt

    resistances = [x["contact_resistance_ohm"] for x in rows]
    excesses = [x["excess_resistance_vs_reference_ohm"] for x in rows]
    resolved_count = sum(
        1 for x in rows
        if x["excess_resistance_vs_reference_ohm"] >
        1.96 * x["excess_resistance_sigma_ohm"]
    )
    mean_excess = statistics.fmean(excesses)
    mean_sigma_excess = math.sqrt(
        statistics.fmean([x["excess_resistance_sigma_ohm"] ** 2 for x in rows]) / len(rows)
    )
    lower = mean_excess - 1.96 * mean_sigma_excess
    upper = mean_excess + 1.96 * mean_sigma_excess

    if lower > 0:
        status = "ABNORMAL_CONTACT_RESISTANCE_RESOLVED"
        interpretation = "Mean contact resistance exceeds the matched healthy reference beyond the approximate 95% uncertainty interval."
    elif upper < 0:
        status = "BELOW_HEALTHY_REFERENCE"
        interpretation = "Mean contact resistance is below the supplied healthy reference; review reference matching."
    else:
        status = "NO_RESOLVED_EXCESS_WITHIN_UNCERTAINTY"
        interpretation = "Excess contact resistance is not resolved from zero at the approximate 95% level."

    return {
        "model": "MODEL_06_ABNORMAL_CONTACT_RESISTANCE",
        "version": VERSION,
        "contact_id": name,
        "status": status,
        "interpretation": interpretation,
        "valid_samples": len(rows),
        "rejected_samples": rejected,
        "temperature_mean_C": mean_temp,
        "temperature_span_C": temp_span,
        "healthy_reference_resistance_ohm": r_ref,
        "healthy_reference_resistance_sigma_ohm": sigma_r_ref,
        "mean_measured_contact_resistance_ohm": statistics.fmean(resistances),
        "minimum_measured_contact_resistance_ohm": min(resistances),
        "maximum_measured_contact_resistance_ohm": max(resistances),
        "mean_excess_resistance_ohm": mean_excess,
        "mean_excess_resistance_approx_95pct_interval_ohm": [lower, upper],
        "samples_resolved_above_reference": resolved_count,
        "mean_contact_power_W": statistics.fmean(x["contact_power_W"] for x in rows),
        "mean_incremental_contact_heat_power_W": statistics.fmean(x["incremental_contact_heat_power_W"] for x in rows),
        "integrated_contact_energy_J": energy_j,
        "integrated_incremental_contact_heat_J": incremental_energy_j,
        "sample_results": rows,
        "limitations": [
            "The voltage measurement must span the identified contact itself; pack terminal voltage is not a substitute.",
            "Four-wire/Kelvin sensing and synchronised current measurement are strongly preferred.",
            "The healthy reference must be matched for temperature, current, direction, ageing, contact pressure and geometry.",
            "First-order uncertainty propagation assumes independent voltage/current/reference uncertainties.",
            "The incremental heat is calculated at the contact and does not predict cell temperature rise or hotspot temperature.",
            "No automatic correction for thermoelectric offsets, lead resistance or current reversal is performed."
        ]
    }


def example_packet():
    # Synthetic four-wire contact measurements; not real hardware data.
    return {
        "contact": {
            "contact_id": "PACK_POSITIVE_BUSBAR_JOINT",
            "minimum_current_for_resistance_A": 0.5,
            "maximum_temperature_span_C": 3.0,
            "healthy_reference": {
                "resistance_ohm": 0.00020,
                "resistance_sigma_ohm": 0.00001,
                "temperature_C": 30.0
            }
        },
        "samples": [
            {"time_s": 0, "current_A": 10.0, "contact_voltage_drop_V": 0.0032,
             "current_sigma_A": 0.02, "voltage_sigma_V": 0.00005,
             "contact_temperature_C": 30.0, "quality": "GOOD"},
            {"time_s": 10, "current_A": 15.0, "contact_voltage_drop_V": 0.0048,
             "current_sigma_A": 0.02, "voltage_sigma_V": 0.00005,
             "contact_temperature_C": 30.5, "quality": "GOOD"},
            {"time_s": 20, "current_A": 20.0, "contact_voltage_drop_V": 0.0064,
             "current_sigma_A": 0.02, "voltage_sigma_V": 0.00005,
             "contact_temperature_C": 31.0, "quality": "GOOD"},
            {"time_s": 30, "current_A": 15.0, "contact_voltage_drop_V": 0.0048,
             "current_sigma_A": 0.02, "voltage_sigma_V": 0.00005,
             "contact_temperature_C": 30.8, "quality": "GOOD"}
        ]
    }


class Model06Tests(unittest.TestCase):
    def test_resistance_calculation(self):
        r = evaluate(example_packet())
        self.assertAlmostEqual(r["sample_results"][0]["contact_resistance_ohm"], 0.00032, places=10)

    def test_abnormal_resistance_detected(self):
        r = evaluate(example_packet())
        self.assertEqual(r["status"], "ABNORMAL_CONTACT_RESISTANCE_RESOLVED")
        self.assertGreater(r["integrated_incremental_contact_heat_J"], 0)

    def test_low_current_sample_rejected(self):
        p = example_packet()
        p["samples"][0]["current_A"] = 0.01
        r = evaluate(p)
        self.assertEqual(r["rejected_samples"], 1)

    def test_bad_quality_rejected(self):
        p = example_packet()
        p["samples"][0]["quality"] = "BAD"
        r = evaluate(p)
        self.assertEqual(r["rejected_samples"], 1)

    def test_temperature_span_rejected(self):
        p = example_packet()
        p["samples"][-1]["contact_temperature_C"] = 50.0
        with self.assertRaises(ValueError):
            evaluate(p)

    def test_duplicate_time_rejected(self):
        p = example_packet()
        p["samples"][1]["time_s"] = 0
        with self.assertRaises(ValueError):
            evaluate(p)


def main():
    parser = argparse.ArgumentParser(description="Model 06 — Abnormal Contact Resistance")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(Model06Tests))
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
