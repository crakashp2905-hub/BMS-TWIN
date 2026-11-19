#!/usr/bin/env python3
"""
Model 04 — Electrolyte Decomposition Assessment
Version: 1.0.0-reference

Purpose
-------
A source-independent, parameterised reduced-order model for estimating
electrolyte consumption by a calibrated parasitic decomposition pathway.

Kinetic model
-------------
d(c/c0)/dt = -k(T, eta) * (c/c0)

k(T, eta) = k_ref * exp[-Ea/R * (1/T - 1/T_ref)]
                    * exp[alpha*n*F*eta/(R*T)]

eta is a signed, reaction-driving overpotential in the decomposition-favouring
direction. It must be produced by a validated cell/electrode observer or
measured with an appropriate reference electrode. It is NOT terminal cell
voltage. The potential acceleration factor is a reduced-order assumption and
must be calibrated for the specific reaction pathway/cell.

Outputs
-------
- Electrolyte concentration and fraction remaining
- Electrolyte amount consumed [mol]
- Electron-equivalent parasitic charge [Ah] (n is pathway-specific)
- Reaction enthalpy estimate [J] when a validated molar enthalpy is supplied
- Rate history and parameterised kinetic exposure

Boundaries
----------
- This is one lumped, parameterised decomposition pathway, not a complete
  electrolyte chemistry network.
- It does not identify decomposition products, gas species, SEI/CEI pathways,
  or uniquely diagnose decomposition from terminal BMS signals.
- The model does not infer reaction kinetics from voltage/current/temperature
  alone; k_ref, Ea, alpha, n and potential definition require calibration.
- Reaction enthalpy is signed: positive means heat released to the cell;
  negative means heat absorbed. Do not double-count with baseline heat.
- No local gas pocket, pressure, spatial concentration or hotspot prediction.
- Research/reference code only; not experimentally validated or safety-certified.

Run:
  python Model04_Electrolyte_Decomposition_Final.py --test
  python Model04_Electrolyte_Decomposition_Final.py --example
  python Model04_Electrolyte_Decomposition_Final.py --input .\\Precursor_Event_Models\\battery_packet\\model04_input.json
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

VERSION = "1.0.0-reference"
R = 8.314462618       # J mol^-1 K^-1
F = 96485.33212       # C mol^-1


def is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def number(d: Dict[str, Any], key: str, *, positive=False, nonnegative=False) -> float:
    x = d.get(key)
    if not is_num(x):
        raise ValueError(f"'{key}' must be a finite number.")
    x = float(x)
    if positive and x <= 0:
        raise ValueError(f"'{key}' must be > 0.")
    if nonnegative and x < 0:
        raise ValueError(f"'{key}' must be >= 0.")
    return x


def validate(packet: Dict[str, Any]):
    if not isinstance(packet, dict):
        raise ValueError("Input root must be a JSON object.")
    p = packet.get("parameters")
    samples = packet.get("samples")
    if not isinstance(p, dict) or not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("Provide 'parameters' and at least two 'samples'.")

    params = {
        "initial_concentration_mol_m3": number(p, "initial_concentration_mol_m3", positive=True),
        "electrolyte_volume_m3": number(p, "electrolyte_volume_m3", positive=True),
        "rate_constant_ref_s": number(p, "rate_constant_ref_s", nonnegative=True),
        "activation_energy_J_mol": number(p, "activation_energy_J_mol", nonnegative=True),
        "reference_temperature_K": number(p, "reference_temperature_K", positive=True),
        "transfer_coefficient": number(p, "transfer_coefficient", nonnegative=True),
        "electron_number": number(p, "electron_number", positive=True),
        "reaction_enthalpy_J_mol": number(p, "reaction_enthalpy_J_mol"),
    }
    if params["transfer_coefficient"] > 2:
        raise ValueError("'transfer_coefficient' outside expected 0–2 range; verify calibration.")
    clean = []
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            raise ValueError(f"Sample {i} must be an object.")
        if str(s.get("quality", "GOOD")).upper() not in {"GOOD", "VALID", "OK"}:
            continue
        t = number(s, "time_s", nonnegative=True)
        temp = number(s, "temperature_K", positive=True)
        eta = number(s, "decomposition_overpotential_V")
        clean.append({"time_s": t, "temperature_K": temp,
                      "decomposition_overpotential_V": eta})
    clean.sort(key=lambda x: x["time_s"])
    if len(clean) < 2:
        raise ValueError("Fewer than two valid samples remain after quality filtering.")
    if any(clean[i+1]["time_s"] <= clean[i]["time_s"] for i in range(len(clean)-1)):
        raise ValueError("Valid timestamps must be strictly increasing.")
    return params, clean


def evaluate(packet: Dict[str, Any]) -> Dict[str, Any]:
    p, samples = validate(packet)
    c0 = p["initial_concentration_mol_m3"]
    volume = p["electrolyte_volume_m3"]
    kref = p["rate_constant_ref_s"]
    ea = p["activation_energy_J_mol"]
    tref = p["reference_temperature_K"]
    alpha = p["transfer_coefficient"]
    n = p["electron_number"]
    dh = p["reaction_enthalpy_J_mol"]

    remaining_fraction = 1.0
    consumed_moles = 0.0
    details = []
    for a, b in zip(samples[:-1], samples[1:]):
        dt = b["time_s"] - a["time_s"]
        temp = 0.5 * (a["temperature_K"] + b["temperature_K"])
        eta = 0.5 * (a["decomposition_overpotential_V"] +
                     b["decomposition_overpotential_V"])
        arrhenius_exp = -ea / R * (1.0 / temp - 1.0 / tref)
        potential_exp = alpha * n * F * eta / (R * temp)
        if abs(arrhenius_exp) > 100 or abs(potential_exp) > 50:
            raise ValueError("Kinetic extrapolation exceeds numerical guard; check units and calibration.")
        k_t = kref * math.exp(arrhenius_exp + potential_exp)
        before = remaining_fraction
        # Exact solution for constant interval-average rate.
        remaining_fraction *= math.exp(-k_t * dt)
        consumed_fraction = before - remaining_fraction
        dmoles = c0 * volume * consumed_fraction
        consumed_moles += dmoles
        details.append({
            "start_time_s": a["time_s"],
            "end_time_s": b["time_s"],
            "duration_s": dt,
            "mean_temperature_K": temp,
            "mean_decomposition_overpotential_V": eta,
            "rate_constant_s": k_t,
            "fraction_electrolyte_consumed_in_interval": consumed_fraction,
            "electrolyte_consumed_mol": dmoles
        })

    initial_moles = c0 * volume
    consumed_fraction_total = consumed_moles / initial_moles
    remaining_concentration = c0 * remaining_fraction
    charge_ah = consumed_moles * n * F / 3600.0
    reaction_heat = consumed_moles * dh
    duration = samples[-1]["time_s"] - samples[0]["time_s"]

    return {
        "model": "MODEL_04_ELECTROLYTE_DECOMPOSITION",
        "version": VERSION,
        "status": "CONDITIONAL_PARAMETERISED_ESTIMATE",
        "time_window_s": duration,
        "valid_samples": len(samples),
        "kinetic_model": "first-order electrolyte consumption with Arrhenius and calibrated potential acceleration",
        "initial_electrolyte_concentration_mol_m3": c0,
        "estimated_final_electrolyte_concentration_mol_m3": remaining_concentration,
        "initial_electrolyte_amount_mol": initial_moles,
        "estimated_electrolyte_consumed_mol": consumed_moles,
        "estimated_electrolyte_consumed_fraction_pct": 100.0 * consumed_fraction_total,
        "estimated_electron_equivalent_parasitic_charge_Ah": charge_ah,
        "reaction_enthalpy_J_mol": dh,
        "estimated_reaction_heat_J": reaction_heat,
        "heat_interpretation": "Positive reaction heat means heat release; negative means heat absorption.",
        "interval_details": details,
        "interpretation": (
            "Conditional estimate for the specified calibrated decomposition pathway; "
            "not an independently detected electrolyte-decomposition event."
        ),
        "required_validation": [
            "Identify the specific decomposition pathway and define its overpotential sign/reference convention.",
            "Calibrate rate_constant_ref_s, activation_energy_J_mol and transfer_coefficient for the target cell and pathway.",
            "Validate electron_number and reaction_enthalpy_J_mol against reaction stoichiometry and calorimetry.",
            "Validate electrolyte volume and initial concentration for the actual cell."
        ],
        "limitations": [
            "A single lumped pathway cannot represent the full coupled electrolyte, SEI and CEI reaction network.",
            "The estimated electron-equivalent charge is pathway-specific and is not automatically equal to measured capacity fade.",
            "The model does not predict gas composition, gas pressure, swelling or local depletion.",
            "Temperature and overpotential inputs must be valid, synchronised and traceable.",
            "Do not add reaction heat to baseline heat unless the baseline model excludes this parasitic reaction."
        ]
    }


def example_packet():
    # Synthetic demonstration values only; not target-cell calibration.
    return {
        "parameters": {
            "initial_concentration_mol_m3": 1000.0,
            "electrolyte_volume_m3": 1.0e-5,
            "rate_constant_ref_s": 1.0e-9,
            "activation_energy_J_mol": 45000.0,
            "reference_temperature_K": 298.15,
            "transfer_coefficient": 0.35,
            "electron_number": 1.0,
            "reaction_enthalpy_J_mol": 25000.0
        },
        "samples": [
            {"time_s": 0, "temperature_K": 298.15, "decomposition_overpotential_V": 0.05, "quality": "GOOD"},
            {"time_s": 3600, "temperature_K": 303.15, "decomposition_overpotential_V": 0.08, "quality": "GOOD"},
            {"time_s": 7200, "temperature_K": 308.15, "decomposition_overpotential_V": 0.10, "quality": "GOOD"},
            {"time_s": 10800, "temperature_K": 303.15, "decomposition_overpotential_V": 0.06, "quality": "GOOD"}
        ]
    }


class Model04Tests(unittest.TestCase):
    def test_consumption_nonnegative_and_bounded(self):
        r = evaluate(example_packet())
        self.assertGreaterEqual(r["estimated_electrolyte_consumed_mol"], 0)
        self.assertLessEqual(r["estimated_electrolyte_consumed_fraction_pct"], 100)

    def test_higher_temperature_increases_rate(self):
        a, b = example_packet(), example_packet()
        for s in a["samples"]:
            s["temperature_K"] = 298.15
        for s in b["samples"]:
            s["temperature_K"] = 318.15
        self.assertGreater(evaluate(b)["estimated_electrolyte_consumed_mol"],
                           evaluate(a)["estimated_electrolyte_consumed_mol"])

    def test_higher_driving_overpotential_increases_rate(self):
        a, b = example_packet(), example_packet()
        for s in a["samples"]:
            s["decomposition_overpotential_V"] = 0.01
        for s in b["samples"]:
            s["decomposition_overpotential_V"] = 0.10
        self.assertGreater(evaluate(b)["estimated_electrolyte_consumed_mol"],
                           evaluate(a)["estimated_electrolyte_consumed_mol"])

    def test_bad_quality_filtered(self):
        p = example_packet()
        p["samples"][1]["quality"] = "BAD"
        self.assertEqual(evaluate(p)["valid_samples"], 3)

    def test_duplicate_time_rejected(self):
        p = example_packet()
        p["samples"][1]["time_s"] = 0
        with self.assertRaises(ValueError):
            evaluate(p)


def main():
    parser = argparse.ArgumentParser(description="Model 04 — Electrolyte Decomposition")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(Model04Tests))
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
        print(f"\\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
