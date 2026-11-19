#!/usr/bin/env python3
"""
Model 03 — SEI Growth Assessment
Version: 1.0.0-reference

A source-independent, parameter-driven SEI growth reference model.

Core growth law (diffusion-limited/parabolic approximation):
    d(L^2)/dt = 2 * k_ref * exp[-Ea/R * (1/T - 1/T_ref)]
where:
    L       = SEI thickness [m]
    k_ref   = calibrated parabolic growth coefficient [m^2/s]
    Ea      = apparent activation energy [J/mol]
    T       = absolute temperature [K]

Derived quantities:
    Delta_Q_SEI = A * Delta_L * q_vol
    Delta_R_SEI = Delta_L / (kappa_SEI * A)
    Delta_P_ohmic = I^2 * Delta_R_SEI

q_vol is a calibrated effective lithium-inventory-loss capacity per SEI
volume [Ah/m^3]. kappa_SEI is effective ionic conductivity [S/m].
The resistance expression assumes a uniform layer over the supplied active
area. The incremental I^2*Delta_R term is additional ohmic heating only;
it is NOT total SEI reaction heat and must not be added to baseline heat
without checking the heat ledger for double counting.

Important boundaries:
- This is a calibrated reduced-order growth law, not a universal SEI model.
- Ordinary terminal voltage/current/temperature cannot uniquely identify
  SEI thickness or growth rate. The model requires a calibrated k_ref and
  material/electrode parameters, or independently measured thickness.
- It does not infer SEI growth from SOH/capacity fade alone.
- No spatial/local hotspot prediction is made.
- Parameter uncertainty and sensor uncertainty are not automatically
  estimated unless the user supplies parameter bounds; outputs are
  deterministic conditional estimates.
- Research/reference implementation only; not safety-certified.

Input JSON example is available separately as model03_input.json.
Run:
    python Model03_SEI_Growth_Final.py --example
    python Model03_SEI_Growth_Final.py --test
    python Model03_SEI_Growth_Final.py --input .\\battery_packet\\model03_input.json
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
R_GAS = 8.314462618  # J mol^-1 K^-1


def is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def require_num(d: Dict[str, Any], key: str, minimum=None, strictly=False) -> float:
    value = d.get(key)
    if not is_num(value):
        raise ValueError(f"'{key}' must be a finite number.")
    value = float(value)
    if minimum is not None and (value <= minimum if strictly else value < minimum):
        op = ">" if strictly else ">="
        raise ValueError(f"'{key}' must be {op} {minimum}.")
    return value


def validate_packet(packet: Dict[str, Any]) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    if not isinstance(packet, dict):
        raise ValueError("Input root must be a JSON object.")
    p = packet.get("parameters")
    samples = packet.get("samples")
    if not isinstance(p, dict) or not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("Provide a 'parameters' object and at least two 'samples'.")

    params = {
        "initial_thickness_m": require_num(p, "initial_thickness_m", 0),
        "parabolic_rate_ref_m2_s": require_num(p, "parabolic_rate_ref_m2_s", 0, True),
        "activation_energy_J_mol": require_num(p, "activation_energy_J_mol", 0),
        "reference_temperature_K": require_num(p, "reference_temperature_K", 0, True),
        "active_area_m2": require_num(p, "active_area_m2", 0, True),
        "sei_ionic_conductivity_S_m": require_num(p, "sei_ionic_conductivity_S_m", 0, True),
        "sei_capacity_per_volume_Ah_m3": require_num(p, "sei_capacity_per_volume_Ah_m3", 0),
        "nominal_capacity_Ah": require_num(p, "nominal_capacity_Ah", 0, True),
    }
    if params["initial_thickness_m"] <= 0:
        raise ValueError("'initial_thickness_m' must be > 0.")
    clean = []
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            raise ValueError(f"Sample {i} must be an object.")
        if str(s.get("quality", "GOOD")).upper() not in {"GOOD", "VALID", "OK"}:
            continue
        t = require_num(s, "time_s", 0)
        temp = require_num(s, "temperature_K", 0, True)
        current = require_num(s, "current_A")
        clean.append({"time_s": t, "temperature_K": temp, "current_A": current})
    clean.sort(key=lambda x: x["time_s"])
    if len(clean) < 2:
        raise ValueError("Fewer than two valid samples remain after quality filtering.")
    if any(clean[i + 1]["time_s"] <= clean[i]["time_s"] for i in range(len(clean) - 1)):
        raise ValueError("Valid sample timestamps must be strictly increasing.")
    return params, clean


def evaluate(packet: Dict[str, Any]) -> Dict[str, Any]:
    p, samples = validate_packet(packet)
    l0 = p["initial_thickness_m"]
    kref = p["parabolic_rate_ref_m2_s"]
    ea = p["activation_energy_J_mol"]
    tref = p["reference_temperature_K"]
    area = p["active_area_m2"]
    conductivity = p["sei_ionic_conductivity_S_m"]
    qvol = p["sei_capacity_per_volume_Ah_m3"]
    qnom = p["nominal_capacity_Ah"]

    # Integrate the temperature-dependent parabolic coefficient over time.
    # Piecewise-constant interval temperature is the arithmetic endpoint mean.
    growth_integral_m2 = 0.0
    interval_rows = []
    for a, b in zip(samples[:-1], samples[1:]):
        dt = b["time_s"] - a["time_s"]
        temp = 0.5 * (a["temperature_K"] + b["temperature_K"])
        exponent = -ea / R_GAS * (1.0 / temp - 1.0 / tref)
        # Guard numerical overflow; such extreme extrapolation is not credible.
        if exponent > 100 or exponent < -100:
            raise ValueError("Arrhenius extrapolation outside numerical guard; check temperature/parameters.")
        k_t = kref * math.exp(exponent)
        increment = 2.0 * k_t * dt
        growth_integral_m2 += increment
        interval_rows.append({
            "start_time_s": a["time_s"],
            "end_time_s": b["time_s"],
            "duration_s": dt,
            "mean_temperature_K": temp,
            "parabolic_rate_m2_s": k_t,
            "delta_thickness_squared_m2": increment
        })

    l_final = math.sqrt(l0 * l0 + growth_integral_m2)
    delta_l = l_final - l0
    q_loss = area * delta_l * qvol
    r_initial = l0 / (conductivity * area)
    r_final = l_final / (conductivity * area)
    delta_r = r_final - r_initial

    # Additional ohmic heat due solely to the estimated SEI resistance increment.
    # Trapezoidal integration of I^2 * delta_R(t), with thickness evolving
    # according to the cumulative Arrhenius integral.
    cumulative = 0.0
    extra_heat_j = 0.0
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        dt = b["time_s"] - a["time_s"]
        tmean = 0.5 * (a["temperature_K"] + b["temperature_K"])
        k_t = kref * math.exp(-ea / R_GAS * (1.0 / tmean - 1.0 / tref))
        l_a = math.sqrt(l0 * l0 + cumulative)
        cumulative_b = cumulative + 2.0 * k_t * dt
        l_b = math.sqrt(l0 * l0 + cumulative_b)
        dr_a = (l_a - l0) / (conductivity * area)
        dr_b = (l_b - l0) / (conductivity * area)
        i2_mean = 0.5 * (a["current_A"] ** 2 + b["current_A"] ** 2)
        extra_heat_j += i2_mean * 0.5 * (dr_a + dr_b) * dt
        cumulative = cumulative_b

    duration = samples[-1]["time_s"] - samples[0]["time_s"]
    return {
        "model": "MODEL_03_SEI_GROWTH",
        "version": VERSION,
        "status": "CONDITIONAL_PARAMETERISED_ESTIMATE",
        "time_window_s": duration,
        "valid_samples": len(samples),
        "growth_model": "temperature-dependent diffusion-limited parabolic law",
        "initial_sei_thickness_m": l0,
        "estimated_final_sei_thickness_m": l_final,
        "estimated_sei_thickness_growth_m": delta_l,
        "estimated_sei_thickness_growth_nm": delta_l * 1e9,
        "initial_sei_resistance_ohm": r_initial,
        "estimated_final_sei_resistance_ohm": r_final,
        "estimated_incremental_sei_resistance_ohm": delta_r,
        "estimated_lithium_inventory_loss_Ah": q_loss,
        "estimated_capacity_fraction_loss_pct": 100.0 * q_loss / qnom,
        "estimated_incremental_sei_ohmic_heat_J": extra_heat_j,
        "interval_details": interval_rows,
        "interpretation": (
            "Conditional model estimate from supplied calibrated parameters and temperature history. "
            "It is not an independently detected SEI event."
        ),
        "required_validation": [
            "Calibrate parabolic_rate_ref_m2_s and activation_energy_J_mol for the target chemistry, cell design and ageing regime.",
            "Validate initial thickness, effective SEI conductivity and capacity-per-volume mapping using independent characterisation.",
            "Compare predicted thickness/capacity loss/resistance growth against suitable reference measurements across temperature and ageing conditions."
        ],
        "limitations": [
            "The parabolic law is a reduced-order approximation; real SEI growth may include reaction-limited, solvent-diffusion-limited and stress-coupled regimes.",
            "No anode-potential dependence, electrolyte depletion, changing porosity, cracking or SEI repair is represented.",
            "Capacity loss is a parameterised mapping from added SEI volume, not a direct electrochemical lithium balance.",
            "Incremental I^2*Delta_R heat is not total SEI reaction heat and must not be double-counted with baseline ohmic heat.",
            "Cell-average resistance/heat does not predict local hotspots."
        ]
    }


def example_packet() -> Dict[str, Any]:
    # Synthetic parameters and temperature/current history for software demonstration only.
    return {
        "parameters": {
            "initial_thickness_m": 20e-9,
            "parabolic_rate_ref_m2_s": 2.0e-22,
            "activation_energy_J_mol": 35000.0,
            "reference_temperature_K": 298.15,
            "active_area_m2": 0.18,
            "sei_ionic_conductivity_S_m": 1.0e-6,
            "sei_capacity_per_volume_Ah_m3": 2.0e8,
            "nominal_capacity_Ah": 2.3
        },
        "samples": [
            {"time_s": 0, "temperature_K": 298.15, "current_A": 0.0, "quality": "GOOD"},
            {"time_s": 3600, "temperature_K": 303.15, "current_A": 1.15, "quality": "GOOD"},
            {"time_s": 7200, "temperature_K": 308.15, "current_A": 1.15, "quality": "GOOD"},
            {"time_s": 10800, "temperature_K": 303.15, "current_A": 0.0, "quality": "GOOD"}
        ]
    }


class Model03Tests(unittest.TestCase):
    def test_positive_growth(self):
        r = evaluate(example_packet())
        self.assertGreater(r["estimated_sei_thickness_growth_m"], 0)
        self.assertGreater(r["estimated_incremental_sei_resistance_ohm"], 0)

    def test_higher_temperature_accelerates_growth(self):
        p1 = example_packet()
        p2 = example_packet()
        for s in p1["samples"]:
            s["temperature_K"] = 298.15
        for s in p2["samples"]:
            s["temperature_K"] = 318.15
        self.assertGreater(evaluate(p2)["estimated_sei_thickness_growth_m"],
                           evaluate(p1)["estimated_sei_thickness_growth_m"])

    def test_capacity_loss_nonnegative(self):
        self.assertGreaterEqual(evaluate(example_packet())["estimated_lithium_inventory_loss_Ah"], 0)

    def test_bad_quality_filtered(self):
        p = example_packet()
        p["samples"][1]["quality"] = "BAD"
        r = evaluate(p)
        self.assertEqual(r["valid_samples"], 3)

    def test_duplicate_timestamp_rejected(self):
        p = example_packet()
        p["samples"][1]["time_s"] = 0
        with self.assertRaises(ValueError):
            evaluate(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Model 03 — SEI Growth Assessment")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(Model03Tests))
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
        parser.error("Specify --input <file>, --example, or --test.")

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
