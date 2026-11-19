#!/usr/bin/env python3
"""
Model 02 — Lithium Plating Assessment
Version: 1.0.0-reference

Purpose
-------
A source-independent, physics-informed lithium-plating assessment module.
It accepts measured signals and/or independently estimated electrochemical
states. It does not require PyBaMM at runtime.

Two assessment paths are implemented:
1) Direct anode-potential criterion:
   During charging, negative-electrode potential vs Li/Li+ approaching or
   crossing 0 V is the strongest available electrochemical indicator.
   Measurement/estimator uncertainty is propagated to estimate the fraction
   of valid samples whose potential is at/below the configurable threshold.
2) Operating-condition risk screen:
   Uses charge temperature, SOC and C-rate as contributing conditions.
   This is a screening score, NOT a plating measurement and NOT a universal
   calibrated probability. Thresholds are explicit configuration parameters.

A post-charge stripping signature can be evaluated only when an independent
voltage trace and a matched reference trace are supplied. The script reports
a descriptive differential relaxation feature; it does not convert that
feature into plated-lithium mass without a validated cell-specific model.

Units
-----
time: s; potential: V vs Li/Li+; temperature: degC; SOC: %;
current: A (positive = charging); capacity: Ah; C-rate: h^-1;
voltage: V.

Input JSON schema (minimal example)
-----------------------------------
{
  "model": "lithium_plating",
  "samples": [
    {
      "time_s": 0,
      "current_A": 2.3,
      "temperature_C": 10.0,
      "soc_pct": 85.0,
      "capacity_Ah": 2.3,
      "anode_potential_V": 0.012,
      "anode_potential_sigma_V": 0.004,
      "quality": "GOOD"
    }
  ],
  "config": {
    "plating_threshold_V": 0.0,
    "near_threshold_margin_V": 0.020,
    "minimum_valid_samples": 3
  }
}

For the direct criterion, anode_potential_V must be an independently measured
or validated model-estimated negative-electrode potential vs Li/Li+. A
terminal cell voltage is NOT a substitute.

Research/reference implementation only. Not experimentally validated,
cell-calibrated, safety-certified, or suitable as a sole protection function.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


VERSION = "1.0.0-reference"


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class PlatingConfig:
    plating_threshold_V: float = 0.0
    near_threshold_margin_V: float = 0.020
    minimum_valid_samples: int = 3
    # These are transparent screening breakpoints, not universal chemistry limits.
    cold_start_C: float = 20.0
    cold_high_risk_C: float = 0.0
    high_soc_start_pct: float = 80.0
    high_soc_max_pct: float = 100.0
    high_crate_start: float = 0.5
    high_crate_max: float = 2.0
    # Weights apply only to the condition-screen score.
    temperature_weight: float = 0.45
    soc_weight: float = 0.25
    crate_weight: float = 0.30

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlatingConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _validate_config(c: PlatingConfig) -> None:
    numeric = [v for v in asdict(c).values() if isinstance(v, (int, float))]
    if not all(finite_number(v) for v in numeric):
        raise ValueError("All numeric config values must be finite.")
    if c.near_threshold_margin_V <= 0:
        raise ValueError("near_threshold_margin_V must be > 0.")
    if c.minimum_valid_samples < 1:
        raise ValueError("minimum_valid_samples must be >= 1.")
    if c.cold_start_C <= c.cold_high_risk_C:
        raise ValueError("cold_start_C must exceed cold_high_risk_C.")
    if c.high_soc_max_pct <= c.high_soc_start_pct:
        raise ValueError("high_soc_max_pct must exceed high_soc_start_pct.")
    if c.high_crate_max <= c.high_crate_start:
        raise ValueError("high_crate_max must exceed high_crate_start.")
    weights = [c.temperature_weight, c.soc_weight, c.crate_weight]
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Condition-screen weights must be non-negative with positive sum.")


def _valid_sample(s: Dict[str, Any]) -> bool:
    if str(s.get("quality", "GOOD")).upper() not in {"GOOD", "VALID", "OK"}:
        return False
    return finite_number(s.get("time_s")) and finite_number(s.get("current_A"))


def _normalised_weighted_score(parts: Dict[str, Optional[float]], cfg: PlatingConfig) -> Optional[float]:
    weights = {
        "temperature": cfg.temperature_weight,
        "soc": cfg.soc_weight,
        "crate": cfg.crate_weight,
    }
    available = [(parts[k], weights[k]) for k in parts if parts[k] is not None and weights[k] > 0]
    if not available:
        return None
    return sum(v * w for v, w in available) / sum(w for _, w in available)


def assess_direct_anode_potential(samples: List[Dict[str, Any]], cfg: PlatingConfig) -> Dict[str, Any]:
    usable = []
    rejected = 0
    for s in samples:
        if not _valid_sample(s):
            rejected += 1
            continue
        # Positive current is defined as charging.
        if float(s["current_A"]) <= 0:
            continue
        p = s.get("anode_potential_V")
        if not finite_number(p):
            continue
        sigma = s.get("anode_potential_sigma_V", 0.0)
        if not finite_number(sigma) or float(sigma) < 0:
            rejected += 1
            continue
        usable.append((float(s["time_s"]), float(p), float(sigma)))

    usable.sort(key=lambda x: x[0])
    if len(usable) < cfg.minimum_valid_samples:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": "Insufficient valid charging samples with anode potential vs Li/Li+.",
            "valid_samples": len(usable),
            "rejected_samples": rejected,
            "warning": "Terminal cell voltage cannot replace negative-electrode potential."
        }

    threshold = cfg.plating_threshold_V
    margins = [p - threshold for _, p, _ in usable]
    # Gaussian probability that the true potential is at/below threshold.
    probs = []
    for _, p, sigma in usable:
        if sigma == 0:
            probs.append(1.0 if p <= threshold else 0.0)
        else:
            z = (threshold - p) / (sigma * math.sqrt(2.0))
            probs.append(0.5 * (1.0 + math.erf(z)))

    below = sum(1 for m in margins if m <= 0)
    near = sum(1 for m in margins if 0 < m <= cfg.near_threshold_margin_V)
    # Integrate only over adjacent valid samples; gaps are not bridged.
    duration_below = 0.0
    duration_near = 0.0
    for i in range(1, len(usable)):
        t0, p0, _ = usable[i - 1]
        t1, p1, _ = usable[i]
        dt = t1 - t0
        if dt <= 0:
            continue
        # Conservative interval classification by endpoint average.
        pm = 0.5 * (p0 + p1)
        if pm <= threshold:
            duration_below += dt
        elif pm <= threshold + cfg.near_threshold_margin_V:
            duration_near += dt

    return {
        "status": "EVALUATED",
        "criterion": "negative-electrode potential vs Li/Li+ during positive-current charging",
        "threshold_V_vs_Li": threshold,
        "valid_samples": len(usable),
        "rejected_samples": rejected,
        "minimum_potential_V": min(p for _, p, _ in usable),
        "mean_potential_V": statistics.fmean(p for _, p, _ in usable),
        "samples_at_or_below_threshold": below,
        "samples_near_threshold": near,
        "fraction_samples_at_or_below_threshold": below / len(usable),
        "mean_probability_potential_at_or_below_threshold": statistics.fmean(probs),
        "estimated_time_at_or_below_threshold_s": duration_below,
        "estimated_time_near_threshold_s": duration_near,
        "interpretation": (
            "Electrochemical plating condition indicated: anode potential reached/crossed threshold."
            if below else
            "No sampled threshold crossing; near-threshold exposure may still warrant review."
            if near else
            "No threshold crossing observed in supplied valid samples."
        ),
        "limitations": [
            "Potential threshold is a thermodynamic indicator, not a direct plated-lithium mass measurement.",
            "Anode potential must be independently measured or produced by a validated electrochemical observer.",
            "Sampling rate, sensor/observer bias, spatial gradients and cell chemistry affect interpretation.",
            "This result does not establish plating morphology, quantity, reversibility or safety."
        ]
    }


def assess_condition_screen(samples: List[Dict[str, Any]], cfg: PlatingConfig) -> Dict[str, Any]:
    valid = [s for s in samples if _valid_sample(s) and float(s["current_A"]) > 0]
    temps = [float(s["temperature_C"]) for s in valid if finite_number(s.get("temperature_C"))]
    socs = [float(s["soc_pct"]) for s in valid if finite_number(s.get("soc_pct"))]
    crates = []
    for s in valid:
        if finite_number(s.get("current_A")) and finite_number(s.get("capacity_Ah")) and float(s["capacity_Ah"]) > 0:
            crates.append(abs(float(s["current_A"])) / float(s["capacity_Ah"]))

    temp_risk = None
    if temps:
        # 0 at/above cold_start; 1 at/below cold_high_risk.
        temp_risk = clamp((cfg.cold_start_C - min(temps)) / (cfg.cold_start_C - cfg.cold_high_risk_C))
    soc_risk = None
    if socs:
        soc_risk = clamp((max(socs) - cfg.high_soc_start_pct) / (cfg.high_soc_max_pct - cfg.high_soc_start_pct))
    crate_risk = None
    if crates:
        crate_risk = clamp((max(crates) - cfg.high_crate_start) / (cfg.high_crate_max - cfg.high_crate_start))

    parts = {"temperature": temp_risk, "soc": soc_risk, "crate": crate_risk}
    score = _normalised_weighted_score(parts, cfg)
    return {
        "status": "EVALUATED" if score is not None else "INSUFFICIENT_DATA",
        "screen_score_0_to_1": score,
        "screen_score_is_probability": False,
        "contributing_condition_scores": parts,
        "observed_min_temperature_C": min(temps) if temps else None,
        "observed_max_soc_pct": max(socs) if socs else None,
        "observed_max_charge_rate_C": max(crates) if crates else None,
        "valid_charging_samples": len(valid),
        "interpretation": (
            "Elevated operating-condition risk screen; this does not confirm lithium plating."
            if score is not None and score >= 0.6 else
            "Some plating-favourable operating conditions are present; this does not confirm plating."
            if score is not None and score > 0 else
            "No elevated condition score from available inputs."
            if score is not None else
            "Insufficient temperature, SOC or capacity/current data for condition screening."
        ),
        "limitations": [
            "Breakpoints and weights are configurable screening assumptions, not universal cell limits.",
            "Risk depends on chemistry, electrode design, ageing, pressure, charge history and cell-specific kinetics.",
            "The score is not a calibrated probability and must not be used as a safety trip threshold."
        ]
    }


def assess_stripping_signature(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Descriptive post-charge differential relaxation feature.
    Requires two time-aligned voltage traces: candidate cell and matched
    non-plating/reference baseline, both under equivalent rest conditions.
    No plated-lithium mass is inferred.
    """
    candidate = data.get("candidate_voltage_V")
    reference = data.get("reference_voltage_V")
    times = data.get("time_s")
    if not (isinstance(candidate, list) and isinstance(reference, list) and isinstance(times, list)):
        return {"status": "NOT_EVALUATED", "reason": "No paired candidate/reference rest traces supplied."}
    if len(times) < 4 or len(candidate) != len(times) or len(reference) != len(times):
        return {"status": "INSUFFICIENT_DATA", "reason": "Paired traces must have equal length and at least four samples."}
    if not all(finite_number(x) for arr in (times, candidate, reference) for x in arr):
        return {"status": "INVALID_DATA", "reason": "Trace contains non-finite or non-numeric values."}
    if any(float(times[i + 1]) <= float(times[i]) for i in range(len(times) - 1)):
        return {"status": "INVALID_DATA", "reason": "time_s must be strictly increasing."}

    residual = [float(v) - float(r) for v, r in zip(candidate, reference)]
    slopes = [(residual[i + 1] - residual[i]) / (float(times[i + 1]) - float(times[i]))
              for i in range(len(times) - 1)]
    # A descriptive feature: largest absolute slope change in differential relaxation.
    curvature = [abs(slopes[i + 1] - slopes[i]) for i in range(len(slopes) - 1)]
    idx = max(range(len(curvature)), key=curvature.__getitem__)
    return {
        "status": "DESCRIPTIVE_FEATURE_ONLY",
        "maximum_differential_relaxation_slope_change_V_per_s2": curvature[idx],
        "feature_time_s": float(times[idx + 1]),
        "differential_voltage_range_V": max(residual) - min(residual),
        "interpretation": "A differential relaxation feature is present in the paired traces; it is not specific proof of lithium stripping.",
        "limitations": [
            "Reference trace must be matched for chemistry, SOC, temperature, age, rest protocol and measurement chain.",
            "Relaxation, hysteresis, thermal drift and other side reactions can produce similar features.",
            "No plated-lithium mass or stripping charge is calculated."
        ]
    }


def evaluate(packet: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(packet, dict):
        raise ValueError("Input root must be a JSON object.")
    samples = packet.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("'samples' must be a list.")
    cfg = PlatingConfig.from_dict(packet.get("config", {}))
    _validate_config(cfg)

    direct = assess_direct_anode_potential(samples, cfg)
    conditions = assess_condition_screen(samples, cfg)
    stripping = assess_stripping_signature(packet.get("stripping", {}))

    if direct["status"] == "EVALUATED" and direct["samples_at_or_below_threshold"] > 0:
        overall = "ELECTROCHEMICAL_PLATING_CONDITION_INDICATED"
    elif direct["status"] == "EVALUATED" and direct["samples_near_threshold"] > 0:
        overall = "NEAR_THRESHOLD_REVIEW"
    elif conditions["status"] == "EVALUATED" and conditions["screen_score_0_to_1"] >= 0.6:
        overall = "ELEVATED_CONDITION_SCREEN_ONLY"
    elif direct["status"] == "EVALUATED":
        overall = "NO_THRESHOLD_CROSSING_OBSERVED"
    else:
        overall = "INSUFFICIENT_DIRECT_ELECTROCHECHEMICAL_EVIDENCE"

    return {
        "model": "MODEL_02_LITHIUM_PLATING",
        "version": VERSION,
        "overall_status": overall,
        "direct_anode_potential_assessment": direct,
        "operating_condition_screen": conditions,
        "post_charge_stripping_feature": stripping,
        "global_limitations": [
            "No plated-lithium mass, morphology or local deposition map is inferred.",
            "A condition screen is not a plating detector.",
            "Do not substitute terminal voltage for anode potential vs Li/Li+.",
            "Cell-specific calibration and validation against reference methods are required before deployment."
        ]
    }


def example_packet() -> Dict[str, Any]:
    # Synthetic data solely to exercise the direct criterion and condition screen.
    samples = []
    for i, p in enumerate([0.018, 0.012, 0.006, -0.003, -0.006, 0.004]):
        samples.append({
            "time_s": float(i * 10),
            "current_A": 2.3,
            "temperature_C": 8.0,
            "soc_pct": 88.0 + 0.2 * i,
            "capacity_Ah": 2.3,
            "anode_potential_V": p,
            "anode_potential_sigma_V": 0.004,
            "quality": "GOOD"
        })
    return {"model": "lithium_plating", "samples": samples}


class Model02Tests(unittest.TestCase):
    def test_terminal_only_is_insufficient_direct(self):
        packet = {"samples": [{"time_s": i, "current_A": 1.0, "voltage_V": 3.7,
                               "temperature_C": 25, "soc_pct": 50, "capacity_Ah": 2.3}
                              for i in range(4)]}
        result = evaluate(packet)
        self.assertEqual(result["direct_anode_potential_assessment"]["status"], "INSUFFICIENT_DATA")

    def test_crossing_detected(self):
        packet = example_packet()
        result = evaluate(packet)
        direct = result["direct_anode_potential_assessment"]
        self.assertEqual(direct["status"], "EVALUATED")
        self.assertGreater(direct["samples_at_or_below_threshold"], 0)
        self.assertEqual(result["overall_status"], "ELECTROCHEMICAL_PLATING_CONDITION_INDICATED")

    def test_condition_score_not_probability(self):
        result = evaluate(example_packet())
        screen = result["operating_condition_screen"]
        self.assertFalse(screen["screen_score_is_probability"])
        self.assertIsNotNone(screen["screen_score_0_to_1"])

    def test_bad_quality_excluded(self):
        packet = example_packet()
        for s in packet["samples"]:
            s["quality"] = "BAD"
        result = evaluate(packet)
        self.assertEqual(result["direct_anode_potential_assessment"]["status"], "INSUFFICIENT_DATA")

    def test_stripping_requires_paired_trace(self):
        result = assess_stripping_signature({})
        self.assertEqual(result["status"], "NOT_EVALUATED")

    def test_invalid_time_rejected(self):
        packet = example_packet()
        packet["samples"][2]["time_s"] = -1
        # Sorting is allowed; direct assessment remains evaluable.
        result = evaluate(packet)
        self.assertEqual(result["direct_anode_potential_assessment"]["status"], "EVALUATED")


def main() -> int:
    parser = argparse.ArgumentParser(description="Model 02 — Lithium Plating Assessment")
    parser.add_argument("--input", type=Path, help="Path to input JSON packet")
    parser.add_argument("--example", action="store_true", help="Run synthetic demonstration")
    parser.add_argument("--test", action="store_true", help="Run built-in unit tests")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Model02Tests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
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
