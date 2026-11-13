#!/usr/bin/env python3
"""
Unified Battery Precursor Evaluation Engine
Version: 0.2.0-six-model-integration

Runs the six standalone precursor reference models through one canonical
source-independent packet and normalises their results into one report.

Folder expectation:
  Precursor_Event_Models/
    Unified_Battery_Precursor_Evaluation_Engine_v0_2.py
    Model01_Internal_MicroShort_Final.py
    Model02_Lithium_Plating_Final.py
    Model03_SEI_Growth_Final.py
    Model04_Electrolyte_Decomposition_Final.py
    Model05_Abnormal_Self_Discharge_Final.py
    Model06_Abnormal_Contact_Resistance_Final.py

Canonical packet:
{
  "schema_version": "1.0",
  "cell": {"cell_id": "...", "chemistry": "LFP_GRAPHITE"},
  "context": {"evaluation_timestamp_utc": "...", "data_provenance": "..."},
  "model_inputs": {
    "internal_micro_short": { ... Model01 evaluate() input ... },
    "lithium_plating": { ... Model02 evaluate() input ... },
    "sei_growth": { ... Model03 evaluate() input ... },
    "electrolyte_decomposition": { ... Model04 evaluate() input ... },
    "abnormal_self_discharge": { ... Model05 evaluate() input ... },
    "abnormal_contact_resistance": { ... Model06 evaluate() input ... }
  }
}

Each model-specific payload preserves its tested standalone schema. This avoids
silently flattening incompatible time-series/parameter contracts. Each
payload should additionally carry provenance metadata in `_provenance`:
source, acquisition_method, timestamp_utc, quality, and notes.

Heat ledger:
- Lists only heat/energy quantities directly returned by a model.
- Keeps local contact/path dissipation separate from cell-level reaction heat.
- Does not infer missing heat or local hotspot temperature.
- Does not automatically sum contributions across mechanisms because
  overlapping physical ownership/double counting has not been experimentally
  excluded. A simple subtotal is shown for traceability, explicitly not total
  cell heat.

Research/reference software only; not safety-certified firmware.
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "0.2.0-six-model-integration"

MODEL_FILES = {
    "internal_micro_short": ("INTERNAL_MICRO_SHORT", "Model01_Internal_MicroShort_Final.py"),
    "lithium_plating": ("LITHIUM_PLATING", "Model02_Lithium_Plating_Final.py"),
    "sei_growth": ("SEI_GROWTH", "Model03_SEI_Growth_Final.py"),
    "electrolyte_decomposition": ("ELECTROLYTE_DECOMPOSITION", "Model04_Electrolyte_Decomposition_Final.py"),
    "abnormal_self_discharge": ("ABNORMAL_SELF_DISCHARGE", "Model05_Abnormal_Self_Discharge_Final.py"),
    "abnormal_contact_resistance": ("ABNORMAL_CONTACT_RESISTANCE", "Model06_Abnormal_Contact_Resistance_Final.py"),
}


def load_model_modules(base_dir: Path) -> Dict[str, Any]:
    modules = {}
    for key, (_, filename) in MODEL_FILES.items():
        path = base_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required model script not found: {path}")
        module_name = f"_precursor_{key}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load model module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not callable(getattr(module, "evaluate", None)):
            raise TypeError(f"{filename} does not expose evaluate(packet).")
        modules[key] = module
    return modules


def validate_root(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Input root must be a JSON object.")
    if not isinstance(raw.get("cell"), dict):
        raise ValueError("'cell' must be an object.")
    if not str(raw["cell"].get("cell_id", "")).strip():
        raise ValueError("cell.cell_id is required.")
    if not str(raw["cell"].get("chemistry", "")).strip():
        raise ValueError("cell.chemistry is required.")
    if not isinstance(raw.get("context", {}), dict):
        raise ValueError("'context' must be an object.")
    if not isinstance(raw.get("model_inputs"), dict):
        raise ValueError("'model_inputs' must be an object.")
    provenance_required = {"source", "acquisition_method", "timestamp_utc", "quality"}
    for model_key, payload in raw["model_inputs"].items():
        if model_key not in MODEL_FILES:
            raise ValueError(f"Unknown model_inputs key: {model_key}")
        if not isinstance(payload, dict):
            raise ValueError(f"model_inputs.{model_key} must be an object.")
        prov = payload.get("_provenance")
        if prov is not None:
            if not isinstance(prov, dict):
                raise ValueError(f"model_inputs.{model_key}._provenance must be an object.")
            missing = provenance_required - set(prov)
            if missing:
                raise ValueError(f"model_inputs.{model_key}._provenance missing: {sorted(missing)}")
            if str(prov["quality"]).upper() not in {"GOOD", "SUSPECT", "BAD", "STALE"}:
                raise ValueError(f"Invalid provenance quality for {model_key}.")
    return raw


def strip_provenance(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "_provenance"}


def status_from(result: Dict[str, Any]) -> str:
    for key in ("overall_status", "status"):
        val = result.get(key)
        if isinstance(val, str):
            return val
    return "UNKNOWN"


def normalise_result(model_key: str, result: Dict[str, Any]) -> Dict[str, Any]:
    model_id = MODEL_FILES[model_key][0]
    status = status_from(result)
    calculated = {}
    findings = []
    required = []
    limitations = list(result.get("global_limitations", [])) if isinstance(result.get("global_limitations"), list) else []

    if model_key == "internal_micro_short":
        assessments = result.get("assessments", {})
        direct = assessments.get("direct_path_assessment", {})
        indirect = assessments.get("indirect_rest_assessment", {})
        calculated = {"direct_path_assessment": direct, "indirect_rest_assessment": indirect}
        statuses = [direct.get("status"), indirect.get("status")]
        status = "EVALUATED" if "EVALUATED" in statuses else "INSUFFICIENT_DATA"
        findings = [x for x in (direct.get("conclusion"), indirect.get("conclusion")) if x]
        required = direct.get("required_evidence", []) + indirect.get("required_evidence", [])
        limitations += result.get("global_limitations", [])
        ident = "PATH_DISSIPATION_OR_EXCESS_REST_LOSS_CONDITIONAL; MICRO_SHORT_ORIGIN_NOT_UNIQUE"
    elif model_key == "lithium_plating":
        direct = result.get("direct_anode_potential_assessment", {})
        screen = result.get("operating_condition_screen", {})
        stripping = result.get("post_charge_stripping_feature", {})
        calculated = {"direct_anode_potential_assessment": direct,
                      "operating_condition_screen": screen,
                      "post_charge_stripping_feature": stripping}
        findings = [direct.get("interpretation", ""), screen.get("interpretation", ""),
                    stripping.get("interpretation", "")]
        required = direct.get("warning", "").splitlines() if direct.get("status") == "INSUFFICIENT_DATA" else []
        limitations += result.get("global_limitations", [])
        ident = "ANODE_POTENTIAL_CRITERION_CONDITIONAL; OPERATING_SCREEN_NOT_CONFIRMATORY"
        status = "EVALUATED" if direct.get("status") == "EVALUATED" or screen.get("status") == "EVALUATED" else "INSUFFICIENT_DATA"
    elif model_key == "sei_growth":
        calculated = {k: v for k, v in result.items() if k not in {"model", "version", "status", "interpretation", "required_validation", "limitations"}}
        findings = [result.get("interpretation", "")]
        required = result.get("required_validation", [])
        limitations += result.get("limitations", [])
        ident = "CONDITIONAL_PARAMETERISED_ESTIMATE_NOT_INDEPENDENT_DETECTION"
        status = result.get("status", "UNKNOWN")
    elif model_key == "electrolyte_decomposition":
        calculated = {k: v for k, v in result.items() if k not in {"model", "version", "status", "interpretation", "required_validation", "limitations"}}
        findings = [result.get("interpretation", "")]
        required = result.get("required_validation", [])
        limitations += result.get("limitations", [])
        ident = "CONDITIONAL_SINGLE_PATHWAY_PARAMETERISED_ESTIMATE"
        status = result.get("status", "UNKNOWN")
    elif model_key == "abnormal_self_discharge":
        calculated = {k: v for k, v in result.items() if k not in {"model", "version", "status", "interpretation", "identifiability", "limitations", "input_integrity_warnings"}}
        findings = [result.get("interpretation", "")]
        required = result.get("input_integrity_warnings", [])
        limitations += result.get("limitations", [])
        ident = result.get("identifiability", "EXCESS_CHARGE_LOSS_NOT_MECHANISM_SPECIFIC")
        status = result.get("status", "UNKNOWN")
    elif model_key == "abnormal_contact_resistance":
        calculated = {k: v for k, v in result.items() if k not in {"model", "version", "status", "interpretation", "limitations"}}
        findings = [result.get("interpretation", "")]
        required = [
            "Four-wire voltage drop across the identified contact",
            "Synchronous current through the same contact",
            "Temperature-matched healthy contact reference"
        ]
        limitations += result.get("limitations", [])
        ident = "IDENTIFIED_CONTACT_RESISTANCE_IF_MEASUREMENT_GEOMETRY_VALID"
        status = result.get("status", "UNKNOWN")

    return {
        "model_id": model_id,
        "status": status,
        "severity": "NOT_ASSESSED",
        "identifiability": ident,
        "findings": [x for x in findings if isinstance(x, str) and x],
        "required_evidence": required,
        "calculated": calculated,
        "limitations": list(dict.fromkeys(x for x in limitations if isinstance(x, str) and x)),
        "native_model_output": result,
    }


def build_heat_ledger(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    contributions = []

    def add(model_key: str, owner: str, location: str,
            power: Optional[float], energy: Optional[float],
            basis: str, status: str, note: str):
        contributions.append({
            "model": MODEL_FILES[model_key][0],
            "physical_owner": owner,
            "location_or_scale": location,
            "power_W": power,
            "energy_J": energy,
            "basis": basis,
            "status": status,
            "aggregation_eligible": False,
            "note": note,
        })

    # Model 01: direct-path electrical dissipation only.
    r = results.get("internal_micro_short", {})
    direct = r.get("native_model_output", {}).get("assessments", {}).get("direct_path_assessment", {})
    heat = direct.get("heat", {}) if isinstance(direct, dict) else {}
    calc = direct.get("calculated", {}) if isinstance(direct, dict) else {}
    power = calc.get("path_power_w")
    energy = calc.get("path_energy_j")
    add("internal_micro_short", "directly_instrumented_conductive_path", "identified path",
        power if isinstance(power, (int, float)) else None,
        energy if isinstance(energy, (int, float)) else None,
        "V_path * I_path; both measurements must refer to same path",
        "IDENTIFIED" if power is not None else "UNAVAILABLE",
        "Electrical path dissipation; physical origin is not uniquely confirmed.")

    # Model 03: only the model's incremental SEI ohmic heat.
    r = results.get("sei_growth", {}).get("native_model_output", {})
    e = r.get("estimated_incremental_sei_ohmic_heat_J")
    add("sei_growth", "incremental_SEI_resistance", "cell-level reduced-order estimate",
        None, e if isinstance(e, (int, float)) and e >= 0 else None,
        "Integrated I^2 * Delta_R_SEI", "CONDITIONAL_ESTIMATE" if e is not None else "UNAVAILABLE",
        "Not total SEI reaction heat; conditional on calibrated SEI parameters.")

    # Model 04: reaction enthalpy; positive is released heat, negative absorbed.
    r = results.get("electrolyte_decomposition", {}).get("native_model_output", {})
    e = r.get("estimated_reaction_heat_J")
    add("electrolyte_decomposition", "specified_electrolyte_decomposition_pathway",
        "cell-level lumped reaction estimate", None,
        e if isinstance(e, (int, float)) and e >= 0 else None,
        "consumed_moles * reaction_enthalpy; signed raw value retained in model output",
        "CONDITIONAL_HEAT_RELEASE_ESTIMATE" if isinstance(e, (int, float)) and e > 0 else
        "CONDITIONAL_HEAT_ABSORPTION_OR_ZERO" if isinstance(e, (int, float)) else "UNAVAILABLE",
        "Only positive released heat is included here; negative/absorbed value remains in native model output.")

    # Model 06: incremental contact heating relative to healthy reference.
    r = results.get("abnormal_contact_resistance", {}).get("native_model_output", {})
    e = r.get("integrated_incremental_contact_heat_J")
    add("abnormal_contact_resistance", "excess_resistance_at_identified_contact",
        "identified external/internal contact", None,
        e if isinstance(e, (int, float)) and e >= 0 else None,
        "Integrated I^2 * max(R_measured - R_healthy, 0)",
        "CONDITIONAL_LOCAL_CONTACT_HEAT" if e is not None else "UNAVAILABLE",
        "Local contact heat; not automatically cell bulk heat.")

    known_energy = [x["energy_J"] for x in contributions
                    if isinstance(x["energy_J"], (int, float)) and x["energy_J"] >= 0]
    return {
        "contributions": contributions,
        "identified_or_conditional_energy_subtotal_J": sum(known_energy),
        "identified_or_conditional_energy_count": len(known_energy),
        "aggregate_cell_heat_J": None,
        "aggregation_status": "NOT_AGGREGATED_POTENTIAL_OVERLAP_AND_SCALE_MISMATCH",
        "warning": [
            "The subtotal is traceability only, not total cell heat.",
            "Contributions span different physical scales and potentially overlapping mechanisms.",
            "No automatic cross-mechanism summation is permitted until physical ownership and time-window compatibility are demonstrated.",
            "Missing heat is not zero; local contact/path heat is not cell-average heat.",
            "Do not combine these estimates with baseline electrothermal heat without a formal double-counting audit."
        ]
    }


def evaluate_packet(raw: Dict[str, Any], base_dir: Optional[Path] = None) -> Dict[str, Any]:
    packet = validate_root(raw)
    modules = load_model_modules(base_dir or Path(__file__).resolve().parent)
    model_reports = {}
    execution_errors = {}

    for key in MODEL_FILES:
        payload = packet["model_inputs"].get(key)
        if payload is None:
            model_reports[key] = {
                "model_id": MODEL_FILES[key][0],
                "status": "INSUFFICIENT_DATA",
                "severity": "NOT_ASSESSED",
                "identifiability": "NOT_IDENTIFIABLE",
                "findings": ["No model-specific evidence payload supplied."],
                "required_evidence": ["Supply this model's documented model-specific input payload."],
                "calculated": {},
                "limitations": ["No missing input is substituted with zero."],
                "native_model_output": {"status": "INSUFFICIENT_DATA"}
            }
            continue
        provenance = payload.get("_provenance")
        if provenance and str(provenance.get("quality", "GOOD")).upper() != "GOOD":
            model_reports[key] = {
                "model_id": MODEL_FILES[key][0],
                "status": "INSUFFICIENT_DATA",
                "severity": "NOT_ASSESSED",
                "identifiability": "NOT_IDENTIFIABLE",
                "findings": [f"Payload provenance quality is {provenance['quality']}; only GOOD data are evaluated."],
                "required_evidence": ["Provide quality-approved model input data."],
                "calculated": {},
                "limitations": ["Payload rejected by unified provenance gate."],
                "native_model_output": {"status": "INSUFFICIENT_DATA"}
            }
            continue
        try:
            native = modules[key].evaluate(strip_provenance(payload))
            model_reports[key] = normalise_result(key, native)
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            execution_errors[key] = str(exc)
            model_reports[key] = {
                "model_id": MODEL_FILES[key][0],
                "status": "INVALID_INPUT",
                "severity": "NOT_ASSESSED",
                "identifiability": "NOT_EVALUATED",
                "findings": [f"Model input rejected: {exc}"],
                "required_evidence": [],
                "calculated": {},
                "limitations": ["Correct this model-specific payload before interpreting the combined report."],
                "native_model_output": {"status": "INVALID_INPUT", "error": str(exc)}
            }

    heat_ledger = build_heat_ledger(model_reports)
    statuses = [v["status"] for v in model_reports.values()]
    overall = "COMPLETED_WITH_EVIDENCE_GATES"
    if execution_errors:
        overall = "COMPLETED_WITH_MODEL_INPUT_ERRORS"
    if all(s == "INSUFFICIENT_DATA" for s in statuses):
        overall = "COMPLETED_NO_MECHANISM_IDENTIFIED"

    return {
        "engine": "UNIFIED_BATTERY_PRECURSOR_EVALUATION_ENGINE",
        "version": VERSION,
        "cell": packet["cell"],
        "context": packet.get("context", {}),
        "evaluation_status": overall,
        "models": list(model_reports.values()),
        "model_execution_errors": execution_errors,
        "heat_ledger": heat_ledger,
        "system_limitations": [
            "This engine orchestrates six research/reference models; it is not a validated diagnostic product.",
            "Native model assumptions, units and limitations remain applicable.",
            "Conditional parameterised estimates are not equivalent to independent defect detection.",
            "No model is experimentally validated for a target cell merely because software tests pass.",
            "No cooling actuation, embedded control or safety function is implemented.",
            "AI/ML is deliberately not part of this integration stage."
        ]
    }


def example_packet() -> Dict[str, Any]:
    # Empty evidence payloads intentionally demonstrate evidence gating.
    return {
        "schema_version": "1.0",
        "cell": {"cell_id": "LFP_EXAMPLE_001", "chemistry": "LFP_GRAPHITE"},
        "context": {
            "evaluation_timestamp_utc": "2026-11-10T00:00:00Z",
            "data_provenance": "synthetic software integration fixture"
        },
        "model_inputs": {}
    }


class UnifiedIntegrationTests(unittest.TestCase):
    def test_empty_packet_runs_six_evidence_gates(self):
        out = evaluate_packet(example_packet())
        self.assertEqual(len(out["models"]), 6)
        self.assertEqual(out["evaluation_status"], "COMPLETED_NO_MECHANISM_IDENTIFIED")
        self.assertIsNone(out["heat_ledger"]["aggregate_cell_heat_J"])

    def test_unknown_model_rejected(self):
        p = example_packet()
        p["model_inputs"]["unknown"] = {}
        with self.assertRaises(ValueError):
            validate_root(p)

    def test_provenance_quality_gate(self):
        p = example_packet()
        p["model_inputs"]["lithium_plating"] = {
            "_provenance": {
                "source": "test", "acquisition_method": "synthetic",
                "timestamp_utc": "2026-11-10T00:00:00Z", "quality": "SUSPECT"
            },
            "samples": []
        }
        out = evaluate_packet(p)
        item = next(x for x in out["models"] if x["model_id"] == "LITHIUM_PLATING")
        self.assertEqual(item["status"], "INSUFFICIENT_DATA")

    def test_all_six_modules_import(self):
        mods = load_model_modules(Path(__file__).resolve().parent)
        self.assertEqual(set(mods), set(MODEL_FILES))

    def test_heat_ledger_does_not_claim_total_cell_heat(self):
        ledger = build_heat_ledger({})
        self.assertIsNone(ledger["aggregate_cell_heat_J"])
        self.assertFalse(any(x["aggregation_eligible"] for x in ledger["contributions"]))


def main():
    parser = argparse.ArgumentParser(description="Unified six-model Battery Precursor Engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path)
    group.add_argument("--example", action="store_true")
    group.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(UnifiedIntegrationTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    try:
        raw = example_packet() if args.example else json.loads(args.input.read_text(encoding="utf-8"))
        report = evaluate_packet(raw)
        rendered = json.dumps(report, indent=2, allow_nan=False)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\\n", encoding="utf-8")
            print(f"\\nSaved: {args.output}")
        return 0
    except (OSError, json.JSONDecodeError, ValueError, TypeError, ImportError) as exc:
        print(f"ENGINE ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
