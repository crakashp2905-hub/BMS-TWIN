#!/usr/bin/env python3
"""Predictive-BTMS six-adapter orchestration layer v1.0.0.

Consumes Unified Data Interface v1.0 packets, invokes the six existing adapter
CLIs unchanged, and uses the existing engine's result normalisation and heat
ledger. It does not map generic measurements into model-specific evidence.
"""
from __future__ import annotations
import argparse, importlib.util, json, subprocess, sys, tempfile
from pathlib import Path
from typing import Any

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
INTERFACE_PATH = HERE / "Unified_Data_Interface_v1.py"
ENGINE_PATH = HERE / "Unified_Battery_Precursor_Evaluation_Engine_v0_2.py"
ADAPTERS = {
    "model_01": ("internal_micro_short", "Model01_Adapter_v1.py", "Model01_Internal_MicroShort_Final.py"),
    "model_02": ("lithium_plating", "Model02_Adapter_v1.py", "Model02_Lithium_Plating_Final.py"),
    "model_03": ("sei_growth", "Model03_Adapter_v1.py", "Model03_SEI_Growth_Final.py"),
    "model_04": ("electrolyte_decomposition", "Model04_Adapter_v1.py", "Model04_Electrolyte_Decomposition_Final.py"),
    "model_05": ("abnormal_self_discharge", "Model05_Adapter_v1.py", "Model05_Abnormal_Self_Discharge_Final.py"),
    "model_06": ("abnormal_contact_resistance", "Model06_Adapter_v1.py", "Model06_Abnormal_Contact_Resistance_Final.py"),
}

def load_py(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def insufficient(model_id: str, reason: str, required=None):
    return {
        "model_id": model_id, "status": "INSUFFICIENT_DATA",
        "severity": "NOT_ASSESSED", "identifiability": "NOT_IDENTIFIABLE",
        "findings": [reason], "required_evidence": required or [],
        "calculated": {}, "limitations": ["No missing input is substituted with zero."],
        "native_model_output": {"status": "INSUFFICIENT_DATA"},
    }

def run_one(adapter_key: str, packet_path: Path, packet: dict, temp_dir: Path,
            engine: Any) -> tuple[dict, dict]:
    model_key, adapter_file, native_file = ADAPTERS[adapter_key]
    model_id = engine.MODEL_FILES[model_key][0]
    block = packet["model_evidence"].get(adapter_key, {})
    if not isinstance(block, dict) or block.get("enabled") is not True:
        return insufficient(model_id, f"{adapter_key} is disabled; no model evaluation performed."), {
            "adapter_status": "DISABLED", "model_evaluated": False
        }
    prov = block.get("provenance")
    if not isinstance(prov, dict) or str(prov.get("quality", "UNKNOWN")).upper() != "GOOD":
        q = prov.get("quality", "MISSING") if isinstance(prov, dict) else "MISSING"
        return insufficient(model_id, f"Evidence gate: {adapter_key} provenance quality is {q}; adapter not executed.",
                            ["Provide model-specific evidence with provenance quality GOOD."]), {
            "adapter_status": "BLOCKED_BY_PROVENANCE_GATE", "model_evaluated": False,
            "provenance_quality": q,
        }
    adapter_path = HERE / adapter_file
    native_path = HERE / native_file
    if not adapter_path.is_file() or not native_path.is_file():
        raise FileNotFoundError(f"Missing adapter or native model for {adapter_key}: {adapter_path}, {native_path}")
    out_path = temp_dir / f"{adapter_key}_adapter_output.json"
    cmd = [sys.executable, str(adapter_path), "--input", str(packet_path), "--output", str(out_path)]
    if adapter_key in {"model_01", "model_02"}:
        cmd += ["--model", str(native_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or f"Adapter exited {proc.returncode}").strip()
        return {
            "model_id": model_id, "status": "ADAPTER_EXECUTION_ERROR",
            "severity": "NOT_ASSESSED", "identifiability": "NOT_EVALUATED",
            "findings": [msg], "required_evidence": [], "calculated": {},
            "limitations": ["Adapter did not complete; no model conclusion is available."],
            "native_model_output": {"status": "ADAPTER_EXECUTION_ERROR", "error": msg},
        }, {"adapter_status": "EXECUTION_ERROR", "model_evaluated": False,
            "return_code": proc.returncode, "stderr": proc.stderr[-4000:]}
    if not out_path.is_file():
        raise RuntimeError(f"{adapter_key} exited successfully but did not create its output report.")
    adapter_result = json.loads(out_path.read_text(encoding="utf-8"))
    if adapter_result.get("model_evaluated") is not True:
        return insufficient(model_id, adapter_result.get("reason") or "; ".join(adapter_result.get("reasons", [])) or "Adapter did not evaluate the model.", adapter_result.get("required_evidence")), adapter_result
    native = adapter_result.get("model_output")
    if not isinstance(native, dict):
        raise TypeError(f"{adapter_key} adapter did not return a model_output object.")
    normalised = engine.normalise_result(model_key, native)
    normalised["adapter_status"] = adapter_result.get("adapter_status")
    normalised["adapter_warnings"] = adapter_result.get("adapter_warnings", [])
    normalised["adapter_note"] = adapter_result.get("note")
    return normalised, adapter_result

def evaluate_packet(packet: dict) -> dict:
    interface = load_py(INTERFACE_PATH, "btms_interface_for_orchestrator")
    engine = load_py(ENGINE_PATH, "btms_existing_engine_for_orchestrator")
    interface_report = interface.validate_packet(packet)
    # Use a temporary on-disk copy because the frozen adapter CLIs consume JSON paths.
    model_reports, adapter_reports = {}, {}
    with tempfile.TemporaryDirectory(prefix="btms_orchestrator_") as td:
        temp_dir = Path(td)
        packet_path = temp_dir / "canonical_packet.json"
        packet_path.write_text(json.dumps(packet, allow_nan=False), encoding="utf-8")
        for evidence_key, (engine_key, _, _) in ADAPTERS.items():
            report, adapter_report = run_one(evidence_key, packet_path, packet, temp_dir, engine)
            model_reports[engine_key] = report
            adapter_reports[evidence_key] = adapter_report
    heat_ledger = engine.build_heat_ledger(model_reports)
    statuses = [r["status"] for r in model_reports.values()]
    execution_errors = {k: r.get("findings", []) for k, r in model_reports.items()
                        if r["status"] in {"ADAPTER_EXECUTION_ERROR", "INVALID_INPUT"}}
    overall = "COMPLETED_WITH_EVIDENCE_GATES"
    if execution_errors:
        overall = "COMPLETED_WITH_ADAPTER_OR_MODEL_ERRORS"
    elif all(s == "INSUFFICIENT_DATA" for s in statuses):
        overall = "COMPLETED_NO_MECHANISM_IDENTIFIED"
    elif any(s == "INSUFFICIENT_DATA" for s in statuses):
        overall = "COMPLETED_WITH_PARTIAL_EVIDENCE"
    return {
        "engine": "PREDICTIVE_BTMS_UNIFIED_SIX_ADAPTER_ORCHESTRATOR",
        "version": VERSION,
        "interface_version": interface.VERSION,
        "source_engine_version": engine.VERSION,
        "packet_id": packet.get("packet_id"),
        "cell": packet.get("cell"),
        "acquisition": packet.get("acquisition"),
        "interface_validation": interface_report,
        "evaluation_status": overall,
        "models": [model_reports[k] for k, _, _ in ADAPTERS.values()],
        "adapter_execution": adapter_reports,
        "model_execution_errors": execution_errors,
        "heat_ledger": heat_ledger,
        "system_limitations": [
            "This is research/reference orchestration, not validated diagnostic firmware.",
            "The six existing adapters and native models are invoked without changing their source contracts.",
            "Only explicit model-specific evidence payloads are passed; generic measurements are not automatically mapped.",
            "A software integration pass using synthetic fixtures is not experimental validation.",
            "Conditional parameterised estimates are not equivalent to independent defect detection.",
            "Heat contributions are not aggregated into total cell heat; local and cell-level scales remain distinct.",
            "No cooling actuation, embedded control, safety function or AI/ML is implemented.",
        ],
    }

def main():
    ap = argparse.ArgumentParser(description="Run all six Predictive-BTMS adapters from one Unified Data Interface packet.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    try:
        packet = json.loads(args.input.read_text(encoding="utf-8"))
        report = evaluate_packet(packet)
        rendered = json.dumps(report, indent=2, allow_nan=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        print(f"\nSaved: {args.output}")
        return 0
    except Exception as exc:
        print(f"ORCHESTRATOR ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
