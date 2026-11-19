#!/usr/bin/env python3
"""Model 04 — Electrolyte Decomposition adapter for Unified Data Interface v1.0."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution for compatibility with dataclasses/type resolution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_model04_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        raise ValueError("model_evidence.model_04.payload must be a JSON object.")

    parameters = payload.get("parameters")
    samples = payload.get("samples")
    if not isinstance(parameters, dict):
        raise ValueError("Model 04 payload requires a 'parameters' object.")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("Model 04 payload requires at least two 'samples'.")

    required_parameters = (
        "initial_concentration_mol_m3",
        "electrolyte_volume_m3",
        "rate_constant_ref_s",
        "activation_energy_J_mol",
        "reference_temperature_K",
        "transfer_coefficient",
        "electron_number",
        "reaction_enthalpy_J_mol",
    )
    missing = [key for key in required_parameters if key not in parameters]
    if missing:
        raise ValueError("Missing Model 04 parameters: " + ", ".join(missing))

    good_samples = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {index} must be a JSON object.")
        quality = str(sample.get("quality", "GOOD")).upper()
        if quality in {"GOOD", "VALID", "OK"}:
            good_samples += 1
            for key in ("time_s", "temperature_K", "decomposition_overpotential_V"):
                if key not in sample or not finite_number(sample[key]):
                    raise ValueError(
                        f"Valid sample {index} requires finite numeric '{key}'."
                    )
    if good_samples < 2:
        raise ValueError(
            "Fewer than two samples have GOOD/VALID/OK quality; "
            "the native model will not have enough valid samples."
        )

    warnings = [
        "Model 04 is a conditional parameterised estimate for one calibrated "
        "decomposition pathway; it does not independently detect decomposition.",
        "decomposition_overpotential_V must use the calibrated reaction-driving "
        "sign/reference convention and must not be substituted with terminal cell voltage.",
        "Reaction heat is signed; positive indicates heat release and negative indicates "
        "heat absorption. Check the baseline heat ledger before aggregation."
    ]
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Model 04 — Electrolyte Decomposition Unified Data Interface adapter"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--interface",
        type=Path,
        default=Path(__file__).with_name("Unified_Data_Interface_v1.py"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("Model04_Electrolyte_Decomposition_Final.py"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        packet = json.loads(args.input.read_text(encoding="utf-8"))
        interface = load_module(args.interface, "unified_data_interface_v1_model04")
        model = load_module(args.model, "model04_electrolyte_decomposition_native")

        interface_report = interface.validate_packet(packet)
        if not isinstance(interface_report, dict):
            raise ValueError("Unified interface returned an unexpected validation result.")

        evidence_root = packet.get("model_evidence", {})
        evidence = evidence_root.get("model_04", {}) if isinstance(evidence_root, dict) else {}
        if not isinstance(evidence, dict) or evidence.get("enabled") is not True:
            raise ValueError("Enable Model 04 using model_evidence.model_04.enabled = true.")

        provenance = evidence.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("model_evidence.model_04.provenance must be an object.")
        quality = str(provenance.get("quality", "UNKNOWN")).upper()
        if quality != "GOOD":
            raise ValueError(
                "Model 04 evidence quality must be GOOD to execute; "
                f"received '{quality}'."
            )

        payload = evidence.get("payload")
        adapter_warnings = validate_model04_payload(payload)
        model_output = model.evaluate(payload)

        result = {
            "adapter": "MODEL_04_ADAPTER",
            "adapter_version": "1.0.0",
            "packet_id": packet.get("packet_id"),
            "adapter_status": "PASSED_TO_MODEL",
            "model_evaluated": True,
            "interface_validation": interface_report,
            "adapter_warnings": adapter_warnings,
            "model_output": model_output,
            "note": (
                "Native Model 04 payload passed unchanged. Estimates are conditional "
                "on the specified calibrated reaction pathway, kinetic parameters and "
                "overpotential convention."
            ),
        }

        rendered = json.dumps(result, indent=2, allow_nan=False)
        print(rendered)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nSaved: {args.output}")
        return 0

    except Exception as exc:
        print(f"Adapter error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
