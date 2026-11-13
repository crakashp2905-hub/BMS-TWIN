#!/usr/bin/env python3
"""Model 03 — SEI Growth adapter for Unified Data Interface v1.0."""

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
    # Required for dataclass/type resolution on newer Python versions.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_model03_payload(payload: Any) -> list[str]:
    warnings = []
    if not isinstance(payload, dict):
        raise ValueError("model_evidence.model_03.payload must be a JSON object.")

    parameters = payload.get("parameters")
    samples = payload.get("samples")
    if not isinstance(parameters, dict):
        raise ValueError("Model 03 payload requires a 'parameters' object.")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("Model 03 payload requires at least two 'samples'.")

    required_parameters = (
        "initial_thickness_m",
        "parabolic_rate_ref_m2_s",
        "activation_energy_J_mol",
        "reference_temperature_K",
        "active_area_m2",
        "sei_ionic_conductivity_S_m",
        "sei_capacity_per_volume_Ah_m3",
        "nominal_capacity_Ah",
    )
    missing = [key for key in required_parameters if key not in parameters]
    if missing:
        raise ValueError("Missing Model 03 parameters: " + ", ".join(missing))

    good_samples = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {index} must be a JSON object.")
        quality = str(sample.get("quality", "GOOD")).upper()
        if quality in {"GOOD", "VALID", "OK"}:
            good_samples += 1
            for key in ("time_s", "temperature_K", "current_A"):
                if key not in sample or not finite_number(sample[key]):
                    raise ValueError(
                        f"Valid sample {index} requires finite numeric '{key}'."
                    )
    if good_samples < 2:
        warnings.append(
            "Fewer than two samples have GOOD/VALID/OK quality; "
            "the native model may reject the payload."
        )

    # Model 03 estimates growth conditionally from supplied calibrated parameters.
    warnings.append(
        "Model 03 produces a conditional parameterised SEI-growth estimate; "
        "it does not independently detect an SEI event."
    )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Model 03 — SEI Growth Unified Data Interface adapter"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--interface", type=Path, default=Path(__file__).with_name("Unified_Data_Interface_v1.py"))
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("Model03_SEI_Growth_Final.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        packet = json.loads(args.input.read_text(encoding="utf-8"))
        interface = load_module(args.interface, "unified_data_interface_v1_model03")
        model = load_module(args.model, "model03_sei_growth_native")

        interface_report = interface.validate_packet(packet)
        if not isinstance(interface_report, dict):
            raise ValueError("Unified interface returned an unexpected validation result.")

        evidence_root = packet.get("model_evidence", {})
        evidence = evidence_root.get("model_03", {}) if isinstance(evidence_root, dict) else {}
        if not isinstance(evidence, dict) or evidence.get("enabled") is not True:
            raise ValueError("Enable Model 03 using model_evidence.model_03.enabled = true.")

        provenance = evidence.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("model_evidence.model_03.provenance must be an object.")
        quality = str(provenance.get("quality", "UNKNOWN")).upper()
        if quality != "GOOD":
            raise ValueError(
                "Model 03 evidence quality must be GOOD to execute; "
                f"received '{quality}'."
            )

        payload = evidence.get("payload")
        adapter_warnings = validate_model03_payload(payload)

        model_output = model.evaluate(payload)

        result = {
            "adapter": "MODEL_03_ADAPTER",
            "adapter_version": "1.0.0",
            "packet_id": packet.get("packet_id"),
            "adapter_status": "PASSED_TO_MODEL",
            "model_evaluated": True,
            "interface_validation": interface_report,
            "adapter_warnings": adapter_warnings,
            "model_output": model_output,
            "note": (
                "Native Model 03 payload passed unchanged. Results are conditional "
                "on the supplied calibrated SEI parameters and temperature history."
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
