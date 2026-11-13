#!/usr/bin/env python3
"""Model 06 — Abnormal Contact Resistance adapter for Unified Data Interface v1.0."""

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


def validate_model06_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        raise ValueError("model_evidence.model_06.payload must be a JSON object.")

    contact = payload.get("contact")
    samples = payload.get("samples")
    if not isinstance(contact, dict):
        raise ValueError("Model 06 payload requires a 'contact' object.")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("Model 06 payload requires at least two 'samples'.")

    required_contact = (
        "contact_id",
        "minimum_current_for_resistance_A",
        "maximum_temperature_span_C",
        "healthy_reference",
    )
    missing_contact = [key for key in required_contact if key not in contact]
    if missing_contact:
        raise ValueError("Missing Model 06 contact fields: " + ", ".join(missing_contact))

    reference = contact.get("healthy_reference")
    if not isinstance(reference, dict):
        raise ValueError("contact.healthy_reference must be an object.")
    required_reference = (
        "resistance_ohm",
        "resistance_sigma_ohm",
        "temperature_C",
    )
    missing_reference = [key for key in required_reference if key not in reference]
    if missing_reference:
        raise ValueError(
            "Missing healthy-reference fields: " + ", ".join(missing_reference)
        )

    good_samples = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {index} must be a JSON object.")
        if str(sample.get("quality", "GOOD")).upper() in {"GOOD", "VALID", "OK"}:
            good_samples += 1
            for key in (
                "time_s",
                "current_A",
                "contact_voltage_drop_V",
                "current_sigma_A",
                "voltage_sigma_V",
                "contact_temperature_C",
            ):
                if key not in sample or not finite_number(sample[key]):
                    raise ValueError(
                        f"Valid sample {index} requires finite numeric '{key}'."
                    )
    if good_samples < 2:
        raise ValueError(
            "Fewer than two samples have GOOD/VALID/OK quality; "
            "the native model needs at least two valid samples."
        )

    return [
        "Contact voltage must be measured directly across the identified joint; "
        "terminal cell or pack voltage is not a substitute.",
        "Current must be measured in the same electrical path and time-aligned "
        "with the contact-voltage measurement.",
        "A four-wire/Kelvin measurement is preferred; the healthy reference must "
        "match temperature, current direction, ageing, clamping force and geometry.",
        "Incremental contact heat is local to the identified contact and is not "
        "cell bulk heat or a cell hotspot prediction."
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Model 06 — Abnormal Contact Resistance Unified Data Interface adapter"
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
        default=Path(__file__).with_name("Model06_Abnormal_Contact_Resistance_Final.py"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        packet = json.loads(args.input.read_text(encoding="utf-8"))
        interface = load_module(args.interface, "unified_data_interface_v1_model06")
        model = load_module(args.model, "model06_abnormal_contact_resistance_native")

        interface_report = interface.validate_packet(packet)
        if not isinstance(interface_report, dict):
            raise ValueError("Unified interface returned an unexpected validation result.")

        evidence_root = packet.get("model_evidence", {})
        evidence = evidence_root.get("model_06", {}) if isinstance(evidence_root, dict) else {}
        if not isinstance(evidence, dict) or evidence.get("enabled") is not True:
            raise ValueError("Enable Model 06 using model_evidence.model_06.enabled = true.")

        provenance = evidence.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("model_evidence.model_06.provenance must be an object.")
        quality = str(provenance.get("quality", "UNKNOWN")).upper()
        if quality != "GOOD":
            raise ValueError(
                "Model 06 evidence quality must be GOOD to execute; "
                f"received '{quality}'."
            )

        payload = evidence.get("payload")
        adapter_warnings = validate_model06_payload(payload)
        model_output = model.evaluate(payload)

        result = {
            "adapter": "MODEL_06_ADAPTER",
            "adapter_version": "1.0.0",
            "packet_id": packet.get("packet_id"),
            "adapter_status": "PASSED_TO_MODEL",
            "model_evaluated": True,
            "interface_validation": interface_report,
            "adapter_warnings": adapter_warnings,
            "model_output": model_output,
            "note": (
                "Native Model 06 payload passed unchanged. Contact resistance is "
                "evaluated from voltage drop across the identified contact and "
                "current through the same path."
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
