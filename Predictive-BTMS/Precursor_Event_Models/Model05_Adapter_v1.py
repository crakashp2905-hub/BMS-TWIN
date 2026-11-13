#!/usr/bin/env python3
"""Model 05 — Abnormal Self-Discharge adapter for Unified Data Interface v1.0."""

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


def validate_model05_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        raise ValueError("model_evidence.model_05.payload must be a JSON object.")

    rest = payload.get("rest_window")
    baseline = payload.get("healthy_baseline")
    if not isinstance(rest, dict) or not isinstance(baseline, dict):
        raise ValueError(
            "Model 05 payload requires 'rest_window' and 'healthy_baseline' objects."
        )

    required_rest = (
        "start_time_s",
        "end_time_s",
        "soc_start_pct",
        "soc_end_pct",
        "soc_start_sigma_pct",
        "soc_end_sigma_pct",
        "usable_capacity_Ah",
        "usable_capacity_sigma_Ah",
        "temperature_start_C",
        "temperature_end_C",
        "maximum_temperature_span_C",
        "maximum_absolute_external_current_A",
        "external_current_samples",
    )
    missing_rest = [key for key in required_rest if key not in rest]
    if missing_rest:
        raise ValueError("Missing Model 05 rest_window fields: " + ", ".join(missing_rest))

    required_baseline = (
        "healthy_self_discharge_current_A",
        "healthy_self_discharge_current_sigma_A",
        "reference_temperature_C",
        "maximum_baseline_temperature_difference_C",
    )
    missing_baseline = [key for key in required_baseline if key not in baseline]
    if missing_baseline:
        raise ValueError(
            "Missing Model 05 healthy_baseline fields: " + ", ".join(missing_baseline)
        )

    samples = rest["external_current_samples"]
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("external_current_samples must contain at least two samples.")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"External-current sample {index} must be an object.")
        if str(sample.get("quality", "GOOD")).upper() in {"GOOD", "VALID", "OK"}:
            for key in ("time_s", "current_A", "current_sigma_A"):
                if key not in sample or not finite_number(sample[key]):
                    raise ValueError(
                        f"Valid external-current sample {index} requires finite '{key}'."
                    )

    warnings = [
        "SOC endpoints must be independent of the current integration being audited; "
        "otherwise the charge-loss estimate is circular.",
        "Healthy self-discharge baseline must be matched to the evaluated cell for "
        "chemistry, design, age/SOH, SOC, temperature and rest protocol.",
        "Excess rest-period charge loss is not mechanism-specific and does not uniquely "
        "identify an internal micro-short.",
        "Terminal-voltage relaxation is descriptive only and is not converted into "
        "leakage current or charge loss."
    ]
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Model 05 — Abnormal Self-Discharge Unified Data Interface adapter"
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
        default=Path(__file__).with_name("Model05_Abnormal_Self_Discharge_Final.py"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        packet = json.loads(args.input.read_text(encoding="utf-8"))
        interface = load_module(args.interface, "unified_data_interface_v1_model05")
        model = load_module(args.model, "model05_abnormal_self_discharge_native")

        interface_report = interface.validate_packet(packet)
        if not isinstance(interface_report, dict):
            raise ValueError("Unified interface returned an unexpected validation result.")

        evidence_root = packet.get("model_evidence", {})
        evidence = evidence_root.get("model_05", {}) if isinstance(evidence_root, dict) else {}
        if not isinstance(evidence, dict) or evidence.get("enabled") is not True:
            raise ValueError("Enable Model 05 using model_evidence.model_05.enabled = true.")

        provenance = evidence.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("model_evidence.model_05.provenance must be an object.")
        quality = str(provenance.get("quality", "UNKNOWN")).upper()
        if quality != "GOOD":
            raise ValueError(
                "Model 05 evidence quality must be GOOD to execute; "
                f"received '{quality}'."
            )

        payload = evidence.get("payload")
        adapter_warnings = validate_model05_payload(payload)
        model_output = model.evaluate(payload)

        result = {
            "adapter": "MODEL_05_ADAPTER",
            "adapter_version": "1.0.0",
            "packet_id": packet.get("packet_id"),
            "adapter_status": "PASSED_TO_MODEL",
            "model_evaluated": True,
            "interface_validation": interface_report,
            "adapter_warnings": adapter_warnings,
            "model_output": model_output,
            "note": (
                "Native Model 05 payload passed unchanged. Excess rest-period charge "
                "loss is evaluated relative to the supplied matched healthy baseline."
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
