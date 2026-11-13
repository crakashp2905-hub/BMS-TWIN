#!/usr/bin/env python3

"""
Predictive-BTMS
Model 01 Adapter — Internal Micro-short
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from pathlib import Path
from typing import Any


# --------------------------------------------------
# DYNAMIC MODULE LOADER
# --------------------------------------------------

def load_module(path: Path):

    spec = importlib.util.spec_from_file_location(
        "model01_frozen",
        path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load Model 01 module: {path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Required for Python 3.13 dataclass compatibility
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)

    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


# --------------------------------------------------
# MODEL 01 EVIDENCE VALIDATION
# --------------------------------------------------

def validate_model01_payload(
    payload: Any,
    provenance: Any
):

    reasons = []

    if not isinstance(payload, dict):

        return False, [
            "model_01.payload must be an object"
        ]

    if not isinstance(provenance, dict):

        reasons.append(
            "model_01.provenance is required"
        )

    elif str(
        provenance.get("quality", "")
    ).upper() != "GOOD":

        reasons.append(
            "model_01.provenance.quality must be GOOD"
        )

    recognized = False

    # ----------------------------------------------
    # DIRECT PATH EVIDENCE
    # ----------------------------------------------

    direct = payload.get("direct_path")

    if direct is not None:

        recognized = True

        if not isinstance(direct, dict):

            reasons.append(
                "direct_path must be an object or null"
            )

        else:

            required = [
                "path_voltage_v",
                "path_voltage_sigma_v",
                "path_current_a",
                "path_current_sigma_a",
                "quality"
            ]

            missing = [
                key for key in required
                if direct.get(key) is None
            ]

            if missing:

                reasons.append(
                    "direct_path missing: "
                    + ", ".join(missing)
                )

            if str(
                direct.get("quality", "")
            ).upper() != "GOOD":

                reasons.append(
                    "direct_path quality must be GOOD"
                )

    # ----------------------------------------------
    # REST WINDOW EVIDENCE
    # ----------------------------------------------

    rest = payload.get("rest_window")

    if rest is not None:

        recognized = True

        if not isinstance(rest, dict):

            reasons.append(
                "rest_window must be an object or null"
            )

        else:

            required = [
                "samples",
                "capacity_ah",
                "capacity_sigma_ah",
                "healthy_baseline_self_discharge_a",
                "healthy_baseline_self_discharge_sigma_a",
                "minimum_rest_duration_s",
                "maximum_abs_external_current_a",
                "maximum_temperature_span_k"
            ]

            missing = [
                key for key in required
                if rest.get(key) is None
            ]

            if missing:

                reasons.append(
                    "rest_window missing: "
                    + ", ".join(missing)
                )

            samples = rest.get("samples")

            if not isinstance(samples, list) or len(samples) < 2:

                reasons.append(
                    "rest_window.samples must contain "
                    "at least two samples"
                )

            else:

                for i, sample in enumerate(samples):

                    if not isinstance(sample, dict):

                        reasons.append(
                            f"rest_window.samples[{i}] "
                            "must be an object"
                        )

                        continue

                    if sample.get(
                        "soc_independent_percent"
                    ) is None:

                        reasons.append(
                            f"rest_window.samples[{i}] "
                            "lacks independent SOC"
                        )

                    if str(
                        sample.get("quality", "")
                    ).upper() != "GOOD":

                        reasons.append(
                            f"rest_window.samples[{i}] "
                            "quality must be GOOD"
                        )

    if not recognized:

        reasons.append(
            "payload must contain direct_path "
            "and/or rest_window"
        )

    return len(reasons) == 0, reasons


# --------------------------------------------------
# MAIN ADAPTER
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Model 01 Adapter"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--model",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path
    )

    args = parser.parse_args()

    # Load unified packet
    packet = json.loads(
        args.input.read_text(encoding="utf-8")
    )

    # Load unified interface
    interface_path = Path(__file__).with_name(
        "Unified_Data_Interface_v1.py"
    )

    interface = load_module(interface_path)

    interface_report = interface.validate_packet(
        packet
    )

    # Retrieve Model 01 evidence
    block = packet.get(
        "model_evidence", {}
    ).get(
        "model_01", {}
    )

    # ----------------------------------------------
    # EVIDENCE GATE
    # ----------------------------------------------

    if (
        not isinstance(block, dict)
        or block.get("enabled") is not True
    ):

        result = {

            "adapter": "MODEL_01_ADAPTER",

            "adapter_version": "1.0.0",

            "packet_id": packet.get("packet_id"),

            "adapter_status": "INSUFFICIENT_DATA",

            "reason": (
                "model_evidence.model_01 "
                "is not enabled"
            ),

            "model_evaluated": False
        }

    else:

        payload = block.get("payload")

        valid, reasons = validate_model01_payload(
            payload,
            block.get("provenance")
        )

        if not valid:

            result = {

                "adapter": "MODEL_01_ADAPTER",

                "adapter_version": "1.0.0",

                "packet_id": packet.get("packet_id"),

                "adapter_status": "INSUFFICIENT_DATA",

                "reasons": reasons,

                "model_evaluated": False
            }

        else:

            # Load frozen physics model
            model = load_module(args.model)

            # Execute physics assessment
            model_result = model.evaluate(payload)

            result = {

                "adapter": "MODEL_01_ADAPTER",

                "adapter_version": "1.0.0",

                "packet_id": packet.get("packet_id"),

                "adapter_status": "PASSED_TO_MODEL",

                "model_evaluated": True,

                "interface_validation": interface_report,

                "model_output": model_result,

                "note": (
                    "Explicit model-specific evidence "
                    "passed to frozen Model 01."
                )
            }

    # ----------------------------------------------
    # SAVE OUTPUT
    # ----------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    args.output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":

    main()