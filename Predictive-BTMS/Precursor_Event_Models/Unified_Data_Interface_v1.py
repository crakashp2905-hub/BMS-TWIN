#!/usr/bin/env python3
"""
Predictive-BTMS Unified Data Interface v1.0
A source-independent ingestion and validation contract for battery precursor models.

This layer validates and normalises acquired measurements, independently estimated
states, cell parameters, provenance, uncertainty and model-specific evidence blocks.
It intentionally does NOT infer missing values, fabricate measurements, or silently
convert unsupported evidence into model inputs.
"""
from __future__ import annotations
import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "1.0.0"
VALID_QUALITY = {"GOOD", "SUSPECT", "BAD", "STALE", "UNKNOWN"}
VALID_SOURCE_TYPES = {"SENSOR", "BMS", "ESTIMATOR", "LAB", "SIMULATION", "OPERATOR", "UNKNOWN"}

# Canonical physical units. Conversion is intentionally not implicit in v1.0.
ALLOWED_UNITS = {
    "V", "mV", "A", "mA", "uA", "K", "degC", "s", "ms", "h",
    "Ah", "mAh", "ohm", "mohm", "W", "mW", "J", "mJ",
    "W/m3", "J/m3", "m2", "m3", "mol/m3", "mol", "V/s",
    "percent", "1", "A/m2", "S/m", "J/mol", "J/K", "W/K",
    "V/A", "A*s", "C"
}

REQUIRED_TOP = {"schema_version", "packet_id", "cell", "acquisition", "measurements",
                "estimated_states", "cell_parameters", "model_evidence"}

def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

def parse_time(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: timestamp_utc must be a non-empty ISO-8601 string")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{label}: invalid ISO-8601 timestamp {value!r}")
    if dt.tzinfo is None:
        raise ValueError(f"{label}: timestamp must include timezone, preferably UTC Z")
    return dt

def validate_common_record(record, label, require_value=True):
    if not isinstance(record, dict):
        raise ValueError(f"{label}: expected object")
    required = {"name", "value", "unit", "timestamp_utc", "source", "quality"}
    if require_value and not required.issubset(record):
        raise ValueError(f"{label}: missing fields {sorted(required - set(record))}")
    if not isinstance(record.get("name"), str) or not record["name"].strip():
        raise ValueError(f"{label}: name must be non-empty")
    if require_value and not is_number(record.get("value")):
        raise ValueError(f"{label}: value must be a finite number; use null only for explicitly optional fields")
    if record.get("unit") not in ALLOWED_UNITS:
        raise ValueError(f"{label}: unsupported/missing unit {record.get('unit')!r}")
    parse_time(record.get("timestamp_utc"), label)
    src = record.get("source")
    if not isinstance(src, dict) or not isinstance(src.get("type"), str):
        raise ValueError(f"{label}: source.type is required")
    if src["type"].upper() not in VALID_SOURCE_TYPES:
        raise ValueError(f"{label}: source.type must be one of {sorted(VALID_SOURCE_TYPES)}")
    if not isinstance(src.get("id"), str) or not src["id"].strip():
        raise ValueError(f"{label}: source.id is required (sensor/channel/estimator/lab identifier)")
    q = str(record.get("quality", "UNKNOWN")).upper()
    if q not in VALID_QUALITY:
        raise ValueError(f"{label}: invalid quality {q!r}")
    if "sigma" in record and record["sigma"] is not None:
        if not is_number(record["sigma"]) or record["sigma"] < 0:
            raise ValueError(f"{label}: sigma must be a finite non-negative number")
        if not record.get("uncertainty_method"):
            raise ValueError(f"{label}: uncertainty_method required when sigma is supplied")
    return q

def validate_packet(packet):
    errors, warnings = [], []
    if not isinstance(packet, dict):
        raise ValueError("Root packet must be a JSON object")
    missing = REQUIRED_TOP - set(packet)
    if missing:
        raise ValueError(f"Missing top-level fields: {sorted(missing)}")
    if packet["schema_version"] != VERSION:
        raise ValueError(f"schema_version must be {VERSION!r}")
    if not isinstance(packet["packet_id"], str) or not packet["packet_id"].strip():
        raise ValueError("packet_id must be non-empty")
    cell = packet["cell"]
    if not isinstance(cell, dict) or not cell.get("cell_id") or not cell.get("chemistry"):
        raise ValueError("cell requires cell_id and chemistry")
    acq = packet["acquisition"]
    if not isinstance(acq, dict) or not acq.get("session_id") or not acq.get("started_at_utc"):
        raise ValueError("acquisition requires session_id and started_at_utc")
    parse_time(acq["started_at_utc"], "acquisition.started_at_utc")

    for group in ("measurements", "estimated_states", "cell_parameters"):
        if not isinstance(packet[group], list):
            raise ValueError(f"{group} must be a list")
        names = set()
        for i, rec in enumerate(packet[group]):
            label = f"{group}[{i}]"
            q = validate_common_record(rec, label)
            key = (rec["name"], rec["timestamp_utc"])
            if key in names:
                errors.append(f"{label}: duplicate name/timestamp pair")
            names.add(key)
            if q != "GOOD":
                warnings.append(f"{label}: quality={q}; do not use as trusted evidence without model-specific policy")
            if group == "estimated_states":
                if not rec.get("estimator"):
                    errors.append(f"{label}: estimated state requires estimator metadata")
                if rec.get("independent_of_current_integration") is False and rec["name"].lower() in {
                    "soc", "soc_pct", "state_of_charge"
                }:
                    warnings.append(f"{label}: SOC is not independent of current integration; do not use for rest-loss inference")
            if group == "cell_parameters" and not rec.get("parameter_status"):
                errors.append(f"{label}: cell parameter requires parameter_status (MEASURED, DATASHEET, IDENTIFIED, PROVISIONAL)")

    evidence = packet["model_evidence"]
    if not isinstance(evidence, dict):
        raise ValueError("model_evidence must be an object")
    allowed_models = {f"model_0{i}" for i in range(1,7)}
    unknown = set(evidence) - allowed_models
    if unknown:
        errors.append(f"Unknown model_evidence keys: {sorted(unknown)}")
    for model_key, block in evidence.items():
        if not isinstance(block, dict):
            errors.append(f"{model_key}: evidence block must be an object")
            continue
        if "enabled" not in block or not isinstance(block["enabled"], bool):
            errors.append(f"{model_key}: enabled boolean is required")
        if block.get("enabled") and not isinstance(block.get("payload"), dict):
            errors.append(f"{model_key}: enabled evidence requires payload object")
        prov = block.get("provenance")
        if prov is not None and not isinstance(prov, dict):
            errors.append(f"{model_key}: provenance must be an object")

    if errors:
        raise ValueError("Packet validation failed:\n - " + "\n - ".join(errors))
    return {"valid": True, "warnings": warnings,
            "counts": {k: len(packet[k]) for k in ("measurements", "estimated_states", "cell_parameters")},
            "enabled_models": [k for k,v in evidence.items() if isinstance(v,dict) and v.get("enabled") is True]}

def build_model_payloads(packet):
    """
    v1.0 deliberately passes explicit model evidence payloads through unchanged.
    It does not guess mappings from generic BMS channels. Dedicated, tested
    adapters will be introduced only where the evidence semantics are equivalent.
    """
    result = {}
    for key, block in packet["model_evidence"].items():
        if block.get("enabled") is True:
            result[key] = block["payload"]
    return result

def main():
    ap = argparse.ArgumentParser(description="Validate a Predictive-BTMS Unified Data Interface packet.")
    ap.add_argument("--input", required=True, type=Path, help="Path to interface JSON")
    ap.add_argument("--output", type=Path, help="Optional path for validation report JSON")
    args = ap.parse_args()
    try:
        packet = json.loads(args.input.read_text(encoding="utf-8"))
        report = validate_packet(packet)
        report["interface_version"] = VERSION
        report["packet_id"] = packet["packet_id"]
        report["model_payloads_ready"] = sorted(build_model_payloads(packet))
        print(json.dumps(report, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}")
        raise SystemExit(2)

if __name__ == "__main__":
    main()
