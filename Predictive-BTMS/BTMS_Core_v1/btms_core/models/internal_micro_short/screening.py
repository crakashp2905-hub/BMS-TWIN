"""Indirect rest-period voltage-residual screening for possible micro-short behaviour.

This is an anomaly screen only. It does not estimate short current, resistance,
short-path voltage, defect heat, or prove that a micro-short is the cause.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, List, Tuple

from btms_core.contracts.types import (
    DataSufficiency, Identifiability, ModelStatus, PrecursorResult, Quantity,
    Quality, SourceType,
)
from btms_core.models.base import PrecursorModel


_PREFIX = "isc_screen."
_PATTERN = re.compile(r"^isc_screen\.(cell_voltage_v|healthy_reference_voltage_v|cell_temperature_k|external_current_a)\.(\d{3,})$")
_REQUIRED_CHANNELS = (
    "cell_voltage_v", "healthy_reference_voltage_v",
    "cell_temperature_k", "external_current_a",
)


@dataclass(frozen=True)
class ScreeningSample:
    index: str
    timestamp_s: float
    cell_voltage_v: float
    reference_voltage_v: float
    temperature_k: float
    external_current_a: float
    residual_sigma_v: float


@dataclass(frozen=True)
class TrendEstimate:
    intercept_v: float
    slope_v_s: float
    slope_std_v_s: float
    lower_95_v_s: float
    upper_95_v_s: float


def _weighted_linear_trend(times_s: List[float], values_v: List[float], sigma_v: List[float]) -> TrendEstimate:
    """Weighted straight-line fit with known independent point uncertainties."""
    if len(times_s) < 3 or not (len(times_s) == len(values_v) == len(sigma_v)):
        raise ValueError("At least three aligned samples are required for trend estimation")
    t0 = times_s[0]
    x = [t - t0 for t in times_s]
    weights = [1.0 / (s * s) for s in sigma_v]
    sw = sum(weights)
    sx = sum(w * xx for w, xx in zip(weights, x))
    sy = sum(w * yy for w, yy in zip(weights, values_v))
    sxx = sum(w * xx * xx for w, xx in zip(weights, x))
    sxy = sum(w * xx * yy for w, xx, yy in zip(weights, x, values_v))
    determinant = sw * sxx - sx * sx
    if not math.isfinite(determinant) or determinant <= 0:
        raise ValueError("Time samples do not provide a solvable linear trend")
    intercept = (sxx * sy - sx * sxy) / determinant
    slope = (sw * sxy - sx * sy) / determinant
    slope_std = math.sqrt(sw / determinant)
    return TrendEstimate(intercept, slope, slope_std,
                         slope - 1.96 * slope_std,
                         slope + 1.96 * slope_std)


class InternalMicroShortScreeningModel(PrecursorModel):
    """Screen a rest window for unexplained negative voltage-residual drift.

    Configuration is mandatory and must be selected from the acquisition
    system's current-sensor zero-offset/noise specification and the test
    protocol. No universal rest-current or temperature threshold is assumed.
    """
    model_id = "INTERNAL_MICRO_SHORT_SCREEN"
    model_version = "1.1.0"

    def __init__(self, *, rest_current_limit_a: float,
                 max_temperature_span_k: float,
                 max_sample_skew_s: float = 0.25,
                 minimum_samples: int = 5):
        values = (rest_current_limit_a, max_temperature_span_k, max_sample_skew_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Screening configuration values must be finite")
        if rest_current_limit_a < 0 or max_temperature_span_k <= 0 or max_sample_skew_s < 0:
            raise ValueError("Invalid screening configuration limits")
        if minimum_samples < 5:
            raise ValueError("minimum_samples must be at least 5 for this screening implementation")
        self.rest_current_limit_a = rest_current_limit_a
        self.max_temperature_span_k = max_temperature_span_k
        self.max_sample_skew_s = max_sample_skew_s
        self.minimum_samples = minimum_samples

    @property
    def required_inputs(self):
        # Samples are variable-length, timestamped channel groups; validated
        # explicitly here rather than represented as fixed scalar requirements.
        return ()

    @property
    def optional_inputs(self):
        return ()

    def evaluate(self, packet, validation):
        if validation.sufficiency != DataSufficiency.SUFFICIENT:
            status = ModelStatus.INVALID_INPUT if validation.sufficiency == DataSufficiency.INVALID else ModelStatus.INSUFFICIENT_DATA
            return self._unavailable(packet, status, validation.sufficiency,
                                     tuple(issue.message for issue in validation.issues))
        try:
            samples, reference_source = self._collect_samples(packet)
            self._validate_window(samples)
            residuals = [s.cell_voltage_v - s.reference_voltage_v for s in samples]
            times = [s.timestamp_s for s in samples]
            sigmas = [s.residual_sigma_v for s in samples]
            trend = _weighted_linear_trend(times, residuals, sigmas)
        except LookupError as exc:
            return self._unavailable(packet, ModelStatus.INSUFFICIENT_DATA,
                                     DataSufficiency.INSUFFICIENT, (str(exc),))
        except ValueError as exc:
            return self._unavailable(packet, ModelStatus.INVALID_INPUT,
                                     DataSufficiency.INVALID, (str(exc),))

        candidate = trend.upper_95_v_s < 0.0
        status = ModelStatus.CANDIDATE if candidate else ModelStatus.NO_CANDIDATE
        slope_mv_h = trend.slope_v_s * 3.6e6
        slope_se_mv_h = trend.slope_std_v_s * 3.6e6
        evidence = {
            "screening_method": "weighted_linear_trend_of_cell_voltage_minus_healthy_reference_voltage",
            "healthy_reference_source_id": reference_source,
            "sample_count": len(samples),
            "window_duration_s": samples[-1].timestamp_s - samples[0].timestamp_s,
            "rest_current_limit_a": self.rest_current_limit_a,
            "maximum_observed_abs_external_current_a": max(abs(s.external_current_a) for s in samples),
            "temperature_span_k": max(s.temperature_k for s in samples) - min(s.temperature_k for s in samples),
            "slope_confidence_interval": "approximate 95% normal interval using supplied independent per-sample voltage standard uncertainties",
            "physical_origin_confirmed": False,
            "micro_short_current_identified": False,
            "micro_short_resistance_identified": False,
            "heat_identified": False,
        }
        diagnostic = (
            "A statistically negative voltage-residual trend was observed during the configured rest window; this is a screening candidate, not a micro-short diagnosis."
            if candidate else
            "No statistically significant negative voltage-residual trend was identified in this window. This does not exclude a micro-short."
        )
        return PrecursorResult(
            model_id=self.model_id, model_version=self.model_version,
            evaluation_timestamp_s=packet.evaluation_timestamp_s,
            status=status, data_sufficiency=DataSufficiency.SUFFICIENT,
            identifiability=Identifiability.NOT_IDENTIFIABLE,
            evidence=evidence,
            magnitude={
                "voltage_residual_slope": Quantity(slope_mv_h, "mV/h"),
                "voltage_residual_slope_standard_uncertainty": Quantity(slope_se_mv_h, "mV/h"),
                "voltage_residual_slope_lower_95": Quantity(trend.lower_95_v_s * 3.6e6, "mV/h"),
                "voltage_residual_slope_upper_95": Quantity(trend.upper_95_v_s * 3.6e6, "mV/h"),
                "final_voltage_residual": Quantity(residuals[-1], "V"),
            },
            heat_power_w=Quantity.unavailable("W", "Voltage-residual drift does not identify a physical leakage current or path voltage."),
            heat_density_w_m3=Quantity.unavailable("W/m^3", "No physical defect power or defect volume is identified."),
            cumulative_heat_j=Quantity.unavailable("J", "No physical defect power is identified."),
            severity=None,
            diagnostics=(diagnostic,
                "The healthy reference must be validated for the same chemistry, cell design, SOC region, temperature, age and relaxation protocol.",
                "Residual drift can also arise from reference-model bias, hysteresis, relaxation mismatch, temperature gradients, sensor drift, aging or other parasitic processes.",
                "No equivalent leakage current, short resistance, micro-short probability or heat is inferred."),
            validity_domain=("Rest-period window only", "Stable cell temperature within configured span", "External current within configured rest limit", "Matched healthy-reference trajectory"),
        )

    def _collect_samples(self, packet) -> Tuple[List[ScreeningSample], str]:
        grouped: Dict[str, Dict[str, object]] = {}
        for key, sig in packet.signals.items():
            match = _PATTERN.match(key)
            if not match:
                continue
            channel, index = match.groups()
            if channel in grouped.setdefault(index, {}):
                raise ValueError(f"Duplicate channel {channel} for sample {index}")
            grouped[index][channel] = sig
        if len(grouped) < self.minimum_samples:
            raise LookupError(f"Need at least {self.minimum_samples} complete indexed samples; found {len(grouped)} sample groups")

        samples = []
        reference_sources = set()
        for index in sorted(grouped, key=lambda x: int(x)):
            group = grouped[index]
            missing = [name for name in _REQUIRED_CHANNELS if name not in group]
            if missing:
                raise LookupError(f"Sample {index} is missing channels: {', '.join(missing)}")
            cell = group["cell_voltage_v"]
            ref = group["healthy_reference_voltage_v"]
            temp = group["cell_temperature_k"]
            current = group["external_current_a"]
            for name, sig, unit, allowed in (
                ("cell_voltage_v", cell, "V", (SourceType.MEASURED,)),
                ("healthy_reference_voltage_v", ref, "V", (SourceType.MODEL_DERIVED, SourceType.ESTIMATED)),
                ("cell_temperature_k", temp, "K", (SourceType.MEASURED,)),
                ("external_current_a", current, "A", (SourceType.MEASURED,)),
            ):
                if sig.unit != unit:
                    raise ValueError(f"Sample {index} {name}: expected {unit}, got {sig.unit}")
                if sig.quality != Quality.VALID:
                    raise ValueError(f"Sample {index} {name} quality must be VALID, got {sig.quality.value}")
                if sig.value is None or not math.isfinite(sig.value):
                    raise ValueError(f"Sample {index} {name} must have a finite value")
                if sig.timestamp_s is None or not math.isfinite(sig.timestamp_s):
                    raise ValueError(f"Sample {index} {name} needs a finite timestamp")
                if sig.timestamp_s > packet.evaluation_timestamp_s:
                    raise ValueError(f"Sample {index} {name} timestamp is after evaluation time")
                if sig.source_type not in allowed:
                    raise ValueError(f"Sample {index} {name} has disallowed source type {sig.source_type.value}")
            if cell.value <= 0 or ref.value <= 0 or temp.value <= 0:
                raise ValueError(f"Sample {index}: voltage and absolute temperature must be positive")
            if cell.uncertainty.standard is None or ref.uncertainty.standard is None:
                raise ValueError(f"Sample {index}: cell and reference voltage standard uncertainties are required")
            if cell.uncertainty.standard <= 0 or ref.uncertainty.standard <= 0:
                raise ValueError(f"Sample {index}: voltage standard uncertainties must be positive")
            timestamps = [cell.timestamp_s, ref.timestamp_s, temp.timestamp_s, current.timestamp_s]
            if max(timestamps) - min(timestamps) > self.max_sample_skew_s:
                raise ValueError(f"Sample {index}: channel timestamp skew exceeds configured limit")
            reference_sources.add(ref.source_id)
            residual_sigma = math.sqrt(cell.uncertainty.standard**2 + ref.uncertainty.standard**2)
            samples.append(ScreeningSample(index, sum(timestamps) / len(timestamps),
                cell.value, ref.value, temp.value, current.value, residual_sigma))
        if len(reference_sources) != 1:
            raise ValueError("Healthy-reference samples must come from one consistent reference-model source_id")
        samples.sort(key=lambda s: s.timestamp_s)
        if any(b.timestamp_s <= a.timestamp_s for a, b in zip(samples, samples[1:])):
            raise ValueError("Sample timestamps must be strictly increasing")
        return samples, next(iter(reference_sources))

    def _validate_window(self, samples):
        if len(samples) < self.minimum_samples:
            raise LookupError("Insufficient complete samples")
        max_current = max(abs(s.external_current_a) for s in samples)
        if max_current > self.rest_current_limit_a:
            raise ValueError("Window is not a valid rest period: external current exceeds configured rest-current limit")
        temp_span = max(s.temperature_k for s in samples) - min(s.temperature_k for s in samples)
        if temp_span > self.max_temperature_span_k:
            raise ValueError("Window temperature variation exceeds configured stability limit")
        if samples[-1].timestamp_s <= samples[0].timestamp_s:
            raise ValueError("Observation window duration must be positive")

    def _unavailable(self, packet, status, sufficiency, diagnostics):
        return PrecursorResult(
            model_id=self.model_id, model_version=self.model_version,
            evaluation_timestamp_s=packet.evaluation_timestamp_s,
            status=status, data_sufficiency=sufficiency,
            identifiability=Identifiability.NOT_IDENTIFIABLE,
            heat_power_w=Quantity.unavailable("W", "Indirect voltage screening does not identify defect power."),
            heat_density_w_m3=Quantity.unavailable("W/m^3", "No physical defect power or volume is identified."),
            cumulative_heat_j=Quantity.unavailable("J", "No physical defect power is identified."),
            diagnostics=diagnostics,
        )
