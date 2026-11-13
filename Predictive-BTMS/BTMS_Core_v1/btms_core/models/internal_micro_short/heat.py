"""Heat accounting for directly measured micro-short-path dissipation."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class HeatEstimate:
    power_w: float
    energy_j: float | None
    density_w_m3: float | None


def estimate_heat(power_w: float, duration_s: float | None = None,
                  effective_volume_m3: float | None = None) -> HeatEstimate:
    if not math.isfinite(power_w) or power_w < 0:
        raise ValueError("Power must be finite and non-negative")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("Observation duration must be finite and positive")
    if effective_volume_m3 is not None and (not math.isfinite(effective_volume_m3) or effective_volume_m3 <= 0):
        raise ValueError("Effective volume must be finite and positive")
    energy = power_w * duration_s if duration_s is not None else None
    density = power_w / effective_volume_m3 if effective_volume_m3 is not None else None
    return HeatEstimate(power_w, energy, density)
