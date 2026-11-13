"""Electrical calculations for a directly observed resistive leakage path."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ElectricalEstimate:
    resistance_ohm: float
    power_w: float
    resistance_std_ohm: float | None
    power_std_w: float | None


def estimate_path(v_v: float, i_a: float, sigma_v: float | None = None,
                  sigma_i: float | None = None) -> ElectricalEstimate:
    """Return R=V/I and P=VI for the same physical path.

    Assumes positive, dissipative voltage/current under passive sign convention.
    First-order uncertainty propagation assumes independent input errors.
    """
    vals = (v_v, i_a)
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("Path voltage/current must be finite")
    if v_v <= 0 or i_a <= 0:
        raise ValueError("Path voltage and current must both be positive")
    if sigma_v is not None and (not math.isfinite(sigma_v) or sigma_v < 0):
        raise ValueError("Voltage standard uncertainty must be finite and non-negative")
    if sigma_i is not None and (not math.isfinite(sigma_i) or sigma_i < 0):
        raise ValueError("Current standard uncertainty must be finite and non-negative")

    resistance = v_v / i_a
    power = v_v * i_a
    r_std = None
    p_std = None
    if sigma_v is not None and sigma_i is not None:
        # Independent-error, first-order propagation. Covariance is not
        # available in the current canonical Signal contract.
        r_std = math.sqrt((sigma_v / i_a) ** 2 + (v_v * sigma_i / i_a**2) ** 2)
        p_std = math.sqrt((i_a * sigma_v) ** 2 + (v_v * sigma_i) ** 2)
    return ElectricalEstimate(resistance, power, r_std, p_std)
