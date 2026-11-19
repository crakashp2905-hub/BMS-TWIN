# Model 01 — Internal Micro-Short Path Assessment

## Scope

This first implementation calculates the electrical resistance and dissipative power of a **directly observed, declared conductive path**. It does not infer a micro-short from ordinary cell-terminal voltage, pack current, SOC drift, or temperature alone.

## Required canonical signals

| Signal | Unit | Meaning |
|---|---:|---|
| `isc_path_voltage_v` | V | Voltage directly across the suspected path |
| `isc_path_current_a` | A | Current directly through that same path; positive under passive sign convention |

Optional:

| Signal | Unit | Use |
|---|---:|---|
| `isc_observation_duration_s` | s | Converts representative instantaneous power to energy |
| `isc_effective_volume_m3` | m³ | Converts path power to volumetric heat generation |

These path-specific signals are not interchangeable with `cell_voltage` and `cell_current`.

## Equations

For a positive dissipative path measurement:

- `R_path = V_path / I_path` [Ω]
- `P_path = V_path × I_path` [W]
- `E_path = P_path × Δt` [J], only when a representative duration is supplied
- `q_path = P_path / V_effective` [W/m³], only when a defensible effective defect volume is supplied

First-order uncertainty propagation is provided for resistance and power when standard uncertainties for **both** voltage and current are supplied. It assumes independent measurement errors; covariance is not currently represented by the core contract.

## Interpretation and limits

- A positive path-current observation is evidence of dissipation in the declared path. It does not, by itself, prove that the physical origin is an internal micro-short rather than another conductive shunt.
- `DETECTED` means the declared path current is positive and its one-standard-uncertainty lower bound is above zero (or uncertainty was not supplied). It is not a safety classification or validated fault diagnosis.
- `CANDIDATE` means positive current was observed but its one-standard-uncertainty lower bound includes zero.
- `PARTIALLY_IDENTIFIABLE` reflects that path resistance and power can be calculated from the direct path measurements, while the physical origin and spatial distribution are not established.
- No local hotspot or cell-average temperature rise is calculated. Heat density is withheld unless an effective defect volume is supplied.
- The heat output is path electrical dissipation. Before adding it to a whole-cell thermal model, verify whether that physical dissipation is already represented in the baseline; do not double-count it.
- No temperature coefficient for path resistance is assumed. No threshold for severity is invented.

## Run

From the project root:

```bash
python main.py
python -m unittest discover -s tests -v
```

The `main.py` values are illustrative interface arithmetic only, not cell measurements or a fault simulation.
