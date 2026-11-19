"""
ELECTROCHEMICAL STATE EXTRACTOR
================================

Purpose:
    Extract electrochemical states from the PyBaMM DFN solution
    for use by the Predictive-BTMS precursor models.

Model:
    LFP / Graphite DFN

Responsibilities:
    1. Extract terminal electrical variables.
    2. Extract electrode potentials.
    3. Extract particle concentrations.
    4. Extract electrolyte concentrations.
    5. Extract interfacial reaction variables.
    6. Calculate spatial averages, minima and maxima for CSV output.
    7. Preserve selected original PyBaMM arrays for spatial analysis.
    8. Export extracted states to CSV, NPZ and JSON metadata.

This module does not modify the electrochemical model.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "electrochemical_states"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SUMMARY VARIABLE MAPPING (CSV)
# ============================================================

# Keep this mapping and its candidate priority unchanged.
# It preserves the existing CSV extraction behaviour.

VARIABLE_MAP = {

    # TERMINAL ELECTRICAL VARIABLES
    "Cell_Voltage": [
        "Voltage [V]"
    ],

    "Cell_Current": [
        "Current [A]"
    ],

    "Discharge_Capacity": [
        "Discharge capacity [A.h]"
    ],

    # THERMAL VARIABLES
    "Cell_Temperature": [
        "Volume-averaged cell temperature [K]"
    ],

    # ELECTRODE POTENTIALS
    "Negative_Electrode_Potential": [
        "Negative electrode potential [V]"
    ],

    "Positive_Electrode_Potential": [
        "Positive electrode potential [V]"
    ],

    "Electrolyte_Potential": [
        "Electrolyte potential [V]"
    ],

    "Negative_Electrode_Surface_Potential_Difference": [
        "Negative electrode surface potential difference [V]"
    ],

    # PARTICLE CONCENTRATIONS
    "Negative_Particle_Surface_Concentration": [
        "X-averaged negative particle surface concentration [mol.m-3]",
        "Negative particle surface concentration [mol.m-3]"
    ],

    "Negative_Particle_Average_Concentration": [
        "X-averaged negative particle concentration [mol.m-3]",
        "Negative particle concentration [mol.m-3]"
    ],

    "Positive_Particle_Surface_Concentration": [
        "X-averaged positive particle surface concentration [mol.m-3]",
        "Positive particle surface concentration [mol.m-3]"
    ],

    # ELECTROLYTE CONCENTRATION
    "Electrolyte_Concentration": [
        "X-averaged electrolyte concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]"
    ],

    # ELECTROCHEMICAL REACTION VARIABLES
    "Negative_Interfacial_Current_Density": [
        "X-averaged negative electrode interfacial current density [A.m-2]",
        "Negative electrode interfacial current density [A.m-2]"
    ],

    "Positive_Interfacial_Current_Density": [
        "X-averaged positive electrode interfacial current density [A.m-2]",
        "Positive electrode interfacial current density [A.m-2]"
    ],

    "Negative_Electrode_Overpotential": [
        "X-averaged negative electrode reaction overpotential [V]",
        "Negative electrode reaction overpotential [V]"
    ],

    "Positive_Electrode_Overpotential": [
        "X-averaged positive electrode reaction overpotential [V]",
        "Positive electrode reaction overpotential [V]"
    ],
}


# ============================================================
# 3. SPATIAL VARIABLE MAPPING (NPZ)
# ============================================================

# This is deliberately separate from VARIABLE_MAP.
# For concentration/reaction quantities, non-X-averaged names
# are requested so that spatial information is not discarded.
# Candidate availability is checked against the solved model.
#
# Note: "X-averaged negative particle concentration" can still
# retain particle-radius dependence. X-averaged refers to
# averaging across electrode thickness, not particle radius.

SPATIAL_VARIABLE_MAP = {

    # Terminal variables (time series)
    "Cell_Voltage": [
        "Voltage [V]"
    ],
    "Cell_Current": [
        "Current [A]"
    ],
    "Discharge_Capacity": [
        "Discharge capacity [A.h]"
    ],
    "Cell_Temperature": [
        "Volume-averaged cell temperature [K]"
    ],

    # Electrode/electrolyte potentials
    "Negative_Electrode_Potential": [
        "Negative electrode potential [V]"
    ],
    "Positive_Electrode_Potential": [
        "Positive electrode potential [V]"
    ],
    "Electrolyte_Potential": [
        "Electrolyte potential [V]"
    ],
    "Negative_Electrode_Surface_Potential_Difference": [
        "Negative electrode surface potential difference [V]"
    ],

    # Particle concentrations: non-X-averaged candidates only
    "Negative_Particle_Surface_Concentration": [
        "Negative particle surface concentration [mol.m-3]"
    ],
    "Negative_Particle_Average_Concentration": [
        "Negative particle concentration [mol.m-3]"
    ],
    "Positive_Particle_Surface_Concentration": [
        "Positive particle surface concentration [mol.m-3]"
    ],

    # Electrolyte concentration
    "Electrolyte_Concentration": [
        "Electrolyte concentration [mol.m-3]"
    ],

    # Interfacial reaction variables
    "Negative_Interfacial_Current_Density": [
        "Negative electrode interfacial current density [A.m-2]"
    ],
    "Positive_Interfacial_Current_Density": [
        "Positive electrode interfacial current density [A.m-2]"
    ],
    "Negative_Electrode_Overpotential": [
        "Negative electrode reaction overpotential [V]"
    ],
    "Positive_Electrode_Overpotential": [
        "Positive electrode reaction overpotential [V]"
    ],
}


# ============================================================
# 4. EXTRACT PYBAMM VARIABLE NAMES
# ============================================================

def get_available_variables(solution):
    """
    Retrieve available variable names from the models
    associated with the PyBaMM solution.
    """

    available_variables = set()

    # Retrieve models associated with the solution
    models = getattr(solution, "all_models", None)

    if models:
        # PyBaMM versions may expose this as a list or dictionary.
        model_iterable = (
            models.values()
            if isinstance(models, dict)
            else models
        )

        for model in model_iterable:
            available_variables.update(model.variables.keys())

    # Fallback: inspect the directly associated model
    model = getattr(solution, "model", None)

    if model is not None:
        available_variables.update(model.variables.keys())

    if not available_variables:
        raise RuntimeError(
            "Could not retrieve model variables from PyBaMM Solution. "
            "Inspect the Solution object to identify its available attributes."
        )

    return available_variables


# ============================================================
# 5. IDENTIFY AVAILABLE VARIABLES
# ============================================================

def identify_variables(solution):
    """
    Match summary/CSV variables against variables available
    in the solved PyBaMM model.
    """

    available_variables = get_available_variables(solution)

    matched_variables = {}
    missing_variables = {}

    for output_name, candidates in VARIABLE_MAP.items():
        matched_name = None

        for candidate in candidates:
            if candidate in available_variables:
                matched_name = candidate
                break

        if matched_name is not None:
            matched_variables[output_name] = matched_name
        else:
            missing_variables[output_name] = candidates

    return matched_variables, missing_variables


def identify_spatial_variables(solution):
    """
    Match spatial-export candidates against the variables
    actually available in the solved PyBaMM model.

    Does not fall back to X-averaged candidates unless they
    are explicitly listed in SPATIAL_VARIABLE_MAP.
    """

    available_variables = get_available_variables(solution)

    matched_variables = {}
    missing_variables = {}

    for output_name, candidates in SPATIAL_VARIABLE_MAP.items():
        matched_name = None

        for candidate in candidates:
            if candidate in available_variables:
                matched_name = candidate
                break

        if matched_name is not None:
            matched_variables[output_name] = matched_name
            print(f"[SPATIAL MATCH] {output_name}: {matched_name}")
        else:
            missing_variables[output_name] = candidates
            print(f"[SPATIAL MISSING] {output_name}: {candidates}")

    return matched_variables, missing_variables


# ============================================================
# 6. EXTRACT TIME-DEPENDENT VARIABLE DATA FOR CSV
# ============================================================

def extract_variable_entries(solution, variable_name, time_length):
    """
    Extract a variable and calculate spatial statistics.

    PyBaMM entries normally have time along the last axis.

    For spatial variables, calculate mean, minimum and maximum
    across all non-time dimensions.
    """

    entries = np.asarray(
        solution[variable_name].entries
    )

    # Scalar time-dependent variable
    if entries.ndim == 1:
        if len(entries) != time_length:
            raise ValueError(
                f"Unexpected data length for {variable_name}: "
                f"{len(entries)} instead of {time_length}"
            )

        return {
            "mean": entries,
            "min": entries,
            "max": entries
        }

    # Identify the time axis. Prefer the last axis because
    # PyBaMM commonly stores time along the final dimension.
    if entries.shape[-1] == time_length:
        time_axis = entries.ndim - 1
    elif entries.shape[0] == time_length:
        time_axis = 0
    else:
        raise ValueError(
            f"Cannot identify time dimension for {variable_name}. "
            f"Shape: {entries.shape}; expected time length: {time_length}"
        )

    if time_axis == 0:
        entries = np.moveaxis(entries, 0, -1)

    # Flatten spatial dimensions while retaining time
    spatial_entries = entries.reshape(-1, time_length)

    return {
        "mean": np.mean(spatial_entries, axis=0),
        "min": np.min(spatial_entries, axis=0),
        "max": np.max(spatial_entries, axis=0)
    }


# ============================================================
# 7. EXPORT ORIGINAL SPATIAL ARRAYS
# ============================================================

def determine_time_axis(entries, time_length):
    """
    Determine which original array axis corresponds to time.

    Returns None if the time dimension cannot be identified.
    """

    if entries.ndim == 0:
        return None

    # A one-dimensional variable is a time series if its length
    # matches the solution time vector.
    if entries.ndim == 1:
        return 0 if entries.shape[0] == time_length else None

    # Prefer PyBaMM's usual final-axis time convention.
    if entries.shape[-1] == time_length:
        return entries.ndim - 1

    if entries.shape[0] == time_length:
        return 0

    return None


def export_spatial_data(solution, matched_variables, time):
    """
    Preserve original PyBaMM variable arrays without spatial
    averaging or flattening.

    Original array shapes and axis order are retained.
    """

    time = np.asarray(time).reshape(-1)

    spatial_data = {
        "Time_s": time
    }

    metadata = {}

    print("\nSPATIAL DATA EXPORT")
    print("-" * 75)

    for output_name, variable_name in matched_variables.items():
        try:
            entries = np.asarray(
                solution[variable_name].entries
            )

            time_axis = determine_time_axis(
                entries,
                len(time)
            )

            if time_axis is None:
                raise ValueError(
                    f"Could not identify time axis. "
                    f"Array shape: {entries.shape}; "
                    f"time samples: {len(time)}"
                )

            # Preserve original array exactly
            spatial_data[output_name] = entries

            units = (
                variable_name.rsplit("[", 1)[-1].rstrip("]")
                if "[" in variable_name
                else "Unknown"
            )

            metadata[output_name] = {
                "pybamm_variable": variable_name,
                "shape": list(entries.shape),
                "dimensions": entries.ndim,
                "units": units,
                "time_axis": time_axis,
                "array_order": "Original PyBaMM entries order"
            }

            print(
                f"{output_name:<45} Shape: {str(entries.shape):<18} "
                f"Time axis: {time_axis}"
            )

        except Exception as error:
            print(
                f"Failed: {output_name} | {error}"
            )

    # Save compressed spatial arrays
    spatial_path = (
        RESULTS_DIR / "spatial_electrochemical_states.npz"
    )

    np.savez_compressed(
        spatial_path,
        **spatial_data
    )

    # Save variable metadata
    metadata_path = (
        RESULTS_DIR / "spatial_electrochemical_metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nSpatial data saved:")
    print(spatial_path)

    print("\nMetadata saved:")
    print(metadata_path)

    print(
        f"Spatial variables exported: {len(metadata)} / "
        f"{len(matched_variables)} matched"
    )

    return metadata


# ============================================================
# 8. MAIN EXTRACTION FUNCTION
# ============================================================

def extract_electrochemical_states(
    solution,
    save_csv=True,
    save_spatial=True
):
    """
    Extract electrochemical states from a solved PyBaMM model.

    Parameters
    ----------
    solution:
        Solved PyBaMM solution object.

    save_csv:
        Whether to export summary statistics to CSV.

    save_spatial:
        Whether to export original selected arrays to NPZ
        and their metadata to JSON.

    Returns
    -------
    pandas.DataFrame
        Time-dependent electrochemical state data.
    """

    print("\n" + "=" * 75)
    print("             ELECTROCHEMICAL STATE EXTRACTION")
    print("=" * 75)

    # Time
    time = np.asarray(solution.t).reshape(-1)

    results = {
        "Time_s": time,
        "Time_min": time / 60.0
    }

    # Identify summary variables (existing CSV mapping)
    matched, missing = identify_variables(solution)

    print("\nAVAILABLE SUMMARY VARIABLES")
    print("-" * 75)

    for output_name, variable_name in matched.items():
        print(f"{output_name:<45}: {variable_name}")

    print("\nUNAVAILABLE SUMMARY VARIABLES")
    print("-" * 75)

    for output_name, candidates in missing.items():
        print(f"{output_name:<45}: {candidates}")

    # Identify and export spatial variables independently
    spatial_matched = {}
    spatial_missing = {}

    if save_spatial:
        print("\nSPATIAL VARIABLE IDENTIFICATION")
        print("-" * 75)

        spatial_matched, spatial_missing = identify_spatial_variables(
            solution
        )

        print("\nSPATIAL IDENTIFICATION SUMMARY")
        print("-" * 75)
        print(f"Spatial variables matched   : {len(spatial_matched)}")
        print(f"Spatial variables missing   : {len(spatial_missing)}")

        export_spatial_data(
            solution,
            spatial_matched,
            time
        )

    # Extract summary variables for CSV
    print("\nEXTRACTING ELECTROCHEMICAL SUMMARY STATES")
    print("-" * 75)

    extraction_errors = {}

    for output_name, variable_name in matched.items():
        try:
            data = extract_variable_entries(
                solution,
                variable_name,
                len(time)
            )

            results[f"{output_name}_Mean"] = data["mean"]
            results[f"{output_name}_Min"] = data["min"]
            results[f"{output_name}_Max"] = data["max"]

            print(f"Extracted: {output_name}")

        except Exception as error:
            extraction_errors[output_name] = str(error)

            print(
                f"Extraction failed: {output_name} | {error}"
            )

    # Create DataFrame
    dataframe = pd.DataFrame(results)

    # Export CSV
    if save_csv:
        output_path = (
            RESULTS_DIR / "electrochemical_states.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False
        )

        print("\nSummary CSV exported:")
        print(output_path)

    # Final summary
    print("\nEXTRACTION SUMMARY")
    print("-" * 75)
    print(f"Time samples                  : {len(time)}")
    print(f"Summary variables matched     : {len(matched)}")
    print(f"Summary variables unavailable : {len(missing)}")
    print(f"Summary extraction errors     : {len(extraction_errors)}")

    if save_spatial:
        print(f"Spatial variables matched     : {len(spatial_matched)}")
        print(f"Spatial variables unavailable : {len(spatial_missing)}")

    print("=" * 75)

    return dataframe


# ============================================================
# 9. STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    print(
        "This module requires a solved PyBaMM solution object."
    )

    print(
        "Import extract_electrochemical_states() "
        "into electrothermal_baseline.py to use it."
    )
