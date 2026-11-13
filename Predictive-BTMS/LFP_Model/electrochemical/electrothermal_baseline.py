"""
LFP / GRAPHITE ELECTROTHERMAL BASELINE
======================================

Physics model:
    Doyle-Fuller-Newman (DFN)

Electrochemical parameterisation:
    Prada2013

Thermal model:
    Lumped thermal model

PyBaMM:
    26.10.1.1

IMPORTANT:
    This is a provisional research baseline, not a validated
    manufacturer-specific cell model.

    Consequently, reversible heat is not physically parameterised.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybamm
from LFP_Model.electrochemical.electrochemical_state_extractor import (
    extract_electrochemical_states
)
# ============================================================
# LITERATURE-BASED ENTROPIC COEFFICIENTS
# Source: Gwak & Ju (2019), Appendix, Equations A3 and A4
# Coefficients expressed in mV/K; converted to V/K.
# ============================================================

def graphite_entropic_change(theta):

    value = (
        0.00291
        - 0.6061 * theta
        - 14.75436 * theta**2
        + 160.37506 * theta**3
        - 782.96735 * theta**4
        + 2226.9794 * theta**5
        - 3818.24879 * theta**6
        + 3852.07546 * theta**7
        - 2098.10884 * theta**8
        + 475.15265 * theta**9
    )

    return value * 1e-3


def lfp_entropic_change(theta):

    value = (
        -0.11853
        + 1.13004 * theta
        - 20.74755 * theta**2
        + 150.66611 * theta**3
        - 544.59353 * theta**4
        + 1096.48601 * theta**5
        - 1257.02758 * theta**6
        + 777.80223 * theta**7
        - 213.43055 * theta**8
        + 9.88834 * theta**9
    )

    return value * 1e-3

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "electrothermal_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AMBIENT_TEMPERATURE = 298.0
INITIAL_TEMPERATURE = 298.0

HEAT_TRANSFER_COEFFICIENT = 5.0

DISCHARGE_RATE = "1C"
LOWER_CUTOFF_VOLTAGE = 2.0

print("=" * 65)
print("       LFP / GRAPHITE ELECTROTHERMAL MODEL")
print("=" * 65)


# ============================================================
# 2. LOAD ELECTROCHEMICAL PARAMETERISATION
# ============================================================

print("\n[1/7] Loading Prada2013 parameterisation...")

parameters = pybamm.ParameterValues("Prada2013")

print("Electrochemical parameterisation loaded.")


# ============================================================
# 3. DEFINE  THERMAL AND MATERIAL PARAMETERS
# ============================================================

print("\n[2/7] Defining thermal material properties...")

thermal_parameters = {

    # Current collectors
    "Negative current collector thickness [m]": 20e-6,
    "Positive current collector thickness [m]": 10e-6,

    "Negative current collector density [kg.m-3]": 8900,
    "Positive current collector density [kg.m-3]": 2700,

    "Negative current collector specific heat capacity [J.kg-1.K-1]": 385,
    "Positive current collector specific heat capacity [J.kg-1.K-1]": 903,

    "Negative current collector conductivity [S.m-1]": 5.96e7,
    "Positive current collector conductivity [S.m-1]": 3.55e7,

    # Negative electrode: graphite
    "Negative electrode density [kg.m-3]": 2660,
    "Negative electrode specific heat capacity [J.kg-1.K-1]": 1437.4,

    # Positive electrode: LFP
    "Positive electrode density [kg.m-3]": 1500,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 1260.2,

    # Separator
    "Separator density [kg.m-3]": 492,
    "Separator specific heat capacity [J.kg-1.K-1]": 1978,

    # Temperature conditions
    "Ambient temperature [K]": AMBIENT_TEMPERATURE,
    "Initial temperature [K]": INITIAL_TEMPERATURE,

    # Lumped thermal boundary condition
    "Total heat transfer coefficient [W.m-2.K-1]":
        HEAT_TRANSFER_COEFFICIENT,

        # Literature-based entropic coefficients
    "Negative electrode OCP entropic change [V.K-1]":
        graphite_entropic_change,

    "Positive electrode OCP entropic change [V.K-1]":
        lfp_entropic_change,
}

parameters.update(
    thermal_parameters,
    check_already_exists=False
)

print("Thermal parameters merged successfully.")

# ============================================================
# 4. ESTABLISH CELL GEOMETRY
# ============================================================

print("\n[3/7] Establishing equivalent cell geometry...")

electrode_width = parameters["Electrode width [m]"]
electrode_height = parameters["Electrode height [m]"]

electrode_area = electrode_width * electrode_height

thickness_parameters = [
    "Negative current collector thickness [m]",
    "Negative electrode thickness [m]",
    "Separator thickness [m]",
    "Positive electrode thickness [m]",
    "Positive current collector thickness [m]",
]

stack_thickness = sum(
    parameters[name] for name in thickness_parameters
)

cell_volume = electrode_area * stack_thickness

cooling_surface_area = 2.0 * electrode_area

parameters.update(
    {
        "Cell volume [m3]": cell_volume,
        "Cell cooling surface area [m2]": cooling_surface_area,
    },
    check_already_exists=False
)

print(f"Electrode area       : {electrode_area:.6f} m2")
print(f"Stack thickness      : {stack_thickness:.8f} m")
print(f"Equivalent volume    : {cell_volume:.8e} m3")
print(f"Cooling surface area : {cooling_surface_area:.6f} m2")


# ============================================================
# GEOMETRY PARAMETER AUDIT
# ============================================================

print("\nGEOMETRY PARAMETER AUDIT")
print("-" * 55)

geometry_keys = [
    "Electrode width [m]",
    "Electrode height [m]",
    "Negative electrode thickness [m]",
    "Positive electrode thickness [m]",
    "Separator thickness [m]",
    "Negative current collector thickness [m]",
    "Positive current collector thickness [m]",
    "Nominal cell capacity [A.h]",
]

for key in geometry_keys:
    if key in parameters:
        print(f"{key:<45}: {parameters[key]}")

print("-" * 55)
print(f"{'Calculated cell volume [m3]':<45}: {cell_volume}")
print(f"{'Calculated cooling area [m2]':<45}: {cooling_surface_area}")

# ============================================================
# THERMAL MASS AUDIT
# ============================================================

print("\nTHERMAL MASS AUDIT")
print("-" * 55)

material_layers = {
    "Negative current collector": (
        parameters["Negative current collector thickness [m]"],
        parameters["Negative current collector density [kg.m-3]"],
        parameters["Negative current collector specific heat capacity [J.kg-1.K-1]"]
    ),
    "Negative electrode": (
        parameters["Negative electrode thickness [m]"],
        parameters["Negative electrode density [kg.m-3]"],
        parameters["Negative electrode specific heat capacity [J.kg-1.K-1]"]
    ),
    "Separator": (
        parameters["Separator thickness [m]"],
        parameters["Separator density [kg.m-3]"],
        parameters["Separator specific heat capacity [J.kg-1.K-1]"]
    ),
    "Positive electrode": (
        parameters["Positive electrode thickness [m]"],
        parameters["Positive electrode density [kg.m-3]"],
        parameters["Positive electrode specific heat capacity [J.kg-1.K-1]"]
    ),
    "Positive current collector": (
        parameters["Positive current collector thickness [m]"],
        parameters["Positive current collector density [kg.m-3]"],
        parameters["Positive current collector specific heat capacity [J.kg-1.K-1]"]
    )
}

total_mass = 0.0
total_heat_capacity = 0.0

for name, (thickness, density, specific_heat) in material_layers.items():

    volume = electrode_area * thickness
    mass = volume * density
    heat_capacity = mass * specific_heat

    total_mass += mass
    total_heat_capacity += heat_capacity

    print(f"\n{name}")
    print(f"Mass          : {mass:.6f} kg")
    print(f"Heat capacity : {heat_capacity:.6f} J/K")

print("\nTOTAL")
print(f"Total mass          : {total_mass:.6f} kg")
print(f"Total heat capacity : {total_heat_capacity:.6f} J/K")
print("-" * 55)

# ============================================================
# PYBAMM THERMAL PARAMETER AUDIT
# ============================================================

print("\nPYBAMM THERMAL PARAMETER AUDIT")
print("=" * 65)

search_terms = [
    "density",
    "specific heat",
    "heat capacity",
    "cell volume",
    "cooling surface",
    "ambient temperature",
    "heat transfer"
]

for key in parameters.keys():
    if any(term in key.lower() for term in search_terms):
        print(f"{key:<55}: {parameters[key]}")

print("=" * 65)

# ============================================================
# 5. INITIALISE DFN ELECTROTHERMAL MODEL
# ============================================================

print("\n[4/7] Initialising DFN model...")

model_options = {
    "cell geometry": "arbitrary",
    "thermal": "lumped",
}

model = pybamm.lithium_ion.DFN(
    options=model_options,
    name="LFP Graphite DFN Lumped Electrothermal Model"
)

print(f"Model                : {model.name}")
print("Chemistry            : LFP / Graphite")
print("Electrochemical model: DFN")
print("Thermal model        : Lumped")

# ============================================================
# 6. CHECK ENTROPIC COEFFICIENT FUNCTIONS
# ============================================================

print("\n[5/7] Checking electrode entropic coefficients...")

theta_values = np.linspace(0, 1, 101)

graphite_values = np.array([
    graphite_entropic_change(theta)
    for theta in theta_values
])

lfp_values = np.array([
    lfp_entropic_change(theta)
    for theta in theta_values
])

print("\nGraphite entropic coefficient:")
print(f"Minimum : {graphite_values.min():.8e} V/K")
print(f"Maximum : {graphite_values.max():.8e} V/K")

print("\nLFP entropic coefficient:")
print(f"Minimum : {lfp_values.min():.8e} V/K")
print(f"Maximum : {lfp_values.max():.8e} V/K")

print("\nBoth entropic coefficient functions are loaded.")


# ============================================================
# 7. BUILD AND SOLVE USING PYBAMM SIMULATION API
# ============================================================

print("\n[6/7] Building and solving electrothermal model...")

experiment = pybamm.Experiment(
    [
        f"Discharge at {DISCHARGE_RATE} "
        f"until {LOWER_CUTOFF_VOLTAGE} V"
    ]
)

# IMPORTANT:
# Use Simulation rather than manually building the model and
# calling the solver. Simulation handles parameter processing,
# geometry, discretisation and solution preparation.

simulation = pybamm.Simulation(
    model,
    parameter_values=parameters,
    experiment=experiment,
)

print("Model build and discretisation initiated...")

solution = simulation.solve()
electrochemical_states = extract_electrochemical_states(
    solution,
    save_csv=True
)

print("Simulation completed successfully!")


# ============================================================
# 8. EXTRACT SOLUTION VARIABLES
# ============================================================

print("\n[7/7] Extracting simulation outputs...")

# Use the solution's time array directly.
time_s = np.asarray(solution.t).reshape(-1)

# These are documented PyBaMM output variable names.
voltage = np.asarray(
    solution["Voltage [V]"].entries
).reshape(-1)

current = np.asarray(
    solution["Current [A]"].entries
).reshape(-1)

temperature = np.asarray(
    solution["Volume-averaged cell temperature [K]"].entries
).reshape(-1)

total_heating = np.asarray(
    solution["Volume-averaged total heating [W.m-3]"].entries
).reshape(-1)

# Individual heat-generation components

ohmic_heating = np.asarray(
    solution[
        "Volume-averaged Ohmic heating [W.m-3]"
    ].entries
).reshape(-1)

irreversible_heating = np.asarray(
    solution[
        "Volume-averaged irreversible electrochemical heating [W.m-3]"
    ].entries
).reshape(-1)

reversible_heating = np.asarray(
    solution[
        "Volume-averaged reversible heating [W.m-3]"
    ].entries
).reshape(-1)

# Check that all extracted arrays have compatible lengths.
array_lengths = {
    "Time": len(time_s),
    "Voltage": len(voltage),
    "Current": len(current),
    "Temperature": len(temperature),
    "Total heating": len(total_heating),
}

print("\nExtracted array lengths:")

for name, length in array_lengths.items():
    print(f"{name:20s}: {length}")

if len(set(array_lengths.values())) != 1:
    raise ValueError(
        "Extracted solution arrays have inconsistent lengths. "
        f"Check output variables: {array_lengths}"
    )

# ============================================================
# ENERGY BALANCE AUDIT
# ============================================================

print("\nENERGY BALANCE AUDIT")
print("=" * 60)

# Simulation outputs
time_seconds = np.asarray(solution.t).reshape(-1)

temperature = np.asarray(
    solution["Volume-averaged cell temperature [K]"].entries
).reshape(-1)

total_heating = np.asarray(
    solution["Volume-averaged total heating [W.m-3]"].entries
).reshape(-1)

# Existing geometry and thermal parameters
ambient_temperature = 298.0
heat_transfer_coefficient = 5.0

# Total heat generation rate (W)
heat_generation_power = total_heating * cell_volume

# Convective heat loss rate (W)
heat_loss_power = (
    heat_transfer_coefficient
    * cooling_surface_area
    * (temperature - ambient_temperature)
)

# Integrate power over simulation time (J)
total_generated_energy = np.trapezoid(
    heat_generation_power, time_seconds
)

total_lost_energy = np.trapezoid(
    heat_loss_power, time_seconds
)

# Change in stored thermal energy (J)
stored_thermal_energy = (
    total_heat_capacity
    * (temperature[-1] - temperature[0])
)

# Energy balance residual
energy_residual = (
    total_generated_energy
    - total_lost_energy
    - stored_thermal_energy
)

# Percentage energy balance error
energy_balance_error = (
    abs(energy_residual) / abs(total_generated_energy)
) * 100 if total_generated_energy != 0 else 0.0

print(f"Simulation duration       : {time_seconds[-1]:.2f} s")
print(f"Total heat generation     : {total_generated_energy:.4f} J")
print(f"Total heat loss           : {total_lost_energy:.4f} J")
print(f"Stored thermal energy     : {stored_thermal_energy:.4f} J")
print(f"Energy balance residual   : {energy_residual:.4f} J")
print(f"Energy balance error      : {energy_balance_error:.4f} %")

print("-" * 60)

print(f"Initial temperature       : {temperature[0]:.4f} K")
print(f"Final temperature         : {temperature[-1]:.4f} K")
print(f"Temperature rise          : {temperature[-1] - temperature[0]:.6f} K")

print("=" * 60)


# ============================================================
# 9. CREATE RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    {
        "Time_s": time_s,
        "Time_min": time_s / 60.0,
        "Voltage_V": voltage,
        "Current_A": current,
        "Cell_Temperature_K": temperature,
        "Cell_Temperature_C": temperature - 273.15,
        "Total_Heating_W_m3": total_heating,
        "Ohmic_Heating_W_m3": ohmic_heating,
        "Irreversible_Heating_W_m3": irreversible_heating,
        "Reversible_Heating_W_m3": reversible_heating,
    }
)

csv_path = RESULTS_DIR / "electrothermal_results.csv"

results.to_csv(csv_path, index=False)

print(f"\nResults saved to:\n{csv_path}")




# ============================================================
# 10. GENERATE SIMULATION PLOTS
# ============================================================

print("\nGenerating simulation plots...")

fig, axes = plt.subplots(
    3,
    1,
    figsize=(11, 12),
    sharex=True
)

# Terminal voltage
axes[0].plot(
    results["Time_min"],
    results["Voltage_V"]
)

axes[0].set_ylabel("Voltage (V)")
axes[0].set_title("LFP / Graphite Cell Voltage")
axes[0].grid(True)

# Cell temperature
axes[1].plot(
    results["Time_min"],
    results["Cell_Temperature_C"]
)

axes[1].set_ylabel("Temperature (°C)")
axes[1].set_title("Volume-Averaged Cell Temperature")
axes[1].grid(True)

# Total volumetric heating
axes[2].plot(
    results["Time_min"],
    results["Total_Heating_W_m3"]
)

axes[2].set_xlabel("Time (minutes)")
axes[2].set_ylabel("Heating (W/m³)")
axes[2].set_title("Volume-Averaged Total Heating")
axes[2].grid(True)

plt.tight_layout()

plot_path = RESULTS_DIR / "electrothermal_response.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Plot saved to:\n{plot_path}")


# ============================================================
# 11. SIMULATION SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("             SIMULATION SUMMARY")
print("=" * 65)

print(f"Simulation duration : {time_s[-1] / 60:.2f} minutes")

print(
    f"Initial voltage     : {voltage[0]:.4f} V"
)

print(
    f"Final voltage       : {voltage[-1]:.4f} V"
)

print(
    f"Initial temperature : {temperature[0]:.2f} K"
)

print(
    f"Maximum temperature : {np.max(temperature):.2f} K"
)

print(
    f"Maximum temperature : {np.max(temperature) - 273.15:.2f} °C"
)

print(
    f"Initial heating     : {total_heating[0]:.4f} W/m3"
)

print(
    f"Maximum heating     : {np.max(total_heating):.4f} W/m3"
)

print(
    f"Minimum voltage     : {np.min(voltage):.4f} V"
)

print("\nOutput files:")
print(f"1. {csv_path}")
print(f"2. {plot_path}")

print("\nSIMULATION FINISHED")
print("=" * 65)