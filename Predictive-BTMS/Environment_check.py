import pybamm
import numpy
import scipy
import pandas
import matplotlib

print("Environment verification")
print("------------------------")

print("PyBaMM:", pybamm.__version__)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("Pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)

print("\nChecking Prada2013 parameter set...")

parameters = pybamm.ParameterValues("Prada2013")

print("Prada2013 loaded successfully.")

print("\nChecking DFN model...")

model = pybamm.lithium_ion.DFN()

print("DFN model initialized successfully.")

print("\nBASELINE ENVIRONMENT READY")