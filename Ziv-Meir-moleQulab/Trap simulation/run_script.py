""" How to run the PaulTrap class """

import numpy as np
import importlib
import PaulTrap
import matplotlib.pyplot as plt
import os
importlib.reload(PaulTrap)
from PaulTrap import PaulTrap
import PyQt6

# experimental parameters
m = 40  # atomic mass of 40Ca+
charge = 1  # charge in elementary charge units
RF_freq = 2 * np.pi * 25e6  # [rad/sec] RF angular frequency

# Define the trap
trap = PaulTrap(mass=m, RF_freq=RF_freq, charge=charge)

# Plot the electrode responses from the COMSOL simulation
trap.plot_electrode_potentials()

# Plot the axial trap potential for given DC (RF = 0) electrode voltages
#graphs (DC only and get trap frequency in z axis)
trap.plot_interactive_trap_potential()
trap.set_DC_voltages(V_EC_L=0,
                     V_DC_L=17,
                     V_BIAS=0,
                     V_DC_R=17,
                     V_EC_R=0)
trap.set_AC_voltage(AC_voltage=0)  # set AC voltage in volts
trap.plot_trap_potential()
trap_freq = trap.get_trap_frequency(0, 0.2)
print(f"Trap frequency (omega_z): 2pi * {(trap_freq / (2*np.pi)) * 1e-6:.2f} MHz")

# Plot the axial trap potential for only RF voltage (no DC) - check the Ponderomotive calculations.
# graph (DC + Bias)
trap.set_DC_voltages(V_EC_L=0,
                     V_DC_L=1,
                     V_BIAS=1,
                     V_DC_R=1,
                     V_EC_R=0)
trap.set_AC_voltage(AC_voltage=0)  # set AC voltage in volts
trap.plot_trap_potential()
# trap_freq = trap.get_trap_frequency(0, 0.1)
# print(f"Trap frequency (omega_z): 2pi * {(trap_freq / (2*np.pi)) * 1e-6:.2f} MHz")
RF_barrier_V, RF_barrier_T = trap.get_RF_barrier()
print(f"RF barrier = {RF_barrier_V:0.2f} V; {RF_barrier_T:0.2f} K")

# trap.plot_interactive_trap_potential()
#graph (Left EC without AC)
trap.set_DC_voltages(V_EC_L=17,
                     V_DC_L=1,
                     V_BIAS=0,
                     V_DC_R=1,
                     V_EC_R=0)
trap.set_AC_voltage(AC_voltage=0)  # set AC voltage in volts
trap.plot_trap_potential()
# graph (left EC with AC)
trap.set_DC_voltages(V_EC_L=17,
                     V_DC_L=2,
                     V_BIAS=0,
                     V_DC_R=5,
                     V_EC_R=0)
trap.set_AC_voltage(AC_voltage=1)  # set AC voltage in volts
trap.plot_trap_potential()
#
trap.set_DC_voltages(V_EC_L=17,
                     V_DC_L=2,
                     V_BIAS=0,
                     V_DC_R=2,
                     V_EC_R=0)
trap.set_AC_voltage(AC_voltage=0)  # set AC voltage in volts
trap.plot_trap_potential()

