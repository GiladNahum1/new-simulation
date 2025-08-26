import numpy as np
import importlib
import PaulTrap
import matplotlib.pyplot as plt
import os

importlib.reload(PaulTrap)
from PaulTrap import PaulTrap

# Params
m = 40  # 40Ca+
charge = 1
RF_freq = 2 * np.pi * 25e6

trap = PaulTrap(mass=m, RF_freq=RF_freq, charge=charge)

trap.plot_electrode_potentials()
# Example DC + AC
trap.set_DC_voltages(V_EC_L=6, V_DC_L=1, V_BIAS=0, V_DC_R=1, V_EC_R=4)
trap.set_AC_voltage(500)
V,K = trap.get_total_voltage_barrier(3,0.2,1,True)
print ("Energy barrier of the trapped ion near local min is : " + str(K) + " Kelvin")
trap.plot_trap_potential()
trap.set_DC_voltages(V_EC_L=0, V_DC_L=0, V_BIAS=0, V_DC_R=0, V_EC_R=0)
trap.set_AC_voltage(500)
trap.plot_trap_potential()

trap.plot_interactive_trap_potential()
xSlice = 0
yy, zz, VV = trap.plot_yz_heatmap_from_slice(filepath=os.path.join(os.getcwd(), "electrodes responses", "Long Axial RF Slice x=" + str(xSlice) + ".txt"),skiprows=8,window_y=0.71/2, window_z=0.71/2,grid_N=301,levels=0)
(s1,V1),(s2,V2) = trap.plot_diagonals(yy,zz,VV)
omega_r = trap.fit_parabola_radial(s1,V1,0.05)
freq = omega_r/(2*np.pi)
print(f"Trap radial frequency (omega_r): 2pi * {freq * 1e-6:.2f} MHz")
