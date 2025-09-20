import numpy as np
import importlib
from scipy.interpolate import interp1d
from scipy.differentiate import derivative

import PaulTrap
import matplotlib.pyplot as plt
import os
from scipy.constants import k
importlib.reload(PaulTrap)
from PaulTrap import PaulTrap
from scipy.constants import epsilon_0
from scipy.constants import pi
from scipy.optimize import root_scalar
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar


"""Trap parameters"""
m = 40  # 40Ca+
charge = 1
RF_freq = 2 * np.pi * 25e6

"""Creates new trap object"""
trap = PaulTrap(mass=m, RF_freq=RF_freq, charge=charge)
"""Find min point for each different assymetric DC voltage"""
# Vdc_L =  [5,3,4,6,7]
# for Vdc in Vdc_L:
#     trap.set_DC_voltages(V_EC_L=10, V_DC_L=Vdc, V_BIAS=11, V_DC_R=5, V_EC_R=10)
#     trap.set_AC_voltage(AC_voltage=0)
#     trap.plot_trap_potential()
#
#     x_r,V,j = trap.find_local_min(x0=1.5,search_width=0.5)
#     x_l, V, j = trap.find_local_min(x0=-1.5, search_width=0.5)
#     print ("Vdc = " + str(Vdc) + "x_r = " +str(x_r) + "x_l = " + str(x_l))
# """checks the legnth between 2 ions near a minimum point: """
# Vbiass = [0,2,4,6,8,10,12,14,16,17,18,20]
# for Vbias in Vbiass:
#     trap.set_DC_voltages(V_EC_L=0, V_DC_L=17, V_BIAS=Vbias, V_DC_R=17, V_EC_R=0)
#     trap.set_AC_voltage(AC_voltage=0)
#     trap.plot_trap_potential()
#     omega_z = trap.get_trap_frequency(center_position= 0 ,width = 0.5)
#     d = (trap.charge**2 / (2 * np.pi * epsilon_0 * trap.mass * omega_z**2))**(1/3)
#     print(d)


# --- set trap configuration ---
trap.set_DC_voltages(20, 17, 20, 17, 20)
e = trap.charge
k_c = 1 / (4 * pi * epsilon_0)

# --- trap data in meters ---
z = trap.position_vector * 1e-3   # if already meters, remove *1e-3
V_vals = trap.get_trap_potential()

# --- restrict data to 0 → 10 µm ---
mask = (z >= 0) & (z <= 10e-6)
z = z[mask]
V_vals = V_vals[mask]

# --- reference value at point closest to zero ---
z0_index = np.argmin(np.abs(z))
z0 = z[z0_index]
V0 = V_vals[z0_index]

# --- interpolator (only valid inside [z.min(), z.max()]) ---
V_interp = interp1d(z, V_vals, kind="cubic", bounds_error=True)

# --- energy functions ---
def U_coul(d):
    """Coulomb energy for two charges separated by 2d"""
    return k_c * e**2 / (2*d)

def U_trap(d):
    """Trap energy relative to the center potential"""
    return e * (V_interp(d) - V0)

def U_tot(d):
    return U_coul(d) + U_trap(d)

# --- scan range restricted to available z data ---
pos = np.linspace(z.min(), z.max(), 500)

U_c_vals = [U_coul(d) for d in pos]
U_t_vals = [U_trap(d) for d in pos]
U_vals   = [U_tot(d)  for d in pos]

# --- find minimum by grid search ---
idx_min = np.argmin(U_vals)
d_min = pos[idx_min]      # half-spacing [m]
E_min = U_vals[idx_min]   # minimum energy [J]

# --- print results ---
print(f"Equilibrium half-spacing = {d_min*1e6:.2f} µm")
print(f"Equilibrium full-spacing = {2*d_min*1e6:.2f} µm")
print(f"Minimum energy = {E_min:.3e} J")

# --- plot contributions ---
plt.plot(pos*1e6, np.array(U_c_vals) - E_min, label="Coulomb")
plt.plot(pos*1e6, np.array(U_t_vals) - E_min, label="Trap")
plt.plot(pos*1e6, np.array(U_vals)   - E_min, label="Total", linewidth=2)

# mark equilibrium point
plt.plot(d_min*1e6, 0, "ro", label="Equilibrium point")

plt.xlabel("Half-spacing d [µm]")
plt.ylabel("Energy relative to min [J]")
plt.title("Coulomb, Trap, and Total potentials (0–10 µm range)")
plt.legend()
plt.grid(True)
plt.show()




