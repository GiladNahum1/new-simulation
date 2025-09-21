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

def trap_length_scale(Z=1):
    nu = trap.get_trap_frequency(0, 0.5)
    M = trap.mass     # ion mass in kg
    l_cubed = (Z**2 * e**2) / (4 * np.pi * epsilon_0 * M * nu**2)
    return l_cubed**(1/3)

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
#
# """checks the legnth between 2 ions near a minimum point: """
# Vbiass = [0,2,4,6,8,10,12,14,16,17,18,20]
# for Vbias in Vbiass:
#     trap.set_DC_voltages(V_EC_L=0, V_DC_L=17, V_BIAS=Vbias, V_DC_R=17, V_EC_R=0)
#     trap.set_AC_voltage(AC_voltage=0)
#     trap.plot_trap_potential()
#     omega_z = trap.get_trap_frequency(center_position= 0 ,width = 0.5)
#     d = (trap.charge**2 / (2 * np.pi * epsilon_0 * trap.mass * omega_z**2))**(1/3)
#     print("distance between the two ions using formula: = " + str(d))

"""In this segment we are adding 2 ions to the system,
 The potential the ion feels is the potential from the trap and the potential from the coulomb force
 by adding them together and finding the minimum points we can get the equilibrium point"""
"""The code works as it should but the COMSOL simulation isn't enough precise in these scales (1~10 μm)
In this case we should maybe try a different approach"""
for v in [1,2,3,4,5,6,7,8]:
    print("Trap v={}".format(v))
    # --- set trap configuration ---
    trap.set_DC_voltages(0, 17, v, 17, 0)
    trap.set_AC_voltage(500)
    e = trap.charge
    k_c = 1 / (4 * pi * epsilon_0)

    # --- trap data in meters ---
    z = trap.position_vector * 1e-3   # if already meters, remove *1e-3
    V_vals = trap.get_trap_potential()

    # --- restrict data to 0 → 10 µm ---
    mask = (z >= -10e-6) & (z <= 10e-6)

    # --- reference value at point closest to zero ---
    z0_index = np.argmin(np.abs(z))
    z0 = z[z0_index]
    V0 = V_vals[z0_index]

    # Raw points from trap.get_trap_potential
    z_mask = z[mask]
    V_mask = V_vals[mask]
    U_points = e*(V_vals - V0)
    U_mask = e*(V_mask - V0)
    plt.scatter(z_mask*1e6, U_mask,s=5, color="red", marker="o", label="points from COMSOL simulation")

    # --- interpolator (only valid inside [z.min(), z.max()]) ---
    V_interp = interp1d(z_mask, V_mask, kind="cubic", bounds_error=True)

    # --- energy functions ---
    def U_coul(d):
        """Coulomb energy for two charges separated by 2d"""
        return k_c * e**2 / (2*d)

    def U_trap(d):
        """Trap energy relative to the center potential"""
        return e * (V_interp(d) - V0 + V_interp(-d) - V0)

    def U_tot(d):
        return U_coul(d) + U_trap(d)

    # --- scan range restricted to available z data ---
    eps = 1e-12
    pos = np.linspace(z_mask.min() + eps, z_mask.max() - eps, 500)

    U_c_vals = [U_coul(d) for d in pos]
    U_t_vals = [U_trap(d) for d in pos]
    U_vals   = [U_tot(d)  for d in pos]

    # restrict scan to positive d only
    mask_pos = pos >= 0
    pos_pos = pos[mask_pos]
    U_vals_pos = np.array(U_vals)[mask_pos]

    # find index of minimum in restricted range
    idx_min = np.argmin(U_vals_pos)
    d_min = pos_pos[idx_min]  # half-spacing [m]
    E_min = U_vals_pos[idx_min]  # minimum energy [J]

    # --- print results ---
    print(f"Equilibrium half-spacing = {d_min*1e6:.2f} µm")
    #print(f"Equilibrium full-spacing = {2*d_min*1e6:.2f} µm")

    # --- plot contributions ---
    plt.plot(pos*1e6, np.array(U_c_vals), label="Coulomb")
    plt.plot(pos*1e6, np.array(U_t_vals), label="Trap")
    plt.plot(pos*1e6, np.array(U_vals), label="Total", linewidth=2)

    # mark equilibrium point
    plt.plot(d_min*1e6, E_min, "ro", label="Equilibrium point")

    plt.xlabel("Half-spacing d [µm]")
    plt.ylabel("Energy relative to min [J]")
    plt.title("Coulomb, Trap, and Total potentials (0–10 µm range)")
    plt.legend()
    plt.grid(True)
    plt.show()



    """The different approach - taylor approximation of the potential (many orders) and from that 
    we can get the equilibrium point without depending on the potential from COMSOL in 1~10 μm"""

    # Restrict to a window near 0
    mask_2 = (z >= -5e-4) & (z <= 5e-4)
    z_small = z[mask_2]
    V_small = V_vals[mask_2]
    U_small = e*(V_small-V0)

    # Fit a polynomial of degree 10
    coeffs = np.polyfit(z_small, V_small, deg=8)
    poly = np.poly1d(coeffs)


    def U_trap_taylor(d):
        """Trap energy relative to the center potential"""
        return e*(poly(d) + poly(-d)-2*poly(0))

    def U_tot_taylor(d):
        return U_coul(d) + U_trap_taylor(d)
    #data for plots
    U_c_vals = [U_coul(d) for d in pos]
    U_t_vals = [U_trap_taylor(d) for d in z_small]
    U_vals   = [U_tot_taylor(d)  for d in pos]

    # restrict scan to positive d only
    mask_pos = pos >= 0
    pos_pos = pos[mask_pos]
    U_vals_pos = np.array(U_vals)[mask_pos]

    # find index of minimum in restricted range
    idx_min = np.argmin(U_vals_pos)
    d_min = pos_pos[idx_min]  # half-spacing [m]
    E_min = U_vals_pos[idx_min]  # minimum energy [J]

    # --- print results ---
    print(f"Equilibrium half-spacing = {d_min*1e6:.2f} µm")
    #print(f"Equilibrium full-spacing = {2*d_min*1e6:.2f} µm")

    # mark equilibrium point
    plt.plot(d_min*1e6, E_min, "ro", label="Equilibrium point Vbias = " + str(v))

    #plots
    plt.scatter(z_small*1e6, U_small,s=5, color="red", marker="o", label="points from COMSOL simulation")
    plt.plot(pos*1e6, np.array(U_c_vals), label="Coulomb")
    plt.plot(z_small*1e6, np.array(U_t_vals), label="Trap")
    plt.plot(pos*1e6, np.array(U_vals), label="Total", linewidth=2)
    plt.xlabel("Half-spacing d [µm]")
    plt.ylabel("Energy relative to min [J]")
    plt.title("Coulomb, Trap, and Total potentials (0–10 µm range)")
    plt.legend()
    plt.show()
    l = trap_length_scale(Z=1)
    print(f"Length scale ℓ = {l:.3e} m ({l * 1e6:.2f} µm)" + "d/2 from article = " + str(0.62996 * l))
