import numpy as np
import importlib
import PaulTrap
import matplotlib.pyplot as plt
import os
from scipy.constants import k
importlib.reload(PaulTrap)
from PaulTrap import PaulTrap

"""Trap parameters"""
m = 40  # 40Ca+
charge = 1
RF_freq = 2 * np.pi * 25e6

"""Creates new trap object"""
trap = PaulTrap(mass=m, RF_freq=RF_freq, charge=charge)

"""Plotting the potential of each electrode"""
trap.plot_electrode_potentials()

"""Example of DC=17V"""
trap.set_DC_voltages(V_EC_L=0, V_DC_L=17, V_BIAS=0, V_DC_R=17, V_EC_R=0)
trap.set_AC_voltage(AC_voltage=0)
trap.plot_trap_potential()
"""Gets the trap frequency"""
omega_z = trap.get_trap_frequency(center_position=0,width = 1)
freq = omega_z/(2*np.pi)
print(f"Trap axial frequency (omega_z): 2pi * {freq * 1e-6:.2f} MHz")

"""Example for another electrode configuration"""
"Sets the electrodes voltages"
trap.set_DC_voltages(V_EC_L=8, V_DC_L=9, V_BIAS=0, V_DC_R=7.5, V_EC_R=0)
trap.set_AC_voltage(AC_voltage=500)

"""Some analysis to get information about the min point, gets the min, max points (in each side) and the dY in each side"""
res = trap.min_max_delta(2,0.8,True)
"""Gets the trap frequency from parabola fit"""
omega_z = trap.get_trap_frequency(res['xmin'],0.1)
freq = omega_z/(2*np.pi)
"""Prints the frequency and the energy barrier"""
print("Kelvin barrier right = " + str(res['delta_y_R'] * trap.charge/k))
print("Kelvin barrier left = " + str(res['delta_y_L'] * trap.charge/k))
print(f"Trap axial frequency (omega_z): 2pi * {freq * 1e-6:.2f} MHz")
print(" (xmin,Vmin) = ", res['xmin'], res['ymin'])
"""at x=0"""
omega_z = trap.get_trap_frequency(0.035,0.1)
freq = omega_z/(2*np.pi)
print(f"Trap axial frequency (omega_z): 2pi * {freq * 1e-6:.2f} MHz")



"""Interactive trap potential"""
trap.plot_interactive_trap_potential()

"""Animation of the trap's potential in the desired configuration"""
voltages_path = [
    [0, 0,0 ,0 ,0 ,0],
    [0, 1, 0, 1,   0,   0],
    [2, 1, 0, 1,   2, 500],
    [2, 1, 8, 1,   2, 500],
    [8, 1, 8, 1,   0, 500],
    [8, 1, 8, 7.5, 0, 500],
    [8, 1, 0, 7.5, 0, 500],
    [8, 9, 0, 7.5, 0, 500],
]

"""Animation execution (saved as gif file in the same folder as run_script)"""
# trap.animate_voltage_path_gif(
#     voltages_path,
#     seconds_per_segment=4.0,    # how long each interval lasts
#     fps=30,                     # frames per second
#     save_path="trap_intervals.gif"  # file name / path
# )

"""In each spot on the axial axis:
1) plot the raw data from COMSOL of the trap's potential in a slice
2) plot the interpolation of the 0.1mm radius in the middle
3) plot the potential along the diagonal
4) parabola fit and derive radial frequency"""
xSlices = (0,0.6,1,2,2.5,2.6,2.7,2.8,2.825,2.85,2.875,2.9,3)
freqs = []
for xSlice in xSlices:
    """enter the window on y,z for the scatter, here x is the axial axis"""
    windowY = 2
    windowZ = 2
    """plot trap potential:
     1)raw data
     2)specific in center with interpolation)"""
    trap.plot_yz_scatter_from_slice(filepath=os.path.join(os.getcwd(), "electrodes responses", "Long Axial RF Slice x=" + str(xSlice) + ".txt"),skiprows=8,window_y=windowY, window_z=windowZ, title = "Potential around x=" + str(xSlice) + " mm (center, raw data)")
    yy, zz, VV = trap.plot_yz_heatmap_from_slice(filepath=os.path.join(os.getcwd(), "electrodes responses", "Long Axial RF Slice x=" + str(xSlice) + ".txt"),skiprows=8,window_y=0.15, window_z=0.15,grid_N=301,levels=0, title = "Potential around x=" + str(xSlice) + " mm (center, interpolated)")

    """plot the potential on the diagonals and calculate the radial frequency of the trap (along the diagonal y=z)"""
    r,V = trap.plot_diagonal_angle(yy, zz, VV, theta_deg=45, plot=True)
    omega_r,q,a,r0 = trap.fit_parabola_radial(r,V,0.02)
    freq = omega_r / (2 * np.pi)
    print(f"Trap radial frequency (omega_r) for 320V RF: 2pi * {freq * 320 * 1e-6:.2f} MHz")
    freqs.append(freq* 320 * 1e-6)
"""Scatter of the frequencies as a function of axial position"""
plt.scatter(xSlices, freqs)
plt.axvline(x=2.6, color='red', linestyle='--')
plt.xlabel("axial position (mm)")
plt.ylabel("radial frequency for 320V(MHz)")
plt.title("Radial frequency vs Axial position")
plt.grid(True)
plt.show()


"""check the difference in the frequency in y=z and y=-z"""
xSlices = (0,0.6,1,2,2.5,2.6,2.7,2.8,2.825,2.85,2.875,2.9,3)
freqs1 = []
freqs2 = []
for xSlice in xSlices:
    """enter the window on y,z for the scatter, here x is the axial axis"""
    windowY = 2
    windowZ = 2
    """plot trap potential:
     1)raw data
     2)specific in center with interpolation"""
    trap.plot_yz_scatter_from_slice(filepath=os.path.join(os.getcwd(), "electrodes responses", "Long Axial RF Slice x=" + str(xSlice) + ".txt"),skiprows=8,window_y=windowY, window_z=windowZ, title = "Potential around x=" + str(xSlice) + " mm (center, raw data)")
    yy, zz, VV = trap.plot_yz_heatmap_from_slice(filepath=os.path.join(os.getcwd(), "electrodes responses", "Long Axial RF Slice x=" + str(xSlice) + ".txt"),skiprows=8,window_y=0.15, window_z=0.15,grid_N=301,levels=0, title = "Potential around x=" + str(xSlice) + " mm (center, interpolated)")
    """plot the potential on the diagonals and calculate the radial frequency of the trap (along the diagonal y=z and y=-z)"""
    r1,V1 = trap.plot_diagonal_angle(yy, zz, VV, theta_deg=45, plot=True)
    r2,V2 = trap.plot_diagonal_angle(yy, zz, VV, theta_deg=-45, plot=True)
    """only the omega_r is relevant"""
    omega_r1,q,a,r0 = trap.fit_parabola_radial(r1,V1,0.09)
    freq1 = omega_r1 / (2 * np.pi)
    print(f"Trap radial frequency (omega_r) for 320V RF: 2pi * {freq1 * 320 * 1e-6:.2f} MHz")
    freqs1.append(freq1* 320 * 1e-6)
    omega_r2, q, a, r0 = trap.fit_parabola_radial(r2, V2, 0.09)
    freq2 = omega_r2 / (2 * np.pi)
    print(f"Trap radial frequency (omega_r) for 320V RF: 2pi * {freq2 * 320 * 1e-6:.2f} MHz")
    freqs2.append(freq2 * 320 * 1e-6)
difference_list = []
for item1,item2 in zip(freqs1,freqs2):
    difference_list.append(np.abs(item1) -np.abs(item2))
    print(difference_list[-1])
plt.scatter(xSlices, difference_list)
plt.axvline(x=2.6, color='red', linestyle='--')
plt.xlabel("axial position (mm)")
plt.ylabel("radial frequency difference for 320V(MHz)")
plt.title("Radial frequency difference vs Axial position")
plt.grid(True)
plt.show()












