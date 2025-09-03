import numpy as np
import importlib
import PaulTrap
import matplotlib.pyplot as plt
import os
from scipy.constants import k
importlib.reload(PaulTrap)
from PaulTrap import PaulTrap

# Params
m = 40  # 40Ca+
charge = 1
RF_freq = 2 * np.pi * 25e6

trap = PaulTrap(mass=m, RF_freq=RF_freq, charge=charge)

trap.plot_electrode_potentials()
# Example DC + AC
trap.set_DC_voltages(V_EC_L=8, V_DC_L=9, V_BIAS=0, V_DC_R=8, V_EC_R=0)
trap.set_AC_voltage(500)
res = trap.min_max_delta(2,0.8,True)
omega_z = trap.get_trap_frequency(res['xmin'],0.1)
freq = omega_z/(2*np.pi)
print("Kelvin barrier right = " + str(res['delta_y_R'] * trap.charge/k))
print("Kelvin barrier left = " + str(res['delta_y_L'] * trap.charge/k))
print(f"Trap axial frequency (omega_z): 2pi * {freq * 1e-6:.2f} MHz")
print(" (xmin,Vmin) = ", res['xmin'], res['ymin'])


#V,K,x = trap.get_total_voltage_barrier(3,0.2,1,True)
#omega_z = trap.get_trap_frequency(x,0.3)
#freq = omega_z/(2*np.pi)
#print(f"Trap radial frequency (omega_z) at x = 3.036mm : 2pi * {freq * 1e-6:.2f} MHz")
#print ("Energy barrier of the trapped ion near local min is : " + str(V) + " Kelvin")

#V,K,x = trap.get_total_voltage_barrier(0,0.2,0.1,True)
#omega_z = trap.get_trap_frequency(0,0.1)
#freq = omega_z/(2*np.pi)
#print(f"Trap radial frequency (omega_z) at x = 0.007mm : 2pi * {freq * 1e-6:.2f} MHz")
#print ("Energy barrier of the trapped ion near local min is : " + str(K) + " Kelvin")


#trap.plot_trap_potential()
trap.set_DC_voltages(V_EC_L=0, V_DC_L=0, V_BIAS=0, V_DC_R=0, V_EC_R=0)
trap.set_AC_voltage(500)
xSlice = 0
windowY = 2
windowZ = 2
trap.plot_yz_scatter_from_slice(filepath=os.path.join(os.getcwd(), "electrodes responses", "Long Axial RF Slice x=" + str(xSlice) + ".txt"),skiprows=8,window_y=windowY, window_z=windowZ, title = "Potential around x=" + str(xSlice) + " mm (center, raw data)")
yy, zz, VV = trap.plot_yz_heatmap_from_slice(filepath=os.path.join(os.getcwd(), "electrodes responses", "Long Axial RF Slice x=" + str(xSlice) + ".txt"),skiprows=8,window_y=0.15, window_z=0.15,grid_N=301,levels=0, title = "Potential around x=" + str(xSlice) + " mm (center, interpolated)")
#plot the potential on the diagonals and calculate the radial frequency of the trap (along the diagonal y=z)
(r,V),(s,T) = trap.plot_diagonals(yy, zz, VV, plot=True)
r = r*np.sqrt(2)
omega_r,qa,q,a,r0,b = trap.fit_parabola_radial(r,V,0.05)
freq = omega_r / (2 * np.pi)
print(f"Trap radial frequency (omega_r) for 320V RF: 2pi * {freq * 320 * 1e-6:.2f} MHz")
#trap.plot_trap_potential()
#trap.plot_interactive_trap_potential()
#insert the x {0,1,2,3,0.6,2.6} to see the slice in the radial surface.
xSlices = (0,0.6,1,2,2.5,2.6,2.7,2.8,2.9,3)
freqs = []
for xSlice in xSlices:
    #enter the window on y,z for the scatter
    windowY = 2
    windowZ = 2
    #plot trap potential (raw data and specific in center with interpolation)
    trap.plot_yz_scatter_from_slice(filepath=os.path.join(os.getcwd(), "electrodes responses", "Long Axial RF Slice x=" + str(xSlice) + ".txt"),skiprows=8,window_y=windowY, window_z=windowZ, title = "Potential around x=" + str(xSlice) + " mm (center, raw data)")
    yy, zz, VV = trap.plot_yz_heatmap_from_slice(filepath=os.path.join(os.getcwd(), "electrodes responses", "Long Axial RF Slice x=" + str(xSlice) + ".txt"),skiprows=8,window_y=0.15, window_z=0.15,grid_N=301,levels=0, title = "Potential around x=" + str(xSlice) + " mm (center, interpolated)")
    #plot the potential on the diagonals and calculate the radial frequency of the trap (along the diagonal y=z)
    r,V = trap.plot_diagonal_angle(yy, zz, VV, theta_deg=45, plot=True)
    omega_r,qa,q,a,r0,b = trap.fit_parabola_radial(r,V,0.05)
    freq = omega_r / (2 * np.pi)
    print(f"Trap radial frequency (omega_r) for 320V RF: 2pi * {freq * 320 * 1e-6:.2f} MHz")
    freqs.append(freq* 320 * 1e-6)
    print("angle = " + str(45)+ "---" + "r0 = " + str(r0*1e3) + "mm")









