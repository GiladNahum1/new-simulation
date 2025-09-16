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
"""Find min point for each different assymetric DC voltage"""
Vdc_L =  [1,2,3,4,5]
for Vdc in Vdc_L:
    trap.set_DC_voltages(V_EC_L=5, V_DC_L=Vdc, V_BIAS=5, V_DC_R=3, V_EC_R=5)
    trap.set_AC_voltage(AC_voltage=0)
    trap.plot_trap_potential()

    x_r,V,j = trap.find_local_min(x0=1.5,search_width=0.5)
    x_l, V, j = trap.find_local_min(x0=-1.5, search_width=0.5)
    print ("Vdc = " + str(Vdc) + "x_r = " +str(x_r) + "x_l = " + str(x_l))



