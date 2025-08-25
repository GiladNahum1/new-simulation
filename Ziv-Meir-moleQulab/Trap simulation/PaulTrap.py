import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from scipy.interpolate import interp1d
from scipy.constants import k
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, griddata

def find_closest_index(array, value):
    differences = np.abs(array - value)
    index = np.argmin(differences)
    return index

def _unique_sorted(x, y):
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    return x, y

def create_position_vector(self):
    x0, x1 = float(self.arc_length_AC_E[0]), float(self.arc_length_AC_E[-1])
    n = len(self.arc_length_AC_E)  # keep same density as source
    self.position_vector = np.linspace(x0, x1, n)


class PaulTrap():
    def __init__(self, mass, RF_freq, charge=1):
        self.mass = mass * 1.66e-27        # amu -> kg
        self.charge = charge * 1.602e-19   # e -> C
        self.RF_freq = RF_freq

        # Load electrode responses
        self.initialize_voltage_responses()
        self.create_position_vector()
        self.interpolate_potentials()
        self.mirror_potentials()
        self.shift_position_vector()

        # Build ponderomotive potential
        self.add_effective_AC_potential()

        # Voltages default
        self.V_EC_L = 0
        self.V_DC_L = 0
        self.V_BIAS = 0
        self.V_DC_R = 0
        self.V_EC_R = 0
        self.AC_Voltage = 0

    def initialize_voltage_responses(self):
        script_dir = os.getcwd()
        file_EC   = os.path.join(script_dir, "electrodes responses", "Shuttling - Axial - E (EC) - Export.txt")
        file_DC   = os.path.join(script_dir, "electrodes responses", "Shuttling - Axial - G+D (DC) - Export.txt")
        file_BIAS = os.path.join(script_dir, "electrodes responses", "Shuttling - Axial - H (BIAS).txt")
        file_AC_E = os.path.join(script_dir, "electrodes responses", "RF Long Axial - Electric Field (RF Barrier).txt")

        data_EC   = np.loadtxt(file_EC, skiprows=8)
        data_DC   = np.loadtxt(file_DC, skiprows=8)
        data_BIAS = np.loadtxt(file_BIAS, skiprows=8)
        data_AC_E = np.loadtxt(file_AC_E, skiprows=8)

        self.arc_length_EC_L = data_EC[:, 0]
        self.EC_L = data_EC[:, 1]

        self.arc_length_DC_L = data_DC[:, 0]
        self.DC_L = data_DC[:, 1]

        self.arc_length_BIAS = data_BIAS[:, 0]
        self.BIAS = data_BIAS[:, 1]

        self.arc_length_AC_E = data_AC_E[:, 0]
        self.AC_E = data_AC_E[:, 1]

    def create_position_vector(self):
        # Use RF axis for common grid
        self.position_vector = np.linspace(self.arc_length_AC_E[0],
                                           self.arc_length_AC_E[-1],
                                           len(self.arc_length_AC_E))

    def shift_position_vector(self):
        self.position_vector -= self.position_vector[-1] / 2

    def interpolate_potentials(self):
        # Dedup first
        self.arc_length_EC_L, self.EC_L = _unique_sorted(self.arc_length_EC_L, self.EC_L)
        self.arc_length_DC_L, self.DC_L = _unique_sorted(self.arc_length_DC_L, self.DC_L)
        self.arc_length_BIAS, self.BIAS = _unique_sorted(self.arc_length_BIAS, self.BIAS)
        self.arc_length_AC_E, self.AC_E = _unique_sorted(self.arc_length_AC_E, self.AC_E)

        x = self.position_vector

        # DC & BIAS can be quadratic (they’re smoother shapes)
        self.EC_L = interp1d(self.arc_length_EC_L, self.EC_L, kind="quadratic",
                             bounds_error=False, fill_value=(self.EC_L[0], self.EC_L[-1]))(x)
        self.DC_L = interp1d(self.arc_length_DC_L, self.DC_L, kind="quadratic",
                             bounds_error=False, fill_value=(self.DC_L[0], self.DC_L[-1]))(x)
        self.BIAS = interp1d(self.arc_length_BIAS, self.BIAS, kind="quadratic",
                             bounds_error=False, fill_value=(self.BIAS[0], self.BIAS[-1]))(x)
        ac_lin = interp1d(self.arc_length_AC_E, self.AC_E, kind="linear",
                          bounds_error=False,
                          fill_value=(self.AC_E[0], self.AC_E[-1]))
        self.AC_E = ac_lin(x)

        # Optional: very gentle smoothing to match the TXT “look”
        # (sigma=1 is small; increase to 2 if you want even smoother)
        self.AC_E = gaussian_filter1d(self.AC_E, sigma=2)

    def mirror_potentials(self):
        self.EC_R = self.EC_L[::-1]
        self.DC_R = self.DC_L[::-1]

    def add_effective_AC_potential(self):
        self.effective_AC_potential = (self.charge / (4 * self.mass * self.RF_freq ** 2)) * (self.AC_E ** 2)
        # If you want, you can also smooth the ponderomotive a touch:
        self.effective_AC_potential = gaussian_filter1d(self.effective_AC_potential, sigma=1)

    def set_DC_voltages(self, V_EC_L, V_DC_L, V_BIAS, V_DC_R, V_EC_R):
        self.V_EC_L = V_EC_L
        self.V_DC_L = V_DC_L
        self.V_BIAS = V_BIAS
        self.V_DC_R = V_DC_R
        self.V_EC_R = V_EC_R

    def set_AC_voltage(self, AC_voltage):
        self.AC_Voltage = AC_voltage

    def get_trap_potential(self):
        return (self.V_EC_R * self.EC_R +
                self.V_EC_L * self.EC_L +
                self.V_DC_R * self.DC_R +
                self.V_DC_L * self.DC_L +
                self.V_BIAS * self.BIAS +
                (self.AC_Voltage**2) * self.effective_AC_potential)

    def plot_trap_potential(self):
        total_potential = self.get_trap_potential()
        plt.figure(figsize=(7, 5))
        plt.plot(self.position_vector, total_potential, label="Total Potential")
        plt.title("Axial Trap Potential")
        plt.xlabel("Axial axis (mm)")
        plt.ylabel("Potential (V)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_electrode_potentials(self):
        plt.figure(figsize=(7, 5))
        plt.plot(self.position_vector, self.EC_R, label="Endcap Right")
        plt.plot(self.position_vector, self.DC_R, label="DC Right")
        plt.plot(self.position_vector, self.BIAS, label="BIAS")
        plt.plot(self.position_vector, self.EC_L, label="Endcap Left")
        plt.plot(self.position_vector, self.DC_L, label="DC Left")
        plt.title("Electrode Potentials")
        plt.xlabel("Axial axis (mm)")
        plt.ylabel("Potential (V)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def find_local_min(self, x0, search_width=1.0):

        V = self.get_trap_potential()
        x = self.position_vector

        left, right = x0 - 0.5 * search_width, x0 + 0.5 * search_width
        mask = (x >= left) & (x <= right)
        idxs = np.nonzero(mask)[0]

        if len(idxs) == 0:
            raise ValueError("No points in the specified window")

        jwin = np.argmin(V[idxs])
        j = idxs[jwin]

        return x[j], V[j], j

    def get_RF_barrier(self):
        RF_barrier = np.max(self.AC_Voltage**2 * self.effective_AC_potential)
        return RF_barrier, RF_barrier * self.charge / k

    def fit_parabola(self, center_position, width):  # all in mm
        center_index = find_closest_index(self.position_vector, center_position)
        gap = (self.position_vector[1] - self.position_vector[0])
        half_window = int(width / (2 * gap))
        start_index = max(0, center_index - half_window)
        end_index   = min(len(self.position_vector), center_index + half_window)
        coefficients = np.polyfit(self.position_vector[start_index:end_index],
                                  self.get_trap_potential()[start_index:end_index], 2)
        self.plot_with_fit(coefficients, start_index, end_index)
        return coefficients

    def plot_with_fit(self, coefficients, start_index, end_index):
        plt.figure(figsize=(7, 5))
        x_slice = self.position_vector[start_index:end_index]
        y_slice = self.get_trap_potential()[start_index:end_index]
        plt.scatter(x_slice, y_slice, label='Potential', color='red', s=6)
        y_fit = np.polyval(coefficients, x_slice)
        plt.plot(x_slice, y_fit, label='Parabola Fit')
        plt.title('Axial Potential with Parabola Fit')
        plt.xlabel('Axial axis (mm)')
        plt.ylabel('Potential (V)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_trap_frequency(self, center_position, width):
        coefficients = self.fit_parabola(center_position, width)
        alpha = coefficients[0] * 1e6  # mm^2 -> m^2 (assuming x is mm)
        omega_z = np.sqrt(2 * alpha * self.charge / self.mass)
        return omega_z

    def find_local_maxima(self, x, V):
        peaks = []
        for i in range(1, len(V) - 1):
            if V[i] > V[i - 1] and V[i] > V[i + 1]:
                peaks.append((x[i], V[i]))
        return peaks

    def mask_local_min(self, x0, search_width=2.0, zoom_width=0.5, plot=False):
        x = self.position_vector
        V = self.get_trap_potential()
        x_min, V_min, idx = self.find_local_min(x0, search_width=search_width)
        mask = (x >= x_min - zoom_width / 2) & (x <= x_min + zoom_width / 2)
        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(x[mask], V[mask], label="Trap potential")
            plt.scatter([x_min], [V_min], color="red", s=60, zorder=3, label="Local min")
            plt.title(f"Zoom near local min at {x_min:.3f} mm")
            plt.xlabel("Axial axis (mm)")
            plt.ylabel("Potential (V)")
            plt.legend()
            plt.tight_layout()
            plt.show()
        return x_min, V_min, x[mask], V[mask]

    def get_total_voltage_barrier(self, x0, search_width=1.0, width=0.5, plot=False):
        x_min, V_min, x_slice, V_slice = self.mask_local_min(x0, search_width, width, plot)
        peaks = self.find_local_maxima(x_slice, V_slice)
        if not peaks:
            raise ValueError("No peak found near the minimum.")
        Vbarrier = float(peaks[0][1]) - float(V_min)
        Kbarrier = self.charge * Vbarrier / k
        return Vbarrier, Kbarrier

    def plot_interactive_trap_potential(self):
        root = tk.Tk()
        root.title("Interactive Trap Potential")

        frame = ttk.Frame(root, padding="15")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Voltages: EC_L, DC_L, BIAS, DC_R, EC_R, AC
        voltages = [self.V_EC_L, self.V_DC_L, self.V_BIAS, self.V_DC_R, self.V_EC_R, self.AC_Voltage]
        labels   = ["V_EC_L", "V_DC_L", "V_BIAS", "V_DC_R", "V_EC_R", "V_AC"]

        sliders, entries = [], []

        fig, ax = plt.subplots(figsize=(6, 4))
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().grid(row=len(labels), column=0, columnspan=3,
                                    sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        def refresh_plot():
            total_potential = self.get_trap_potential()
            ax.clear()
            ax.plot(self.position_vector, total_potential, label='Electric Potential')
            ax.set_title('Axial axis vs Electric Potential')
            ax.set_xlabel('Axial axis (mm)')
            ax.set_ylabel('Electric Potential (V))')
            ax.legend()
            canvas.draw()

        def update_potential(index, value):
            value = float(value)
            voltages[index] = value
            self.set_DC_voltages(*voltages[:5])
            self.set_AC_voltage(voltages[5])
            if index < len(entries):
                entries[index].delete(0, tk.END)
                entries[index].insert(0, f"{value:.2f}")
            refresh_plot()

        def update_from_entry(index, entry_widget):
            try:
                value = float(entry_widget.get())
                voltages[index] = value
                self.set_DC_voltages(*voltages[:5])
                self.set_AC_voltage(voltages[5])
                sliders[index].set(value)
                refresh_plot()
            except ValueError:
                pass

        for i, label_text in enumerate(labels):
            ttk.Label(frame, text=label_text, font=("Arial", 12)).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            slider_range = (0, 20) if i < 5 else (0, 500)
            slider = ttk.Scale(frame, from_=slider_range[0], to=slider_range[1],
                               orient=tk.HORIZONTAL, length=250,
                               command=lambda value, idx=i: update_potential(idx, value))
            slider.set(voltages[i])
            slider.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
            sliders.append(slider)

            entry = ttk.Entry(frame, width=8, font=("Arial", 12))
            entry.insert(0, str(voltages[i]))
            entry.grid(row=i, column=2, padx=5, pady=5)
            entries.append(entry)
            entry.bind("<Return>", lambda event, idx=i, e=entry: update_from_entry(idx, e))

        frame.columnconfigure(1, weight=1)
        refresh_plot()
        root.mainloop()

    def plot_yz_heatmap_from_slice(self,filepath,skiprows=8,delimiter=None,window_y=None,window_z=None,grid_N=301,center_crosshair=True,levels=15,cmap="viridis",title=None):
        data = np.loadtxt(filepath, skiprows=skiprows, delimiter=delimiter)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"Expected 3 or 4 columns, got shape {data.shape}")
        y = data[:, 1]
        z = data[:, 2]
        V = data[:, 3]
        # Determine symmetric plot window around (0,0)
        y_full = max(abs(np.min(y)), abs(np.max(y)))
        z_full = max(abs(np.min(z)), abs(np.max(z)))
        ymax = abs(window_y) if window_y is not None else y_full
        zmax = abs(window_z) if window_z is not None else z_full

        # Build uniform grid
        yy = np.linspace(-ymax, ymax, grid_N)
        zz = np.linspace(-zmax, zmax, grid_N)
        YY, ZZ = np.meshgrid(yy, zz, indexing="xy")

        # Interpolate scattered samples onto the grid
        # cubic for smooth interior + linear fallback on edges
        from scipy.interpolate import griddata
        pts = np.column_stack([y, z])
        VV_cubic = griddata(pts, V, (YY, ZZ), method="cubic")
        VV_lin = griddata(pts, V, (YY, ZZ), method="linear")
        VV = np.where(np.isnan(VV_cubic), VV_lin, VV_cubic)

        # Symmetric color scale around median (optional but looks nice)
        v_center = np.nanmedian(VV)
        v_span = np.nanmax(np.abs(VV - v_center))
        vmin, vmax = v_center - v_span, v_center + v_span

        # Plot
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        im = ax.imshow(
            VV.T,
            extent=[-ymax, ymax, -zmax, zmax],
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Potential (V)")


        if center_crosshair:
            ax.axhline(0, color="w", lw=0.8, alpha=0.8)
            ax.axvline(0, color="w", lw=0.8, alpha=0.8)

        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_title(title if title else f"V(y,z) heatmap — {os.path.basename(filepath)}")
        plt.tight_layout()
        plt.show()

        # Return grid & field for further analysis if needed
        return yy, zz, VV

