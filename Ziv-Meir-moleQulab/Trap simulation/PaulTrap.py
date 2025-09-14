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
from scipy.interpolate import RegularGridInterpolator
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

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
        """Initialize the trap parameters"""
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
        """Gets data from COMSOL simulation for each electrode"""
        script_dir = os.getcwd()
        file_EC   = os.path.join(script_dir, "electrodes responses", "Shuttling - Axial - E (EC) - Export.txt")
        file_DC   = os.path.join(script_dir, "electrodes responses", "High resolution G+D (DC) with fine mesh.txt")
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
        #
        """Creates position vector of the trap (linspace
        from 0 to 10mm and not -5mm to 5mm because that's the data we get from COMSOL simulation).Use RF axis for common grid"""
        self.position_vector = np.linspace(self.arc_length_AC_E[0],
                                           self.arc_length_AC_E[-1],
                                           len(self.arc_length_AC_E))


    def shift_position_vector(self):
        """Makes the center 0 (-5mm to 5mm position vector), it's easier to work with symmetric position vector."""
        self.position_vector -= self.position_vector[-1] / 2


    def interpolate_potentials(self):
        """Match the COMSOL data to the position vector created via interpolation"""
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

        """very gentle smoothing """
        """""""(sigma=1 is small; increase to 2 if you want even smoother)"""
        self.AC_E = gaussian_filter1d(self.AC_E, sigma=2)
        """self.DC_L = gaussian_filter1d(self.DC_L, sigma=2)"""
        self.EC_L = gaussian_filter1d(self.EC_L, sigma=2)


    def mirror_potentials(self):
        """Uses the symmetry of DC and EC electrodes to infer their twins potential(shortcut)"""
        self.EC_R = self.EC_L[::-1]
        self.DC_R = self.DC_L[::-1]


    def add_effective_AC_potential(self):
        """Calculation of the Ponderomotive Potential from the RF electrodes"""
        #self.AC_E is the electric field
        self.effective_AC_potential = (self.charge / (4 * self.mass * self.RF_freq ** 2)) * (self.AC_E ** 2)
        # If you want, you can also smooth the ponderomotive a touch:
        self.effective_AC_potential = gaussian_filter1d(self.effective_AC_potential, sigma=1)



    def set_DC_voltages(self, V_EC_L, V_DC_L, V_BIAS, V_DC_R, V_EC_R):
        """Sets the voltages on each electrode (DC, EC and Bias)"""
        self.V_EC_L = V_EC_L
        self.V_DC_L = V_DC_L
        self.V_BIAS = V_BIAS
        self.V_DC_R = V_DC_R
        self.V_EC_R = V_EC_R

    def set_AC_voltage(self, AC_voltage):
        """Sets the voltage on the RF electrode"""
        self.AC_Voltage = AC_voltage


    def get_trap_potential(self):
        """Uses the linearity of the laplace equation to calculate the total trap
        potential for a given voltage setup"""
        return (self.V_EC_R * self.EC_R +
                self.V_EC_L * self.EC_L +
                self.V_DC_R * self.DC_R +
                self.V_DC_L * self.DC_L +
                self.V_BIAS * self.BIAS +
                (self.AC_Voltage**2) * self.effective_AC_potential)

    def plot_trap_potential(self):
        """Plotting the total trap potential"""
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
        """Plotting the effect of each electrode"""
        plt.figure(figsize=(7, 5))
        plt.plot(self.position_vector, self.EC_R, label="Endcap Right", color = "red")
        plt.plot(self.position_vector, self.DC_R, label="DC Right", color = "blue")
        plt.plot(self.position_vector, self.BIAS, label="BIAS", color = "green")
        plt.plot(self.position_vector, self.EC_L, label="Endcap Left", color ="orange")
        plt.plot(self.position_vector, self.DC_L, label="DC Left", color = "purple")
        plt.plot(self.position_vector, self.effective_AC_potential*20000, label="RF")
        plt.title("Electrode Potentials")
        plt.xlabel("Axial axis (mm)")
        plt.ylabel("Potential (V)")
        plt.legend()
        plt.tight_layout()
        ax = plt.gca()
        self._add_electrode_boxes(ax)
        plt.show()

    def _add_electrode_boxes(self, ax, spans=None, height=0.08):
        """Putting boxes on the grid to see the electrodes position"""
        if spans is None:
            # Non-overlapping, left→right:
            spans = [
                (-5.0, -2.7, "EC_L"),
                (-2.60, -0.6, "DC_L"),
                (-0.50, 0.50, "Bias"),
                (0.6, 2.6, "DC_R"),
                (2.7, 5.00, "EC_R"),  # ← EC starts after the DC span
            ]

        colors = {"EC_L": "tab:orange", "DC_L": "tab:purple", "Bias": "tab:green","EC_R": "tab:red", "DC_R": "tab:blue"}

        # x: data coords, y: axes coords (0..1)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

        y0 = 1 - height  # stick to the very top
        for x0, x1, label in spans:
            if x0 > x1:
                x0, x1 = x1, x0
            rect = Rectangle((x0, y0), x1 - x0, height,
                             transform=trans, clip_on=False,
                             facecolor=colors.get(label, "gray"),
                             alpha=0.35, edgecolor="none")
            ax.add_patch(rect)
            xm = 0.5 * (x0 + x1)
            ax.text(xm, y0 + 0.5 * height, label,
                    transform=trans, ha="center", va="center",
                    fontsize=10, color="black", weight="bold", clip_on=False)


    def find_local_min(self, x0, search_width=1.0):
        """Finding local minimum point (not using it in code)"""
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
        """Getting the voltage and energy barrier only from the RF electrode (not using in code)"""
        RF_barrier = np.max(self.AC_Voltage**2 * self.effective_AC_potential)
        return RF_barrier, RF_barrier * self.charge / k


    def fit_parabola(self, center_position, width):  # all in mm
        """Fit a minimum point to harmonic potential and plot it. Be careful, the coefficients need unit adjustment"""
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
        """Use coefficients of the parabola to plot the fit over the potential and position vector"""
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
        """Calculation: using parabola coefficients to get the axial frequency of the trap"""
        coefficients = self.fit_parabola(center_position, width)
        alpha = coefficients[0] * 1e6  # mm^2 -> m^2 (assuming x is mm)
        omega_z = np.sqrt(2 * alpha * self.charge / self.mass)
        return omega_z

    def find_local_maxima(self, x, V):
        """Finding maximum point  (not using in code)"""
        return np.max(V)

    def mask_local_min(self, x0, search_width=2.0, zoom_width=0.5, plot=False):
        """gets part of the position vector around a minimum point (not using in code)"""
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
        """Getting the total voltage and energy barrier (in volts and kelvin)"""
        x_min, V_min, x_slice, V_slice = self.mask_local_min(x0, search_width, width, plot)
        Vmax = self.find_local_maxima(x_slice, V_slice)
        Vbarrier = float(Vmax) - float(V_min)
        Kbarrier = self.charge * Vbarrier / k
        return Vbarrier, Kbarrier, x_min

    def plot_interactive_trap_potential(self):
        """Plotting an interactive GUI that allows us to easily change the voltage
         in each electrode and see the total potential curve"""
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
            self._add_electrode_boxes(ax)
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

    def plot_yz_scatter_from_slice(self, filepath, *,skiprows=8, delimiter=None,y_col=1, z_col=2, v_col=3,window_y=None, window_z=None,cmap="viridis",center_crosshair=True,centered_colors=False,point_size=6,axes_in_mm=False,title=None):
        """Plotting the raw data from COMSOL of a thin section in the radial axis of the trap
        (here x is the axial axis) We can see the RF electrode with 1V and the DC/bias/EC electrodes potential
        is held at 0. The interesting points we need to analyze are in the middle.
        We need to get the barrier and the radial frequency of the trap from it."""
        data = np.loadtxt(filepath, skiprows=skiprows, delimiter=delimiter)
        if data.ndim != 2 or data.shape[1] <= max(y_col, z_col, v_col):
            raise ValueError(f"{os.path.basename(filepath)} has shape {data.shape}; check column indices.")

        y = data[:, y_col].astype(float)
        z = data[:, z_col].astype(float)
        V = data[:, v_col].astype(float)


        # Window around origin: if None, use true symmetric extent of data
        y_full = max(abs(np.nanmin(y)), abs(np.nanmax(y)))
        z_full = max(abs(np.nanmin(z)), abs(np.nanmax(z)))
        ymax = abs(window_y) if window_y is not None else y_full
        zmax = abs(window_z) if window_z is not None else z_full

        # Ignore all points outside the window
        maskY = (y >= -ymax) & (y <= ymax)
        maskZ = (z >= -zmax) & (z <= zmax)
        mask = maskY & maskZ
        y_win, z_win, V_win = y[mask], z[mask], V[mask]

        # Color scaling
        if centered_colors:
            v_center = np.nanmedian(V_win)
            v_span = np.nanmax(np.abs(V_win - v_center))
            vmin, vmax = v_center - v_span, v_center + v_span
        else:
            vmin, vmax = float(np.nanmin(V_win)), float(np.nanmax(V_win))

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(y, z, c=V, s=point_size, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlim(-ymax, ymax)
        ax.set_ylim(-zmax, zmax)
        ax.set_aspect("equal", adjustable="box")

        cbar = plt.colorbar(sc, ax=ax, shrink=0.9)
        cbar.set_label("Potential (V)")

        if center_crosshair:
            ax.axhline(0, color="w", lw=0.8, alpha=0.8)
            ax.axvline(0, color="w", lw=0.8, alpha=0.8)

        ax.set_xlabel("y (mm)" if not axes_in_mm else "y (mm)")
        ax.set_ylabel("z (mm)" if not axes_in_mm else "z (mm)")
        if axes_in_mm:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v * 1e3:.1f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v * 1e3:.1f}"))

        fname = os.path.basename(filepath)
        ax.set_title(title if title else f"Raw V(y,z) scatter — {fname}")
        y_line = np.linspace(-ymax, ymax, 400)
        ax.plot(y_line, y_line, 'r--', lw=1, label="y = z")
        plt.tight_layout()
        plt.show()

        return y, z, V

    def plot_yz_heatmap_from_slice(self, filepath, skiprows=8, delimiter=None,window_y=None, window_z=None, grid_N=301, levels=15, cmap="viridis",title=None, point_size=8, point_color="k"):
        """Plotting a heatmap of the potential in small area in the radial axis (interpolated data)"""

        data = np.loadtxt(filepath, skiprows=skiprows, delimiter=delimiter)
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(f"Expected at least 4 columns, got shape {data.shape}")

        y = data[:, 1]
        z = data[:, 2]
        V = data[:, 3]

        # Determine symmetric plot window around (0,0)
        y_full = max(abs(np.min(y)), abs(np.max(y)))
        z_full = max(abs(np.min(z)), abs(np.max(z)))
        ymax = abs(window_y) if window_y is not None else y_full
        zmax = abs(window_z) if window_z is not None else z_full

        # Ignore all points outside the window
        maskY = (y >= -ymax) & (y <= ymax)
        maskZ = (z >= -zmax) & (z <= zmax)
        mask = maskY & maskZ
        y_win, z_win, V_win = y[mask], z[mask], V[mask]

        # Build uniform grid from the points
        yy = np.linspace(-ymax, ymax, grid_N)
        zz = np.linspace(-zmax, zmax, grid_N)
        YY, ZZ = np.meshgrid(yy, zz, indexing="xy")

        # Interpolate scattered samples onto the grid
        from scipy.interpolate import griddata
        pts = np.column_stack([y_win, z_win])
        VV_cubic = griddata(pts, V_win, (YY, ZZ), method="cubic")
        VV_lin = griddata(pts, V_win, (YY, ZZ), method="linear")
        VV = np.where(np.isnan(VV_cubic), VV_lin, VV_cubic)

        # Symmetric color scale around median
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

        # Overlay the original points for interpolation accuracy check
        ax.scatter(y_win, z_win, c=V_win, cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   s=point_size, edgecolor="w", linewidths=0.3,
                   marker="o", label="Original data")

        cbar = plt.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Potential (V)")

        ax.set_xlabel("y (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title(title if title else f"V(y,z) heatmap — {os.path.basename(filepath)}")
        plt.tight_layout()
        plt.show()

        return yy, zz, VV

    def plot_diagonals(self,yy, zz, VV, n=501,plot = True):

        """Plotting the potential on the diagonals (should get parabola)
        not giving the exact diagonals, instead using the plot_diagonal_angle with the 45 degrees angle"""
        yy = np.asarray(yy)
        zz = np.asarray(zz)
        VV = np.asarray(VV)

        # Build interpolator
        f = RegularGridInterpolator((yy, zz), VV, bounds_error=False, fill_value=np.nan)

        # --- y = z ---
        s_lo = max(yy.min(), zz.min())
        s_hi = min(yy.max(), zz.max())
        s = np.linspace(s_lo, s_hi, n)
        pts = np.column_stack([s, s])
        V_y_eq_z = f(pts)
        plt.figure(figsize=(6, 4))
        plt.plot(s, V_y_eq_z, lw=1.8)
        plt.xlabel("s along y=z (mm)")
        plt.ylabel("Potential V (V)")
        plt.title("Potential along y = z")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- y = -z ---
        s_lo = max(yy.min(), -zz.max())
        s_hi = min(yy.max(), -zz.min())
        s2 = np.linspace(s_lo, s_hi, n)
        pts2 = np.column_stack([s2, -s2])
        V_y_eq_minus_z = f(pts2)
        plt.figure(figsize=(6, 4))
        plt.plot(s2, V_y_eq_minus_z, lw=1.8, color="orange")
        plt.xlabel("s along y=-z (mm)")
        plt.ylabel("Potential V (V)")
        plt.title("Potential along y = -z")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        return (s, V_y_eq_z), (s2, V_y_eq_minus_z)

    #fitting the diagonals data to parabola and from the coefficients we derive the radial frequency, the barrier is just (detla V * m / Boltzman constant) Kelvins

    def fit_parabola_radial(self,x,V,range):
        mask = (x >= -range) & (x <= range)
        x_slice = x[mask]
        V_slice = V[mask]
        coefficients = np.polyfit(x_slice,V_slice,2)
        y_fit = np.polyval(coefficients, x_slice)
        plt.plot(x_slice, y_fit, label="Parabola fit", color = "red")
        plt.plot(x_slice,V_slice,'o',ms =1,label='Potential', color = "blue")
        plt.title('Axial Potential with Parabola Fit')
        plt.xlabel('Axial axis (mm)')
        plt.ylabel('Interpolated potential (V)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        r0 = 0.0005
        V0 = 1
        a = coefficients[0]*1e6 #from mm^2 to m^2
        omega = (np.sqrt(2) * a * self.charge / (self.mass*self.RF_freq))
        qAna =  2*self.charge*V0/(self.mass*(self.RF_freq**2)*(r0**2))
        qNum = 4*self.charge*(a*V0)/(self.mass*(self.RF_freq**2))
        r0 = 1/np.sqrt(2*a)
        b = coefficients[1]*1e3
        return omega,qAna,qNum,a,r0,b

    #Searching for a minimum point near x0 and after that marking the maximum points near it on each side, plotting it and returning min point, 2 max points and the voltage barrier on each side. we need to be accurate on the window and on the search width

    def min_max_delta(self,x0, search_width=None, plot = False):

        x = self.position_vector
        y = self.get_trap_potential()
        left, right = x0 - 0.5 * search_width, x0 + 0.5 * search_width
        mask = (x >= left) & (x <= right)
        idx = np.flatnonzero(mask)
        xs, ys = x[idx], y[idx]
        imin_rel = int(np.argmin(ys))
        imin = idx[imin_rel]
        left, right = x[imin] - 0.5 * search_width, x[imin] + 0.5 * search_width
        mask_L, mask_R = (x >= left) & (x <= x[imin]) , (x >= x[imin]) & (x<= right)
        idx_L,idx_R = np.flatnonzero(mask_L), np.flatnonzero(mask_R)
        xs_L,xs_R,ys_L,ys_R = x[idx_L], x[idx_R], y[idx_L], y[idx_R]
        imax_rel_L,imax_rel_R = int(np.argmax(ys_L)), int(np.argmax(ys_R))
        imax_L,imax_R = idx_L[imax_rel_L], idx_R[imax_rel_R]
        result = {
            'xmin': float(x[imin]), 'ymin': float(y[imin]),
            'xmax_L': float(x[imax_L]), 'ymax_L': float(y[imax_L]),
            'xmax_R': float(x[imax_R]), 'ymax_R': float(y[imax_R]),
            'delta_y_L': float(y[imax_L] - y[imin]),
            'delta_y_R': float(y[imax_R] - y[imin])
        }
        if plot:
            plt.figure(figsize=(7, 4))
            plt.plot(x, y, label="Data")
            plt.plot(x[imin], y[imin], 'ro',ms = 2, label="Local min")
            plt.plot(x[imax_L], y[imax_L], 'go', ms = 2, label="Local max_L")
            plt.plot(x[imax_R], y[imax_R], 'bo', ms=2, label="Local max_R")
            # vertical delta line
            #plt.vlines(x[imin], y[imin], np.min(y[imax_R],y[imax_L]), colors='k', linestyles='--',
                       #label=f"ΔV = {result['delta_y']:.3g}")
            plt.xlabel("z(mm)")
            plt.ylabel("V (V)")
            plt.legend()
            plt.title("Local Min & Max around x0 = " + str(x[imin]) + "mm")
            plt.tight_layout()
            plt.show()
        return result

    def plot_diagonal_angle(self, yy, zz, VV, theta_deg=45, r_max=None, npts=401, plot=True):
        """
        Sample V(y,z) along a diagonal at angle theta (deg) through (0,0).
        theta =  45 → y =  z
        theta = -45 → y = -z
        theta =   0 → along +y
        theta =  90 → along +z
        """

        yy = np.asarray(yy).ravel()  # 1-D coords
        zz = np.asarray(zz).ravel()
        VV = np.asarray(VV)  # 2-D field on (zz, yy) grid because meshgrid(..., indexing="xy")

        # RegularGridInterpolator expects the axes in the same order as VV's dimensions.
        # With indexing="xy", VV.shape == (len(zz), len(yy)). Use VV.T if you want (yy, zz).
        f = RegularGridInterpolator((yy, zz), VV.T, bounds_error=False, fill_value=np.nan)

        theta = np.deg2rad(theta_deg)
        if r_max is None:
            r_max = min(yy.max(), zz.max())
        r = np.linspace(-r_max, r_max, npts)

        # Parametrize the line
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        pts = np.column_stack([y, z])  # order (y, z) to match f's (yy, zz)

        V_line = f(pts)

        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(r, V_line, lw=1.8)
            plt.xlabel("r (mm along diagonal)")
            plt.ylabel("Potential V (V)")
            plt.title(f"Diagonal cut θ={theta_deg}°")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        return r, V_line
    def animate_voltage_path_gif(self,voltage_points,*,seconds_per_segment=2.0,fps=30,save_path="transition.gif",include_boxes=True,dpi=120,):
        # Ensure proper array format
        voltage_points = np.asarray(voltage_points, dtype=float)
        assert voltage_points.ndim == 2 and voltage_points.shape[1] == 6, \
            "voltage_points must be shape (N, 6)"

        n_segments = len(voltage_points) - 1
        frames_per_segment = max(2, int(round(seconds_per_segment * fps)))
        total_frames = n_segments * frames_per_segment

        # Build interpolated path
        path = []
        for i in range(n_segments):
            seg = np.linspace(
                voltage_points[i], voltage_points[i + 1],
                frames_per_segment, endpoint=False
            )
            path.append(seg)
        path.append(voltage_points[-1][None, :])  # include last point
        path = np.vstack(path)

        # Initialize trap with first point
        self.set_DC_voltages(*path[0, :5])
        self.set_AC_voltage(path[0, 5])

        fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
        (line,) = ax.plot([], [], lw=2, label="Total Potential")
        ax.set_xlabel("Axial axis (mm)")
        ax.set_ylabel("Potential (V)")
        ax.set_xlim(float(self.position_vector.min()), float(self.position_vector.max()))
        ax.legend(loc="best")

        # Initial y-limits
        V0 = self.get_trap_potential()
        pad = 0.05 * max(1e-12, np.ptp(V0))
        ax.set_ylim(float(V0.min() - pad), float(V0.max() + pad))

        # Optional electrode boxes
        if include_boxes and hasattr(self, "_add_electrode_boxes"):
            try:
                self._add_electrode_boxes(ax)
            except Exception:
                pass

        def init():
            line.set_data([], [])
            return (line,)

        def update(i):
            vals = path[i]
            self.set_DC_voltages(*vals[:5])
            self.set_AC_voltage(vals[5])

            V = self.get_trap_potential()
            line.set_data(self.position_vector, V)

            pad = 0.05 * max(1e-12, np.ptp(V))
            ax.set_ylim(float(V.min() - pad), float(V.max() + pad))

            ax.set_title(
                f"Frame {i + 1}/{total_frames} | "
                f"EC_L={vals[0]:.1f}, DC_L={vals[1]:.1f}, "
                f"Bias={vals[2]:.1f}, DC_R={vals[3]:.1f}, "
                f"EC_R={vals[4]:.1f}, AC={vals[5]:.0f}"
            )
            return (line,)

        ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True)
        ani.save(save_path, writer=PillowWriter(fps=fps))
        plt.close(fig)







