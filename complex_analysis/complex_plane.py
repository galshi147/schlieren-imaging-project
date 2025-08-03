from complex_numbers import Cmplxcar, Cmplxrad, generate_cmplxcar_arr, generate_cmplxrad_arr, extract_x_y_from_cmplxcar_arr
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

PI = np.pi

class Airfoil:
    def __init__(self, arr):
        self.arr = arr

    def get_arr(self):
        return self.arr

    def normalize(self):
        x_arr, y_arr = extract_x_y_from_cmplxcar_arr(self.arr)
        shift = min(x_arr)
        if shift >= 0:
            xs = x_arr - shift
        else:
            xs = x_arr + abs(shift)
        norm_factor = max(xs)
        xn, yn = xs / norm_factor, y_arr / norm_factor
        return Airfoil(generate_cmplxcar_arr(xn, yn))

    def get_x_arr_y_arr(self):
        return extract_x_y_from_cmplxcar_arr(self.arr)


class Circle:
    def __init__(self, radius, center=(0, 0), points_num=1000):
        self.radius = radius
        self.points_num = points_num
        cx, cy = 0, 0
        if isinstance(center, tuple):
            cx, cy = center
        elif isinstance(center, Cmplxcar):
            cx, cy = center.real(), center.imag()
        elif isinstance(center, Cmplxrad):
            new = center.tocar()
            cx, cy = new.real(), new.imag()
        self.center = (cx, cy)
        angles = np.linspace(0, 2 * PI, self.points_num)
        radii = np.full_like(angles, self.radius)
        rad_arr = generate_cmplxrad_arr(radii, angles)
        converted_arr_to_car = np.vectorize(Cmplxrad.tocar)
        shift = Cmplxcar(cx, cy)
        self.arr = converted_arr_to_car(rad_arr) + shift

    def get_arr(self):
        return self.arr

    def get_center(self):
        return self.center

    def joukowski(self, k):
        jouk = lambda z: z + k ** 2 * (z.conj() / abs(z)**2)
        vec_j = np.vectorize(jouk)
        return Airfoil(vec_j(self.arr))


def get_shape(cmplxcar_arr):
    real_arr = np.vectorize(Cmplxcar.real)
    imag_arr = np.vectorize(Cmplxcar.imag)
    x = real_arr(cmplxcar_arr)
    y = imag_arr(cmplxcar_arr)
    return x, y


def load_printed():
    path = r"C:\university\year_2\Lab_T\simulations\conformal_map\printed.csv"
    df = pd.read_csv(path)
    x = df["X(mm)"] / 100
    y = df["Y(mm)"] / 100
    return x, y


def get_data(k, r, c_x, c_y):
    printed_x, printed_y = load_printed()
    length = len(printed_x)
    cyl_center = Cmplxcar(c_x, c_y)
    cyl = Circle(r, cyl_center, points_num=length)
    cx, cy = cyl.get_center()
    airfoil = cyl.joukowski(k)
    cyl_x, cyl_y = get_shape(cyl.get_arr())
    airfoil_x, airfoil_y = get_shape(airfoil.get_arr())
    return cx, cy, cyl_x, cyl_y, airfoil_x, airfoil_y, printed_x, printed_y


bright_mode = {
    "fig_color": "white",
    "ax_color": "white",
    "axis_color": "black",
    "title_color": "black",
    "circle_color": "black",
    "center_color": "green",
    "airfoil_color": "orange",
    "NACA_color": "#1f77b4",
    "printed_color": "#1f77b4",
    "jouk_color": "orange",
    "legend_color": "black"
}

dark_mode = {
    "fig_color": "black",
    "ax_color": "black",
    "axis_color": "white",
    "title_color": "white",
    "circle_color": "white",
    "center_color": "white",
    "airfoil_color": "orange",
    "NACA_color": "#17becf",
    "printed_color": "#17becf",
    "jouk_color": "orange",
    "legend_color": "white"
}


def set_ax_style(ax, ax_color, axis_color):
    ax.set_facecolor(ax_color)
    ax.spines['bottom'].set_color(axis_color)
    ax.spines['left'].set_color(axis_color)
    plt.xticks(color=axis_color)
    plt.yticks(color=axis_color)
    return ax


def design_ax1(ax1, r, cx, cy, cyl_x, cyl_y, airfoil_x, airfoil_y,
               title_style, circle_color, airfoil_color, center_color,
               ax_color, axis_color):
    ax1.plot(cyl_x, cyl_y, color=circle_color)
    ax1.plot(airfoil_x, airfoil_y, color=airfoil_color)
    ax1.scatter(cx, cy, color=center_color, marker='.')
    ax1.set_aspect('equal')
    ax1.grid(which="both")
    ax1.axvline(x=0, color="dimgray")
    ax1.axhline(y=0, color="dimgray")
    ax1.set_title(
        "Joukowski Transform\n" + r"$J(z)=z+\frac{k^2}{z}$" + f"\t, R={r}±0.1cm",
        **title_style)
    ax1 = set_ax_style(ax1, ax_color, axis_color)
    return ax1


def design_ax2(ax2, printed_x, printed_y, title_style, NACA_color, ax_color,
               axis_color):
    ax2.plot(printed_x, printed_y, color=NACA_color)
    ax2.set_aspect('equal')
    ax2.grid(which="both")
    ax2.axvline(x=0, color="dimgray")
    ax2.axhline(y=0, color="dimgray")
    ax2.set_title("NACA = 0020", **title_style)
    ax2 = set_ax_style(ax2, ax_color, axis_color)
    return ax2


def design_ax3(ax3, k, c_x, c_y, airfoil_x, airfoil_y, printed_x, printed_y,
               title_style, printed_color, jouk_color, legend_color, ax_color,
               axis_color):
    norm_airfoil = Airfoil(
        generate_cmplxcar_arr(airfoil_x, airfoil_y)).normalize()
    norm_printed = Airfoil(
        generate_cmplxcar_arr(printed_x, printed_y)).normalize()
    n_airf_x, n_airf_y = norm_airfoil.get_x_arr_y_arr()
    n_printed_x, n_printed_y = norm_printed.get_x_arr_y_arr()
    ax3.plot(n_printed_x, n_printed_y, label="printed", color=printed_color)
    ax3.scatter(n_airf_x, n_airf_y, label="Joukowski", color=jouk_color,
                alpha=0.5)
    ax3.legend(labelcolor=legend_color, facecolor=ax_color)
    ax3.set_title(f"Fit: k={k}±0.001, center=({c_x}±0.01, {c_y})", **title_style)
    ax3 = set_ax_style(ax3, ax_color, axis_color)
    return ax3


def plot_all(k, r, c_x, c_y, fig_color, ax_color, axis_color, title_color,
             circle_color, center_color, airfoil_color, NACA_color,
             printed_color, jouk_color, legend_color):
    fig = plt.figure(figsize=(6, 6), dpi=300, facecolor=fig_color)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212)
    title_style = {"size": 14, "pad": 15, "color": title_color}
    cx, cy, cyl_x, cyl_y, airfoil_x, airfoil_y, printed_x, printed_y = get_data(
        k, r, c_x, c_y)

    ax1 = design_ax1(ax1, r, cx, cy, cyl_x, cyl_y, airfoil_x, airfoil_y,
                     title_style, circle_color, airfoil_color, center_color,
                     ax_color, axis_color)
    ax2 = design_ax2(ax2, printed_x, printed_y, title_style, NACA_color,
                     ax_color, axis_color)

    ax3 = design_ax3(ax3, k, c_x, c_y, airfoil_x, airfoil_y, printed_x,
                     printed_y, title_style, printed_color, jouk_color,
                     legend_color, ax_color, axis_color)

    plt.show()


plot_all(k=1.165, r=1.4, c_x=-0.16, c_y=0, **dark_mode)
