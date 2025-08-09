from matplotlib import pyplot as plt
from complex_numbers import Cmplxcar, generate_cmplxcar_arr, extract_x_y_from_cmplxcar_arr
from complex_plane import Circle, Airfoil, PrintedAirfoil

class AirfoilComparePlotter:
    def __init__(self, jouk_k: float, radius: float, center_x: float, center_y: float):
        self.jouk_k = jouk_k
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y
        self.bright_mode ={
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
            "legend_label_color": "black",
            "legend_face_color": "white"
            }
        
        self.dark_mode = {
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
            "legend_label_color": "white",
            "legend_face_color": "black"
            }
        
        self.fig = plt.figure(figsize=(6, 6), dpi=300)
        self.jouk_ax = plt.subplot(221)
        self.naca_ax = plt.subplot(222)
        self.compare_ax = plt.subplot(212)

    def set_axes_style(self, ax_color, axis_color):
        for ax in [self.jouk_ax, self.naca_ax, self.compare_ax]:
            ax.set_facecolor(ax_color)
            ax.spines['bottom'].set_color(axis_color)
            ax.spines['left'].set_color(axis_color)
            plt.xticks(color=axis_color)
            plt.yticks(color=axis_color)


    def design_jouk_ax(self, cyl: Circle, jouk_airfoil: Airfoil,
                       title_style: dict, circle_color: str, airfoil_color: str, center_color: str) -> None:
        cyl_x, cyl_y = cyl.get_x_y_arrays()
        airfoil_x, airfoil_y = jouk_airfoil.get_x_y_arrays()
        self.jouk_ax.plot(cyl_x, cyl_y, color=circle_color)
        self.jouk_ax.plot(airfoil_x, airfoil_y, color=airfoil_color)
        self.jouk_ax.scatter(self.center_x, self.center_y, color=center_color, marker='.')
        self.jouk_ax.set_aspect('equal')
        self.jouk_ax.grid(which="both")
        self.jouk_ax.axvline(x=0, color="dimgray")
        self.jouk_ax.axhline(y=0, color="dimgray")
        self.jouk_ax.set_title(
            "Joukowski Transform\n" + r"$J(z)=z+\frac{k^2}{z}$" + f"\t, R={self.radius}±0.1cm",
            **title_style)


    def design_naca_ax(self, printed_airfoil: PrintedAirfoil, title_style: dict, NACA_color: str) -> None:
        printed_x, printed_y = printed_airfoil.get_x_y_arrays()
        self.naca_ax.plot(printed_x, printed_y, color=NACA_color)
        self.naca_ax.set_aspect('equal')
        self.naca_ax.grid(which="both")
        self.naca_ax.axvline(x=0, color="dimgray")
        self.naca_ax.axhline(y=0, color="dimgray")
        self.naca_ax.set_title("NACA = 0020", **title_style)


    def design_compare_ax(self, jouk_airfoil: Airfoil, printed_airfoil: PrintedAirfoil,
                          title_style: dict, printed_color: str, jouk_color: str, legend_label_color: str, legend_face_color: str) -> None:
        jouk_airfoil.normalize()
        norm_printed_airfoil = printed_airfoil.normalize()
        norm_airf_x, norm_airf_y = jouk_airfoil.get_x_y_arrays()
        norm_printed_x, norm_printed_y = norm_printed_airfoil.get_x_y_arrays()
        self.compare_ax.plot(norm_printed_x, norm_printed_y, label="printed", color=printed_color)
        self.compare_ax.scatter(norm_airf_x, norm_airf_y, label="Joukowski", color=jouk_color,
                    alpha=0.5)
        self.compare_ax.legend(labelcolor=legend_label_color, facecolor=legend_face_color)
        self.compare_ax.set_title(f"Fit: k={self.jouk_k}±0.001, center=({self.center_x}±0.01, {self.center_y})", **title_style)
        return self.compare_ax

    def get_data(self) -> tuple:
        printed_airfoil = PrintedAirfoil("complex_analysis/printed.csv")
        cyl = Circle(radius=self.radius, center=(self.center_x, self.center_y), points_num=printed_airfoil.get_num_points())
        jouk_airfoil = cyl.apply_joukowski(self.jouk_k)
        return printed_airfoil, cyl, jouk_airfoil

    def plot_all(self, fig_color, ax_color, axis_color, title_color,
                circle_color, center_color, airfoil_color, NACA_color,
                printed_color, jouk_color, legend_label_color, legend_face_color):
        
        self.fig.set_facecolor(fig_color)
        self.set_axes_style(ax_color, axis_color)
        title_style = {"size": 14, "pad": 15, "color": title_color}

        printed_airfoil, cyl, jouk_airfoil = self.get_data()

        self.design_jouk_ax(cyl, jouk_airfoil, 
                            title_style, circle_color, airfoil_color, center_color)

        self.design_naca_ax(printed_airfoil, title_style, NACA_color)

        self.design_compare_ax(jouk_airfoil, printed_airfoil,
                               title_style, printed_color, jouk_color,
                               legend_label_color, legend_face_color)

        plt.show()

    def run(self):
        self.plot_all(**self.dark_mode)
    
plotter= AirfoilComparePlotter(jouk_k=1.165, radius=1.4, center_x=-0.16, center_y=0)
plotter.run()