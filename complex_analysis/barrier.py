import numpy as np
import matplotlib.pyplot as plt
from complex_numbers import Cmplxcar, Cmplxrad, extract_x_y_from_cmplxcar_arr, generate_cmplxcar_arr
from complex_plane import Circle, Airfoil


class Barrier:
    def __init__(self, name: str, color_scheme: str = "dark", 
                 field_limit_x: float = 5, field_limit_y: float = 5, steps: int = 100, num_lines: int = 35):
        self.name = name
        if color_scheme not in ["dark", "bright"]:
            raise ValueError("color_scheme must be either 'dark' or 'bright'")
        self.color_scheme = color_scheme
        if self.color_scheme == "dark":
            self.style_dict = {
                "barrier_color": "white",
                "barrier_edge_color": "white",
            }
        elif self.color_scheme == "bright":
            self.style_dict = {
                "barrier_color": "black",
                "barrier_edge_color": "black",
            }
        self.field_limit_x = field_limit_x
        self.field_limit_y = field_limit_y
        self.steps = steps
        self.num_lines = num_lines

        x = np.linspace(-self.field_limit_x, self.field_limit_x, self.steps)
        y = np.linspace(-self.field_limit_y, self.field_limit_y, self.steps)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = generate_cmplxcar_arr(self.X, self.Y)
        self.barrier_arr = np.vectorize(self.conformal_map)(self.Z)
        self.field_x, self.field_y = extract_x_y_from_cmplxcar_arr(self.barrier_arr)
    
    def get_name(self):
        return self.name

    def conformal_map(self, z: Cmplxcar) -> Cmplxcar:
        pass

    def get_field_y(self):
        return self.field_y

    def fill_barrier_color(self, ax: plt.Axes):
        pass

    def plot_scalar_field(self, ax: plt.Axes, contour_style):
        pass

class EmptyBarrier(Barrier):
    def __init__(self):
        super().__init__(name="Plane", color_scheme="dark")

    def conformal_map(self, z):
        return z
    
    def plot_scalar_field(self, ax: plt.Axes, contour_style):
        ax.contour(self.X, self.Y, self.Y, self.num_lines, cmap=contour_style)


class PrismBarrier(Barrier):
    def __init__(self):
        super().__init__(name="Prism", color_scheme="dark")

    def conformal_map(self, z):
        _z = z.torad()
        rotate = Cmplxrad(1, np.pi / 4)
        res = rotate * (_z ** (3 / 4))
        old_r = res.get_r()
        res.set_new_radius(old_r * 2)
        return res.tocar()
    
    def fill_barrier_color(self, ax: plt.Axes):
        barrier_color, barrier_edge_color = self.style_dict.values()
        pass

    def plot_scalar_field(self, ax: plt.Axes, contour_style):
        ax.contour(self.field_x, self.field_y, self.Y, self.num_lines, cmap=contour_style)


class CylinderBarrier(Barrier):
    def __init__(self, asymptotic_velocity, radius, cyl_shift):
        self.asymptotic_velocity = asymptotic_velocity
        self.radius = radius
        self.cyl_shift = cyl_shift
        super().__init__(name="Cylinder", color_scheme="dark")

    def conformal_map(self, z: Cmplxcar) -> Cmplxcar:
        z = z + self.cyl_shift
        if abs(z) <= self.radius or (z.real() == 0):
            return Cmplxcar(0, 0)
        return self.asymptotic_velocity * (z + self.radius ** 2 * (z.conj() / abs(z)**2))

    def fill_barrier_color(self, ax: plt.Axes):
        barrier_color, barrier_edge_color = self.style_dict.values()
        circle = plt.Circle(
            (-self.cyl_shift, 0),
            self.radius,
            color=barrier_color,
            edgecolor=barrier_edge_color,
            zorder=2
        )
        ax.add_artist(circle)

    def plot_scalar_field(self, ax: plt.Axes, contour_style):
        ax.contour(self.X, self.Y, self.field_y, self.num_lines, cmap=contour_style)


class AirfoilBarrier(Barrier):
    def __init__(self, asymptotic_velocity=53, radius=1.4, jouk_const=1.165, cyl_shift=0.16):
        self.radius = radius
        self.cyl_shift = cyl_shift
        self.jouk_const = jouk_const
        self.original_cyl = CylinderBarrier(
            asymptotic_velocity=asymptotic_velocity,
            radius=radius,
            cyl_shift=cyl_shift
        )
        super().__init__(name="Airfoil", color_scheme="dark")

    def conformal_map(self, z: Cmplxcar) -> Cmplxcar:
        return z + self.jouk_const ** 2 * (z.conj() / abs(z)**2)

    def fill_barrier_color(self, ax: plt.Axes):
        barrier_color, barrier_edge_color = self.style_dict.values()
        cyl = Circle(self.radius, (-self.cyl_shift, 0))
        airfoil = cyl.apply_joukowski(self.jouk_const)
        x, y = airfoil.get_x_y_arrays()
        ax.plot(x, y, color=barrier_edge_color)
        ax.fill(x, y, barrier_color, zorder=2)

    def plot_scalar_field(self, ax: plt.Axes, contour_style):
        cyl_solution_field_y = self.original_cyl.get_field_y()
        ax.contour(self.field_x, self.field_y, cyl_solution_field_y, self.num_lines, cmap=contour_style)
