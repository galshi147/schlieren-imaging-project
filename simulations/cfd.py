# IMPORTS #
import numpy as np
from matplotlib import pyplot as plt, cm
from abc import ABC, abstractmethod

# CONSTANTS #
NX = 100  # num of points in x direction
NY = 100  # num of points in y direction
TIME_ITERATIONS = 20  # nt
NIT = 100  # nit - virtual time for poisson eq.
LENGTH = 2  # square blocks the mirror with diameter of 8 cm
DX = LENGTH / (NX - 1)  # step length in x direction
DY = LENGTH / (NY - 1)  # step length in y direction
DT = 1/10000  # time steps (s)
C = 53  # fluid velocity (m/s)
AIR_STATIC_DENSITY = 1  # \rho
AIR_KINEMATIC_VISCOSITY = 0.01  # \nu
shape_style = {"color": "black", "fill": True}


def get_Re_num(c, length, nu):
    """
    :param c: fluid velocity
    :param nu: fluid kinematic viscosity
    :param length: characteristic length
    :return: Re num
    """
    return f"\nReynold's number = {c * length / nu}\n"


# BARRIER SHAPES #

class Shape(ABC):
    @abstractmethod
    def apply_pressure_condition(self, arr, dx, dy, length):
        pass

    @abstractmethod
    def apply_no_slip_condition(self, u, v, dx, dy, length):
        pass

    @abstractmethod
    def plot(self, center, ax=None):
        pass


def create_mesh(nx, ny, length):
    x = np.linspace(0, length, nx)
    y = np.linspace(0, length, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def _get_arr_data(arr, length):
    nx, ny = arr.shape
    cx, cy = length / 2, length / 2
    X, Y = create_mesh(nx, ny, length)
    return nx, ny, cx, cy, X, Y


class Cyl(Shape):
    def __init__(self, radius):
        self.radius = radius

    def get_size(self):
        return f"radius={self.radius}"

    def plot(self, center: tuple, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        circle = plt.Circle(center, self.radius, **shape_style)
        ax.add_artist(circle)
        ax.set_aspect('equal')
        return ax

    def apply_pressure_condition(self, p, dx, dy, length):
        nx, ny, cx, cy, x, y = _get_arr_data(p, length)
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= self.radius ** 2
        p[mask] = 0

        p[:, 0] = 0  # p=0 at x=0
        p = self._p_on_edge(p, dx, dy, length)
        return p

    def _create_edge_mask(self, x, y, cx, cy, dx, dy):
        dr = np.sqrt(dx ** 2 + dy ** 2)
        distance_squared = (x - cx) ** 2 + (y - cy) ** 2
        mask_1 = (self.radius - dr) ** 2 <= distance_squared
        mask_2 = distance_squared <= (self.radius + dr) ** 2
        edge_mask = mask_1 & mask_2
        return edge_mask

    def _create_shifted_edge_mask(self, edge_mask, edge_x, edge_y, shift_type):
        new_edge_x = edge_x
        new_edge_y = edge_y
        if shift_type == 'x':
            new_edge_x = edge_x + 1
        elif shift_type == 'y':
            new_edge_y = edge_y + 1
        else:
            raise "unsupported shifted type, must be 'x' or 'y'"
        shifted_mask = np.full_like(edge_mask, False)
        shifted_mask[new_edge_y, new_edge_x] = True
        return shifted_mask

    def _get_edge(self, arr, dx, dy, length):
        nx, ny, cx, cy, x, y = _get_arr_data(arr, length)
        edge_mask = self._create_edge_mask(x, y, cx, cy, dx, dy)
        # Get the indices of the edge cells
        edge_indices = np.nonzero(edge_mask)
        edge_x = edge_indices[0]  # Get the x coordinates
        edge_y = edge_indices[1]  # Get the y coordinates
        # Get inner mask to force zero velocity inside shape
        inner_mask = (x - cx) ** 2 + (y - cy) ** 2 < self.radius ** 2
        return edge_mask, edge_x, edge_y, inner_mask

    def _u_on_edge(self, y):
        return (C / self.radius) * np.abs(y)

    def _v_on_edge(self, x, y):
        return (C / self.radius) * x * np.sign(y)

    def _p_on_edge(self, _p, dx, dy, length):
        p = np.copy(_p)
        nx, ny, cx, cy, x, y = _get_arr_data(p, length)
        edge_mask, edge_x, edge_y, inner_mask = self._get_edge(p, dx, dy, length)
        edge_mask_x_shift = self._create_shifted_edge_mask(edge_mask, edge_x, edge_y, 'x')
        edge_mask_y_shift = self._create_shifted_edge_mask(edge_mask, edge_x, edge_y, 'y')
        coef = 1 / (edge_x/dx + edge_y/dy)
        p[edge_mask] = (edge_x * coef / dx) * _p[edge_mask_x_shift] + (edge_y * coef / dy) * _p[edge_mask_y_shift]
        return p

    def apply_no_slip_condition(self, _u, _v, dx, dy, length):
        u = np.copy(_u)
        v = np.copy(_v)
        mask_u, edge_x_u, edge_y_u, inner_mask_u = self._get_edge(u, dx, dy, length)
        u[mask_u] = self._u_on_edge(edge_y_u)
        mask_v, edge_x_v, edge_y_v, inner_mask_v = self._get_edge(v, dx, dy, length)
        v[mask_v] = self._v_on_edge(edge_x_v, edge_y_v)
        # force zero inside shape
        u[inner_mask_u] = 0
        v[inner_mask_v] = 0
        return u, v


class Prism(Shape):
    def __init__(self, base, side):
        self.base = base
        self.side = side

    def plot(self):
        fig, ax = plt.subplots()
        # circle = plt.Circle(self.radius, color='black', fill=True)
        ax.add_artist()
        ax.set_aspect('equal')
        return ax


class Drop(Shape):
    pass


class Airfoil(Shape):
    pass







