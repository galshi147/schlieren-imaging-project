import numpy as np
import matplotlib.pyplot as plt
from complex_numbers import Cmplxcar, Cmplxrad, generate_cmplxcar_arr
from barrier import Barrier, EmptyBarrier, PrismBarrier, CylinderBarrier, AirfoilBarrier

class PotentialFlowPlotter:
    def __init__(self, barrier1: Barrier, barrier2: Barrier):
        self.barrier1 = barrier1
        self.barrier2 = barrier2
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, dpi=100)
        self.ax1: plt.Axes
        self.ax2: plt.Axes
        self.dark_mode = {
            "fig_color": "black",
            "ax_color": "black",
            "axis_color": "white",
            "title_color": "white",
            "contour_style": "Blues"
            }
        self.light_mode = {
            "fig_color": "white",
            "ax_color": "white",
            "axis_color": "black",
            "title_color": "black",
            "contour_style": "Blues"
        }

    def set_ax_style(self, ax_color, axis_color, title_color, field_limit_x, field_limit_y):
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor(ax_color)
            ax.spines['bottom'].set_color(axis_color)
            ax.spines['left'].set_color(axis_color)
            ax.spines['right'].set_color(axis_color)
            ax.spines['top'].set_color(axis_color)
            ax.set_aspect('equal')
            ax.tick_params(axis='x', colors=axis_color)
            ax.tick_params(axis='y', colors=axis_color)
            ax.set_xlim((-field_limit_x, field_limit_x))
            ax.set_ylim((-field_limit_y, field_limit_y))
        self.ax1.set_title(self.barrier1.get_name(), color=title_color, fontsize=14, pad=15)
        self.ax2.set_title(self.barrier2.get_name(), color=title_color, fontsize=14, pad=15)

    def plot_flow(self, style_dict: dict, field_limit_x=5, field_limit_y=5):
        fig_color, ax_color, axis_color, title_color, contour_style, = style_dict.values()
        self.fig.set_facecolor(fig_color)
        self.set_ax_style(ax_color, axis_color, title_color, field_limit_x, field_limit_y)
       
        self.barrier1.plot_scalar_field(self.ax1, contour_style)
        self.barrier2.plot_scalar_field(self.ax2, contour_style)
        self.barrier1.fill_barrier_color(self.ax1)
        self.barrier2.fill_barrier_color(self.ax2)

        plt.show()

    def run(self, color_scheme='dark'):
        if color_scheme == 'dark':
            self.plot_flow(style_dict=self.dark_mode)
        else:
            self.plot_flow(style_dict=self.light_mode)
    