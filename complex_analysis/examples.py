from airfoil_comparison_plotter import AirfoilComparePlotter
from barrier import AirfoilBarrier, CylinderBarrier, EmptyBarrier, PrismBarrier
from potential_flow_plotter import PotentialFlowPlotter

def example_usage_airfoil_comparison():
    plotter = AirfoilComparePlotter(jouk_k=1.165, radius=1.4, center_x=-0.16, center_y=0)
    plotter.run(color_mode="dark")

def example_usage_potential_flow_plotter_1():
    empty = EmptyBarrier()
    prism = PrismBarrier()
    plotter2 = PotentialFlowPlotter(barrier1=empty, barrier2=prism)
    plotter2.run(color_scheme='dark')
    
def example_usage_potential_flow_plotter_2():
    airfoil_barrier = AirfoilBarrier(asymptotic_velocity=53, radius=1.4, jouk_const=1.165, cyl_shift=0.16)
    cylinder_barrier = CylinderBarrier(asymptotic_velocity=53, radius=1.4, cyl_shift=0.16)
    plotter = PotentialFlowPlotter(barrier1=airfoil_barrier, barrier2=cylinder_barrier)
    plotter.run(color_scheme='dark')

