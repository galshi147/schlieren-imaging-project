"""
Demo script showing how to use the animated CFD plotter
"""

import numpy as np
from simulations.old.cfd_solver import CfdSolver
from cfd_obstacles import CfdCircle, CfdTriangle, CfdAirfoil
from simulations.old.cfd_plotter import CfdPlotter

def demo_circle_flow():
    """Demonstrate flow around a circular obstacle"""
    print("Setting up flow around a circle...")
    
    # Create circular obstacle
    circle = CfdCircle(radius=0.25, center=(1.0, 1.0))
    
    # Create solver with the obstacle
    solver = CfdSolver(obstacle=circle, nx=128, ny=64, Lx=4.0, Ly=2.0, dt=5e-3)
    
    # Create plotter
    plotter = CfdPlotter(solver)
    
    # Run animation
    print("Starting animation...")
    animation = plotter.animate(frames=500, interval=50, save_animation=False)
    
    return solver, plotter, animation

def demo_triangle_flow():
    """Demonstrate flow around a triangular obstacle"""
    print("Setting up flow around a triangle...")
    
    # Create triangular obstacle
    vertices = np.array([[0.8, 0.8], [1.2, 1.0], [0.8, 1.2]])
    triangle = CfdTriangle(vertices=vertices)
    
    # Create solver with the obstacle
    solver = CfdSolver(obstacle=triangle, nx=128, ny=64, Lx=4.0, Ly=2.0, dt=5e-3)
    
    # Create plotter
    plotter = CfdPlotter(solver)
    
    # Run animation
    print("Starting animation...")
    animation = plotter.animate(frames=500, interval=50, save_animation=False)
    
    return solver, plotter, animation

def demo_airfoil_flow():
    """Demonstrate flow around an airfoil"""
    print("Setting up flow around an airfoil...")
    
    # Create NACA airfoil obstacle
    airfoil = CfdAirfoil(NACA_code='0012')
    
    # Create solver with the obstacle
    solver = CfdSolver(obstacle=airfoil, nx=128, ny=64, Lx=4.0, Ly=2.0, dt=5e-3)
    
    # Create plotter
    plotter = CfdPlotter(solver)
    
    # Run animation
    print("Starting animation...")
    animation = plotter.animate(frames=500, interval=50, save_animation=False)
    
    return solver, plotter, animation

def demo_static_analysis():
    """Demonstrate static analysis with statistics"""
    print("Running static analysis...")
    
    # Create circular obstacle
    circle = CfdCircle(radius=0.2, center=(1.0, 1.0))
    solver = CfdSolver(obstacle=circle, nx=64, ny=32, Lx=4.0, Ly=2.0, dt=5e-3)
    plotter = CfdPlotter(solver)
    
    # Run simulation for several steps and collect statistics
    print("Running simulation steps...")
    for i in range(100):
        solver.step()
        if i % 20 == 0:
            stats = plotter.get_simulation_stats()
            print(f"Step {i}: Max vel = {stats['max_velocity']:.3f}, "
                  f"Avg vel = {stats['avg_velocity']:.3f}, "
                  f"Pressure range = [{stats['min_pressure']:.3f}, {stats['max_pressure']:.3f}]")
    
    # Create static plot
    plotter.plot_static()
    
    # Save frame
    plotter.save_frame('final_state.png')
    
    return solver, plotter

def demo_save_animation():
    """Demonstrate saving animation to file"""
    print("Creating and saving animation...")
    
    # Create simple circular obstacle
    circle = CfdCircle(radius=0.15, center=(0.8, 1.0))
    solver = CfdSolver(obstacle=circle, nx=64, ny=32, Lx=3.0, Ly=1.5, dt=5e-3)
    plotter = CfdPlotter(solver)
    
    # Run and save animation
    print("Saving animation (this may take a while)...")
    animation = plotter.animate(frames=200, interval=100, 
                               save_animation=True, filename='circle_flow.gif')
    
    return solver, plotter, animation

if __name__ == "__main__":
    print("CFD Animation Demo")
    print("==================")
    
    # Uncomment the demo you want to run:
    
    # Demo 1: Circle flow animation
    demo_circle_flow()
    
    # Demo 2: Triangle flow animation  
    # demo_triangle_flow()
    
    # Demo 3: Airfoil flow animation
    # demo_airfoil_flow()
    
    # Demo 4: Static analysis with statistics
    # demo_static_analysis()
    
    # Demo 5: Save animation to file
    # demo_save_animation()
    
    print("Demo complete!")
