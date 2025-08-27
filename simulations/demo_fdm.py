"""
Demo script for the FDM-based CFD solver with ready-to-run examples
for different obstacle types with optimized parameters.
"""

import numpy as np
from .cfd_solver_fdm import CfdSolverFdm
from .cfd_obstacles import CfdCircle, CfdTriangle, CfdAirfoil, CfdSquare
from .cfd_plotter_fdm import CfdPlotterFdm

def demo_circle_flow(save_animation=False):
    """Flow around a circular cylinder - classic von Kármán vortex street"""
    print("🔵 Demo: Flow around circular cylinder")
    print("=" * 40)
    
    # Create circular obstacle
    circle = CfdCircle(radius=0.1, center=(0.8, 0.5))
    
    # Create solver with optimized parameters for vortex shedding
    solver = CfdSolverFdm(
        obstacle=circle, 
        nx=128, ny=64,           # Good resolution 
        Lx=2.5, Ly=1.25,         # Long domain for wake development
        dt=0.001,                # Small time step for stability
        nu=0.005,                # Low viscosity for Re ≈ 40
        initial_velocity=1.0
    )
    
    print(f"Grid: {solver.nx}×{solver.ny}, Domain: {solver.Lx}×{solver.Ly}")
    print(f"Reynolds number: {circle.radius * solver.initial_velocity / solver.nu:.0f}")
    
    # Initialize smooth flow
    solver.initialize_flow(steps=30)
    
    # Create plotter and run animation
    plotter = CfdPlotterFdm(solver, circle)
    print("✓ Starting animation... Close window when done.")
    plotter.animate(frames=300, interval=30, save_animation=save_animation, filename='circle_flow.gif')

    return solver, plotter

def demo_triangle_flow(save_animation=False):
    """Flow around a triangular obstacle - sharp body aerodynamics"""
    print("🔺 Demo: Flow around triangular obstacle")
    print("=" * 40)
    
    # Create triangular obstacle (streamlined shape)
    vertices = np.array([[0.6, 0.4], [1.0, 0.5], [0.6, 0.6]])
    triangle = CfdTriangle(vertices=vertices)
    
    # Create solver
    solver = CfdSolverFdm(
        obstacle=triangle,
        nx=120, ny=60,
        Lx=2.0, Ly=1.0,
        dt=0.0008,
        nu=0.008,                # Slightly higher viscosity
        initial_velocity=0.8
    )
    
    print(f"Grid: {solver.nx}x{solver.ny}, Domain: {solver.Lx}x{solver.Ly}")
    
    # Initialize flow
    solver.initialize_flow(steps=25)
    
    # Create plotter and run
    plotter = CfdPlotterFdm(solver, triangle)
    print("✓ Starting animation... Close window when done.")
    plotter.animate(frames=250, interval=40, save_animation=save_animation, filename='triangle_flow.gif')

    return solver, plotter

def demo_airfoil_flow(save_animation=False):
    """Flow around NACA airfoil - aerodynamic lift generation"""
    print("✈️  Demo: Flow around NACA 0012 airfoil")
    print("=" * 40)
    
    # Create NACA airfoil with angle of attack
    airfoil = CfdAirfoil(
        NACA_code='0012',        # Symmetric airfoil
        center=(1.0, 0.75),      # Positioned in flow
        chord_length=0.4,        # Good size relative to domain
        angle_of_attack=8        # Moderate angle for lift
    )
    
    # Create solver with high resolution for airfoil details
    solver = CfdSolverFdm(
        obstacle=airfoil,
        nx=160, ny=80,           # Higher resolution
        Lx=3.0, Ly=1.5,          # Long domain
        dt=0.0005,               # Very small time step
        nu=0.003,                # Low viscosity for realistic Re
        initial_velocity=1.2
    )

    print(f"Grid: {solver.nx}x{solver.ny}, Domain: {solver.Lx}x{solver.Ly}")
    print(f"Airfoil: NACA {airfoil.NACA_code}, Angle of Attack: {np.degrees(airfoil.angle_of_attack):.1f}°")
    
    # Initialize flow carefully
    solver.initialize_flow(steps=40)
    
    # Create plotter and run
    plotter = CfdPlotterFdm(solver, airfoil)
    print("✓ Starting animation... Close window when done.")
    plotter.animate(frames=400, interval=25, save_animation=save_animation, filename='airfoil_flow.gif')

    return solver, plotter

def demo_square_flow(save_animation=False):
    """Flow around a square obstacle - bluff body with sharp corners"""
    print("⬜ Demo: Flow around square obstacle")
    print("=" * 40)
    
    # Create square obstacle
    square = CfdSquare(center=(0.7, 0.5), side_length=0.15)
    
    # Create solver
    solver = CfdSolverFdm(
        obstacle=square,
        nx=100, ny=50,
        Lx=2.0, Ly=1.0,
        dt=0.001,
        nu=0.006,
        initial_velocity=0.9
    )

    print(f"Grid: {solver.nx}x{solver.ny}, Domain: {solver.Lx}x{solver.Ly}")

    # Initialize flow
    solver.initialize_flow(steps=30)
    
    # Create plotter and run
    plotter = CfdPlotterFdm(solver, square)
    print("✓ Starting animation... Close window when done.")
    plotter.animate(frames=200, interval=50, save_animation=save_animation, filename='square_flow.gif')

    return solver, plotter

def demo_comparison():
    """Compare flow around different obstacle shapes"""
    print("📊 Demo: Flow comparison between different obstacles")
    print("=" * 50)
    
    # Common parameters for fair comparison
    common_params = {
        'nx': 80, 'ny': 40,
        'Lx': 1.8, 'Ly': 0.9,
        'dt': 0.001,
        'nu': 0.01,
        'initial_velocity': 1.0
    }
    
    # Create different obstacles at same position
    center = (0.6, 0.45)
    circle = CfdCircle(radius=0.08, center=center)
    square = CfdSquare(center=center, side_length=0.16)
    
    # Run circle simulation
    print("Running circle simulation...")
    solver1 = CfdSolverFdm(obstacle=circle, **common_params)
    solver1.initialize_flow(steps=20)
    solver1.run(iterations=100)
    
    # Run square simulation  
    print("Running square simulation...")
    solver2 = CfdSolverFdm(obstacle=square, **common_params)
    solver2.initialize_flow(steps=20)
    solver2.run(iterations=100)
    
    # Show final results
    plotter1 = CfdPlotterFdm(solver1, circle)
    plotter2 = CfdPlotterFdm(solver2, square)
    
    print("✓ Showing final flow patterns...")
    plotter1.plot_static()
    plotter2.plot_static()
    
    # Print statistics
    stats1 = solver1.get_simulation_stats()
    stats2 = solver2.get_simulation_stats()
    
    print(f"\nCircle - Max velocity: {stats1['max_velocity']:.3f}, Max pressure: {stats1['max_pressure']:.3f}")
    print(f"Square - Max velocity: {stats2['max_velocity']:.3f}, Max pressure: {stats2['max_pressure']:.3f}")

def quick_demo(save_animation=False):
    """Quick demo with optimized parameters for fast visualization"""
    print("⚡ Quick Demo: Fast circular cylinder flow")
    print("=" * 35)
    
    circle = CfdCircle(radius=0.08, center=(0.5, 0.25))
    solver = CfdSolverFdm(
        obstacle=circle,
        nx=64, ny=32,            # Lower resolution for speed
        Lx=1.5, Ly=0.75,
        dt=0.002,                # Larger time step
        nu=0.01,
        initial_velocity=1.0
    )
    
    solver.initialize_flow(steps=15)
    plotter = CfdPlotterFdm(solver, circle)
    plotter.animate(frames=150, interval=50, save_animation=save_animation, filename='quick_demo.gif')

    return solver, plotter

def interactive_demo():
    """Interactive demo - choose your obstacle type"""
    print("🎮 Interactive Demo")
    print("=" * 20)
    print("""Choose obstacle type:
          1. Circle (classic vortex shedding)
          2. Triangle (sharp body)
          3. Airfoil (aerodynamic)
          4. Square (bluff body)
          5. Quick demo (fast)
          6. Comparison (multiple obstacles)""")
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == '1':
        return demo_circle_flow()
    elif choice == '2':
        return demo_triangle_flow()
    elif choice == '3':
        return demo_airfoil_flow()
    elif choice == '4':
        return demo_square_flow()
    elif choice == '5':
        return quick_demo()
    elif choice == '6':
        demo_comparison()
        return None, None
    else:
        print("Invalid choice, running circle demo...")
        return demo_circle_flow()

