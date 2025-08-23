"""
Test script to compare CfdSolver and CfdSolverFdm implementations
"""

import numpy as np
import matplotlib.pyplot as plt
from simulations.old.cfd_solver import CfdSolver
from cfd_solver_fdm import CfdSolverFdm
from cfd_obstacles import CfdCircle
import time

def test_both_solvers():
    """Compare the original and FDM-based solvers"""
    print("Comparing CfdSolver vs CfdSolverFdm")
    print("=" * 40)
    
    # Create identical setup for both solvers
    circle = CfdCircle(radius=0.2, center=(1.0, 0.5))
    nx, ny = 64, 32
    Lx, Ly = 2.0, 1.0
    dt = 1e-3
    
    print(f"Grid: {nx} x {ny}")
    print(f"Domain: {Lx} x {Ly}")
    print(f"Time step: {dt}")
    print()
    
    # Initialize both solvers
    print("Initializing solvers...")
    solver_original = CfdSolver(obstacle=circle, nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt)
    solver_fdm = CfdSolverFdm(obstacle=circle, nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt)
    
    print("✓ Both solvers initialized successfully")
    print()
    
    # Run both solvers for same number of steps
    n_steps = 50
    print(f"Running both solvers for {n_steps} steps...")
    
    # Time original solver
    start_time = time.time()
    for i in range(n_steps):
        solver_original.step()
    original_time = time.time() - start_time
    
    # Time FDM solver
    start_time = time.time()
    for i in range(n_steps):
        solver_fdm.step()
    fdm_time = time.time() - start_time
    
    print(f"✓ Original solver time: {original_time:.3f} seconds")
    print(f"✓ FDM solver time: {fdm_time:.3f} seconds")
    print(f"✓ Speed ratio: {original_time/fdm_time:.2f}x")
    print()
    
    # Compare results
    print("Comparing results...")
    
    # Get velocity magnitudes
    u_orig_center = 0.5 * (solver_original.u[1:, :] + solver_original.u[:-1, :])
    v_orig_center = 0.5 * (solver_original.v[:, 1:] + solver_original.v[:, :-1])
    vel_mag_orig = np.sqrt(u_orig_center**2 + v_orig_center**2)
    
    vel_mag_fdm = solver_fdm.get_velocity_magnitude()
    
    # Calculate differences
    max_vel_diff = np.max(np.abs(vel_mag_orig - vel_mag_fdm))
    mean_vel_diff = np.mean(np.abs(vel_mag_orig - vel_mag_fdm))
    pressure_diff = np.max(np.abs(solver_original.p - solver_fdm.p))
    
    print(f"✓ Max velocity magnitude difference: {max_vel_diff:.6f}")
    print(f"✓ Mean velocity magnitude difference: {mean_vel_diff:.6f}")
    print(f"✓ Max pressure difference: {pressure_diff:.6f}")
    print()
    
    # Print statistics
    stats_orig = {
        'max_velocity': np.max(vel_mag_orig),
        'mean_velocity': np.mean(vel_mag_orig),
        'max_pressure': np.max(solver_original.p),
        'min_pressure': np.min(solver_original.p)
    }
    
    stats_fdm = solver_fdm.get_simulation_stats()
    
    print("Final Statistics Comparison:")
    print("-" * 30)
    print(f"Max velocity    - Original: {stats_orig['max_velocity']:.4f}, FDM: {stats_fdm['max_velocity']:.4f}")
    print(f"Mean velocity   - Original: {stats_orig['mean_velocity']:.4f}, FDM: {stats_fdm['mean_velocity']:.4f}")
    print(f"Max pressure    - Original: {stats_orig['max_pressure']:.4f}, FDM: {stats_fdm['max_pressure']:.4f}")
    print(f"Min pressure    - Original: {stats_orig['min_pressure']:.4f}, FDM: {stats_fdm['min_pressure']:.4f}")
    
    return solver_original, solver_fdm

def plot_comparison(solver_orig, solver_fdm):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get velocity magnitudes
    u_orig_center = 0.5 * (solver_orig.u[1:, :] + solver_orig.u[:-1, :])
    v_orig_center = 0.5 * (solver_orig.v[:, 1:] + solver_orig.v[:, :-1])
    vel_mag_orig = np.sqrt(u_orig_center**2 + v_orig_center**2)
    vel_mag_fdm = solver_fdm.get_velocity_magnitude()
    
    # Create coordinate grids
    x = np.linspace(0, solver_orig.Lx, solver_orig.nx)
    y = np.linspace(0, solver_orig.Ly, solver_orig.ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Plot velocity magnitude - Original
    im1 = axes[0, 0].contourf(X.T, Y.T, vel_mag_orig.T, levels=20, cmap='viridis')
    axes[0, 0].set_title('Original Solver - Velocity Magnitude')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot velocity magnitude - FDM
    im2 = axes[0, 1].contourf(X.T, Y.T, vel_mag_fdm.T, levels=20, cmap='viridis')
    axes[0, 1].set_title('FDM Solver - Velocity Magnitude')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot velocity difference
    vel_diff = np.abs(vel_mag_orig - vel_mag_fdm)
    im3 = axes[0, 2].contourf(X.T, Y.T, vel_diff.T, levels=20, cmap='Reds')
    axes[0, 2].set_title('Velocity Magnitude Difference')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot pressure - Original
    im4 = axes[1, 0].contourf(X.T, Y.T, solver_orig.p.T, levels=20, cmap='RdBu_r')
    axes[1, 0].set_title('Original Solver - Pressure')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Plot pressure - FDM
    im5 = axes[1, 1].contourf(X.T, Y.T, solver_fdm.p.T, levels=20, cmap='RdBu_r')
    axes[1, 1].set_title('FDM Solver - Pressure')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Plot pressure difference
    pressure_diff = np.abs(solver_orig.p - solver_fdm.p)
    im6 = axes[1, 2].contourf(X.T, Y.T, pressure_diff.T, levels=20, cmap='Reds')
    axes[1, 2].set_title('Pressure Difference')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im6, ax=axes[1, 2])
    
    # Add obstacle visualization
    for ax in axes.flat:
        obstacle_mask = (1 - solver_orig.obstacle).astype(bool)
        ax.contourf(X.T, Y.T, obstacle_mask.T, levels=[0.5, 1.5], colors=['black'], alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('solver_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_readability():
    """Demonstrate the improved readability of the FDM solver"""
    print("Code Readability Comparison")
    print("=" * 30)
    
    print("Original solver convection calculation:")
    print("- Complex array indexing and interpolation")
    print("- Manual finite difference stencils")
    print("- Difficult to debug and modify")
    print()
    
    print("FDM solver convection calculation:")
    print("- Clear method names: gradient_x(), gradient_y(), laplacian()")
    print("- Separated interpolation functions")
    print("- Easy to understand physics")
    print("- Modular and extensible")
    print()
    
    print("Key improvements:")
    print("✓ FiniteDifferenceOperators class encapsulates all FD operations")
    print("✓ Method names clearly indicate physical operations")
    print("✓ Better documentation and comments")
    print("✓ Easier to add new operators or modify existing ones")
    print("✓ More maintainable and educational code")

def main():
    """Run all tests"""
    print("CFD Solver Comparison Test")
    print("=" * 50)
    
    # Test both solvers
    solver_orig, solver_fdm = test_both_solvers()
    
    # Test readability
    test_readability()
    
    # Create comparison plots
    print("Creating comparison plots...")
    plot_comparison(solver_orig, solver_fdm)
    
    print("Test completed successfully!")
    print("Check 'solver_comparison.png' for visual comparison.")

if __name__ == "__main__":
    main()
