"""
Test script for the CFD animation system
"""

import sys
import os
import numpy as np

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic functionality without animation"""
    print("Testing basic CFD functionality...")
    
    try:
        from simulations.old.cfd_solver import CfdSolver
        from cfd_obstacles import CfdCircle
        from simulations.old.cfd_plotter import CfdPlotter
        
        print("✓ Imports successful")
        
        # Create a simple circular obstacle
        circle = CfdCircle(radius=0.1, center=(1.0, 0.5))
        print("✓ Obstacle created")
        
        # Create solver
        solver = CfdSolver(obstacle=circle, nx=32, ny=16, Lx=2.0, Ly=1.0, dt=1e-3)
        print("✓ Solver created")
        
        # Create plotter
        plotter = CfdPlotter(solver)
        print("✓ Plotter created")
        
        # Run a few simulation steps
        for i in range(5):
            solver.step()
        print("✓ Simulation steps completed")
        
        # Get statistics
        stats = plotter.get_simulation_stats()
        print(f"✓ Statistics: Max vel = {stats['max_velocity']:.4f}")
        
        # Test velocity interpolation
        u_center, v_center = plotter.interpolate_velocity_to_center()
        print(f"✓ Velocity interpolation: u shape = {u_center.shape}, v shape = {v_center.shape}")
        
        print("All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_obstacles():
    """Test different obstacle types"""
    print("\nTesting different obstacle types...")
    
    try:
        from cfd_obstacles import CfdCircle, CfdTriangle, CfdAirfoil
        
        # Test circle
        circle = CfdCircle(radius=0.1, center=(0.5, 0.5))
        print("✓ Circle obstacle created")
        
        # Test triangle
        vertices = np.array([[0.4, 0.4], [0.6, 0.4], [0.5, 0.6]])
        triangle = CfdTriangle(vertices=vertices)
        print("✓ Triangle obstacle created")
        
        # Test airfoil
        airfoil = CfdAirfoil(NACA_code='0012')
        print("✓ Airfoil obstacle created")
        
        print("All obstacle tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Obstacle test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("CFD Animation System Tests")
    print("=" * 30)
    
    success = True
    
    # Test basic functionality
    success &= test_basic_functionality()
    
    # Test obstacles
    success &= test_obstacles()
    
    if success:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nTo run animations, use demo_animation.py:")
        print("python demo_animation.py")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()
