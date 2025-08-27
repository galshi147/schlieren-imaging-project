from simulations.demo_fdm import (
    demo_circle_flow, demo_triangle_flow, demo_airfoil_flow, demo_square_flow,
    demo_comparison, quick_demo, interactive_demo
)

def main():
    """Main demo function with all examples"""
    print("🌊 CFD FDM Solver - Ready-to-Run Examples")
    print("=" * 45)
    print()
    
    # Uncomment the demo you want to run:
    
    # 1. Classic vortex shedding
    # solver, plotter = demo_circle_flow(save_animation=True)
    
    # 2. Sharp body aerodynamics
    # solver, plotter = demo_triangle_flow()
    
    # 3. Airfoil aerodynamics
    solver, plotter = demo_airfoil_flow(save_animation=True)
    
    # 4. Bluff body flow
    # solver, plotter = demo_square_flow()
    
    # 5. Quick visualization
    # solver, plotter = quick_demo()
    
    # 6. Compare multiple obstacles
    # demo_comparison()
    
    # 7. Interactive choice
    # solver, plotter = interactive_demo()
    
    print("\n✓ Demo completed!")
    return solver, plotter

if __name__ == "__main__":
    solver, plotter = main()
    
