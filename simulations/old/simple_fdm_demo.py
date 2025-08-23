"""
Simple working FDM solver demonstration
"""

import numpy as np
from cfd_obstacles import CfdCircle
import matplotlib.pyplot as plt

class SimpleFdmSolver:
    """
    A simplified FDM solver for demonstration of readable code structure.
    This focuses on clarity over performance.
    """
    
    def __init__(self, obstacle, nx=64, ny=32, Lx=2.0, Ly=1.0, dt=1e-3):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.nu = 1e-3
        self.rho = 1.0
        
        # Initialize fields on regular grid (not staggered for simplicity)
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.p = np.zeros((nx, ny))
        
        # Initialize obstacle
        self.init_obstacle(obstacle)
        
        # Set initial conditions
        self.u[0, :] = 1.0  # Inflow
    
    def init_obstacle(self, obstacle):
        """Initialize obstacle on regular grid"""
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        chi = np.ones((self.nx, self.ny), dtype=np.int8)
        self.obstacle = obstacle.get_mask(X, Y, chi)
    
    def gradient_x(self, field):
        """Compute x-gradient using central differences"""
        grad = np.zeros_like(field)
        grad[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * self.dx)
        return grad
    
    def gradient_y(self, field):
        """Compute y-gradient using central differences"""
        grad = np.zeros_like(field)
        grad[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * self.dy)
        return grad
    
    def laplacian(self, field):
        """Compute Laplacian using 5-point stencil"""
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1] = ((field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.dx**2 +
                           (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / self.dy**2)
        return lap
    
    def divergence(self, u_field, v_field):
        """Compute divergence"""
        div = np.zeros_like(u_field)
        div[1:-1, 1:-1] = ((u_field[2:, 1:-1] - u_field[:-2, 1:-1]) / (2 * self.dx) +
                           (v_field[1:-1, 2:] - v_field[1:-1, :-2]) / (2 * self.dy))
        return div
    
    def apply_boundary_conditions(self):
        """Apply simple boundary conditions"""
        # Inflow
        self.u[0, :] = 1.0
        self.v[0, :] = 0.0
        
        # Walls
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0
        
        # Outflow
        self.u[-1, :] = self.u[-2, :]
        self.v[-1, :] = self.v[-2, :]
        
        # Obstacle
        self.u[self.obstacle == 0] = 0.0
        self.v[self.obstacle == 0] = 0.0
    
    def compute_convection(self):
        """Compute convective terms clearly"""
        # u * du/dx + v * du/dy
        conv_u = self.u * self.gradient_x(self.u) + self.v * self.gradient_y(self.u)
        
        # u * dv/dx + v * dv/dy  
        conv_v = self.u * self.gradient_x(self.v) + self.v * self.gradient_y(self.v)
        
        return conv_u, conv_v
    
    def compute_diffusion(self):
        """Compute diffusion terms clearly"""
        diff_u = self.nu * self.laplacian(self.u)
        diff_v = self.nu * self.laplacian(self.v)
        return diff_u, diff_v
    
    def solve_pressure_poisson(self, rhs):
        """Solve pressure Poisson equation using simple iteration"""
        p = np.zeros_like(rhs)
        
        for _ in range(50):  # Simple iteration
            p_old = p.copy()
            p[1:-1, 1:-1] = 0.25 * (p_old[2:, 1:-1] + p_old[:-2, 1:-1] + 
                                    p_old[1:-1, 2:] + p_old[1:-1, :-2] - 
                                    self.dx**2 * rhs[1:-1, 1:-1])
            
            # Neumann BC on obstacles
            p[self.obstacle == 0] = 0.0
            
            # Reference pressure
            p[1, 1] = 0.0
        
        return p
    
    def step(self):
        """Perform one time step using projection method"""
        self.apply_boundary_conditions()
        
        # Compute convective and diffusive terms
        conv_u, conv_v = self.compute_convection()
        diff_u, diff_v = self.compute_diffusion()
        
        # Intermediate velocity (without pressure)
        u_star = self.u + self.dt * (-conv_u + diff_u)
        v_star = self.v + self.dt * (-conv_v + diff_v)
        
        # Apply BC to intermediate velocity
        u_star[self.obstacle == 0] = 0.0
        v_star[self.obstacle == 0] = 0.0
        
        # Solve pressure Poisson equation
        div_u_star = self.divergence(u_star, v_star)
        rhs = self.rho / self.dt * div_u_star
        self.p = self.solve_pressure_poisson(rhs)
        
        # Correct velocity
        self.u = u_star - self.dt / self.rho * self.gradient_x(self.p)
        self.v = v_star - self.dt / self.rho * self.gradient_y(self.p)
        
        # Apply final BC
        self.apply_boundary_conditions()
    
    def run(self, steps):
        """Run simulation for given number of steps"""
        for i in range(steps):
            self.step()
            if i % 10 == 0:
                vel_mag = np.sqrt(self.u**2 + self.v**2)
                print(f"Step {i}: Max velocity = {np.max(vel_mag):.4f}")
    
    def plot(self):
        """Create a simple plot"""
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        X, Y = np.meshgrid(x, y)
        
        # Velocity magnitude
        im1 = ax1.contourf(X, Y, vel_mag.T, levels=20, cmap='viridis')
        ax1.set_title('Velocity Magnitude')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1)
        
        # Add obstacle
        obstacle_mask = (1 - self.obstacle).astype(bool)
        ax1.contourf(X, Y, obstacle_mask.T, levels=[0.5, 1.5], colors=['black'], alpha=0.8)
        
        # Pressure
        im2 = ax2.contourf(X, Y, self.p.T, levels=20, cmap='RdBu_r')
        ax2.set_title('Pressure')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2)
        
        # Add obstacle
        ax2.contourf(X, Y, obstacle_mask.T, levels=[0.5, 1.5], colors=['black'], alpha=0.8)
        
        plt.tight_layout()
        plt.show()

def main():
    """Demonstrate the simple FDM solver"""
    print("Simple FDM Solver Demonstration")
    print("=" * 35)
    
    # Create circular obstacle
    circle = CfdCircle(radius=0.2, center=(1.0, 0.5))
    
    # Create solver
    solver = SimpleFdmSolver(obstacle=circle, nx=64, ny=32)
    
    print("Key advantages of this readable implementation:")
    print("✓ Clear method names: gradient_x(), laplacian(), divergence()")
    print("✓ Separate functions for each physical process")
    print("✓ Easy to understand and modify")
    print("✓ Educational and research-friendly")
    print()
    
    # Run simulation
    print("Running simulation...")
    solver.run(steps=50)
    
    # Show results
    print("Creating plot...")
    solver.plot()
    
    print("✓ Simple FDM solver demonstration completed!")
    print()
    print("This shows how finite difference operators can be")
    print("implemented in a clear, readable way that makes")
    print("the underlying physics transparent.")

if __name__ == "__main__":
    main()
