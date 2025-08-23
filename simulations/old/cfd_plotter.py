from simulations.old.cfd_solver import CfdSolver
from matplotlib import pyplot as plt, cm
from matplotlib.animation import FuncAnimation
import numpy as np

class CfdPlotter:
    def __init__(self, solver: CfdSolver):
        self.solver = solver
        self.fig, (self.velocity_ax, self.pressure_ax) = plt.subplots(2, 1, figsize=(12, 10), dpi=100)
        
        # Setup grid for plotting
        self.setup_grid()
        
        # Initialize plot elements
        self.velocity_magnitude = None
        self.pressure_contour = None
        self.streamlines = None
        self.quiver = None
        
        # Animation properties
        self.animation = None
        self.frame_count = 0
        
    def setup_grid(self):
        """Setup coordinate grids for plotting"""
        # Cell-centered grid for pressure and obstacles
        self.x_center = np.linspace(0, self.solver.Lx, self.solver.nx)
        self.y_center = np.linspace(0, self.solver.Ly, self.solver.ny)
        self.X_center, self.Y_center = np.meshgrid(self.x_center, self.y_center, indexing='ij')
        
        # Staggered grids for velocity components
        self.x_u = np.linspace(-self.solver.dx/2, self.solver.Lx + self.solver.dx/2, self.solver.nx + 1)
        self.y_u = np.linspace(self.solver.dy/2, self.solver.Ly - self.solver.dy/2, self.solver.ny)
        
        self.x_v = np.linspace(self.solver.dx/2, self.solver.Lx - self.solver.dx/2, self.solver.nx)
        self.y_v = np.linspace(-self.solver.dy/2, self.solver.Ly + self.solver.dy/2, self.solver.ny + 1)
    
    def interpolate_velocity_to_center(self):
        """Interpolate staggered velocity components to cell centers"""
        # Interpolate u from faces to centers
        u_center = np.zeros((self.solver.nx, self.solver.ny))
        u_center[1:-1, :] = 0.5 * (self.solver.u[1:-1, :] + self.solver.u[2:, :])
        u_center[0, :] = self.solver.u[0, :]
        u_center[-1, :] = self.solver.u[-1, :]
        
        # Interpolate v from faces to centers
        v_center = np.zeros((self.solver.nx, self.solver.ny))
        v_center[:, 1:-1] = 0.5 * (self.solver.v[:, 1:-1] + self.solver.v[:, 2:])
        v_center[:, 0] = self.solver.v[:, 0]
        v_center[:, -1] = self.solver.v[:, -1]
        
        return u_center, v_center
    
    def plot_static(self):
        """Create a static plot of the current simulation state"""
        u_center, v_center = self.interpolate_velocity_to_center()
        velocity_magnitude = np.sqrt(u_center**2 + v_center**2)
        
        # Clear previous plots
        self.velocity_ax.clear()
        self.pressure_ax.clear()
        
        # Plot velocity magnitude
        self.velocity_ax.set_title(f'Velocity Magnitude (Frame {self.frame_count})')
        vel_contour = self.velocity_ax.contourf(self.X_center.T, self.Y_center.T, velocity_magnitude.T, 
                                               levels=20, cmap='viridis')
        
        # Add velocity vectors (subsample for clarity)
        skip = max(1, self.solver.nx // 20)
        self.velocity_ax.quiver(self.X_center[::skip, ::skip].T, self.Y_center[::skip, ::skip].T,
                               u_center[::skip, ::skip].T, v_center[::skip, ::skip].T,
                               scale=20, alpha=0.7, color='white', width=0.003)
        
        # Plot streamlines
        self.velocity_ax.streamplot(self.X_center.T, self.Y_center.T, u_center.T, v_center.T,
                                   color='red', density=1.5, linewidth=0.8, alpha=0.6)
        
        # Plot obstacle
        obstacle_mask = (1 - self.solver.obstacle).astype(bool)
        self.velocity_ax.contourf(self.X_center.T, self.Y_center.T, obstacle_mask.T, 
                                 levels=[0.5, 1.5], colors=['black'], alpha=0.8)
        
        self.velocity_ax.set_xlabel('x')
        self.velocity_ax.set_ylabel('y')
        self.velocity_ax.set_aspect('equal')
        plt.colorbar(vel_contour, ax=self.velocity_ax, label='Velocity Magnitude')
        
        # Plot pressure
        self.pressure_ax.set_title(f'Pressure Field (Frame {self.frame_count})')
        pressure_contour = self.pressure_ax.contourf(self.X_center.T, self.Y_center.T, self.solver.p.T,
                                                    levels=20, cmap='RdBu_r')
        
        # Plot obstacle on pressure plot
        self.pressure_ax.contourf(self.X_center.T, self.Y_center.T, obstacle_mask.T,
                                 levels=[0.5, 1.5], colors=['black'], alpha=0.8)
        
        self.pressure_ax.set_xlabel('x')
        self.pressure_ax.set_ylabel('y')
        self.pressure_ax.set_aspect('equal')
        plt.colorbar(pressure_contour, ax=self.pressure_ax, label='Pressure')
        
        plt.tight_layout()
        plt.show()
    
    def animate_step(self, frame):
        """Animation function called for each frame"""
        # Perform one simulation step
        self.solver.step()
        self.frame_count += 1
        
        # Update plots
        u_center, v_center = self.interpolate_velocity_to_center()
        velocity_magnitude = np.sqrt(u_center**2 + v_center**2)
        
        # Clear axes
        self.velocity_ax.clear()
        self.pressure_ax.clear()
        
        # Update velocity plot
        self.velocity_ax.set_title(f'Velocity Magnitude (Frame {self.frame_count})')
        vel_contour = self.velocity_ax.contourf(self.X_center.T, self.Y_center.T, velocity_magnitude.T,
                                               levels=20, cmap='viridis', vmin=0, vmax=2.0)
        
        # Add velocity vectors (subsample for clarity)
        skip = max(1, self.solver.nx // 20)
        self.velocity_ax.quiver(self.X_center[::skip, ::skip].T, self.Y_center[::skip, ::skip].T,
                               u_center[::skip, ::skip].T, v_center[::skip, ::skip].T,
                               scale=20, alpha=0.7, color='white', width=0.003)
        
        # Add streamlines
        try:
            self.velocity_ax.streamplot(self.X_center.T, self.Y_center.T, u_center.T, v_center.T,
                                       color='red', density=1.5, linewidth=0.8, alpha=0.6)
        except:
            pass  # Skip streamlines if they fail (can happen with complex flows)
        
        # Plot obstacle
        obstacle_mask = (1 - self.solver.obstacle).astype(bool)
        self.velocity_ax.contourf(self.X_center.T, self.Y_center.T, obstacle_mask.T,
                                 levels=[0.5, 1.5], colors=['black'], alpha=0.8)
        
        self.velocity_ax.set_xlabel('x')
        self.velocity_ax.set_ylabel('y')
        self.velocity_ax.set_aspect('equal')
        self.velocity_ax.set_xlim(0, self.solver.Lx)
        self.velocity_ax.set_ylim(0, self.solver.Ly)
        
        # Update pressure plot
        self.pressure_ax.set_title(f'Pressure Field (Frame {self.frame_count})')
        pressure_contour = self.pressure_ax.contourf(self.X_center.T, self.Y_center.T, self.solver.p.T,
                                                    levels=20, cmap='RdBu_r')
        
        # Plot obstacle on pressure plot
        self.pressure_ax.contourf(self.X_center.T, self.Y_center.T, obstacle_mask.T,
                                 levels=[0.5, 1.5], colors=['black'], alpha=0.8)
        
        self.pressure_ax.set_xlabel('x')
        self.pressure_ax.set_ylabel('y')
        self.pressure_ax.set_aspect('equal')
        self.pressure_ax.set_xlim(0, self.solver.Lx)
        self.pressure_ax.set_ylim(0, self.solver.Ly)
        
        plt.tight_layout()
        return []
    
    def animate(self, frames=1000, interval=50, save_animation=False, filename='cfd_animation.gif'):
        """Start the animation"""
        self.frame_count = 0
        
        # Create animation
        self.animation = FuncAnimation(self.fig, self.animate_step, frames=frames,
                                      interval=interval, blit=False, repeat=False)
        
        if save_animation:
            print(f"Saving animation as {filename}...")
            self.animation.save(filename, writer='pillow', fps=20)
            print("Animation saved!")
        
        plt.show()
        return self.animation
    
    def save_frame(self, filename='cfd_frame.png'):
        """Save current frame as image"""
        self.plot_static()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Frame saved as {filename}")
    
    def get_simulation_stats(self):
        """Get current simulation statistics"""
        u_center, v_center = self.interpolate_velocity_to_center()
        velocity_magnitude = np.sqrt(u_center**2 + v_center**2)
        
        # Calculate statistics only in fluid regions
        fluid_mask = self.solver.obstacle.astype(bool)
        
        stats = {
            'frame': self.frame_count,
            'max_velocity': velocity_magnitude[fluid_mask].max() if fluid_mask.any() else 0,
            'avg_velocity': velocity_magnitude[fluid_mask].mean() if fluid_mask.any() else 0,
            'max_pressure': self.solver.p[fluid_mask].max() if fluid_mask.any() else 0,
            'min_pressure': self.solver.p[fluid_mask].min() if fluid_mask.any() else 0,
        }
        
        return stats
