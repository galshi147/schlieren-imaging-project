from cfd_solver_fdm import CfdSolverFdm
from matplotlib import pyplot as plt, cm
from matplotlib.animation import FuncAnimation
import numpy as np

from cfd_obstacles import CfdObstacle

class CfdPlotterFdm:
    """
    Plotter class for the FDM-based CFD solver with clear and readable visualization methods.
    """

    def __init__(self, solver: CfdSolverFdm, obstacle: CfdObstacle):
        self.solver = solver
        self.nx, self.ny, self.Lx, self.Ly = self.solver.get_grid_data()
        self.obstacle = obstacle
        self.fig, (self.velocity_ax, self.pressure_ax) = plt.subplots(1, 2, figsize=(10, 8), dpi=100)
        self.velocity_ax: plt.Axes
        self.pressure_ax: plt.Axes

        # Setup grid for plotting
        self.setup_grid()
        
        # Animation properties
        self.animation = None
        self.frame_count = 0
        
    def setup_grid(self):
        """Setup coordinate grids for plotting"""
        # Cell-centered grid for pressure and visualization
        self.x_center = np.linspace(0, self.Lx, self.nx)
        self.y_center = np.linspace(0, self.Ly, self.ny)
        self.X_center, self.Y_center = np.meshgrid(self.x_center, self.y_center, indexing='ij')
    
    def get_velocity_at_centers(self):
        """Get velocity components interpolated to cell centers"""
        u, v = self.solver.get_velocities()
        # Interpolate u from faces to centers
        u_center = np.zeros((self.nx, self.ny))
        u_center[:, :] = 0.5 * (u[1:, :] + u[:-1, :])
        
        # Interpolate v from faces to centers
        v_center = np.zeros((self.nx, self.ny))
        v_center[:, :] = 0.5 * (v[:, 1:] + v[:, :-1])
        
        return u_center, v_center
    
    def plot_static(self):
        """Create a static plot of the current simulation state"""
        u_center, v_center = self.get_velocity_at_centers()
        velocity_magnitude = np.sqrt(u_center**2 + v_center**2)
        
        # Clear previous plots
        self.velocity_ax.clear()
        self.pressure_ax.clear()
        
        # Plot velocity magnitude with streamlines
        self.velocity_ax.set_title(f'Velocity Field (Frame {self.frame_count})')
        
        # Velocity magnitude contour
        vel_contour = self.velocity_ax.contourf(self.X_center.T, self.Y_center.T, velocity_magnitude.T, 
                                               levels=20, cmap='viridis')
        
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
            print("Streamline plotting failed.")
        
        # Plot obstacle
        obstacle_mask = (1 - self.obstacle).astype(bool)
        self.velocity_ax.contourf(self.X_center.T, self.Y_center.T, obstacle_mask.T, 
                                 levels=[0.5, 1.5], colors=['black'], alpha=0.8)
        
        self.velocity_ax.set_xlabel('x')
        self.velocity_ax.set_ylabel('y')
        self.velocity_ax.set_aspect('equal')
        self.velocity_ax.set_xlim(0, self.Lx)
        self.velocity_ax.set_ylim(0, self.Ly)
        plt.colorbar(vel_contour, ax=self.velocity_ax, label='Velocity Magnitude')
        
        # Plot pressure field
        self.pressure_ax.set_title(f'Pressure Field (Frame {self.frame_count})')
        pressure_contour = self.pressure_ax.contourf(self.X_center.T, self.Y_center.T, self.solver.get_pressure().T,
                                                    levels=20, cmap='RdBu_r')
        
        # Plot obstacle on pressure plot
        self.pressure_ax.contourf(self.X_center.T, self.Y_center.T, obstacle_mask.T,
                                 levels=[0.5, 1.5], colors=['black'], alpha=0.8)
        
        self.pressure_ax.set_xlabel('x')
        self.pressure_ax.set_ylabel('y')
        self.pressure_ax.set_aspect('equal')
        self.pressure_ax.set_xlim(0, self.solver.Lx)
        self.pressure_ax.set_ylim(0, self.solver.Ly)
        plt.colorbar(pressure_contour, ax=self.pressure_ax, label='Pressure')
        
        plt.tight_layout()
        plt.show()
    
    def design_velocity_ax(self):
        u_center, v_center = self.get_velocity_at_centers()
        velocity_magnitude = np.sqrt(u_center**2 + v_center**2)
        self.velocity_ax.set_title(f'Velocity Field (Frame {self.frame_count})')
        vel_contour = self.velocity_ax.contourf(self.X_center.T, self.Y_center.T, velocity_magnitude.T,
                                               levels=100, cmap='viridis')
        # # Add velocity vectors (subsample for clarity)
        # skip = max(1, self.solver.nx // 20)
        # self.velocity_ax.quiver(self.X_center[::skip, ::skip].T, self.Y_center[::skip, ::skip].T,
        #                        u_center[::skip, ::skip].T, v_center[::skip, ::skip].T,
        #                        scale=20, alpha=0.7, color='white', width=0.003)
        # Add streamlines
        try:
            self.velocity_ax.streamplot(self.X_center.T, self.Y_center.T, u_center.T, v_center.T,
                                       color='white', density=1.5, linewidth=0.8)
        except Exception as e:
            print(f"Streamline plotting failed: {e}")
        
        # Only create colorbar on first frame, then update it
        if not hasattr(self, 'velocity_cbar'):
            self.velocity_cbar = self.fig.colorbar(vel_contour, ax=self.velocity_ax, label='Velocity Magnitude')
        else:
            # Update existing colorbar with new data
            self.velocity_cbar.update_normal(vel_contour)


    def design_pressure_ax(self):
        pressure_field = self.solver.get_pressure()
        self.pressure_ax.set_title(f'Pressure Field (Frame {self.frame_count})')
        pressure_contour = self.pressure_ax.contourf(self.X_center.T, self.Y_center.T, pressure_field.T,
                                                    levels=100, cmap='RdBu_r')
        
        # Only create colorbar on first frame, then update it
        if not hasattr(self, 'pressure_cbar'):
            self.pressure_cbar = self.fig.colorbar(pressure_contour, ax=self.pressure_ax, label='Pressure')
        else:
            # Update existing colorbar with new data
            self.pressure_cbar.update_normal(pressure_contour)

    def animate_step(self, frame):
        """Animation function called for each frame"""
        # Perform one simulation step
        for _ in range(3):
            self.solver.step()
        self.frame_count += 1
        
        # Clear axes
        self.velocity_ax.clear()
        self.pressure_ax.clear()
                
        # Update plots
        self.design_velocity_ax()
        self.design_pressure_ax()
        for ax in [self.velocity_ax, self.pressure_ax]:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            ax.set_xlim(0, self.Lx)
            ax.set_ylim(0, self.Ly)
            # Plot obstacle
            # obstacle_mask = (1 - self.obstacle).astype(bool)
            # ax.contourf(self.X_center.T, self.Y_center.T, obstacle_mask.T,
            #                      levels=[0.5, 1.5], colors=['black'], alpha=0.8)
            self.obstacle.plot_obstacle(ax)

        plt.tight_layout()
        return []
    
    def animate(self, frames=100, interval=50, save_animation=False, filename='cfd_fdm_animation.gif'):
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
    
    def save_frame(self, filename='cfd_fdm_frame.png'):
        """Save current frame as image"""
        self.plot_static()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Frame saved as {filename}")
    
    def get_simulation_stats(self):
        """Get current simulation statistics"""
        return self.solver.get_simulation_stats()
    
    def plot_comparison_with_analytical(self, analytical_solution=None):
        """Plot comparison with analytical solution if available"""
        if analytical_solution is None:
            print("No analytical solution provided")
            return
        
        u_center, v_center = self.get_velocity_at_centers()
        velocity_magnitude = np.sqrt(u_center**2 + v_center**2)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot numerical solution
        im1 = axes[0].contourf(self.X_center.T, self.Y_center.T, velocity_magnitude.T, 
                              levels=20, cmap='viridis')
        axes[0].set_title('Numerical Solution')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot analytical solution
        im2 = axes[1].contourf(self.X_center.T, self.Y_center.T, analytical_solution.T, 
                              levels=20, cmap='viridis')
        axes[1].set_title('Analytical Solution')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot difference
        difference = np.abs(velocity_magnitude - analytical_solution)
        im3 = axes[2].contourf(self.X_center.T, self.Y_center.T, difference.T, 
                              levels=20, cmap='Reds')
        axes[2].set_title('Absolute Difference')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[2])
        
        # Add obstacle to all plots
        obstacle_mask = (1 - self.solver.obstacle).astype(bool)
        for ax in axes:
            ax.contourf(self.X_center.T, self.Y_center.T, obstacle_mask.T,
                       levels=[0.5, 1.5], colors=['black'], alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print error statistics
        fluid_mask = self.solver.obstacle.astype(bool)
        if fluid_mask.any():
            max_error = difference[fluid_mask].max()
            mean_error = difference[fluid_mask].mean()
            print(f"Maximum error: {max_error:.6f}")
            print(f"Mean error: {mean_error:.6f}")
            print(f"Relative error: {mean_error/analytical_solution[fluid_mask].mean()*100:.2f}%")

