from cfd_obstacles import CfdObstacle
import numpy as np

class FiniteDifferenceOperators:
    """
    A utility class for finite difference operators on staggered grids.
    This provides clean, readable implementations of vector calculus operators.
    """
    
    @staticmethod
    def gradient_x(field, dx):
        """Compute gradient in x-direction using central differences"""
        grad = np.zeros_like(field)
        grad[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dx)
        return grad
    
    @staticmethod
    def gradient_y(field, dy):
        """Compute gradient in y-direction using central differences"""
        grad = np.zeros_like(field)
        grad[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dy)
        return grad
    
    @staticmethod
    def divergence(u_field, v_field, dx, dy):
        """Compute divergence of velocity field on staggered grid"""
        nx, ny = u_field.shape[0] - 1, v_field.shape[1] - 1
        div = np.zeros((nx, ny))
        
        # ∇·u = ∂u/∂x + ∂v/∂y on MAC grid
        div[:, :] = (u_field[1:, :] - u_field[:-1, :]) / dx + \
                    (v_field[:, 1:] - v_field[:, :-1]) / dy
        return div
    
    @staticmethod
    def laplacian(field, dx, dy):
        """Compute Laplacian using 5-point stencil"""
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1] = ((field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / dx**2 +
                           (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / dy**2)
        return lap
    
    @staticmethod
    def interpolate_v_to_u_faces(v_field):
        """Interpolate v-velocity to u-face locations"""
        nx_u, ny_u = v_field.shape[0] + 1, v_field.shape[1] - 1
        v_on_u = np.zeros((nx_u, ny_u))
        
        # Average v from surrounding cell faces - fix the indexing
        if nx_u > 2 and ny_u > 0:
            v_on_u[1:-1, :] = 0.25 * (v_field[:-1, :-1] + v_field[:-1, 1:] + 
                                       v_field[1:, :-1] + v_field[1:, 1:])
        return v_on_u
    
    @staticmethod
    def interpolate_u_to_v_faces(u_field):
        """Interpolate u-velocity to v-face locations"""
        nx_v, ny_v = u_field.shape[0] - 1, u_field.shape[1] + 1
        u_on_v = np.zeros((nx_v, ny_v))
        
        # Average u from surrounding cell faces - fix the indexing
        if nx_v > 0 and ny_v > 2:
            u_on_v[:, 1:-1] = 0.25 * (u_field[:-1, :-1] + u_field[1:, :-1] + 
                                       u_field[:-1, 1:] + u_field[1:, 1:])
        return u_on_v


class CfdSolverFdm:
    """
    A class to solve the 2D incompressible Navier-Stokes equations using Projection Method
    with clear and readable finite difference operators.
    
    The equations solved are:
    ∂u/∂t + (u·∇)u = -1/ρ ∇p + ν∇²u
    ∇·u = 0 (incompressibility constraint)
    
    Using the projection method:
    1. Compute intermediate velocity without pressure: u* = u + dt*(-∇u + ν∇²u)
    2. Solve pressure Poisson equation: ∇²p = ρ/dt * ∇·u*
    3. Correct velocity: u^(n+1) = u* - dt/ρ * ∇p
    """
    
    def __init__(self, obstacle: CfdObstacle, nx=128, ny=64, Lx=4.0, Ly=2.0, dt=0.005, rho=1, nu=0.01, initial_velocity=1.0):
        # Physical parameters
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx/nx
        self.dy = Ly/ny
        self.dt = dt
        self.rho = rho
        self.nu = nu
        self.initial_velocity = initial_velocity
        
        # Create finite difference operators
        self.fd_ops = FiniteDifferenceOperators()
        
        # Initialize velocity and pressure fields (MAC grid)
        self.u = np.ones((nx+1, ny)) * self.initial_velocity  # u-velocity at i+1/2,j
        self.v = np.zeros((nx, ny+1))      # v-velocity at i,j+1/2
        self.p = np.zeros((nx, ny))        # pressure at cell centers
        
        # Intermediate variables for projection method
        self.u_star = None
        self.v_star = None
        self.convection_u = None
        self.convection_v = None
        self.diffusion_u = None
        self.diffusion_v = None

        # Initialize obstacle
        self.init_obstacle(obstacle)
    
    def init_obstacle(self, obstacle: CfdObstacle):
        """Initialize obstacle mask at pressure grid points (cell centers)"""
        # Create coordinate arrays for pressure grid
        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy
        Xc, Yc = np.meshgrid(x, y, indexing='ij')
        
        # Initialize fluid mask (1=fluid, 0=solid)
        chi = np.ones((self.nx, self.ny), dtype=np.int8)
        self.obstacle = obstacle.get_mask(Xc, Yc, chi)
    
    def get_grid_data(self):
        return self.nx, self.ny, self.Lx, self.Ly
    
    def get_velocities(self):
        return self.u, self.v
    
    def get_pressure(self):
        return self.p
    
    def get_obstacle(self):
        return self.obstacle
    
    def initialize_flow(self, steps=50):
        """Initialize flow field gradually to avoid shock"""
        # Store original parameters
        original_dt = self.dt
        original_nu = self.nu
        
        # Use smaller time step and higher viscosity for initialization
        self.dt = original_dt * 0.1
        self.nu = original_nu * 10  # Higher viscosity for smooth startup
        
        print("Initializing flow field...")
        for i in range(steps):
            # Only apply boundary conditions and diffusion (no convection)
            self.apply_boundary_conditions()
            self.apply_obstacle_boundary_conditions(self.u, self.v)
            
            # Compute only diffusion terms
            self.compute_diffusion_terms()
            
            # Simple explicit step
            self.u += self.dt * self.diffusion_u
            self.v += self.dt * self.diffusion_v
            
            # Apply boundary conditions
            self.apply_boundary_conditions()
            self.apply_obstacle_boundary_conditions(self.u, self.v)
            
            if i % 10 == 0:
                max_u = np.max(np.abs(self.u))
                print(f"  Step {i}: max |u| = {max_u:.4f}")
        
        # Restore original parameters
        self.dt = original_dt
        self.nu = original_nu
        print("Flow initialization complete.")

    def apply_boundary_conditions(self):
        """Apply boundary conditions for flow domain"""
        # Inflow (left boundary): uniform flow
        self.u[0, :] = self.initial_velocity  # Inflow velocity
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
        
        # No-slip walls (top and bottom)
        self.u[:, 0] = 0.0    # Bottom wall
        self.u[:, -1] = 0.0   # Top wall
        self.v[:, 0] = 0.0    # Bottom wall
        self.v[:, -1] = 0.0   # Top wall
        
        # Outflow (right boundary): zero gradient
        self.u[-1, :] = self.u[-2, :]
    
    def apply_obstacle_boundary_conditions(self, u_field, v_field):
        """Apply no-slip boundary conditions on obstacle surfaces"""
        # Zero velocity on faces adjacent to solid cells
        
        # For u-faces (between pressure cells)
        solid_left = np.pad(1 - self.obstacle, ((1, 0), (0, 0)), constant_values=0)
        solid_right = np.pad(1 - self.obstacle, ((0, 1), (0, 0)), constant_values=0)
        u_mask = np.logical_or(solid_left, solid_right)
        u_field[u_mask] = 0.0
        
        # For v-faces (between pressure cells)
        solid_bottom = np.pad(1 - self.obstacle, ((0, 0), (1, 0)), constant_values=0)
        solid_top = np.pad(1 - self.obstacle, ((0, 0), (0, 1)), constant_values=0)
        v_mask = np.logical_or(solid_bottom, solid_top)
        v_field[v_mask] = 0.0
    
    def compute_convective_terms(self):
        """
        Compute convective terms: (u·∇)u and (u·∇)v
        Using proper interpolation for staggered grids
        """
        # Initialize convection arrays
        self.convection_u = np.zeros_like(self.u)
        self.convection_v = np.zeros_like(self.v)

        # Add artificial viscosity for stability
        artificial_viscosity = 0.01 * max(self.dx, self.dy) * np.max(np.abs(self.u))
        
        # For u-momentum: u * ∂u/∂x + v * ∂u/∂y
        if self.u.shape[0] > 2 and self.u.shape[1] > 2:
            # u * ∂u/∂x (both u and ∂u/∂x are at u-faces)
            du_dx = self.fd_ops.gradient_x(self.u, self.dx)
            self.convection_u += self.u * du_dx

        # Add artificial viscosity term
        if artificial_viscosity > 0:
            self.convection_u += artificial_viscosity * self.fd_ops.laplacian(self.u, self.dx, self.dy)
            
            # v * ∂u/∂y (need to interpolate v to u-faces)
            v_at_u = self.fd_ops.interpolate_v_to_u_faces(self.v)
            du_dy = self.fd_ops.gradient_y(self.u, self.dy)
            self.convection_u += v_at_u * du_dy
        
        # For v-momentum: u * ∂v/∂x + v * ∂v/∂y
        if self.v.shape[0] > 2 and self.v.shape[1] > 2:
            # v * ∂v/∂y (both v and ∂v/∂y are at v-faces)
            dv_dy = self.fd_ops.gradient_y(self.v, self.dy)
            self.convection_v += self.v * dv_dy

        # Add artificial viscosity term
        if artificial_viscosity > 0:
            self.convection_v += artificial_viscosity * self.fd_ops.laplacian(self.v, self.dx, self.dy)

            # u * ∂v/∂x (need to interpolate u to v-faces)
            u_at_v = self.fd_ops.interpolate_u_to_v_faces(self.u)
            dv_dx = self.fd_ops.gradient_x(self.v, self.dx)
            self.convection_v += u_at_v * dv_dx
    
    def compute_convective_terms_old(self):
        """Compute convective terms clearly"""
        # u * du/dx + v * du/dy
        self.convection_u = self.u * self.fd_ops.gradient_x(self.u, self.dx) + self.fd_ops.interpolate_v_to_u_faces(self.v) * self.fd_ops.gradient_y(self.u, self.dy)

        # u * dv/dx + v * dv/dy
        self.convection_v = self.fd_ops.interpolate_u_to_v_faces(self.u) * self.fd_ops.gradient_x(self.v, self.dx) + self.v * self.fd_ops.gradient_y(self.v, self.dy)

    def compute_diffusion_terms(self):
        """Compute viscous diffusion terms: ν∇²u and ν∇²v"""
        self.diffusion_u = self.nu * self.fd_ops.laplacian(self.u, self.dx, self.dy)
        self.diffusion_v = self.nu * self.fd_ops.laplacian(self.v, self.dx, self.dy)
    
    def compute_intermediate_velocity(self):
        """
        Compute intermediate velocity field without pressure correction
        u* = u^n + dt * (-convection + diffusion)
        """
        self.u_star = self.u + self.dt * (-self.convection_u + self.diffusion_u)
        self.v_star = self.v + self.dt * (-self.convection_v + self.diffusion_v)
        
        # Apply boundary conditions to intermediate velocity
        self.apply_obstacle_boundary_conditions(self.u_star, self.v_star)
    
    def solve_pressure_poisson(self):
        """
        Solve pressure Poisson equation: ∇²p = ρ/dt * ∇·u*
        Using Gauss-Seidel iteration with successive over-relaxation (SOR)
        """
        # Compute divergence of intermediate velocity
        div_u_star = self.fd_ops.divergence(self.u_star, self.v_star, self.dx, self.dy)
        rhs = (self.rho / self.dt) * div_u_star
        
        # Solve using SOR iteration
        self.p = self._solve_poisson_sor(rhs, max_iterations=100, omega=1.7)
    
    def _solve_poisson_sor(self, rhs, max_iterations=100, omega=1.7):
        """
        Solve Poisson equation using Successive Over-Relaxation (SOR)
        with Neumann boundary conditions on obstacles
        """
        p = np.zeros_like(rhs)
        dx2_inv, dy2_inv = 1.0 / self.dx**2, 1.0 / self.dy**2
        denominator = 2 * (dx2_inv + dy2_inv)
        
        # Ensure solvability: set mean of RHS over fluid domain to zero
        fluid_mask = self.obstacle.astype(bool)
        if fluid_mask.any():
            mean_rhs = rhs[fluid_mask].mean()
            rhs = rhs - mean_rhs
        
        # SOR iteration
        for _ in range(max_iterations):
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    if self.obstacle[i, j] == 0:  # Skip solid cells
                        continue
                    
                    # Neumann BC on obstacles: use current value if neighbor is solid
                    p_west = p[i-1, j] if self.obstacle[i-1, j] else p[i, j]
                    p_east = p[i+1, j] if self.obstacle[i+1, j] else p[i, j]
                    p_south = p[i, j-1] if self.obstacle[i, j-1] else p[i, j]
                    p_north = p[i, j+1] if self.obstacle[i, j+1] else p[i, j]
                    
                    # 5-point stencil for Laplacian
                    p_new = ((p_east + p_west) * dx2_inv + 
                            (p_north + p_south) * dy2_inv - rhs[i, j]) / denominator
                    
                    # SOR update
                    p[i, j] = (1 - omega) * p[i, j] + omega * p_new
            
            # Fix reference pressure (set one fluid cell to zero)
            if fluid_mask.any():
                i0, j0 = np.argwhere(fluid_mask)[0]
                p[i0, j0] = 0.0
        
        return p
    
    def correct_velocity(self):
        """
        Correct velocity using pressure gradient: u^(n+1) = u* - dt/ρ * ∇p
        """
        # Compute pressure gradients at face locations
        # For u-faces: gradient between adjacent pressure cells
        dp_dx = np.zeros_like(self.u)
        dp_dx[1:-1, :] = (self.p[1:, :] - self.p[:-1, :]) / self.dx
        
        # For v-faces: gradient between adjacent pressure cells  
        dp_dy = np.zeros_like(self.v)
        dp_dy[:, 1:-1] = (self.p[:, 1:] - self.p[:, :-1]) / self.dy
        
        # Velocity correction
        self.u = self.u_star - (self.dt / self.rho) * dp_dx
        self.v = self.v_star - (self.dt / self.rho) * dp_dy
    
        # self.u = self.u_star - (self.dt / self.rho) * self.fd_ops.gradient_x(self.p, self.dx)
        # self.v = self.v_star - (self.dt / self.rho) * self.fd_ops.gradient_y(self.p, self.dy)

    def step(self):
        """Perform one time step of the Navier-Stokes solver"""
        # Apply boundary conditions
        self.apply_boundary_conditions()
        self.apply_obstacle_boundary_conditions(self.u, self.v)
        
        # Compute convective and diffusive terms
        self.compute_convective_terms()
        self.compute_diffusion_terms()
        
        # Projection method step 1: compute intermediate velocity
        self.compute_intermediate_velocity()
        
        # Projection method step 2: solve pressure Poisson equation
        self.solve_pressure_poisson()
        
        # Projection method step 3: correct velocity with pressure gradient
        self.correct_velocity()
        
        # Apply boundary conditions again
        self.apply_boundary_conditions()
        self.apply_obstacle_boundary_conditions(self.u, self.v)
    
    def run(self, iterations):
        """Run the simulation for a specified number of iterations"""
        for _ in range(iterations):
            self.step()
        return self.u, self.v, self.p
    
    def get_velocity_magnitude(self):
        """Compute velocity magnitude at cell centers"""
        # Interpolate velocities to cell centers
        u_center = 0.5 * (self.u[1:, :] + self.u[:-1, :])
        v_center = 0.5 * (self.v[:, 1:] + self.v[:, :-1])
        return np.sqrt(u_center**2 + v_center**2)
    
    def get_simulation_stats(self):
        """Get current simulation statistics"""
        vel_mag = self.get_velocity_magnitude()
        fluid_mask = self.obstacle.astype(bool)
        
        return {
            'max_velocity': vel_mag[fluid_mask].max() if fluid_mask.any() else 0,
            'mean_velocity': vel_mag[fluid_mask].mean() if fluid_mask.any() else 0,
            'max_pressure': self.p[fluid_mask].max() if fluid_mask.any() else 0,
            'min_pressure': self.p[fluid_mask].min() if fluid_mask.any() else 0,
            'time_step': self.dt
        }