from cfd_obstacles import CfdObstacle
from operators import FiniteDifferenceOperators
import numpy as np


class CfdSolverFdm:
    """
    A class to solve the 2D incompressible Navier-Stokes equations using Projection Method.    
    The equations solved are:
    (-1-) ∂u/∂t + (⊽·∇)⊽ = -1/ρ ∇p + 𝜈∇²⊽
    (-2-) ∇·⊽ = 0 (incompressibility constraint)
    
    connective term: (⊽·∇)⊽
    diffusion term: 𝜈∇²⊽

    Projection method steps:
    1. Compute intermediate velocity without pressure: ⊽* = ⊽ + dt(-(⊽·∇)⊽ + 𝜈∇²⊽)
    2. Solve pressure Poisson equation: ∇²p = ρ/dt * (∇·⊽*)
    3. Correct velocity: ⊽^(n+1) = ⊽* - dt/ρ * ∇p

    self.u is the x-velocity field on a staggered MAC grid
    self.v is the y-velocity field on a staggered MAC grid
    self.p is the pressure field at cell centers

    ┌─────┬─────┬─────┐
    │  P  │  P  │  P  │  ← Pressure at cell centers
    ├←─u──┼←─u──┼←─u──┤  ← u-velocity at cell faces (vertical edges)
    │  P  │  P  │  P  │
    ├←─u──┼←─u──┼←─u──┤
    │  P  │  P  │  P  │
    └─────┴─────┴─────┘
       ↑     ↑     ↑
       v     v     v      ← v-velocity at cell faces (horizontal edges)
    """
    
    def __init__(self, obstacle: CfdObstacle, nx=128, ny=64, Lx=4.0, Ly=2.0, dt=0.005, rho=1, nu=0.01, initial_velocity=1.0) -> None:
        """Initialize the CFD solver with the given parameters.

        Args:
            obstacle (CfdObstacle): The obstacle object defining the fluid domain.
            nx (int, optional): Number of grid points in the x-direction. Defaults to 128.
            ny (int, optional): Number of grid points in the y-direction. Defaults to 64.
            Lx (float, optional): Length of the domain in the x-direction. Defaults to 4.0.
            Ly (float, optional): Length of the domain in the y-direction. Defaults to 2.0.
            dt (float, optional): Time step size. Defaults to 0.005.
            rho (int, optional): (ρ) Density of the fluid. Defaults to 1.
            nu (float, optional): (𝜈) Kinematic viscosity of the fluid. Defaults to 0.01.
            initial_velocity (float, optional): Initial velocity magnitude. Defaults to 1.0.
        """
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

        self.init_obstacle(obstacle)
    
    def init_obstacle(self, obstacle: CfdObstacle) -> None:
        """Initialize obstacle mask at pressure grid points (cell centers)"""
        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy
        Xc, Yc = np.meshgrid(x, y, indexing='ij')
        # Initialize fluid mask (1=fluid, 0=solid)
        chi = np.ones((self.nx, self.ny), dtype=np.int8)
        self.obstacle = obstacle.get_mask(Xc, Yc, chi)

    def get_grid_data(self) -> tuple[int, int, float, float]:
        """Get grid data for the simulation.

        Returns:
            tuple[int, int, float, float]: Grid dimensions (nx, ny) and domain sizes (Lx, Ly).
        """
        return self.nx, self.ny, self.Lx, self.Ly

    def get_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        """Get velocity fields for the simulation.

        Returns:
            tuple[np.ndarray, np.ndarray]: u-velocity and v-velocity fields.
        """
        return self.u, self.v

    def get_pressure(self) -> np.ndarray:
        """Get pressure field for the simulation.

        Returns:
            np.ndarray: Pressure field.
        """
        return self.p
    
    def get_obstacle(self):
        """Get obstacle mask for the simulation.

        Returns:
            np.ndarray: Obstacle mask.
        """
        return self.obstacle

    def initialize_flow(self, steps=50, time_step_coef=0.1, viscosity_coef=10, print_info=False):
        """Gradually initialize flow field to avoid shock.

        Args:
            steps (int, optional): Number of initialization steps. Defaults to 50.
            time_step_coef (float, optional): Coefficient to reduce time step. Defaults to 0.1.
            viscosity_coef (float, optional): Coefficient to increase viscosity. Defaults to 10.
            print_info (bool, optional): Whether to print initialization info. Defaults to False.
        """
        # Store original parameters
        original_dt = self.dt
        original_nu = self.nu
        
        # Use smaller time step and higher viscosity for initialization
        self.dt = original_dt * time_step_coef
        self.nu = original_nu * viscosity_coef  # Higher viscosity for smooth startup

        if print_info: print("Initializing flow field...")
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
                if print_info: print(f"  Step {i}: max |u| = {max_u:.4f}")

        # Restore original parameters
        self.dt = original_dt
        self.nu = original_nu
        if print_info: print("Flow initialization complete.")

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

    def apply_obstacle_boundary_conditions(self, u_field: np.ndarray, v_field: np.ndarray) -> None:
        """Apply no-slip boundary conditions on obstacle surfaces 
        (Zero velocity on faces adjacent to solid cells)"""
        
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
    
    def compute_convective_terms(self, artificial_viscosity_coef=0.01):
        """
        Compute convective terms: (⊽·∇)⊽ and (⊽·∇)v
        Using proper interpolation for staggered grids
        """
        # Initialize convection arrays
        self.convection_u = np.zeros_like(self.u)
        self.convection_v = np.zeros_like(self.v)

        # Add artificial viscosity for stability
        artificial_viscosity = artificial_viscosity_coef * max(self.dx, self.dy) * np.max(np.abs(self.u))

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

    def compute_diffusion_terms(self):
        """Compute viscous diffusion terms: 𝜈∇²u and 𝜈∇²v"""
        self.diffusion_u = self.nu * self.fd_ops.laplacian(self.u, self.dx, self.dy)
        self.diffusion_v = self.nu * self.fd_ops.laplacian(self.v, self.dx, self.dy)
    
    def compute_intermediate_velocity(self) -> None:
        """
        Compute intermediate velocity field without pressure correction
        ⊽* = ⊽^n + dt * (-convection + diffusion)
        """
        self.u_star = self.u + self.dt * (-self.convection_u + self.diffusion_u)
        self.v_star = self.v + self.dt * (-self.convection_v + self.diffusion_v)
        
        # Apply boundary conditions to intermediate velocity
        self.apply_obstacle_boundary_conditions(self.u_star, self.v_star)
    
    def solve_pressure_poisson(self):
        """
        Solve pressure Poisson equation: ∇²p = ρ/dt * (∇·⊽*)
        Using Gauss-Seidel iteration with successive over-relaxation (SOR) method
        """
        # Compute divergence of intermediate velocity
        div_u_star = self.fd_ops.divergence(self.u_star, self.v_star, self.dx, self.dy)
        rhs = (self.rho / self.dt) * div_u_star
        
        # Solve using SOR iteration
        self.p = self._solve_poisson_sor(rhs)

    def _solve_poisson_sor(self, rhs: np.ndarray, max_iterations: int = 100, omega: float = 1.7) -> np.ndarray:
        """
        Solve Poisson equation using Successive Over-Relaxation (SOR)
        with Neumann boundary conditions on obstacles
        Args:
            rhs (np.ndarray): Right-hand side of the Poisson equation (Inhomogeneous term).
            max_iterations (int, optional): Maximum number of SOR iterations. Defaults to 100.
            omega (float, optional): Relaxation factor (1 < omega < 2). Defaults to 1.7.
        Returns:
            np.ndarray: Solved pressure field.
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

    def correct_velocity(self) -> None:
        """
        Correct velocity using pressure gradient: ⊽^(n+1) = ⊽* - dt/ρ * ∇p
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
    
    def step(self) -> None:
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

    def run(self, iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the simulation for a specified number of iterations

        Args:
            iterations (int): Number of iterations to run the simulation

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Final velocity and pressure fields
        """
        for _ in range(iterations):
            self.step()
        return self.u, self.v, self.p
    
    def get_velocity_magnitude(self) -> np.ndarray:
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