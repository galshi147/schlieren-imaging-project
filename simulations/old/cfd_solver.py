import numpy as np
from cfd_obstacles import CfdObstacle

class CfdSolver:
    """A class to solve the 2D incompressible Navier-Stokes equations using Projection Method.
    """
    def __init__(self, obstacle: CfdObstacle, nx=128, ny=256, Lx=4.0, Ly=2.0, dt=5e-3, rho=1, nu=1e-3):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx/nx
        self.dy = Ly/ny
        self.dt = dt
        self.rho = rho
        self.nu = nu
        # MAC grid arrays
        self.u = np.zeros((nx+1, ny))      # u at i+1/2,j  -> shape (nx+1, ny)
        self.v = np.zeros((nx, ny+1))      # v at i,j+1/2  -> shape (nx, ny+1)
        self.p = np.zeros((nx, ny))        # p at i,j
        self.u_star = None
        self.v_star = None
        self.advect_u = None
        self.advect_v = None
        self.diffuse_u = None
        self.diffuse_v = None
        self.init_obstacle(obstacle)
    
    def init_obstacle(self, obstacle: CfdObstacle):
        # --- obstacle mask at pressure (cell centers) ---
        x = (np.arange(self.nx)+0.5)*self.dx
        y = (np.arange(self.ny)+0.5)*self.dy
        Xc, Yc = np.meshgrid(x, y, indexing='ij')

        chi = np.ones((self.nx, self.ny), dtype=np.int8)  # 1=fluid, 0=solid
        self.obstacle = obstacle.get_mask(Xc, Yc, chi)

    def apply_frame_boundary_conditions(self):
        # Inflow (left): uniform profile
        self.u[0, :] = 1.0        # set ghost/leftmost u-face
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0

        # No-slip top/bottom walls
        self.u[:, 0]  = 0.0; self.u[:, -1] = 0.0
        self.v[:, 0]  = 0.0; self.v[:, -1] = 0.0

        # Outflow (right): du/dx=0 -> copy interior
        self.u[-1, :] = self.u[-2, :]
    
    def apply_no_slip_obstacle(self, u, v):
        # zero any face adjacent to a solid cell center
        # u face between p[i-1,j] and p[i,j]
        solidL = np.pad(1-self.obstacle, ((1,0),(0,0)), constant_values=0)
        solidR = np.pad(1-self.obstacle, ((0,1),(0,0)), constant_values=0)
        mask_u = np.logical_or(solidL, solidR)
        u[mask_u] = 0.0

        # v face between p[i,j-1] and p[i,j]
        solidB = np.pad(1-self.obstacle, ((0,0),(1,0)), constant_values=0)
        solidT = np.pad(1-self.obstacle, ((0,0),(0,1)), constant_values=0)
        mask_v = np.logical_or(solidB, solidT)
        v[mask_v] = 0.0
    
    def _calc_advect_u(self):
        """
        Calculate the convective term (u ∂u/∂x + v ∂u/∂y) for the u-velocity component
        using a blend of central and upwind differences (theta controls blend).
        """
        theta = 0.1  # upwind blending factor

        # Central differences for (u ∂u/∂x)
        u_center = self.u
        du_dx = np.zeros_like(u_center)
        du_dx[1:-1, :] = (u_center[2:, :] - u_center[:-2, :]) / (2 * self.dx)
        advect_x = u_center[1:-1, :] * du_dx[1:-1, :]

        # Interpolate v to u-face locations for (v ∂u/∂y)
        v_on_u = 0.5 * (self.v[1:-1, :-1] + self.v[1:-1, 1:])  # shape (nx-1, ny)
        du_dy = np.zeros_like(u_center[1:-1, :])
        du_dy[:, 1:-1] = (u_center[1:-1, 2:] - u_center[1:-1, :-2]) / (2 * self.dy)
        advect_y = v_on_u[:, :] * du_dy

        # Combine convective terms
        advect = np.zeros_like(self.u)
        advect[1:-1, 1:-1] = advect_x[:, 1:-1] + advect_y[:, 1:-1]

        # Upwind dissipation (for stability)
        upwind = np.zeros_like(self.u)
        upwind[1:-1, 1:-1] = (
            np.abs(self.u[1:-1, 1:-1]) * (self.u[2:, 1:-1] - 2 * self.u[1:-1, 1:-1] + self.u[:-2, 1:-1]) / self.dx
            + 0.25 * np.abs(v_on_u[:, 1:-1]) * (self.u[1:-1, 2:] - 2 * self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) / self.dy
        )

        # Blend central and upwind
        return (1 - theta) * advect + theta * upwind

    def _calc_advect_v(self):
        theta = 0.1
        dv2dy = (self.v[:,1:]**2 - self.v[:,:-1]**2)/self.dy
        u_on_v_iph = 0.5*(self.u[:-1,1:] + self.u[:-1,:-1])
        u_on_v_imh = 0.5*(self.u[1:,1:]  + self.u[1:,:-1])
        v_iph = 0.5*(self.v[1:,:] + self.v[:-1,:])
        v_imh = 0.5*(self.v[1:,:] + self.v[:-1,:])
        uv_iph = u_on_v_iph[:,1:-1]*v_iph[:,1:-1]
        uv_imh = u_on_v_imh[:,1:-1]*v_imh[:,1:-1]
        duvdx = (uv_iph - uv_imh)/self.dx

        Av = np.zeros_like(self.v)
        Av[1:-1,1:-1] = duvdx[1:-1,:] + dv2dy[1:-1,1:-1]

        upwind = np.zeros_like(Av)
        upwind[1:-1,1:-1] = (np.abs(self.v[1:-1,1:-1])*(self.v[1:-1,2:]-2*self.v[1:-1,1:-1]+self.v[1:-1,:-2])/self.dy
                            + 0.25*np.abs(self.u[1:-1,1:-1])*(self.v[2:,1:-1]-2*self.v[1:-1,1:-1]+self.v[:-2,1:-1])/self.dx)
        return (1-theta)*Av + theta*upwind

    def _calc_diffuse(self, q):
        lap = np.zeros_like(q)
        lap[1:-1,1:-1] = ((q[2:,1:-1]-2*q[1:-1,1:-1]+q[:-2,1:-1])/self.dx**2 +
                          (q[1:-1,2:]-2*q[1:-1,1:-1]+q[1:-1,:-2])/self.dy**2)
        return self.nu * lap
    
    def calc_convective_terms(self):
        self.advect_u = self._calc_advect_u()
        self.advect_v = self._calc_advect_v()

    def calc_diffusion_terms(self):
        self.diffuse_u = self._calc_diffuse(self.u)
        self.diffuse_v = self._calc_diffuse(self.v)

    def calc_intermediate_velocity(self):
        self.u_star = self.u + self.dt*(-self.advect_u + self.diffuse_u)
        self.v_star = self.v + self.dt*(-self.advect_v + self.diffuse_v)

    def _calc_divergence(self, x_component, y_component):
        div = np.zeros((self.nx, self.ny))
        div[1:-1, 1:-1] = (x_component[1:, 1:-1] - x_component[:-1, 1:-1]) / self.dx + \
                         (y_component[1:-1, 1:] - y_component[1:-1, :-1]) / self.dy
        return div

    def _solve_pressure_poisson_helper(self, rhs, iters=200, omega=1.7):
        # 5-point Laplacian with Neumann on solids via masking
        p = np.zeros_like(rhs)
        mx, my = rhs.shape
        inv_dx2, inv_dy2 = 1.0/self.dx**2, 1.0/self.dy**2
        denom = 2*(inv_dx2+inv_dy2)

        # ensure solvability: set mean(rhs over fluid)=0
        fluid = self.obstacle.astype(bool)
        mean_rhs = rhs[fluid].mean() if fluid.any() else 0.0
        rhs = rhs - mean_rhs

        for _ in range(iters):
            # Gauss–Seidel with SOR over interior fluid cells
            for i in range(1, mx-1):
                for j in range(1, my-1):
                    if self.obstacle[i,j]==0:   # solid -> skip; Neumann implied
                        continue
                    pxm = p[i-1,j] if self.obstacle[i-1,j] else p[i,j]
                    pxp = p[i+1,j] if self.obstacle[i+1,j] else p[i,j]
                    pym = p[i,j-1] if self.obstacle[i,j-1] else p[i,j]
                    pyp = p[i,j+1] if self.obstacle[i,j+1] else p[i,j]
                    p_new = ( (pxp+pxm)*inv_dx2 + (pyp+pym)*inv_dy2 - rhs[i,j] )/denom
                    p[i,j] = (1-omega)*p[i,j] + omega*p_new
            # fix reference pressure at one fluid cell
            if fluid.any():
                i0, j0 = np.argwhere(fluid)[0]
                p[i0,j0] = 0.0
        return p

    def solve_pressure_poisson(self):
        div_intermediate_velocity = self._calc_divergence(self.u_star, self.v_star)
        rhs = (self.rho / self.dt) * div_intermediate_velocity
        self.p = self._solve_pressure_poisson_helper(rhs, self.obstacle)

    def correct_velocity(self):
        pressure_grad_x = (self.p[1:,:]-self.p[:-1,:])/self.dx
        pressure_grad_y = (self.p[:,1:]-self.p[:,:-1])/self.dy
        self.u = self.u_star.copy()
        self.v = self.v_star.copy()
        self.u[1:-1,:] -= (self.dt/self.rho)*pressure_grad_x
        self.v[:,1:-1] -= (self.dt/self.rho)*pressure_grad_y

    def step(self):
        self.apply_frame_boundary_conditions()
        self.apply_no_slip_obstacle(self.u, self.v)

        # Calculate convective and diffusion terms
        self.calc_convective_terms()
        self.calc_diffusion_terms()

        # Calculate velocity without pressure correction
        self.calc_intermediate_velocity()

        # Solve pressure Poisson equation: \nabla ^2 p = (rho/dt) * div(u*, v*)
        self.solve_pressure_poisson()

        # Correct velocity based on pressure gradient: velocity = velocity* - (dt/rho) * grad(p)
        self.correct_velocity()

        self.apply_frame_boundary_conditions()
        self.apply_no_slip_obstacle(self.u, self.v)
    
    def run(self, iterations):
        # --- time loop (example) ---
        for _ in range(iterations):
            self.step()
        
        return self.u, self.v, self.p

