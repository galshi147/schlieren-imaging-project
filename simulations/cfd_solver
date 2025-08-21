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
        x = (np.arange(nx)+0.5)*dx
        y = (np.arange(ny)+0.5)*dy
        Xc, Yc = np.meshgrid(x, y, indexing='ij')

        chi = np.ones((nx, ny), dtype=np.int8)  # 1=fluid, 0=solid
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
        # centered with simple upwind blend (theta)
        theta = 0.1
        uc = self.u.copy()
        # derivatives in conservative flux form
        # (quick & readable; vectorized carefully in production)
        # du2/dx at u-locations
        du2dx = (uc[1:,:]**2 - uc[:-1,:]**2)/self.dx
        # (uv) at (i+1/2, j+1/2)
        v_on_u_jph = 0.5*(self.v[1:,:-1] + self.v[:-1,:-1])  # shape (nx, ny)
        v_on_u_jmh = 0.5*(self.v[1:,1:]  + self.v[:-1,1:])   # notice indexing flip
        u_jph = 0.5*(self.u[:,1:] + self.u[:,:-1])
        u_jmh = 0.5*(self.u[:,1:] + self.u[:,:-1])
        uv_jph = u_jph[1:-1,:]*v_on_u_jph[1:-1,:]
        uv_jmh = u_jmh[1:-1,:]*v_on_u_jmh[1:-1,:]
        duvdy = (uv_jph - uv_jmh)/self.dy

        Au = np.zeros_like(self.u)
        Au[1:-1,1:-1] = du2dx[1:-1,1:-1] + duvdy[:,1:-1]

        # simple upwind damping
        upwind = np.zeros_like(Au)
        upwind[1:-1,1:-1] = (np.abs(self.u[1:-1,1:-1])*(self.u[2:,1:-1]-2*self.u[1:-1,1:-1]+self.u[:-2,1:-1])/self.dx
                            + 0.25*np.abs(self.v[1:-1,1:-1])*(self.u[1:-1,2:]-2*self.u[1:-1,1:-1]+self.u[1:-1,:-2])/self.dy)
        return (1-theta)*Au + theta*upwind

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



    


# --- grid ---
nx, ny = 256, 128
Lx, Ly = 4.0, 2.0
dx, dy = Lx/nx, Ly/ny

# MAC grid arrays
u = np.zeros((nx+1, ny))      # u at i+1/2,j  -> shape (nx+1, ny)
v = np.zeros((nx, ny+1))      # v at i,j+1/2  -> shape (nx, ny+1)
p = np.zeros((nx, ny))        # p at i,j

rho = 1.0
nu  = 1e-3
dt  = 5e-3

# --- obstacle mask at pressure (cell centers) ---
x = (np.arange(nx)+0.5)*dx
y = (np.arange(ny)+0.5)*dy
Xc, Yc = np.meshgrid(x, y, indexing='ij')

chi = np.ones((nx, ny), dtype=np.int8)  # 1=fluid, 0=solid

# Example: circular obstacle
cx, cy, R = 1.0, 1.0, 0.25
chi[(Xc-cx)**2 + (Yc-cy)**2 <= R**2] = 0

def apply_bc(u, v, p):
    # Inflow (left): uniform profile
    u[0, :] = 1.0        # set ghost/leftmost u-face
    v[0, :] = 0.0
    v[-1, :] = 0.0

    # No-slip top/bottom walls
    u[:, 0]  = 0.0; u[:, -1] = 0.0
    v[:, 0]  = 0.0; v[:, -1] = 0.0

    # Outflow (right): du/dx=0 -> copy interior
    u[-1, :] = u[-2, :]

def zero_faces_in_solid(u, v, chi):
    # zero any face adjacent to a solid cell center
    # u face between p[i-1,j] and p[i,j]
    solidL = np.pad(1-chi, ((1,0),(0,0)), constant_values=0)
    solidR = np.pad(1-chi, ((0,1),(0,0)), constant_values=0)
    mask_u = np.logical_or(solidL, solidR)
    u[mask_u] = 0.0

    # v face between p[i,j-1] and p[i,j]
    solidB = np.pad(1-chi, ((0,0),(1,0)), constant_values=0)
    solidT = np.pad(1-chi, ((0,0),(0,1)), constant_values=0)
    mask_v = np.logical_or(solidB, solidT)
    v[mask_v] = 0.0

def advect_u(u, v, dx, dy):
    # centered with simple upwind blend (theta)
    theta = 0.1
    uc = u.copy()
    # derivatives in conservative flux form
    # (quick & readable; vectorized carefully in production)
    # du2/dx at u-locations
    du2dx = (u[1:,:]**2 - u[:-1,:]**2)/dx
    # (uv) at (i+1/2, j+1/2)
    v_on_u_jph = 0.5*(v[1:,:-1] + v[:-1,:-1])  # shape (nx, ny)
    v_on_u_jmh = 0.5*(v[1:,1:]  + v[:-1,1:])   # notice indexing flip
    u_jph = 0.5*(u[:,1:] + u[:,:-1])
    u_jmh = 0.5*(u[:,1:] + u[:,:-1])
    uv_jph = u_jph[1:-1,:]*v_on_u_jph[1:-1,:]
    uv_jmh = u_jmh[1:-1,:]*v_on_u_jmh[1:-1,:]
    duvdy = (uv_jph - uv_jmh)/dy

    Au = np.zeros_like(u)
    Au[1:-1,1:-1] = du2dx[1:-1,1:-1] + duvdy[:,1:-1]

    # simple upwind damping
    upwind = np.zeros_like(Au)
    upwind[1:-1,1:-1] = (np.abs(u[1:-1,1:-1])*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1])/dx
                         + 0.25*np.abs(v[1:-1,1:-1])*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])/dy)
    return (1-theta)*Au + theta*upwind

def advect_v(u, v, dx, dy):
    theta = 0.1
    dv2dy = (v[:,1:]**2 - v[:,:-1]**2)/dy
    u_on_v_iph = 0.5*(u[:-1,1:] + u[:-1,:-1])
    u_on_v_imh = 0.5*(u[1:,1:]  + u[1:,:-1])
    v_iph = 0.5*(v[1:,:] + v[:-1,:])
    v_imh = 0.5*(v[1:,:] + v[:-1,:])
    uv_iph = u_on_v_iph[:,1:-1]*v_iph[:,1:-1]
    uv_imh = u_on_v_imh[:,1:-1]*v_imh[:,1:-1]
    duvdx = (uv_iph - uv_imh)/dx

    Av = np.zeros_like(v)
    Av[1:-1,1:-1] = duvdx[1:-1,:] + dv2dy[1:-1,1:-1]

    upwind = np.zeros_like(Av)
    upwind[1:-1,1:-1] = (np.abs(v[1:-1,1:-1])*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])/dy
                         + 0.25*np.abs(u[1:-1,1:-1])*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1])/dx)
    return (1-theta)*Av + theta*upwind

def diffuse_face(q, dx, dy, nu):
    lap = np.zeros_like(q)
    lap[1:-1,1:-1] = ((q[2:,1:-1]-2*q[1:-1,1:-1]+q[:-2,1:-1])/dx**2 +
                      (q[1:-1,2:]-2*q[1:-1,1:-1]+q[1:-1,:-2])/dy**2)
    return nu*lap

def divergence(u, v, dx, dy):
    div = (u[1:,:]-u[:-1,:])/dx + (v[:,1:]-v[:,:-1])/dy
    return div  # shape (nx, ny)

def solve_pressure_poisson(rhs, chi, dx, dy, iters=200, omega=1.7):
    # 5-point Laplacian with Neumann on solids via masking
    p = np.zeros_like(rhs)
    mx, my = rhs.shape
    inv_dx2, inv_dy2 = 1.0/dx**2, 1.0/dy**2
    denom = 2*(inv_dx2+inv_dy2)

    # ensure solvability: set mean(rhs over fluid)=0
    fluid = chi.astype(bool)
    mean_rhs = rhs[fluid].mean() if fluid.any() else 0.0
    rhs = rhs - mean_rhs

    for _ in range(iters):
        # Gauss–Seidel with SOR over interior fluid cells
        for i in range(1, mx-1):
            for j in range(1, my-1):
                if chi[i,j]==0:   # solid -> skip; Neumann implied
                    continue
                pxm = p[i-1,j] if chi[i-1,j] else p[i,j]
                pxp = p[i+1,j] if chi[i+1,j] else p[i,j]
                pym = p[i,j-1] if chi[i,j-1] else p[i,j]
                pyp = p[i,j+1] if chi[i,j+1] else p[i,j]
                p_new = ( (pxp+pxm)*inv_dx2 + (pyp+pym)*inv_dy2 - rhs[i,j] )/denom
                p[i,j] = (1-omega)*p[i,j] + omega*p_new
        # fix reference pressure at one fluid cell
        if fluid.any():
            i0, j0 = np.argwhere(fluid)[0]
            p[i0,j0] = 0.0
    return p

def step(u, v, p, chi, dt):
    apply_bc(u, v, p)
    zero_faces_in_solid(u, v, chi)

    Au = advect_u(u, v, dx, dy)
    Av = advect_v(u, v, dx, dy)
    Du = diffuse_face(u, dx, dy, nu)
    Dv = diffuse_face(v, dx, dy, nu)

    u_star = u + dt*(-Au + Du)
    v_star = v + dt*(-Av + Dv)

    zero_faces_in_solid(u_star, v_star, chi)

    rhs = (rho/dt)*divergence(u_star, v_star, dx, dy)
    p_new = solve_pressure_poisson(rhs, chi, dx, dy)

    # correct
    gradp_x = (p_new[1:,:]-p_new[:-1,:])/dx
    gradp_y = (p_new[:,1:]-p_new[:,:-1])/dy
    u_corr = u_star.copy()
    v_corr = v_star.copy()
    u_corr[1:-1,:] -= (dt/rho)*gradp_x
    v_corr[:,1:-1] -= (dt/rho)*gradp_y

    apply_bc(u_corr, v_corr, p_new)
    zero_faces_in_solid(u_corr, v_corr, chi)
    return u_corr, v_corr, p_new

# --- time loop (example) ---
# for n in range(1000):
#     u, v, p = step(u, v, p, chi, dt)
#     # (optional) adaptive dt by CFL, residual checks, outputs, etc.
