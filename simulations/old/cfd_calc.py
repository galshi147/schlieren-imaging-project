from simulations.old.cfd import *
# from tqdm import tqdm


def build_poisson_eq_inhomogeneous_part(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                             (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (
                                         2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))
    return b


def pressure_poisson(shape, p, dx, dy, b, nit, length):
    pn = np.empty_like(p)
    pn = p.copy()

    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) /
                         (2 * (dx ** 2 + dy ** 2)) -
                         dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                         b[1:-1, 1:-1])

        p = shape.apply_pressure_condition(p, dx, dy, length)  # p=0 inside the shape

    return p


def velocity_u_update(u, dx, dy, dt, rho, p, un, vn, nu):
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx ** 2 *
                           (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1,
                                                                0:-2]) +
                           dt / dy ** 2 *
                           (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2,
                                                                1:-1])))
    return u


def velocity_v_update(v, dx, dy, dt, rho, p, un, vn, nu):
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt / dx ** 2 *
                           (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1,
                                                                0:-2]) +
                           dt / dy ** 2 *
                           (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2,
                                                                1:-1])))
    return v


def flow(shape, nx, ny, nt, u, v, dt, dx, dy, p, rho, nu, nit, length):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    pn = np.empty_like(p)
    b = np.zeros((ny, nx))
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        pn = p.copy()
        b = build_poisson_eq_inhomogeneous_part(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(shape, p, dx, dy, b, nit, length)

        u = velocity_u_update(u, dx, dy, dt, rho, p, un, vn, nu)
        v = velocity_v_update(v, dx, dy, dt, rho, p, un, vn, nu)

        u[:, 0] = C
        # u[0, :] = C
        # u[-1, :] = C
        v[:, 0] = 0
        # v[0, :] = 0
        # v[-1, :] = 0

        u, v = shape.apply_no_slip_condition(u, v, dx, dy, length)

    return u, v, p


def initiate_u_v_p(nx, ny):
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    return u, v, p


def run_simulation(u, v, p, shape, nx, ny, nt, dt, dx, dy, rho, nu, nit, length):
    b = np.zeros((ny, nx))
    u_update, v_update, p_update = flow(shape, nx, ny, nt, u, v, dt, dx, dy, p, rho, nu, nit, length)
    return u_update, v_update, p_update
