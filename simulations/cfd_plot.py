from cfd_calc import *
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch


def create_mirror(radius):
    fig, ax_mirror = plt.subplots()
    circle = plt.Circle((0, 0), radius, color='gray', fill=False)
    ax_mirror.add_artist(circle)
    ax_mirror.set_aspect('equal')
    return ax_mirror


velocity_field_style = {"density": 2, "linewidth": 0.5, "color": "navy"}


def plot_simulation(shape, length, nx, ny, nt, dt, dx, dy, rho, nu, nit):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    X, Y = create_mesh(nx, ny, length)
    u, v, p = initiate_u_v_p(nx, ny)
    u[0, :] = C
    pressure_field = ax.contourf(X, Y, p, alpha=0.5, cmap=cm.coolwarm)
    fig.colorbar(pressure_field, ax=ax)
    velocity_field = ax.streamplot(X, Y, u, v, **velocity_field_style)
    center = (length/2, length/2)
    shape.plot(center, ax=ax)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_title(f"{shape.get_size()} ; length = {length} ; C = {C} ; nt = {nt} ; DT = {dt} ; nx = {nx}, ny = {ny}")

    def init_state():
        return pressure_field, velocity_field

    def update_state(time):
        nonlocal pressure_field, velocity_field  # Allow access to outer scope variables

        u_i, v_i, p_i = initiate_u_v_p(nx, ny)
        u, v, p = run_simulation(u_i, v_i, p_i, shape, nx, ny, time, dt, dx, dy, rho, nu, nit, length)
        for contour in pressure_field.collections:
            contour.remove()  # Remove old contour
        pressure_field = ax.contourf(X, Y, p, alpha=0.5, cmap=cm.coolwarm)

        # Remove old streamlines (LineCollection)
        for collection in ax.collections:
            if isinstance(collection, LineCollection):
                collection.remove()

        # Remove old arrows (FancyArrowPatch) from ax.patches
        for patch in ax.patches:
            if isinstance(patch, FancyArrowPatch):
                patch.remove()

        # Update the velocity streamplot with the new data
        velocity_field = ax.streamplot(X, Y, u, v, **velocity_field_style)

        return pressure_field, velocity_field

    frames = np.arange(0, nt, 1)
    animation = FuncAnimation(fig, update_state, frames=frames,
                          init_func=init_state, blit=False, interval=400)
    simulation_data = f"{shape.get_size()}_len{length}_C{C}_nt{nt}_DT{dt}_nxy={nx,ny}"
    animation.save(f"cfd_{simulation_data}.mp4")
    return simulation_data


def save_simulation(animation):
    print(type(animation))
    command = input(
        """would you like to save the animation? press 'Y' or 'N': """)
    if command == 'Y':
        animation.save("cfd.mp4")
