import matplotlib.pyplot as plt
from complex_numbers import *
from complex_plane import Circle

# constants
V = 53
KAPPA = 263000
PI = np.pi


def jouk(z: Cmplxcar, jouk_const):
    return z + jouk_const ** 2 * (z.conj() / abs(z)**2)

##############################################################################
def phi_vector_field(x, y, k, v):
    phi_x = v + (k / (2 * PI)) * ((y ** 2 - x ** 2) / (x ** 2 + y ** 2) ** 2)
    phi_y = (-k / PI) * ((x * y) / (x ** 2 + y ** 2) ** 2)
    return phi_x, phi_y

def apply_jouk(x, y, jouk_const):
    original_arr = generate_cmplxcar_arr(x, y)
    print("original_arr: ", original_arr)
    jouk_arr = np.vectorize(jouk)(original_arr, jouk_const)
    print("\njouk_arr: ", jouk_arr)
    x_jouk, y_jouk = extract_x_y_from_cmplxcar_arr(jouk_arr)
    return x_jouk, y_jouk


def plot_jouk(field_limit_x, field_limit_y, steps, kappa, jouk_const, v):
    # 1D arrays
    x = np.linspace(-field_limit_x, field_limit_x, steps)
    y = np.linspace(-field_limit_y, field_limit_y, steps)

    # Meshgrid
    X, Y = np.meshgrid(x, y)

    # Assign vector field
    phi_x, phi_y = phi_vector_field(X, Y, kappa, v)
    x_jouk, y_jouk = apply_jouk(phi_x, phi_y, jouk_const)
    print(phi_x)
    print(x_jouk)

    # Plot Vector Field
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("BEFORE")
    # ax1.quiver(X, Y, phi_x, phi_y)
    ax1.streamplot(X, Y, phi_x, phi_y)

    ax2.set_title("AFTER")
    # ax2.quiver(X, Y, x_jouk, y_jouk)
    ax2.streamplot(X, Y, x_jouk, y_jouk)
    plt.show()


# plot_jouk(100, 100, 50, KAPPA, 1.165, V)

##############################################################################

bright_mode = {
    "fig_color": "white",
    "ax_color": "white",
    "axis_color": "black",
    "title_color": "black",
    "circle_color": "black",
    "airfoil_color": "white",
    "airfoil_edge_color": "black",
    "contour_style": "Blues"
}

dark_mode = {
    "fig_color": "black",
    "ax_color": "black",
    "axis_color": "white",
    "title_color": "white",
    "circle_color": "white",
    "airfoil_color": "white",
    "airfoil_edge_color": "white",
    "contour_style": "Blues"
}


def potential_field(z: Cmplxcar, v, r):
    if abs(z) <= r or (z.real() == 0):
        return Cmplxcar(0, 0)
    return v * (z + r ** 2 * (z.conj() / abs(z)**2))


def set_ax_style(ax, ax_color, axis_color):
    ax.set_facecolor(ax_color)
    ax.spines['bottom'].set_color(axis_color)
    ax.spines['left'].set_color(axis_color)
    ax.spines['right'].set_color(axis_color)
    ax.spines['top'].set_color(axis_color)
    ax.tick_params(axis='x', colors=axis_color)
    ax.tick_params(axis='y', colors=axis_color)
    return ax


def design_cyl_ax(ax1, X, Y, f_y, lines_num, cyl_shift, r, ax_color,
                  axis_color, title_color, circle_color, contour_style):
    ax1.set_title("Cylinder", color=title_color, fontsize=14, pad=15)
    ax1.set_aspect('equal')
    ax1.contour(X, Y, f_y, lines_num, cmap=contour_style)
    circle = plt.Circle((-cyl_shift, 0), r, color=circle_color, zorder=2)
    ax1.add_artist(circle)
    ax1 = set_ax_style(ax1, ax_color, axis_color)
    return ax1


def design_drop_ax(ax2, J_X, J_Y, f_y, lines_num, field_limit_x,
                   field_limit_y, r, cyl_shift, jouk_const, ax_color,
                   axis_color, title_color, airfoil_color, airfoil_edge_color,
                   contour_style):
    ax2.set_title("Streamlined Body", color=title_color, fontsize=14, pad=15)
    ax2.set_aspect('equal')
    ax2.contour(J_X, J_Y, f_y, lines_num, cmap=contour_style, linewidths=1)
    ax2.set_xlim((-field_limit_x, field_limit_x))
    ax2.set_ylim((-field_limit_y, field_limit_y))
    cyl = Circle(r, (-cyl_shift, 0))
    airfoil = cyl.joukowski(jouk_const)
    x, y = airfoil.get_x_arr_y_arr()
    ax2.plot(x, y, color=airfoil_edge_color)
    ax2.fill(x, y, airfoil_color, zorder=2)
    ax2 = set_ax_style(ax2, ax_color, axis_color)
    return ax2


def plot_drop(field_limit_x, field_limit_y, steps, jouk_const, cyl_shift,
             lines_num, fig_color, ax_color, axis_color, title_color,
             circle_color, airfoil_color, airfoil_edge_color, contour_style,
             v=V, r=1.4):

    x = np.linspace(-field_limit_x, field_limit_x, steps)
    y = np.linspace(-field_limit_y, field_limit_y, steps)

    X, Y = np.meshgrid(x, y)
    Z = generate_cmplxcar_arr(X, Y)
    jouk_arr = np.vectorize(jouk)(Z, jouk_const)
    J_X, J_Y = extract_x_y_from_cmplxcar_arr(jouk_arr)

    f = np.vectorize(potential_field)(Z+cyl_shift, v, r)
    f_x, f_y = extract_x_y_from_cmplxcar_arr(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300, facecolor=fig_color)

    ax1 = design_cyl_ax(ax1, X, Y, f_y, lines_num, cyl_shift, r, ax_color,
                        axis_color, title_color, circle_color,
                        contour_style)
    ax2 = design_drop_ax(ax2, J_X, J_Y, f_y, lines_num, field_limit_x,
                         field_limit_y, r, cyl_shift, jouk_const, ax_color,
                         axis_color, title_color, airfoil_color,
                         airfoil_edge_color, contour_style)

    plt.show()


plot_drop(5, 5, 100, 1.165, cyl_shift=0.16, lines_num=35, **dark_mode, v=53,
         r=1.4)
