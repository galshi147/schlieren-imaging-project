from cfd import *
from cfd_plot import *
from timeit import default_timer as timer

def main():
    start = timer()

    my_cyl = Cyl(0.1)
    simulation_data = plot_simulation(my_cyl, LENGTH, NX, NY, TIME_ITERATIONS, DT,
                                DX, DY, AIR_STATIC_DENSITY,
                                AIR_KINEMATIC_VISCOSITY, NIT)
    # save_simulation(animation)
    end = timer()
    print(simulation_data, "\nexecution time (s): ", end - start)
    print(get_Re_num(C, LENGTH, AIR_KINEMATIC_VISCOSITY))


if __name__ == '__main__':
    main()

