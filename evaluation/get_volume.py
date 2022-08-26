from visualization.plot_dvh import get_dvh


def get_volume(dose, my_plan, organ, dose_value, weight_flag=True):
    x, y = get_dvh(dose, my_plan, organ, weight_flag=weight_flag)

    return
