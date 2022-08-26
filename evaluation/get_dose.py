import visualization
from scipy import interpolate


def get_dose(dose, my_plan, organ, volume_per, weight_flag=True):
    x, y = visualization.get_dvh(dose, my_plan, organ, weight_flag=weight_flag)
    f = interpolate.interp1d(100*y, x)

    return f(volume_per)