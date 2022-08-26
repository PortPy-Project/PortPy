import visualization
import numpy as np
from scipy import interpolate


def get_volume(dose, my_plan, organ, dose_value, weight_flag=True):
    x, y = visualization.get_dvh(dose, my_plan, organ, weight_flag=weight_flag)
    x1, indices = np.unique(x, return_index=True)
    y1 = y[indices]
    f = interpolate.interp1d(x1, 100*y1)

    return f(dose_value)
