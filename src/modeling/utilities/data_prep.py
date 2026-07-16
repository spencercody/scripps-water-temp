import pandas as pd
import numpy as np
from typing import Callable

def f_to_c(temp_f: float)->float:
    """
    Convert the temperature from farenheit to celcius
    """
    temp_c = (temp_f - 32) * (5/9)

    return temp_c

# ---------------------------------------------------------------------------- #

def convert_to_cyclical(trig_func: Callable, x: float, max_x: float):
    """
    this function is to convert cyclical data to radial data
    """
    radian_value = trig_func(2*np.pi* (x/max_x))

    return radian_value