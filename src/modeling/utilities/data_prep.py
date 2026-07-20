import pandas as pd
import numpy as np
from typing import Callable
from datetime import timedelta

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


# ---------------------------------------------------------------------------- #

def create_split_dfs(df: pd.DataFrame)->list[pd.DataFrame]:
    """
    The purpose of this function is to find lapses in the dates and
    slice the dataframes into individual dfs.

    date column is required
    """
    slices_index = []

    for i, day in enumerate(df.date.diff()):
        
        if day != timedelta(days=1):
            slices_index.append(i)

    # ---------------------------------------------------------------------------- #
    slice_dfs = []

    for i, idx in enumerate(slices_index):
        final_idx = slices_index[-1]
        if idx != final_idx:
            end_idx = slices_index[i+1]
            slice_df = df[idx:end_idx]
        else:
            slice_df = df[idx:]
        slice_dfs.append(slice_df)

    return slice_dfs

# ---------------------------------------------------------------------------- #

def setup_sequential_data(x_df, y_df, seq_size):
    X, y = [], []
    for i in range(seq_size, len(x_df) + 1):
        seq_x = x_df.iloc[(i-seq_size): i]
        
        X.append(np.array(seq_x))
        y.append(y_df.target.iloc[i-1])
    
    return np.array(X), np.array(y)