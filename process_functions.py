import numpy as np
import pandas as pd

def nearest_value(lst, val):
    min_diff = abs(lst[0] - val)
    nearest = lst[0]

    for item in lst[1:]:
        if abs(item - val) < min_diff:
            min_diff = abs(item - val)
            nearest = item
    
    return nearest


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def import_data(data, run_data = False):
    required_length = len(data["FX"])
    for item in list(data.keys()):
        if not len(data[item]) == required_length:
            del data[item]
        
        else:
            data[item] = data[item].transpose()[0] if run_data else data[item][0]
    
    return pd.DataFrame.from_dict(data)