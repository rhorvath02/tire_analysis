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