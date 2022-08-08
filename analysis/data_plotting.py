from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def fitting(lst1, lst2):

    model1 = np.poly1d(np.polyfit(lst1, lst2, 4))

    polyline = np.linspace(lst1[0], lst1[-1])

    plt.plot(polyline, model1(polyline), color='green')

    plt.show()

SR = [-0.0399, -0.0405, -0.0403, -0.0399, -0.0404, -0.0389, -0.0414, -0.0392, -0.0405, -0.0416]
FX = [-414.61, -415.92, -475.45, -506.56, -482.78, -558.66, -541.61, -418.85, -462.45, -451.87]

fitting(SR, FX)