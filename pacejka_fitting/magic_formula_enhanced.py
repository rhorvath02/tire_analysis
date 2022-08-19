import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D


def fit(data, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17):
    FZ = data[0]
    SA = data[1]
    IA = data[2]

    C = 1.4
    D = FZ * (a1 * FZ + a2) * (1 - a15 * IA**2)
    
    BCD = a3 * np.sin(np.arctan(FZ / a4) * 2) * (1 - a5 * abs(IA))
    B = BCD / (C * D)
    H = a8 * FZ + a9 + a10 * IA

    E = (a6 * FZ + a7) * (1 - (a16 * IA + a17) * np.sign(SA + H))

    V = a11 * FZ + a12 + (a13 * FZ + a14) * IA * FZ
    Bx1 = B * (SA + H)

    return D * np.sin(C * np.arctan(Bx1 - E * (Bx1 - np.arctan(Bx1)))) + V


tires = {"hoosier_r25b_18x7-5_10x8":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1


for name, tire in tires.items():
    try:
        df = pd.read_csv(f"./tire_data/processed_data/braking_{name}.csv")
        tire["long"] = df[(df["pressure"] == pressure) & (df["velocity"] == velocity)]
        # print(tire["long"])
        
    except:
        print("Error getting long data for {0}".format(name))

    try:
        df = pd.read_csv(f"./tire_data/processed_data/cornering_{name}.csv")
        tire["lat"] = df[(df["velocity"] == velocity) & (df["pressure"] == pressure)]
        # print(tire["lat"])

    except:
        print("Error getting lateral data for {0}".format(name))


df = tires["hoosier_r25b_18x7-5_10x8"]["lat"]

x_lst = df["FZ"].tolist()
y_lst = df["SA"].tolist()
z_lst = df["IA"].tolist()

w_lst = df["FY"].tolist()

# print(x_lst, y_lst, z_lst)

# a_vals = [8, 1, -4.5]

a_vals = [0, 1100, 1100, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

parameters, covariance = curve_fit(fit, [x_lst, y_lst, z_lst], w_lst, a_vals, maxfev = 10000)

for i in range(2000):
    predicted = fit([x_lst[i], y_lst[i], z_lst[i]], *parameters)

    # print("FZ =", x_lst[i], "SA =", y_lst[i], "FY =", w_lst[i])

    plt.scatter(y_lst[i], predicted)

    # print(parameters)
    # print(predicted)
    # print(w_lst[i])

plt.show


# model_x_data = np.linspace(min(x_lst), max(x_lst), 30)
# model_y_data = np.linspace(min(y_lst), max(y_lst), 30)
# model_z_data = np.linspace(min(z_lst), max(z_lst), 30)
# # create coordinate arrays for vectorized evaluations
# X, Y, Z = np.meshgrid(model_x_data, model_y_data, model_z_data)
# # calculate Z coordinate array
# X = X[0]
# Y = Y[0]
# Z = Z[0]
# W = fit(np.array([X, Y, Z]), *parameters)

# # setup figure object
# fig = plt.figure()
# # setup 3d object
# ax = Axes3D(fig)
# # plot surface
# ax.plot_surface(X, Y, W)
# # plot input data
# # ax.scatter(x_lst, y_lst, w_lst, color='red')
# # set plot descriptions
# ax.set_xlabel('Normal Load (N)')
# ax.set_ylabel('Slip Angle (deg)')
# ax.set_zlabel('Lateral Force (N)')

# plt.show()