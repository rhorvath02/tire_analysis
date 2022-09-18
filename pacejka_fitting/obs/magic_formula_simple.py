import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

def fit(data, B, D, E):
    C = 1.3
    FZ = data[0]
    SA = data[1]

    return FZ * D * np.sin(C * np.arctan(B * SA - E * (B * SA - np.arctan(B * SA))))

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
# z_lst = df["IA"].tolist()

w_lst = df["FY"].tolist()

# print(x_lst, y_lst, z_lst)

a_vals = [8, 1, -4.5]

parameters, covariance = curve_fit(fit, [x_lst, y_lst], w_lst, a_vals)

predicted = fit([x_lst[100], y_lst[100]], *parameters)

print("FZ =", x_lst[100], "SA =", y_lst[100], "FY =", w_lst[100])

print(parameters)
print(predicted)
print(w_lst[100])

model_x_data = np.linspace(min(x_lst), max(x_lst), 30)
model_y_data = np.linspace(min(y_lst), max(y_lst), 30)
# create coordinate arrays for vectorized evaluations
X, Y = np.meshgrid(model_x_data, model_y_data)
# calculate Z coordinate array
Z = fit(np.array([X, Y]), *parameters)

# setup figure object
fig = plt.figure()
# setup 3d object
ax = Axes3D(fig)
# plot surface
ax.plot_surface(X, Y, Z)
# plot input data
# ax.scatter(x_lst, y_lst, w_lst, color='red')
# set plot descriptions
ax.set_xlabel('Normal Load (N)')
ax.set_ylabel('Slip Angle (deg)')
ax.set_zlabel('Lateral Force (N)')

plt.show()
