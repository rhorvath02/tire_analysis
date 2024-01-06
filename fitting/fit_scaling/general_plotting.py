import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

tires = {"Hoosier_20.5x7.0-13_R20_7_cornering":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1

for name, tire in tires.items():
    try:
        df = pd.read_csv(f"././processing/results/{name}.csv")
        tire["long"] = df[(df["pressure"] == pressure) & (df["velocity"] == velocity) & (df["camber"] == 0)]
        # print(tire["long"])
        
    except:
        print("Error getting long data for {0}".format(name))

df = tires[list(tires.keys())[0]]["long"]

x_lst = (df["FZ"]).tolist()
y_lst = (df["SA"]).tolist()
z_lst = (df["MX"]).tolist()

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)

ax = plt.axes(projection='3d')

ax.scatter3D(x_lst, y_lst, z_lst, cmap='Greens')

fig.add_axes(ax)

ax.set_xlabel('SA')
ax.set_ylabel('SL')
ax.set_zlabel('MZ')

plt.show()