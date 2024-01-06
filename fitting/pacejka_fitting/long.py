import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import basinhopping
from mpl_toolkits.mplot3d import Axes3D

def lat_fit(data, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17):
    FZ = data[0]
    SA = data[1]
    IA = data[2]

    C = a0
    D = FZ * (a1 * FZ + a2) * (1 - a15 * IA**2)
    
    BCD = a3 * np.sin(np.arctan(FZ / a4) * 2) * (1 - a5 * abs(IA))
    B = BCD / (C * D)
    H = a8 * FZ + a9 + a10 * IA

    E = (a6 * FZ + a7) * (1 - (a16 * IA + a17) * np.sign(SA + H))

    V = a11 * FZ + a12 + (a13 * FZ + a14) * IA * FZ
    Bx1 = B * (SA + H)

    return D * np.sin(C * np.arctan(Bx1 - E * (Bx1 - np.arctan(Bx1)))) + V

def long_fit(data, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13):
    FZ = data[0] / 1000
    SR = data[1] * 100

    # if FZ <= 0:
    #     return 0
    # else:
    C = b0
    D = FZ * (b1 * FZ + b2)
    
    BCD = (b3 * FZ**2 + b4 * FZ) * np.exp(-1 * b5 * FZ)
    B = BCD / (C * D)
    H = b9 * FZ + b10

    E = (b6 * FZ**2 + b7 * FZ + b8) * (1 - b13 * np.sign(SR + H))

    V = b11 * FZ + b12
    Bx1 = B * (SR + H)

    return (D * np.sin(C * np.arctan(Bx1 - E * (Bx1 - np.arctan(Bx1)))) + V)


tires = {"Hoosier_16x7.5-10_R20_7_braking":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1

for name, tire in tires.items():
    try:
        df = pd.read_csv(f"././processing/results/{name}.csv")
        # tire["long"] = df[(df["pressure"] == pressure) & (df["velocity"] == velocity) & (df["slip_a"] == slip_angle)]
        tire["long"] = df
        
    except:
        print("Error getting long data for {0}".format(name))

    try:
        df = pd.read_csv(f"././processing/results/{name}.csv")
        tire["lat"] = df[(df["velocity"] == velocity) & (df["pressure"] == pressure)]

    except:
        print("Error getting lateral data for {0}".format(name))

optimal1 = [1.6273891721090612, -698.8419595678488, 3484.805352225537, 0.05134138933309365, 767.1901442937605, 0.3582710405478557, -1.6410632545280734, 3.169678791610308, -1.1006443264258339, -0.18232465799105665, 0.255295041808798, 20.326956445391467, -42.92101885459951, 0.07200020453003882]

optimal2 = [1.6273891721090612, -698.8419595678488, 3484.805352225537, 0.05134138933309365, 767.1901442937605, 0.3582710405478557, -1.6410632545280734, 3.169678791610308, -1.1006443264258339, 0, 0, 0, 0, 0]

df = tires["Hoosier_16x7.5-10_R20_7_braking"]["long"]

x_lst = list(abs(df["load"]))
y_lst = list(df["SL"])

z_lst = list(df["FX"])

# data = df["load"].unique()

# for normal_load in data:
#     test_df = df[(df["load"] == normal_load)]
#     print(f"Load: {normal_load}")
#     print(f"Max FX: {max(test_df['FX'])}")
#     print(f"Min FX: {min(test_df['FX'])}")
#     print()

bounds = [[-1112, [-3130, 3027]], [-890, [-2657, 2540]], [-667, [-2069, 1961]], [-445, [-1199, 1175]], [-222, [-889, 771]]]

# Hacky solution to force desired behavior
# for bound in bounds:
#     for i in range(25):
#         x_lst.append(abs(bound[0]))
#         y_lst.append(1)
#         z_lst.append(bound[1][1] * 0.70)
        
#         x_lst.append(abs(bound[0]))
#         y_lst.append(-1)
#         z_lst.append(bound[1][0] * 0.70)

# for bound in bounds:
#     for i in range(25):
#         x_lst.append(abs(bound[0]))
#         y_lst.append(0.50)
#         z_lst.append(bound[1][1] * 0.85)
        
#         x_lst.append(abs(bound[0]))
#         y_lst.append(-0.50)
#         z_lst.append(bound[1][0] * 0.85)

params1 = optimal1
params2 = optimal2

model_x_data = np.linspace(min(x_lst), max(x_lst), 1000)
model_y_data = np.linspace(-1, 1, 1000)

X, Y = np.meshgrid(model_x_data, model_y_data)

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)

Z1 = long_fit([X, Y], *params1)
Z2 = long_fit([X, Y], *params2)

ax = plt.axes(projection='3d')

ax.scatter3D(x_lst, y_lst, z_lst, cmap='Greens')

fig.add_axes(ax)
ax.plot_surface(X, Y, Z1)
ax.plot_surface(X, Y, Z2)

ax.set_xlabel('Normal Load (N)')
ax.set_ylabel('Slip Ratio')
ax.set_zlabel('Longitudinal Force (N)')

plt.show()