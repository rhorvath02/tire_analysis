import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import basinhopping
from mpl_toolkits.mplot3d import Axes3D

scaling_coeffs = [1 for x in range(28)]

pure_long_coeffs = [1.4, -3, 0, 0, 0, 0, 0, 0, 30, 3, 0, 0, 0, 0, 0]

# current_solution = [1.7522465581415032, -3.0543141420289697, 0.3615401388276578, 0.0, 0.6893889814926916, -0.005537730182135393, 0.012739965935907486, 0.00627354178712224, 64.84540366507208, 0.0007919860986186593, -0.28559075010725293, 0.001349213974115627, -0.00043721353267875233, -0.05807363023474984, 0.037410219013107086]

def _pure_long(data: list[float]) -> float:
    [CFX1, CFX2, CFX3, CFX4, CFX5, CFX6, CFX7, CFX8, CFX9, CFX10, CFX11, CFX12, CFX13, CFX14, CFX15] = pure_long_coeffs
    FZ_nom = data[0]
    FZ = data[1]
    SR = data[2]
    IA = data[3]

    IA_x = IA * scaling_coeffs[7]
    df_z = (FZ - FZ_nom * scaling_coeffs[0]) / (FZ_nom * scaling_coeffs[0])
    mu_x = (CFX2 + CFX3 * df_z) * (1 - CFX4 * IA_x**2) * scaling_coeffs[2]

    C_x = CFX1 * scaling_coeffs[1]
    D_x = mu_x * FZ
    K_x = FZ * (CFX9 + CFX10 * df_z) * np.exp(CFX11 * df_z) * scaling_coeffs[4]
    B_x = K_x / (C_x * D_x)

    S_Hx = (CFX12 + CFX13 * df_z) * scaling_coeffs[5]
    S_Vx = FZ * (CFX14 + CFX15 * df_z) * scaling_coeffs[6] * scaling_coeffs[2]
    SR_x = SR + S_Hx

    E_x = (CFX5 + CFX6 * df_z + CFX7 * df_z**2) * (1 - CFX8 * np.sign(SR_x)) * scaling_coeffs[3]

    F_X0 = D_x * np.sin(C_x * np.arctan(B_x * SR_x - E_x * (B_x * SR_x - np.arctan(B_x * SR_x)))) + S_Vx
    F_X = F_X0
    
    return F_X

tires = {"Hoosier_18x7.5-10_R25B_7_braking":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1

for name, tire in tires.items():
    try:
        df = pd.read_csv(f"././processing/results/{name}.csv")
        # tire["long"] = df[(df["pressure"] == pressure) & (df["velocity"] == velocity) & (df["camber"] == camber) & (df["SA"] < 1) & (df["SA"] > -1)]
        tire["long"] = df[(df["pressure"] == pressure) & (df["velocity"] == velocity) & (df["camber"] == camber) & (df["FZ"] < -1000)]
        # print(tire["long"])
        
    except:
        print("Error getting long data for {0}".format(name))

df = tires[list(tires.keys())[0]]["long"]
x_lst = (df["load"] * -1).tolist()
y_lst = (df["SL"]).tolist()
y2_lst = (df["SA"]).tolist()
z_lst = (df["camber"] * np.pi / 180).tolist()

w_lst = (df["FX"]).tolist()
w2_lst = (df["FY"]).tolist()

# plt.plot([x for x in range(len(x_lst))], x_lst)

fig = plt.figure()

ax = Axes3D(fig, auto_add_to_figure=False)

ax = plt.axes(projection='3d')

# ax.scatter3D(x_lst, y_lst, w_lst, cmap='Greens')

ax.scatter3D(y_lst, y2_lst, w2_lst, cmap='Greens')

plt.show()

bounds = [-1112, [-3130, 3027]], [-890, [-2657, 2540]], [-667, [-2069, 1961]], [-445, [-1199, 1175]], [-222, [-889, 771]]

# Hacky solution to force desired behavior
# for bound in bounds:
#     for i in range(25):
#         x_lst.append(abs(bound[0]))
#         y_lst.append(1)
#         z_lst.append(0)
#         w_lst.append(bound[1][1] * 0.70)
        
#         x_lst.append(abs(bound[0]))
#         y_lst.append(-1)
#         z_lst.append(0)
#         w_lst.append(bound[1][0] * 0.70)

# for bound in bounds:
#     for i in range(25):
#         x_lst.append(abs(bound[0]))
#         y_lst.append(0.50)
#         z_lst.append(0)
#         w_lst.append(bound[1][1] * 0.85)
        
#         x_lst.append(abs(bound[0]))
#         y_lst.append(-0.50)
#         z_lst.append(0)
#         w_lst.append(bound[1][0] * 0.85)

model_x_data = np.linspace(abs(max(x_lst)), abs(min(x_lst)), 1000)
model_y_data = np.linspace(-1, 1, 1000)

X, Y = np.meshgrid(model_x_data, model_y_data)

Z = _pure_long([1000, X, Y, 0])

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)

ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z)
ax.scatter3D(x_lst, y_lst, w_lst, cmap='Greens')

fig.add_axes(ax)

ax.set_xlabel('Normal Load (N)')
ax.set_ylabel('Slip Ratio')
ax.set_zlabel('Longitudinal Force (N)')

plt.show()