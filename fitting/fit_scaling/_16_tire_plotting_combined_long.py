import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

scaling_coeffs = [1 for x in range(28)]

# coeffs = [1.6198223342770743, 1.024674814374011, -2.477237671962809, 36.998983223111196, -41.272526689753484, -114.4561213331932, -77.65005250329699, -0.023434330378624434, 5.4706410020754666, 2.692862788822841, -3.2047888753217126, -0.027402729728334622, -0.03396210143074468, 0.18609729679969367, 0.22180474455698637, 5.000160398317436, 7.999959195181089, 1.0008810780626822, -3.706554347530667e-05, 2.6697133269336703e-05, -0.008333579867423765]

coeffs = [1.1756372529166366, 2.275693693135797, -0.6501009126133899, 17.15239682104082, -2.3290570234385215, -8.321786521216294, -5.708815286024726, 0.17545348128614113, 39.4346378819371, 33.92968608125336, -2.010829751105287, -0.0021163086632554535, -0.002049018916192311, -0.09099235392041341, -0.04812593772160824]

pure_long_coeffs = coeffs[0:15]

combined_long_coeffs = [6.701679824892427, 8.017951012667416, 0.99, -7.620470441623948, 0.0, 0.0]

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

def _combined_long(data: list[float]) -> float:
    FZ_nom = data[0]
    FZ = data[1]
    SA = data[2]
    SR = data[3]
    IA = data[4]

    [CCFX1, CCFX2, CCFX3, CCFX4, CCFX5, CCFX6] = combined_long_coeffs

    df_z = (FZ - FZ_nom * scaling_coeffs[0]) / (FZ_nom * scaling_coeffs[0])
    
    C_xSA = CCFX3
    B_xSA = CCFX1 * np.cos(np.arctan(CCFX2 * SR)) * scaling_coeffs[21]
    E_xSA = CCFX4 + CCFX5 * df_z
    S_HxSA = CCFX6

    SA_s = SA + S_HxSA

    G_xSA = (np.cos(C_xSA * np.arctan(B_xSA * SA_s - E_xSA * (B_xSA * SA_s - np.arctan(B_xSA * SA_s))))) / (np.cos(C_xSA * np.arctan(B_xSA * S_HxSA - E_xSA * (B_xSA * S_HxSA - np.arctan(B_xSA * S_HxSA)))))
    FX_0 = _pure_long([FZ_nom, FZ, SR, IA])

    FX_adj = FX_0 * G_xSA
    
    return FX_adj

tires = {"Hoosier_18x6.0-10_R20_7_braking":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1

for name, tire in tires.items():
    try:
        df = pd.read_csv(f"././processing/results/{name}.csv")
        # tire["long"] = df[(df["pressure"] == pressure) & (df["velocity"] == velocity) & (df["slip"] == 0) & (df["camber"] == 0)]
        tire["long"] = df[(df["pressure"] == pressure) & (df["velocity"] == velocity) & (df["load"] == -1112.0551223483046) & (df["camber"] == 0)]
        # print(tire["long"])
        
    except:
        print("Error getting long data for {0}".format(name))

df = tires[list(tires.keys())[0]]["long"]
x_lst = (df["FZ"] * -1).tolist()
y_lst = (df["SL"]).tolist()
y2_lst = (df["SA"]).tolist()
z_lst = (df["camber"] * np.pi / 180).tolist()

w_lst = (df["FX"]).tolist()


bounds1 = []
loads1 = list(df["load"].unique())

# plt.scatter(df[df["load"] == df["load"].unique()[0]]["SL"], df[df["load"] == df["load"].unique()[0]]["FX"], s=0.5)

# plt.show()

for load in loads1:
    FX_slice = df[(df["load"] == load)]

    max_FX = max(FX_slice["FX"])
    min_FX = min(FX_slice["FX"])

    if ((max_FX - min_FX) / 2 * 1.5 < (max_FX)) or ((max_FX - min_FX) / 2 * 1.5 < (abs(min_FX))):
        continue

    bounds1.append([load, [min_FX, max_FX]])


print(bounds1)

# for bound in bounds1:
#     for i in range(int(1000/1)):
#         x_lst.append(abs(bound[0]))
#         y_lst.append(1)
#         y2_lst.append(0)
#         z_lst.append(0)
#         w_lst.append(bound[1][0] * -1 * 0.80)

#         # x_lst.append(abs(bound[0]) * 0.75)
#         # y_lst.append(1)
#         # z_lst.append(0)
#         # w_lst.append(bound[1][0] * -1 * 0.70)
        
#         x_lst.append(abs(bound[0]))
#         y_lst.append(-1)
#         y2_lst.append(0)
#         z_lst.append(0)
#         w_lst.append(bound[1][1] * -1 * 0.80)
        
#         # x_lst.append(abs(bound[0]) * 0.75)
#         # y_lst.append(-1)
#         # z_lst.append(0)
#         # w_lst.append(bound[1][1] * -1 * 0.70)

# for bound in bounds1:
#     for i in range(int(1000/1)):
#         x_lst.append(abs(bound[0]))
#         y_lst.append(0.5)
#         y2_lst.append(0)
#         z_lst.append(0)
#         w_lst.append(bound[1][0] * -1 * 0.90)

#         # x_lst.append(abs(bound[0]) * 0.75)
#         # y_lst.append(0.5)
#         # z_lst.append(0)
#         # w_lst.append(bound[1][0] * -1 * 0.85)
        
#         x_lst.append(abs(bound[0]))
#         y_lst.append(-0.5)
#         y2_lst.append(0)
#         z_lst.append(0)
#         w_lst.append(bound[1][1] * -1 * 0.90)

#         # x_lst.append(abs(bound[0]) * 0.75)
#         # y_lst.append(-0.5)
#         # z_lst.append(0)
#         # w_lst.append(bound[1][1] * -1 * 0.85)

model_x_data = np.linspace(3000, abs(min(x_lst)), 1000)
model_y_data = np.linspace(-1, 1, 1000)
# model_y_data = np.linspace(-0.15, 0.15, 1000)
model_z_data = np.linspace(-np.pi / 2, np.pi / 2, 1000)
# model_z_data = np.linspace(-6 * np.pi / 180, 0 * np.pi / 180, 1000)

X, Y = np.meshgrid(model_x_data, model_y_data)
X2, Y2 = np.meshgrid(model_z_data, model_y_data)

Z = _pure_long([250 * 4.44822, X, Y, 0])
# Z2 = _combined_long([3000, 1112.0551223483046, X2, Y2, 0])

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)

ax = plt.axes(projection='3d')

# ax.plot_surface(X, Y, Z)
# ax.scatter3D(x_lst, y_lst, w_lst, cmap='Greens', s=0.1)

for normal_load in [250 * 4.44822]:
    Z2 = _combined_long([250 * 4.44822, normal_load, X2, Y2, 0])
    ax.plot_surface(X2 * 180 / np.pi, Y2, Z2)
    # plt.legend()

# ax.plot_surface(X2 * 180 / np.pi, Y2, Z2)
ax.scatter3D(y2_lst, y_lst, w_lst, cmap='Greens', s=0.1)

fig.add_axes(ax)

# ax.set_xlabel('Normal Load (N)')
# ax.set_ylabel('Slip Ratio')
# ax.set_zlabel('Longitudinal Force (N)')
ax.set_xlabel('Slip Angle (deg)')
ax.set_ylabel('Slip Ratio')
ax.set_zlabel('Longitudinal Force (N)')

plt.show()