import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

scaling_coeffs = [1 for x in range(28)]

# save_this = [1.5920766052828437, 1.3388449042445494, -2.088995774772945, 36.95776952551025, -41.78837118835335, -114.68959776031716, -77.02883197949176, 0.017369999849801185, 5.940862173346288, 3.1325394082364886, -3.483652816561306, 0.0023514393989217015, 0.0050753961796519455, 0.16218678053209926, 0.16715878473618767]\

# fit from given data alone: [1.1909414020454832, 2.3308909810824296, -0.5042181746446683, 17.152126857248817, -2.107196442986876, -8.502247630670709, -5.563398957903864, 0.02445769003456203, 39.438291772391764, 33.92712849221479, -2.04670700924917, 0.0028133647433124026, 0.003973376948261052, -0.09076816870771447, -0.04893326654694003]

pure_long_coeffs = [1.1756372529166366, 2.275693693135797, -0.6501009126133899, 17.15239682104082, -2.3290570234385215, -8.321786521216294, -5.708815286024726, 0.17545348128614113, 39.4346378819371, 33.92968608125336, -2.010829751105287, -0.0021163086632554535, -0.002049018916192311, -0.09099235392041341, -0.04812593772160824]

pure_long_coeffs = pure_long_coeffs[0:15]

# pure_long_coeffs += [0 for x in range(6)]

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

tires = {"Hoosier_18x6.0-10_R20_7_braking":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1

for name, tire in tires.items():
    try:
        df = pd.read_csv(f"././processing/results/{name}.csv")
        tire["long"] = df[(df["pressure"] == pressure) & (df["velocity"] == velocity) & (df["camber"] == 0) & (df["slip"] == 0) & (df["camber"] == 0) & (df["load"] < -190)]
        # print(tire["long"])
        
    except:
        print("Error getting long data for {0}".format(name))

df = tires[list(tires.keys())[0]]["long"]
length = int(len(df) / 2)
x_lst = (df["load"] * -1).tolist()[length:]
y_lst = (df["SL"]).tolist()[length:]
z_lst = (df["camber"] * np.pi / 180).tolist()[length:]

w_lst = (df["FX"]).tolist()[length:]


bounds1 = []
loads1 = list(df["load"].unique())

# print(loads1)

# plt.scatter(df[df["load"] == df["load"].unique()[0]]["SL"], df[df["load"] == df["load"].unique()[0]]["FX"], s=0.5)

# plt.show()

for load in loads1:
    FX_slice = df[(df["load"] == load)]

    max_FX = max(FX_slice["FX"])
    min_FX = min(FX_slice["FX"])

    if ((max_FX - min_FX) / 2 * 1.5 < (max_FX)) or ((max_FX - min_FX) / 2 * 1.5 < (abs(min_FX))):
        continue

    bounds1.append([load, [min_FX, max_FX]])

for bound in bounds1:
    for i in range(1000):
        x_lst.append(abs(bound[0]))
        y_lst.append(1)
        z_lst.append(0)
        w_lst.append(bound[1][0] * -1 * 0.50)
        
        x_lst.append(abs(bound[0]))
        y_lst.append(-1)
        z_lst.append(0)
        w_lst.append(bound[1][1] * -1 * 0.50)

for bound in bounds1:
    for i in range(1000):
        x_lst.append(abs(bound[0]))
        y_lst.append(0.5)
        z_lst.append(0)
        w_lst.append(bound[1][0] * -1 * 0.70)
        
        x_lst.append(abs(bound[0]))
        y_lst.append(-0.5)
        z_lst.append(0)
        w_lst.append(bound[1][1] * -1 * 0.70)

model_x_data = np.linspace(3000, abs(min(x_lst)), 1000)
model_y_data = np.linspace(-1, 1, 1000)

X, Y = np.meshgrid(model_x_data, model_y_data)

Z = _pure_long([350 * 4.44822, X, Y, 0])

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