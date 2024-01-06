import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

scaling_coeffs = [1 for x in range(28)]

coeffs = [1.3621019747547105, -2.435216396304886, 0.65, 9.423678477311629, 0.14799132747634844, -0.062214373627922905, 0.8075703695312021, 42.81227067232975, 22.19707765833606, 0.904357602882356, 0.6108200318681702, -0.0026565847602868942, -0.0020774251539237916, -0.07082967061402329, 0.009860302616368338, -0.030866236429790993, -1.8590779944132787, -1.519168529358441]

pure_lat_coeffs = coeffs[0:18]

# combined_lat_coeffs = [7, 2.5, 0, 1, 0, 0, 0.02, 0, 0, 0, -0.2, 14, 1.9, 10]
combined_lat_coeffs = [7, -0.17964306958425666, 0.2691732017495078, 1.0003194244762001, 0.3182331326059698, 3.129613765556598, 0.00466860938556589, 0.005539973668806044, 0.053304401957131244, -0.07571180014922728, -3.549342272076268, 92.80755348460904, 1.7236478746185968, 0.5023986661210857]

def _pure_lat(data: list[float]) -> float:
    FZ_nom = data[0]
    FZ = data[1]
    SA = data[2]
    IA = data[3]

    [CFY1, CFY2, CFY3, CFY4, CFY5, CFY6, CFY7, CFY8, CFY9, CFY10, \
        CFY11, CFY12, CFY13, CFY14, CFY15, CFY16, CFY17, CFY18] = pure_lat_coeffs
    
    IA_y = IA * scaling_coeffs[14]
    df_z = (FZ - FZ_nom * scaling_coeffs[0]) / (FZ_nom * scaling_coeffs[0])
    mu_y = (CFY2 + CFY3 * df_z) * (1 - CFY4 * IA_y**2) * scaling_coeffs[9]

    C_y = CFY1 * scaling_coeffs[8]
    D_y = mu_y * FZ
    K_y = CFY9 * FZ_nom * np.sin(2 * np.arctan(FZ / (CFY10 * FZ_nom * scaling_coeffs[0]))) * \
        (1 - CFY11 * abs(IA_y)) * scaling_coeffs[0] * scaling_coeffs[11]
    B_y = K_y / (C_y * D_y)

    S_Hy = (CFY12 + CFY13 * df_z) * scaling_coeffs[12] + CFY14 * IA_y
    S_Vy = FZ * ((CFY15 + CFY16 * df_z) * scaling_coeffs[13] + (CFY17 + CFY18 * df_z) * IA_y) * scaling_coeffs[9]
    SA_y = SA + S_Hy

    E_y = (CFY5 + CFY6 * df_z) * (1 - (CFY7 + CFY8 * IA_y) * np.sign(SA_y)) * scaling_coeffs[10]

    F_Y0 = D_y * np.sin(C_y * np.arctan(B_y * SA_y - E_y * (B_y * SA_y - np.arctan(B_y * SA_y)))) + S_Vy
    F_Y = F_Y0

    return F_Y

def _combined_lat(data: list[float]) -> float:
    FZ_nom = data[0]
    FZ = data[1]
    SA = data[2]
    SR = data[3]
    IA = data[4]

    [CCFY1, CCFY2, CCFY3, CCFY4, CCFY5, CCFY6, CCFY7, CCFY8, CCFY9, CCFY10, CCFY11, CCFY12, CCFY13, CCFY14] = combined_lat_coeffs
    [CFY1, CFY2, CFY3, CFY4, CFY5, CFY6, CFY7, CFY8, CFY9, CFY10, CFY11, CFY12, CFY13, CFY14, CFY15, CFY16, CFY17, CFY18] = pure_lat_coeffs
    
    df_z = (FZ - FZ_nom * scaling_coeffs[0]) / (FZ_nom * scaling_coeffs[0])
    IA_y = IA * scaling_coeffs[14]
    mu_y = (CFY2 + CFY3 * df_z) * (1 - CFY4 * IA_y**2) * scaling_coeffs[9]

    C_ySR = CCFY4
    B_ySR = CCFY1 * np.cos(np.arctan(CCFY2 * (SA - CCFY3))) * scaling_coeffs[22]
    E_ySR = CCFY5 + CCFY6 * df_z
    S_HySR = CCFY7 + CCFY8 * df_z

    D_VySR = mu_y * FZ * (CCFY9 + CCFY10 * df_z + CCFY11 * IA) * np.cos(np.arctan(CCFY12 * SA))

    S_VySR = D_VySR * np.sin(CCFY13 * np.arctan(CCFY14 * SR)) * scaling_coeffs[23]

    SR_s = SR + S_HySR
    
    # The Delft paper defines FY_adj this way, but we can decompose this into force output from the pure fit * some scaling coefficient
    # D_ySR = pure_lat_coeffs([FZ_nom, FZ, SA, IA]) / (np.cos(C_ySR * np.arctan(B_ySR * S_HySR - E_ySR * (B_ySR * S_HySR - np.arctan(B_ySR * S_HySR)))))
    # FY_adj = D_ySR * np.cos(C_ySR * np.arctan(B_ySR * SR_s - E_ySR * (B_ySR * SR_s - np.arctan(B_ySR * SR_s)))) + S_VySR

    # This is the same calculation, but the variables are shown a more intuitive way
    G_ySR = (np.cos(C_ySR * np.arctan(B_ySR * SR_s - E_ySR * (B_ySR * SR_s - np.arctan(B_ySR * SR_s))))) / (np.cos(C_ySR * np.arctan(B_ySR * S_HySR - E_ySR * (B_ySR * S_HySR - np.arctan(B_ySR * S_HySR)))))
    FY_0 = _pure_lat([FZ_nom, FZ, SA, IA])

    FY_adj = FY_0 * G_ySR + S_VySR

    return FY_adj

tires = {"Hoosier_18x6.0-10_R20_7_braking":{"long":None, "lat":None}}
        #  "Hoosier_16x7.5-10_R20_7_cornering":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1

for name, tire in tires.items():
    try:
        df = pd.read_csv(f"././processing/results/{name}.csv")
        tire["lat"] = df[(df["velocity"] == velocity) & (df["pressure"] == pressure) & (df["load"] == -1112.0551223483046) & (df["camber"] == 0)]

    except:
        print("Error getting lateral data for {0}".format(name))

df1 = tires[list(tires.keys())[0]]["lat"]
x1_lst = (df1["FZ"] * -1).tolist()[::50]
y1_lst = (df1["SA"] * np.pi / 180).tolist()[::50]
y2_lst = (df1["SL"]).tolist()[::50]
z1_lst = (df1["IA"] * np.pi / 180).tolist()[::50]

w1_lst = (df1["FY"] * -1).tolist()[::50]

bounds1 = []
loads1 = list(df1["load"].unique())

print(loads1)

# for load in loads1:
#     FY_slice = df1[(df1["load"] == load)]

#     max_FY = max(FY_slice["FY"])
#     min_FY = min(FY_slice["FY"])

#     bounds1.append([load, [min_FY, max_FY]])

# for bound in bounds1:
#     for i in range(1000):
#         x1_lst.append(abs(bound[0]))
#         y1_lst.append(np.pi / 2)
#         z1_lst.append(0)
#         w1_lst.append(bound[1][0] * -1 * 0.70)
        
#         x1_lst.append(abs(bound[0]))
#         y1_lst.append(-np.pi / 2)
#         z1_lst.append(0)
#         w1_lst.append(bound[1][1] * -1 * 0.70)

# for bound in bounds1:
#     for i in range(1000):
#         x1_lst.append(abs(bound[0]))
#         y1_lst.append(np.pi / 4)
#         z1_lst.append(0)
#         w1_lst.append(bound[1][0] * -1 * 0.85)
        
#         x1_lst.append(abs(bound[0]))
#         y1_lst.append(-np.pi / 4)
#         z1_lst.append(0)
#         w1_lst.append(bound[1][1] * -1 * 0.85)

model_x_data = np.linspace(3000, abs(min(x1_lst)), 1000)
model_y_data = np.linspace(-1, 1, 1000)
# model_y_data = np.linspace(-0.20, 0.20, 1000)
model_z_data = np.linspace(-90 * np.pi / 180, 90 * np.pi / 180, 1000)
# model_z_data = np.linspace(-6 * np.pi / 180, 0 * np.pi / 180, 1000)

X, Y = np.meshgrid(model_x_data, model_y_data)
X2, Y2 = np.meshgrid(model_z_data, model_y_data)

# Z1 = _pure_lat([250 * 4.44822, X, Y, 0])
# Z2 = _combined_lat([350 * 4.44822, 250 * 4.44822, X2, Y2, 0])

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)

ax = plt.axes(projection='3d')

for normal_load in [250 * 4.44822]:
    Z2 = _combined_lat([350 * 4.44822, normal_load, X2, Y2, 0])
    ax.plot_surface(X2 * 180 / np.pi, Y2, Z2)
    # plt.legend()

# ax.plot_surface(X2, Y2, Z2)
ax.scatter3D(np.array(y1_lst) * 180 / np.pi, y2_lst, w1_lst, cmap='Greens')

fig.add_axes(ax)

ax.set_xlabel('Slip Angle (deg)')
ax.set_ylabel('Slip Ratio')
ax.set_zlabel('Lateral Force (N)')

plt.show()