import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import basinhopping
from mpl_toolkits.mplot3d import Axes3D

scaling_coeffs = [1 for x in range(28)]

# pure_lat_coeffs = [1.4, -3, 0, 0, 0, 0, 0, 0, 30, 3, 0, 0, 0, 0, 0, 0, 0, 0]

# 16 inch: [1.462644363336092, -2.4525488894188525, 0.07875259157828247, 11.665598056159695, -0.0007085356603566561, -0.0003936037282383536, 78.65359240952695, -7842.704918479082, 56.50858901077719, 2.42841666180457, 0.6581244608544692, 0.00016212560860950413, -0.0010386186529656692, -0.07982161655436235, -0.0037501542948406765, 0.02152965494317117, -0.7891081372943167, -1.1113888934215281]

pure_lat_coeffs = [1.462644363336092, -2.4525488894188525, 0.07875259157828247, 11.665598056159695, -0.0007085356603566561, -0.0003936037282383536, 78.65359240952695, -7842.704918479082, 56.50858901077719, 2.42841666180457, 0.6581244608544692, 0.00016212560860950413, -0.0010386186529656692, -0.07982161655436235, -0.0037501542948406765, 0.02152965494317117, -0.7891081372943167, -1.1113888934215281]

def _pure_lat(data: list[float], coeffs) -> float:
    FZ_nom = data[0]
    FZ = data[1]
    SA = data[2]
    IA = data[3]

    [CFY1, CFY2, CFY3, CFY4, CFY5, CFY6, CFY7, CFY8, CFY9, CFY10, \
        CFY11, CFY12, CFY13, CFY14, CFY15, CFY16, CFY17, CFY18] = coeffs
    
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

tires = {"Hoosier_20.5x7.0-13_R20_cornering":{"long":None, "lat":None}}
        #  "Hoosier_16x7.5-10_R20_7_cornering":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1

for name, tire in tires.items():
    try:
        df = pd.read_csv(f"././processing/results/{name}.csv")
        tire["lat"] = df[(df["velocity"] == velocity) & (df["pressure"] == pressure)]

    except:
        print("Error getting lateral data for {0}".format(name))

df1 = tires[list(tires.keys())[0]]["lat"]
x1_lst = (df1["FZ"] * -1).tolist()
y1_lst = (df1["SA"] * np.pi / 180).tolist()
z1_lst = (df1["IA"] * np.pi / 180).tolist()

w1_lst = (df1["FY"] * -1).tolist()

bounds1 = []
loads1 = list(df1["load"].unique())

for load in loads1:
    FY_slice = df1[(df1["load"] == load)]

    max_FY = max(FY_slice["FY"])
    min_FY = min(FY_slice["FY"])

    bounds1.append([load, [min_FY, max_FY]])

for bound in bounds1:
    for i in range(1000):
        x1_lst.append(abs(bound[0]))
        y1_lst.append(np.pi / 2)
        z1_lst.append(0)
        w1_lst.append(bound[1][0] * -1 * 0.70)
        
        x1_lst.append(abs(bound[0]))
        y1_lst.append(-np.pi / 2)
        z1_lst.append(0)
        w1_lst.append(bound[1][1] * -1 * 0.70)

for bound in bounds1:
    for i in range(1000):
        x1_lst.append(abs(bound[0]))
        y1_lst.append(np.pi / 4)
        z1_lst.append(0)
        w1_lst.append(bound[1][0] * -1 * 0.85)
        
        x1_lst.append(abs(bound[0]))
        y1_lst.append(-np.pi / 4)
        z1_lst.append(0)
        w1_lst.append(bound[1][1] * -1 * 0.85)

model_x_data = np.linspace(abs(max(x1_lst)), abs(min(x1_lst)), 1000)
model_y_data = np.linspace(-90 * np.pi / 180, 90 * np.pi / 180, 1000)

X, Y = np.meshgrid(model_x_data, model_y_data)

Z1 = _pure_lat([1000, X, Y, 0], pure_lat_coeffs)

# scaling_coeffs[8:15] = [0.8017659201734039, 0.8934978936290433, -0.09790039056025218, 1.1994136955660188, -0.951818681073718, -0.16885699108151495, 1.4242086020133868] 

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)

ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z1)
ax.scatter3D(x1_lst, y1_lst, w1_lst, cmap='Greens')

fig.add_axes(ax)

ax.set_xlabel('Normal Load (N)')
ax.set_ylabel('Slip Angle')
ax.set_zlabel('Lateral Force (N)')

plt.show()