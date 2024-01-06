import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import basinhopping
from mpl_toolkits.mplot3d import Axes3D

scaling_coeffs = [1 for x in range(28)]

pure_lat_coeffs = [1.462644363336092, -2.4525488894188525, 0.07875259157828247, 11.665598056159695, -0.0007085356603566561, -0.0003936037282383536, 78.65359240952695, -7842.704918479082, 56.50858901077719, 2.42841666180457, 0.6581244608544692, 0.00016212560860950413, -0.0010386186529656692, -0.07982161655436235, -0.0037501542948406765, 0.02152965494317117, -0.7891081372943167, -1.1113888934215281]

pure_aligning_coeffs = [pure_aligning_coeffs[x] if x < 9 else 0 for x in range(25)]

pure_aligning_coeffs = [-9.645786583079877, 5.650486243093343, 8.771555906212312, 115.61127816375134, -115.60898485597117, 61.08571530896941, 3.1449796678478394, 1.3270126298289837, 0.16384794241846673, -0.12769225216682936, -0.1420044172387813, -12.85054065245455, 0.012735955800908277, 0.002477164835327709, -3.422534867913855, 2.523749634198314]

pure_aligning_coeffs = pure_aligning_coeffs + [0 for x in range(9)]

# recent = [9.704000789172332, 6.785442241112642, -3.5673948457155817, 75.49357391040512, -59.95568477067915, -364.03090012814664, -20.06826031910382, 1.1665783368548153, 0.12362899913664448, -0.0059914505616676895, 72.83572337975993, 0.9113581583888881, 0.001962206438988134, -0.0035953155615178864, 1.7535489825803574, -5.337595386286589, -1.9185794596808825, 4.8650168464868395, -3.205219194626394, -0.30404559838107476, 22.17602705584904, 0.010899787095383567, -0.005920676417427969, -1.0640866160392524, -0.8552263058958373]

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

def _pure_aligning(data: list[float]) -> float:
    R_nom = data[0]
    FZ_nom = data[1]
    FZ = data[2]
    SA = data[3]
    IA = data[4]

    # Lateral Dependencies
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
    F_y0 = _pure_lat([FZ_nom, FZ, SA, IA])

    B_y = K_y / (C_y * D_y)
    S_Hy = (CFY12 + CFY13 * df_z) * scaling_coeffs[12] + CFY14 * IA_y
    S_Vy = FZ * ((CFY15 + CFY16 * df_z) * scaling_coeffs[13] + (CFY17 + CFY18 * df_z) * IA_y) * scaling_coeffs[9]

    # Pure Aligning Moment
    [CMZ1, CMZ2, CMZ3, CMZ4, CMZ5, CMZ6, CMZ7, CMZ8, CMZ9, CMZ10, CMZ11, CMZ12, \
        CMZ13, CMZ14, CMZ15, CMZ16, CMZ17, CMZ18, CMZ19, CMZ20, CMZ21, CMZ22, CMZ23, CMZ24, CMZ25] = pure_aligning_coeffs
    
    IA_z = IA * scaling_coeffs[17]
    df_z = (FZ - FZ_nom * scaling_coeffs[0]) / (FZ_nom * scaling_coeffs[0])

    S_Ht = CMZ22 + CMZ23 * df_z + (CMZ24 + CMZ25 * df_z) * IA_z
    SA_t = SA + S_Ht

    D_r = FZ * ((CMZ13 + CMZ14 * df_z) * scaling_coeffs[16] + (CMZ15 + CMZ16 * df_z) * IA_z) * R_nom * scaling_coeffs[9]
    B_r = CMZ6 * scaling_coeffs[11] / scaling_coeffs[9] + CMZ7 * B_y * C_y

    D_t = FZ * (CMZ9 + CMZ10 * df_z) * (1 + CMZ11 * IA_z + CMZ12 * IA_z**2) * (R_nom / FZ_nom) * scaling_coeffs[15]
    C_t = CMZ8
    B_t = (CMZ1 + CMZ2 * df_z + CMZ3 * df_z**2) * (1 + CMZ4 * IA_z + CMZ5 * abs(IA_z)) * scaling_coeffs[11] / scaling_coeffs[9]

    E_t = (CMZ17 + CMZ18 * df_z + CMZ19 * df_z**2) * (1 + (CMZ20 + CMZ21 * IA_z) * (2 / np.pi) * np.arctan(B_t * C_t * SA_t))

    # S_Hf = S_Hy + S_Vy / K_y
    S_Hr = S_Hy + S_Vy / K_y
    
    # Residual Torque

    # No definition in the Delft paper, so ignore for now
    S_Hr = 0 
    SA_r = SA + S_Hr
    M_zr = D_r * np.cos(np.arctan(B_r * SA_r)) * np.cos(SA)

    # Pneumatic trail
    t = D_t * np.cos(C_t * np.arctan(B_t * SA_t - E_t * (B_t * SA_t - np.arctan(B_t * SA_t)))) * np.cos(SA)

    M_Z0 = -t * F_y0 + M_zr
    M_Z = M_Z0

    return M_Z

tires = {"Hoosier_16x7.5-10_R20_7_cornering":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1

for name, tire in tires.items():
    try:
        df = pd.read_csv(f"././processing/results/{name}.csv")
        tire["lat"] = df[(df["velocity"] == velocity) & (df["pressure"] == pressure) & (df["SA"] < 10) & (df["SA"] > -10)]

    except:
        print("Error getting lateral data for {0}".format(name))

df = tires[list(tires.keys())[0]]["lat"]
x_lst = (df["FZ"] * -1).tolist()[0::10]
y_lst = (df["SA"] * np.pi / 180).tolist()[0::10]
z_lst = (df["IA"] * np.pi / 180).tolist()[0::10]

w_lst = (df["MZ"] * -1).tolist()[0::10]

print(len(w_lst))

model_x_data = np.linspace(abs(max(x_lst)), abs(min(x_lst)), 1000)
print(abs(max(x_lst)), abs(min(x_lst)))
model_y_data = np.linspace(-10 * np.pi / 180, 10 * np.pi / 180, 1000)

X, Y = np.meshgrid(model_x_data, model_y_data)

Z = _pure_aligning([8 * 0.0254, 1000, X, Y, 0])
Z1 = _pure_aligning([8 * 0.0254, 1000, X, Y, 1 * np.pi / 180])
Z2 = _pure_aligning([8 * 0.0254, 1000, X, Y, 2 * np.pi / 180])
Z3 = _pure_aligning([8 * 0.0254, 1000, X, Y, 3 * np.pi / 180])
Z4 = _pure_aligning([8 * 0.0254, 1000, X, Y, 4 * np.pi / 180])

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)

ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z)
ax.plot_surface(X, Y, Z1)
ax.plot_surface(X, Y, Z2)
ax.plot_surface(X, Y, Z3)
ax.plot_surface(X, Y, Z4)
ax.scatter3D(x_lst, y_lst, w_lst, cmap='Greens', s = 1)

fig.add_axes(ax)

ax.set_xlabel('Normal Load (N)')
ax.set_ylabel('Slip Angle (rad)')
ax.set_zlabel('Aligning Moment (Nm)')

plt.show()

def _aligning_tire_eval(pure_aligning_coeffs, R_nom, FZ_nom, FZ, SA, IA):
    pure_aligning_coeffs = list(pure_aligning_coeffs) + [0 for x in range(16)]
    pure_aligning_coeffs = pure_aligning_coeffs

    MZ = _pure_aligning(data = [R_nom, FZ_nom, FZ, SA, IA])

    return MZ

def _aligning_residual_calc(aligning_coeffs):
    residuals = []

    count = 0

    count += 1
    # for i in range(len(y_lst)):
    #     current_step = _aligning_tire_eval(aligning_coeffs, 8 * 0.0254, 1000, x_lst[i], y_lst[i], 0)
    #     # zero_eval = self._aligning_tire_eval(aligning_coeffs, self.R_nom, self.FZ_nom, self.FZ[i], 0, self.IA[i])

    #     residual = w_lst[i] - current_step

    #     if (y_lst[i] > 3.5 * np.pi / 180) and (y_lst[i] < 5 * np.pi / 180):
    #         residuals.append(residual**2)
    #     elif (y_lst[i] < -3.5 * np.pi / 180) and (y_lst[i] > -5 * np.pi / 180):
    #         residuals.append(residual**2)
    #     elif (y_lst[i] < 1 * np.pi / 180) or (y_lst[i] > -1 * np.pi / 180):
    #         residuals.append(residual**2)
    #     else:
    #         residuals.append(residual)

    # print(np.linalg.norm(residuals))

    # return residuals

    for i in range(len(y_lst)):
        current_step = _aligning_tire_eval(aligning_coeffs, 8 * 0.0254, 1000, x_lst[i], y_lst[i], z_lst[i])

        residuals.append(w_lst[i] - current_step)

    print(np.linalg.norm(residuals))

    return residuals

residual_vals = _aligning_residual_calc(pure_aligning_coeffs)

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)

ax = plt.axes(projection='3d')

ax.scatter3D(x_lst, y_lst, residual_vals, cmap='Greens', s = 1)

fig.add_axes(ax)

ax.set_xlabel('Normal Load (N)')
ax.set_ylabel('Slip Ratio')
ax.set_zlabel('Longitudinal Force (N)')

plt.show()