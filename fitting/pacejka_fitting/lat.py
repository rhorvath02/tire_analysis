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

def long_fit(data, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, IA_coeff_mult, IA_coeff_shift):
    FZ = data[0] / 1000 * -1
    SR = data[1]
    IA = data[2]

    C = b0
    D = FZ * (b1 * FZ + b2)
    
    BCD = (b3 * FZ**2 + b4 * FZ) * np.exp(-1 * b5 * FZ)
    B = BCD / (C * D)
    H = b9 * FZ + b10

    E = (b6 * FZ**2 + b7 * FZ + b8) * (1 - b13 * np.sign(SR + H))

    V = b11 * FZ + b12
    Bx1 = B * (SR + H)

    return (D * np.sin(C * np.arctan(Bx1 - E * (Bx1 - np.arctan(Bx1)))) + V) * (abs(IA) * IA_coeff_mult + IA_coeff_shift)
    # return (D * np.sin(C * np.arctan(Bx1 - E * (Bx1 - np.arctan(Bx1)))) + V) * (IA_coeff_shift)


tires = {"Hoosier_16x7.5-10_R20_7_cornering":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1

for name, tire in tires.items():
    try:
        df = pd.read_csv(f"./tire_data/processed_data/braking_{name}.csv")
        tire["long"] = df[(df["pressure"] == pressure) & (df["velocity"] == velocity) & (df["slip_a"] == slip_angle)]
        # print(tire["long"])
        
    except:
        print("Error getting long data for {0}".format(name))

    try:
        df = pd.read_csv(f"././processing/results/{name}.csv")
        tire["lat"] = df[(df["velocity"] == velocity) & (df["pressure"] == pressure)]
        tire["lat"] = df
        print(tire["lat"])

    except:
        print("Error getting lateral data for {0}".format(name))

# optimal = [0.39830083710963915, 0.0014393087708314588, 7.71265763884129, 580.0454819644223, 1317.9872318215967, 0.010267523595846822, 0.0003825324881518819, 1.0089330008243167, -0.00010586685563578281, 0.12849611983691872, -0.1533438020447101, -0.0035070986751402013, -11.340688484385716, -3.10472941443183e-05, 0.042842643562589405, 0.0039455024323146225, -0.018924538207664387, 0.012425246407185168]

# optimal = [0.05381525833576014, 0.29621203400913704, -1106.3490214778724, 667.5938503985029, 4993.007666973025, 0.009923438946849938, 0.046373040467766254, 152.77496414902092, 0.005959760753357412, 96.74580143257889, 24.02999984496768, -28.07566891920053, 279.18542572914686, 0.0005010329961510464, -5.8810447448081495, -1.100992695739943, 0.0011145475151656424, 1.0112508971102616]

optimal1 = [0.3828513134342068, 5.2004464027784356e-05, 7.736780948786636, 634.4300745542466, 1382.3479345192316, 0.009928553088627485, 3.69045647708646e-05, 1.013456233264579, 7.398415229413076e-05, -0.09587885248980442, -0.1336338249732939, -0.06309105968029514, 63.18099000275936, 5.170184683813135e-06, 0.0006185767547265901, 0.004267447047470915, -0.01783403753353505, -0.00035378037090950004]

# optimal = [3.06312013e-01, -4.84039800e-04, 9.28662043e+00, 7.26482752e+02, 1.37727265e+03, 4.12679421e-02, 2.53429368e-05, 1.01054136e+00, 1.46808723e-04, -1.59143742e-01, -1.29418581e-01, -1.00903302e-01, 9.57798247e+01, 2.27490918e-05, -1.83997391e-02, -1.13901618e-03, -1.56217480e-02, -4.89395709e-04]

optimal2 = [3.06312013e-01, -4.84039800e-04, 9.28662043e+00, 7.26482752e+02, 1.37727265e+03, 4.12679421e-02, 2.53429368e-05, 1.01054136e+00, 0, 0, 0, 0, 0, 0, 0, -1.13901618e-03, -1.56217480e-02, -4.89395709e-04]

df = tires[list(tires.keys())[0]]["lat"]
x_lst = (df["FZ"] * -1).tolist()
y_lst = df["SA"].tolist()
z_lst = df["IA"].tolist()

w_lst = (df["FY"] * -1).tolist()

# data = df["load"].unique()

# for normal_load in data:
#     test_df = df[(df["load"] == normal_load)]
#     print(f"Load: {normal_load}")
#     print(f"Max FY: {max(test_df['FY'])}")
#     print(f"Min FY: {min(test_df['FY'])}")
#     print()

bounds = [[-1112, [-2776, 2718]], [-890, [-2270, 2455]], [-667, [-1740, 1710]], [-445, [-1199, 1175]], [-222, [-1066, 612]]]

# for bound in bounds:
#     for i in range(1000):
#         x_lst.append(abs(bound[0]))
#         y_lst.append(np.pi / 2 * 180 / np.pi)
#         z_lst.append(0)
#         w_lst.append(bound[1][0] * -1 * 0.70)
        
#         x_lst.append(abs(bound[0]))
#         y_lst.append(-np.pi / 2 * 180 / np.pi)
#         z_lst.append(0)
#         w_lst.append(bound[1][1] * -1 * 0.70)

# for bound in bounds:
#     for i in range(1000):
#         x_lst.append(abs(bound[0]))
#         y_lst.append(np.pi / 4 * 180 / np.pi)
#         z_lst.append(0)
#         w_lst.append(bound[1][0] * -1 * 0.85)
        
#         x_lst.append(abs(bound[0]))
#         y_lst.append(-np.pi / 4 * 180 / np.pi)
#         z_lst.append(0)
#         w_lst.append(bound[1][1] * -1 * 0.85)

params1 = optimal1
params2 = optimal2

model_x_data = np.linspace(abs(max(x_lst)), abs(min(x_lst)), 1000)
model_y_data = np.linspace(-15, 15, 1000)

X, Y = np.meshgrid(model_x_data, model_y_data)

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)

W1 = lat_fit([X, Y, 0], *params1)
W2 = lat_fit([X, Y, 0], *params2)
# W2 = lat_fit([X, Y, 1], *params)
# W3 = lat_fit([X, Y, 2], *params)

ax = plt.axes(projection='3d')

ax.scatter3D(x_lst, y_lst, w_lst, cmap='Greens')

fig.add_axes(ax)
ax.plot_surface(X, Y, W1)
ax.plot_surface(X, Y, W2)
# ax.plot_surface(X, Y, W2)
# ax.plot_surface(X, Y, W3)

ax.set_xlabel('Normal Load (N)')
ax.set_ylabel('Slip Angle')
ax.set_zlabel('Lateral Force (N)')

plt.show()