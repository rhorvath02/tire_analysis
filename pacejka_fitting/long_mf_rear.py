import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from mpl_toolkits.mplot3d import Axes3D

def fit(data, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13):
    FZ = data[0] / 1000 * -1
    SR = data[1]

    C = b0
    D = FZ * (b1 * FZ + b2)
    
    BCD = (b3 * FZ**2 + b4 * FZ) * np.exp(-1 * b5 * FZ)
    B = BCD / (C * D)
    H = b9 * FZ + b10

    E = (b6 * FZ**2 + b7 * FZ + b8) * (1 - b13 * np.sign(SR + H))

    V = b11 * FZ + b12
    Bx1 = B * (SR + H)

    return (D * np.sin(C * np.arctan(Bx1 - E * (Bx1 - np.arctan(Bx1)))) + V)

tires = {"hoosier_r25b_18x7-5_10x7":{"long":None, "lat":None}}

camber = 0 # default camber
pressure = 12 * 6.89476 # default pressure
velocity = 25 * 1.60934 # default velocity
slip_angle = 0
run_num = 1

for name, tire in tires.items():
    try:
        df = pd.read_csv(f"./tire_data/processed_data/braking_{name}.csv")
        tire["long"] = df[(df["pressure"] == pressure) & (df["velocity"] == velocity) & (df["slip"] == 0) & (df["camber"] == 0)]
        # print(tire["long"])
        
    except:
        print("Error getting long data for {0}".format(name))

    try:
        df = pd.read_csv(f"./tire_data/processed_data/cornering_{name}.csv")
        tire["lat"] = df[(df["velocity"] == velocity) & (df["pressure"] == pressure)]
        # print(tire["lat"])

    except:
        print("Error getting lateral data for {0}".format(name))

df = tires["hoosier_r25b_18x7-5_10x7"]["long"]
x_lst = [x for x in df["FZ"].tolist()]
y_lst = [x * 100 for x in df["SR"].tolist()]
z_lst = df["FX"].tolist()

a_vals = [1.8, 840, 5000, 400, 700, -0.5, 0, 0, 0, -2, 0, 0, 0, 0]
a_vals2 = [1.65, -5000, 2000, 400, 700, 0.7, 0, 0, -3, 0, 0, 0, 0, 0]
hand_fit = [1.8, 200, 3000, -400, 700, -0.5, 0, 0, 0, -2, 2, 0, 0, 0]
optimal = [ 6.37312726e-01,  1.94939660e+02,  3.00088541e+03, -4.03320823e+02,
        7.02486660e+02, -6.64921922e-01,  2.47703186e-01, -4.39641395e-01,
       -1.08916763e+00, -8.10952295e-01,  3.54937175e+00,  7.73535131e-01,
       -2.70299041e+00, -4.43023736e-01]

# labels = [b0 , b1 , b2  ,  b3 , b4 ,  b5, b6,b7,b8, b9,b10,b11,b12,b13]

# parameters, covariance = curve_fit(fit, [x_lst, y_lst], z_lst, hand_fit, maxfev = 10000, bounds = ((1.65, 500, 1000, 0, 0, -2, -100, -0.00001, -0.00001, -1000, -1000, -5000, -5000, -0.00001), (2, 4000, 10000, 1000, 1500, 3, 100, 0, 0, 1000, 1000, 5000, 5000, 0)))

model_x_data = np.linspace(min(x_lst), max(x_lst), 1000)
model_y_data = np.linspace(-50, 50, 100)

X, Y = np.meshgrid(model_x_data, model_y_data)

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)

# params = parameters.tolist()

# params[8] = 0


def accuracy(coeffs):
    ia_count = 0
    for i in range(200):
        predicted = fit([x_lst[i], y_lst[i]], *coeffs)

        error = (z_lst[i] - predicted) / predicted * 100

        if abs(error) >= 10:
            ia_count += 1
        
    print(((1.000001 - ia_count / 200) * 100))

    return ((1.000001 - ia_count / 200) * 100)**-1

params = basinhopping(accuracy, optimal).x
print(list(params))

accuracy(optimal)

Z = fit([X, Y], *optimal)

ax = plt.axes(projection='3d')

ax.scatter3D(x_lst, y_lst, z_lst, cmap='Greens')

fig.add_axes(ax)
ax.plot_surface(X, Y, Z)

ax.set_xlabel('Normal Load (N)')
ax.set_ylabel('Slip Ratio')
ax.set_zlabel('Longitudinal Force (N)')

# print("Accuracy: " + str((1 - ia_count / 25000) * 100) + "%")

params = basinhopping(accuracy, optimal)
print(params)

accuracy(optimal)

# print("Accuracy: " + str((1 - ia_count / 25000) * 100) + "%")

plt.show()

# print(parameters)

print()