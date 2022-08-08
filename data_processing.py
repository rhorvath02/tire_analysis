import pandas as pd
import numpy as np
import process_functions

###########################################################
# General conditions for fitting
###########################################################

sweeps = ["FZ", "V", "P", "SA", "IA"]

# Normal loads
L1 = [x / 0.224809 for x in [-50, -100, -150, -200, -250]]
L3 = [x / 0.224809 for x in [-50, -150, -250, -350]]


# Pressures
P = [x * 6.89476 for x in [8, 10, 12, 14]] 


# Velocity
V_25 = [x * 1.60934 for x in [25]] 
V1 = [x * 1.60934 for x in [0, 25, 2]]
V3 = [x * 1.60934 for x in [25, 15, 45]]


# Slip angles
S1 = [0, -3, -6]

# Inclination angle
l1 = [0, 2, 4]


###########################################################
# Raw import and file initialization
###########################################################

# Open and process file
raw_file = open("tire_data/raw_data/Round9/B2356raw72.dat")

imported_data = raw_file.readlines()

titles = imported_data[1].split()
df_setup = dict()

for title in titles:
    df_setup[title] = []


###########################################################
# Data processing
###########################################################
def import_data(FZ, P, V, SA, IA):
    for line in [line.split() for line in imported_data[3:]]:
        modified_line = [float(x) for x in line]
        i = -1
        for point in modified_line:
            i += 1

            if titles[i] == "FZ":
                df_setup[titles[i]] += [process_functions.nearest(FZ, point)]

            elif titles[i] == "P":
                df_setup[titles[i]] += [process_functions.nearest(P, point)]

            elif titles[i] == "V":
                df_setup[titles[i]] += [process_functions.nearest(V, point)]
            
            elif titles[i] == "SA":
                df_setup[titles[i]] += [process_functions.nearest(SA, point)]
            
            # elif titles[i] == "IA":
            #     df_setup[titles[i]] += [process_functions.nearest(IA, point)]
            
            else:
                df_setup[titles[i]] += [point]


    df = pd.DataFrame(df_setup)

    output_directory = "tire_data/processed_data/"

    df.to_csv(f'{output_directory}test_braking.csv')

    print(f".csv written to {output_directory}")

import_data(L3, P, V_25, S1, l1)