import pandas as pd
import numpy as np
import process_functions


###########################################################
# General conditions for fitting
###########################################################

# Normal loads
L_Continuous = [x / 0.224809 for x in np.arange(-500, 0, 1)]
L1 = [x / 0.224809 for x in [-50, -100, -150, -200, -250]]
L3 = [x / 0.224809 for x in [-50, -150, -250, -350]]
L4 = [x / 0.224809 for x in [-50, -100, -150, -200, -250]]
L6 = [x / 0.224809 for x in [-50, -150, -200, -250]]


# Pressures
P = [x * 6.89476 for x in [8, 10, 12, 14]]


# Velocity
V_25 = [x * 1.60934 for x in [25]] 
V1 = [x * 1.60934 for x in [0, 25, 2]]
V3 = [x * 1.60934 for x in [25, 15, 45]]


# Slip angles
S1 = [-1, 1, 6]
S2 = [-12, -8, -4, 0, 4, 8, 12]
S3 = [0, -3, -6]
S4 = [0, -3, -6]

# Inclination angle
l1 = [0, 2, 4]
l2 = [0, 2, 4]

######################################################################################################################
################################################## INPUTS ############################################################
######################################################################################################################

# {"TIRE_NAME": ["FILE NAME", [CONDITIONS]]}

files = {"cornering_hoosier_r25b_16x7-5_10x8": ["B2356run8.dat", [L1, P, V_25, S1, l1]]}
        # , "cornering_hoosier_r25b_18x7-5_10x8": ["B1654run24.dat", [L4, P, V_25, S2, l2]]}

input_location = "tire_data/raw_data/Round9/"

###########################################################
# Raw import and file initialization
###########################################################

def import_data():
    for tire, conditions in files.items():
        # Open and process file
        raw_file = open(f"{input_location}{conditions[0]}")

        imported_data = raw_file.readlines()

        titles = imported_data[1].split()
        titles += ["load", "pressure", "velocity", "slip", "camber"]
        df_setup = dict()

        for title in titles:
            df_setup[title] = []


###########################################################
# Data processing
###########################################################

        for line in [line.split() for line in imported_data[3:]]:
            modified_line = [float(x) for x in line]
            i = -1
            for point in modified_line:
                i += 1

                if titles[i] == "FZ":
                    df_setup["load"] += [process_functions.nearest_value(conditions[1][0], point)]

                elif titles[i] == "P":
                    df_setup["pressure"] += [process_functions.nearest_value(conditions[1][1], point)]

                elif titles[i] == "V":
                    df_setup["velocity"] += [process_functions.nearest_value(conditions[1][2], point)]
                
                elif titles[i] == "SA":
                    df_setup["slip"] += [process_functions.nearest_value(conditions[1][3], point)]
                
                elif titles[i] == "IA":
                    df_setup["camber"] += [process_functions.nearest_value(conditions[1][4], point)]
                
                df_setup[titles[i]] += [point]


        df = pd.DataFrame(df_setup)

        output_directory = "tire_data/processed_data/"

        df.to_csv(f'{output_directory}{tire}.csv')

        print(f".csv written to {output_directory}{tire}")

import_data()