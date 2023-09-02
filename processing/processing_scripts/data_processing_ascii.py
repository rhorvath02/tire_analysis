import pandas as pd
import numpy as np
from processing_scripts import process_functions

###########################################################
# General Conditions for Processing
###########################################################

# Normal loads
L_dict = {
    "L_Continuous": [x / 0.224809 for x in np.arange(-500, 0, 1)], 
    "L1": [x / 0.224809 for x in [-50, -100, -150, -200, -250, -350]],
    "L2": [x / 0.224809 for x in [-50, -100, -150, -200, -250, -350]],
    "L3": [x / 0.224809 for x in [-50, -100, -150, -250, -350]], 
    "L4": [x / 0.224809 for x in [-50, -100, -150, -200, -250]], 
    "L6": [x / 0.224809 for x in [-50, -150, -200, -250]]
    }


# Pressures
P_dict = {
    "P": [x * 6.89476 for x in [8, 10, 12, 14]]
    }


# Velocity
V_dict = {
    "V_25": [x * 1.60934 for x in [25]], 
    "V1": [x * 1.60934 for x in [0, 25, 2]], 
    "V3": [x * 1.60934 for x in [25, 15, 45]]
    }


# Slip angles
SA_dict = {
    "S_cont": np.arange(-12, 13, 1), 
    "S1": [-1, 1, 6], 
    "S2": np.arange(-12, 13, 1), 
    "S3": [0, -3, -6], 
    "S4": [0, -3, -6]
    }


# Inclination angle
IA_dict = {
    "I1": [0, 2, 4],
    "I2": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    "I3": [-4, 0, 2, 4]
    }


# Slip ratio
SR_dict = {
    "const": [-1],
    "B2": [-0.15 -0.10, -0.05, 0, 0.05, 0.10, 0.15],
    "B3": [-0.15 -0.10, -0.05, 0, 0.05, 0.10, 0.15]
    }


###########################################################
# Inputs
###########################################################

inputs = pd.read_csv("inputs.csv")

def import_files(file_df):

    tire_dict = {}

    for i in range(len(file_df)):
        current_row = file_df.iloc[i]

        tire_dict[current_row["tire_name"]] = [current_row["file_location"], current_row["data_file_name"], [L_dict[current_row["FZ"]], 
        P_dict[current_row["P"]], V_dict[current_row["V"]], SA_dict[current_row["SA"]], IA_dict[current_row["IA"]], 
        SR_dict[current_row["SR"]]], current_row["corner/brake"]]

    return tire_dict

###########################################################
# Raw Import and File Initialization
###########################################################

def import_data(prepared_files):
    for tire, conditions in prepared_files.items():
        # Open and process file

        raw_file = open(f"./{conditions[0]}{conditions[1]}.dat")

        imported_data = raw_file.readlines()

        titles = imported_data[1].split()
        titles += ["load", "pressure", "velocity", "slip", "camber", "slip_ratio"]
        df_setup = dict()

        for title in titles:
            df_setup[title] = []

        ###########################################################
        # Data Processing
        ###########################################################

        for line in [line.split() for line in imported_data[3:]]:
            modified_line = [float(x) for x in line]
            i = -1
            for point in modified_line:
                i += 1

                if titles[i] == "FZ":
                    df_setup["load"] += [process_functions.nearest_value(conditions[2][0], point)]

                elif titles[i] == "P":
                    df_setup["pressure"] += [process_functions.nearest_value(conditions[2][1], point)]

                elif titles[i] == "V":
                    df_setup["velocity"] += [process_functions.nearest_value(conditions[2][2], point)]
                
                elif titles[i] == "SA":
                    df_setup["slip"] += [process_functions.nearest_value(conditions[2][3], point)]
                
                elif titles[i] == "IA":
                    df_setup["camber"] += [process_functions.nearest_value(conditions[2][4], point)]
                
                elif titles[i] == "SR":
                    df_setup["slip_ratio"] += [process_functions.nearest_value(conditions[2][5], point)]
                
                df_setup[titles[i]] += [point]

        df = pd.DataFrame(df_setup)

        output_directory = "./results/"

        df.to_csv(f'{output_directory}{conditions[3]}_{tire}.csv')

        print(f".csv written to {output_directory}{conditions[3]}_{tire}")