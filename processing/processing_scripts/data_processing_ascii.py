import pandas as pd
import numpy as np
from processing_scripts import process_functions

###########################################################
# General Conditions for Processing
###########################################################

input_location = "./data/"

# Normal loads
L_dict = {"L_Continuous": [x / 0.224809 for x in np.arange(-500, 0, 1)], "L1": [x / 0.224809 for x in [-50, -100, -150, -200, -250]],
"L3": [x / 0.224809 for x in [-50, -150, -250, -350]], "L4": [x / 0.224809 for x in [-50, -100, -150, -200, -250]], 
"L6": [x / 0.224809 for x in [-50, -150, -200, -250]]}


# Pressures
P_dict = {"P": [x * 6.89476 for x in [8, 10, 12, 14]]}


# Velocity
V_dict = {"V_25": [x * 1.60934 for x in [25]], "V1": [x * 1.60934 for x in [0, 25, 2]], "V3": [x * 1.60934 for x in [25, 15, 45]]}


# Slip angles
SA_dict = {"S1": [-1, 1, 6], "S2": [-12, -8, -4, 0, 4, 8, 12], "S3": [0, -3, -6], "S4": [0, -3, -6], "N/A": [0]}


# Inclination angle
IA_dict = {"I1": [0, 2, 4], "I2": [0, 2, 4]}


# Slip ratio
SR_dict = {"B2": [-0.15 -0.10, -0.05, 0, 0.05, 0.10, 0.15], "N/A": [0]}


###########################################################
# Inputs
###########################################################

inputs = pd.read_csv("inputs.csv")

def import_files(file_df):

    tire_dict = {}

    for i in range(len(file_df)):
        current_row = file_df.iloc[i]

        if type(current_row["SR"]) == np.float64:
            tire_dict[current_row["tire_name"]] = [current_row["data_file_name"], [L_dict[current_row["FZ"]], 
            P_dict[current_row["P"]], V_dict[current_row["V"]], SA_dict[current_row["SA"]], IA_dict[current_row["IA"]]]]

        elif type(current_row["SA"] == np.float64):
            tire_dict[current_row["tire_name"]] = [current_row["data_file_name"], [L_dict[current_row["FZ"]], 
            P_dict[current_row["P"]], V_dict[current_row["V"]], IA_dict[current_row["IA"]], 
            SR_dict[current_row["SR"]]]]

    return tire_dict

###########################################################
# Raw Import and File Initialization
###########################################################

def import_data(prepared_files):
    for tire, conditions in prepared_files.items():
        # Open and process file
        raw_file = open(f"{input_location}{conditions[0]}.dat")

        imported_data = raw_file.readlines()

        titles = imported_data[1].split()
        titles += ["load", "pressure", "velocity", "slip_a", "camber"]
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
                    df_setup["load"] += [process_functions.nearest_value(conditions[1][0], point)]

                elif titles[i] == "P":
                    df_setup["pressure"] += [process_functions.nearest_value(conditions[1][1], point)]

                elif titles[i] == "V":
                    df_setup["velocity"] += [process_functions.nearest_value(conditions[1][2], point)]
                
                elif titles[i] == "SA":
                    df_setup["slip_a"] += [process_functions.nearest_value(conditions[1][3], point)]
                
                elif titles[i] == "IA":
                    df_setup["camber"] += [process_functions.nearest_value(conditions[1][4], point)]
                
                df_setup[titles[i]] += [point]

        df = pd.DataFrame(df_setup)

        output_directory = "./results/"

        df.to_csv(f'{output_directory}{tire}.csv')

        print(f".csv written to {output_directory}{tire}")