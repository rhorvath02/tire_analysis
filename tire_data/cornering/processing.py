import pandas as pd

files = {"braking_hoosier_r25b_18x7-5_10x7": []}

input_location = "cornering.csv"

def import_data(file_location):
    raw_df = pd.read_csv(f"{file_location}")

    df = pd.DataFrame({'Tire': [], 'Spring Rate (lbs/in)': [], 'P (psi)': [], 'IA (deg)': [], 'Nominal Load (lbs)': [], 'CS (lbs/deg)': [], 'CSC': [], 'FY-0 (lbs)': []})

    previous_row = []
    for i in range(8):
        previous_row.append(raw_df.iloc[0][i])

    new_row = []
    for i in range(1, len(raw_df)):
        new_row.clear()

        for j in range(8):
            new_row.append(raw_df.iloc[i][j])

        for k in range(len(new_row)):
            if (str(new_row[k]) == "nan") or (new_row[k] == "-      "):
                new_row[k] = previous_row[k]

        df2 = {'Tire': [new_row[0]], 'Spring Rate (lbs/in)': [new_row[1]], 'P (psi)': [new_row[2]], 'IA (deg)': [new_row[3]], 'Nominal Load (lbs)': [new_row[4]], 'CS (lbs/deg)': [new_row[5]], 'CSC': [new_row[6]], 'FY-0 (lbs)': [new_row[7]]}

        df = df.append(df2, ignore_index = True)

        previous_row = new_row.copy()

import_data(input_location)