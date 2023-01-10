import pandas as pd
from processing_scripts import data_processing_ascii as process

if __name__ == '__main__':
    inputs = pd.read_csv("inputs.csv")

    tire_dict = process.import_files(inputs)

    process.import_data(tire_dict)