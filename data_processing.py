import scipy.io as sio
import numpy as np
import pandas as pd
import data_helpers

def main():
    # Normal load sweeps
    L1 = np.array([-50, -100, -150, -200, -250]) / 0.224809
    L2 = np.array([-50, -100, -150, -250, -350]) / 0.224809
    L3 = np.array([-350, -150, -250, -50]) / 0.224809
    L4 = np.array([-50, -100, -150, -200, -250]) / 0.224809
    L6 = np.array([-50, -150, -200, -250]) / 0.224809
    L7 = np.array([-50, -150, -250, -350]) / 0.224809

    # Camber sweeps
    l1 = np.array([0, 2, 4])

    # velocity sweeps
    V1 = np.array([0, 25, 2]) * 1.60934
    V3 = np.array([15, 25, 45]) * 1.60934
    
    # pressure sweep
    P = np.array([8, 10, 12, 14]) * 6.89476 # includes P1r and P2r
    
    # slip angle sweep (for long/combined data)
    S1 = np.array([-1, 1, 6])
    S4 = np.array([0, -3, -6])

    def create_sweep_dict(normal_load, camber, pressure, velocity, slip_angle = None):
        if slip_angle is not None:
            return {"load" : {"sweep" : normal_load, "label" : "FZ" },
                    "camber" : {"sweep" : camber, "label" : "IA"},
                    "pressure" : {"sweep" : pressure, "label" : "P"},
                    "velocity" : {"sweep" : velocity, "label" : "V"},
                    "slip" : {"sweep" : slip_angle, "label" : "SA"}}
        else:
            return {"load" : {"sweep" : normal_load, "label" : "FZ" },
                    "camber" : {"sweep" : camber, "label" : "IA"},
                    "pressure" : {"sweep" : pressure, "label" : "P"},
                    "velocity" : {"sweep" : velocity, "label" : "V"}}
    
    output_directory = "tire_data/processed_data/"

    data_map = {"cornering_test": {"file_path" : "tire_data/raw_data/Round9/B2356raw2.mat", 
                                                            "sweeps" : create_sweep_dict(L1, l1, P, V1, S1), "avg": True},
                "braking_test": {"file_path" : "tire_data/raw_data/Round9/B2356raw50.mat", 
                                                            "sweeps" : create_sweep_dict(L1, l1, P, V1, S1), "avg": True}
                }
    
    for output_name, data_info in data_map.items():
        # load matlab file and convert to pandas df
        ## NOTEE - if multiple sweeps call the same matlab file, this can cause this to stop working, dont do that
        df = data_helpers.import_data(sio.loadmat(data_info["file_path"]), run_data = True)

        # classify sweeps on data
        for variable, info in data_info["sweeps"].items():
            temp_nearest_func = lambda x: get_nearest_value(info["sweep"], x)
            df[variable] = df[info["label"]].apply(temp_nearest_func)

        # period of oscillation is ~ 10.5 data points, remove oscillation
        # TODO: Use FFT to find oscillation period, and remove it
        if data_info["avg"]:
            for target_var in ["FY", "FX", "FZ"]:
                df[target_var] = moving_average(df[target_var], 10)

        # export data to CSV
        df.to_csv(f'{output_directory}{output_name}.csv')
    
def get_nearest_value(possible_values, input_value):
    closest_value, distance = None, 0
    for value in possible_values:
        test_dist = abs(value - input_value)
        if not distance or test_dist < distance:
            distance = test_dist
            closest_value = value
    return closest_value

def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

if __name__ == "__main__":
    main()