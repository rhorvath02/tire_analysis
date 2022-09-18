import scipy.io as sio
import process_functions

###########################################################
# All inputs
###########################################################

# Normal load
L1 = [x / 0.224809 for x in [-50, -100, -150, -200, -250]]
L2 = [x / 0.224809 for x in [-50, -100, -150, -250, -350]]
L3 = [x / 0.224809 for x in [-50, -100, -150, -250, -350]]
L4 = [x / 0.224809 for x in [-50, -100, -150, -200, -250]]

L6 = [x / 0.224809 for x in [-50, -150, -200, -250]]


# Pressure
P_12 = [x * 6.89476 for x in [12]]
P = [x * 6.89476 for x in [8, 10, 12, 14]] 


# Velocity
V_25 = [x * 1.60934 for x in [25]] 
V1 = [x * 1.60934 for x in [0, 25, 2]]
V3 = [x * 1.60934 for x in [25, 15, 45]]


# Slip angle
S1 = [0, -3, -6]
S2 = [-4, 0, 4, 8, 12]
S3 = [0, -3, -6]
S4 = [0, -3, -6]

# Inclination angle
l1 = [0, 2, 4]
l2 = [0, 2, 4]
l3 = [-2, 0, 2, 6]


###########################################################
# Initialization
###########################################################

output_directory = "tire_data/processed_data/"

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

data = {"cornering_placeholder": {"file_path" : "tire_data/raw_data/Round9/B2356run31.mat", 
                                                            "sweeps" : create_sweep_dict(L1, l1, P, V_25), "avg": True},
        
        "braking_placeholder": {"file_path" : "tire_data/raw_data/Round9/B2356run72.mat", 
                                                            "sweeps" : create_sweep_dict(L3, l1, P, V_25, S3)},

        "cornering_hoosier_r25b_18x7-5_10x7_run1": {"file_path" : "tire_data/raw_data/Round9/B1654run21.mat", 
                                                            "sweeps" : create_sweep_dict(L4, l2, P, V1), "avg": True},

        "braking_hoosier_r25b_18x7-5_10x7_run1": {"file_path" : "tire_data/raw_data/Round9/B1654run35.mat",
                                                            "sweeps" : create_sweep_dict(L6, l2, P, V1, S4)}}
        

###########################################################
# Data processing
###########################################################

for output_name, data_info in data.items():
        # load matlab file and convert to pandas df
        ## NOTEE - if multiple sweeps call the same matlab file, this can cause this to stop working, dont do that
        df = process_functions.import_data(sio.loadmat(data_info["file_path"]), run_data = True)
        
        # classify sweeps on data
        for variable, info in data_info["sweeps"].items():
            temp_nearest_func = lambda x: process_functions.nearest_value(info["sweep"], x)
            df[variable] = df[info["label"]].apply(temp_nearest_func)

        # Remove oscillation
        if data_info["avg"]:
            for target_var in ["FY", "FX", "FZ"]:
                df[target_var] = process_functions.nearest_value(df[target_var], 10)


        # export data to csv
        df.to_csv(f'{output_directory}{output_name}.csv')


        print(f".csv written to {output_directory}{output_name}")
