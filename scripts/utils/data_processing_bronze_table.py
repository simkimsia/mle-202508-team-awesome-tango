import os
from datetime import datetime
import h5py
import numpy as np
import pandas as pd



def process_bronze_table(snapshot_date_str, bronze_lms_directory):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    all_W, all_X_s, all_X_v, all_Y, all_A = [], [], [], [], []

    print(f"Processing bronze table for snapshot date: {snapshot_date_str}")

    d = snapshot_date_str[-1] 

    print(d)

    filename = f"Data/N-CMAPSS_DS0{d}.h5"

    table_name = "N_CMAPSS".lower()

    print(f"Loading data from {filename}")

    with h5py.File(filename, 'r') as hdf:
        W_dev = np.array(hdf.get('W_dev'))
        X_s_dev = np.array(hdf.get('X_s_dev'))
        X_v_dev = np.array(hdf.get('X_v_dev'))
        Y_dev = np.array(hdf.get('Y_dev'))
        A_dev = np.array(hdf.get('A_dev'))

        W_test = np.array(hdf.get('W_test'))
        X_s_test = np.array(hdf.get('X_s_test'))
        X_v_test = np.array(hdf.get('X_v_test'))
        Y_test = np.array(hdf.get('Y_test'))
        A_test = np.array(hdf.get('A_test'))

        # Combine dev and test for this dataset
        W = np.concatenate((W_dev, W_test), axis=0)
        X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
        X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
        Y = np.concatenate((Y_dev, Y_test), axis=0)
        A = np.concatenate((A_dev, A_test), axis=0)


    # Combine into a simple dict DataFrame for quick storage
    df = pd.DataFrame({
        "W": list(W),
        "X_s": list(X_s),
        "X_v": list(X_v),
        "Y": list(Y),
        "A": list(A)
    })

    df["snapshot_date"] = snapshot_date

    print(f"{table_name}_{snapshot_date_str} row count: {len(df)}")

    dataset_dir = os.path.join(bronze_lms_directory, table_name, f"snapshot_date={snapshot_date_str}")
    os.makedirs(dataset_dir, exist_ok=True)
    
    filepath = os.path.join(dataset_dir, "data.parquet")
    df.to_parquet(filepath, index=False)
    print('saved to:', filepath)
        
    return df