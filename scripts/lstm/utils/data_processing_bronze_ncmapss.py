from datetime import datetime
import os

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def process_bronze_table(snapshot_date_str, bronze_lms_directory):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    table_name = "n_cmapss"
    filename = f"../data/N-CMAPSS_DS_{snapshot_date_str}.h5"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    print(f"Processing Bronze table for snapshot date: {snapshot_date_str}")
    print(f"Loading data from {filename}")
    dataset_dir = os.path.join(
        bronze_lms_directory, table_name, f"snapshot_date={snapshot_date_str}"
    )
    if os.path.exists(dataset_dir):
        print(f"Overwriting existing Bronze snapshot: {dataset_dir}")
        for f in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, f)
            if os.path.isfile(file_path) and f.endswith(".parquet"):
                os.remove(file_path)
        for f in os.listdir(dataset_dir):
            if f.startswith("_"):
                os.remove(os.path.join(dataset_dir, f))
    else:
        os.makedirs(dataset_dir)
    with h5py.File(filename, "r") as hdf:
        dev_rows = hdf["W_dev"].shape[0]
        test_rows = hdf["W_test"].shape[0]
        total_rows = dev_rows + test_rows
        print(f"Total rows: {total_rows} (dev: {dev_rows}, test: {test_rows})")
        print("Loading dev split...")
        W_dev = hdf["W_dev"][:]
        X_s_dev = hdf["X_s_dev"][:]
        X_v_dev = hdf["X_v_dev"][:]
        Y_dev = hdf["Y_dev"][:]
        A_dev = hdf["A_dev"][:]
        T_dev = hdf["T_dev"][:]
        print("Loading test split...")
        W_test = hdf["W_test"][:]
        X_s_test = hdf["X_s_test"][:]
        X_v_test = hdf["X_v_test"][:]
        Y_test = hdf["Y_test"][:]
        A_test = hdf["A_test"][:]
        T_test = hdf["T_test"][:]
        print("Concatenating dev and test splits...")
        W = np.concatenate((W_dev, W_test), axis=0)
        X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
        X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
        Y = np.concatenate((Y_dev, Y_test), axis=0)
        A = np.concatenate((A_dev, A_test), axis=0)
        T = np.concatenate((T_dev, T_test), axis=0)
        del W_dev, W_test, X_s_dev, X_s_test, X_v_dev, X_v_test
        del Y_dev, Y_test, A_dev, A_test, T_dev, T_test
        print(f"Creating DataFrame with {len(W)} rows...")
        df_chunk = pd.DataFrame({
            "W": list(W),
            "X_s": list(X_s),
            "X_v": list(X_v),
            "T": list(T),
            "Y": list(Y),
            "A": list(A)
        })
        df_chunk["snapshot_date"] = snapshot_date
        filepath = os.path.join(dataset_dir, "data.parquet")
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        pq.write_table(table, filepath)
        print(f"  ➤ Saved {filepath} ({len(df_chunk)} rows)")
        del df_chunk, table, W, X_s, X_v, T, Y, A
    print(f"Completed Bronze partitioned write to: {dataset_dir}")
    return dataset_dir