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
    filename = f"data/N-CMAPSS_DS_{snapshot_date_str}.h5"
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
    chunk_size = 500000
    with h5py.File(filename, "r") as hdf:
        total_rows = hdf["W_dev"].shape[0]
        print(f"Total rows: {total_rows}")
        for i, start in enumerate(range(0, total_rows, chunk_size)):
            end = min(start + chunk_size, total_rows)
            print(f"Processing chunk {i+1}: rows {start}:{end}")
            W = np.concatenate((hdf["W_dev"][start:end], hdf["W_test"][start:end]), axis=0)
            X_s = np.concatenate((hdf["X_s_dev"][start:end], hdf["X_s_test"][start:end]), axis=0)
            X_v = np.concatenate((hdf["X_v_dev"][start:end], hdf["X_v_test"][start:end]), axis=0)
            Y = np.concatenate((hdf["Y_dev"][start:end], hdf["Y_test"][start:end]), axis=0)
            A = np.concatenate((hdf["A_dev"][start:end], hdf["A_test"][start:end]), axis=0)
            T = np.concatenate((hdf["T_dev"][start:end], hdf["T_test"][start:end]), axis=0)
            df_chunk = pd.DataFrame({
                "W": list(W),
                "X_s": list(X_s),
                "X_v": list(X_v),
                "T": list(T),
                "Y": list(Y),
                "A": list(A)
            })
            df_chunk["snapshot_date"] = snapshot_date
            filepath = os.path.join(dataset_dir, f"data_part_{i+1:04d}.parquet")
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            pq.write_table(table, filepath)
            print(f"  ➤ Saved {filepath} ({len(df_chunk)} rows)")
            del df_chunk, table, W, X_s, X_v, T, Y, A
    print(f"Completed Bronze partitioned write to: {dataset_dir}")
    return dataset_dir