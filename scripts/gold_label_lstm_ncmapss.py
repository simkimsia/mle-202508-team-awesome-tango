#!/usr/bin/env python3
"""
Gold Layer - LSTM Label Store: Convert base labels to TimeSeries format for LSTM
Usage:
    python3 gold_label_lstm_ncmapss.py --snapshotdate "2023-01-01"
"""

import os
import argparse
import pandas as pd
# from darts import TimeSeries
# from darts.dataprocessing.transformers import Scaler
# from utils import data_processing_gold_label_lstm


def main(snapshotdate):
    print("\n\n--- Starting Gold LSTM Label N-CMAPSS job ---\n\n")

    # Define input/output directories
    snapshot_date_str = snapshotdate
    gold_label_base_dir = f"datamart/gold/label_base/n_cmapss/snapshot_date={snapshot_date_str}"
    gold_label_lstm_dir = f"datamart/gold/label/lstm/n_cmapss/snapshot_date={snapshot_date_str}"

    # Create output directories for each split
    for split in ['train', 'val', 'test', 'oot']:
        os.makedirs(f"{gold_label_lstm_dir}/{split}", exist_ok=True)

    print(f"Input directory: {gold_label_base_dir}")
    print(f"Output directory: {gold_label_lstm_dir}")

    # TODO: Implement LSTM label formatting
    # Tasks:
    # 1. Load base label DataFrames (df_train, df_val, df_test, df_oot)
    # 2. For each split:
    #    a. Group by engine unit
    #    b. Create Darts TimeSeries objects (one per engine)
    #       - Each TimeSeries contains RUL_Clipped values indexed by time
    #       - Format: TimeSeries with columns=['RUL_Clipped'], index=time
    #    c. Collect all engine TimeSeries into a list
    # 3. Normalize labels:
    #    a. Fit Scaler on training labels → target_scaler
    #    b. Transform train/val/test/oot labels
    # 4. Save outputs:
    #    - train/targets.pkl (list of 39 TimeSeries objects)
    #    - train/units.pkl (list of 39 unit IDs)
    #    - val/targets.pkl (list of 7 TimeSeries objects)
    #    - val/units.pkl (list of 7 unit IDs)
    #    - test/targets.pkl (list of 9 TimeSeries objects)
    #    - test/units.pkl (list of 9 unit IDs)
    #    - oot/targets.pkl (list of 19 TimeSeries objects)
    #    - oot/units.pkl (list of 19 unit IDs)
    #    - target_scaler.pkl (fitted Scaler object)

    # Expected output structure per engine:
    # - Each engine: TimeSeries (time_steps, 1)
    # - All values normalized to [0, 1]
    # - Indexed by time (cycle number)

    # Example for one engine:
    # unit = "DS01_001"
    # df_engine = df_train[df_train['unit'] == unit][['time', 'RUL_Clipped']]
    # ts = TimeSeries.from_dataframe(df_engine, time_col='time', value_cols=['RUL_Clipped'])

    # Call processing function (to be implemented in utils)
    # data_processing_gold_label_lstm.process_gold_label_lstm(
    #     gold_label_base_dir,
    #     gold_label_lstm_dir,
    #     snapshot_date_str
    # )

    print("\n\n--- Completed Gold LSTM Label N-CMAPSS job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gold LSTM Label formatting for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
