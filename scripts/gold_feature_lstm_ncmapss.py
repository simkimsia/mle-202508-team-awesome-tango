#!/usr/bin/env python3
"""
Gold Layer - LSTM Feature Store: Create time series features for LSTM model
Usage:
    python3 gold_feature_lstm_ncmapss.py --snapshotdate "2023-01-01"
"""

import os
import argparse
import pandas as pd
import numpy as np
# from darts import TimeSeries
# from darts.dataprocessing.transformers import Scaler
# from utils import data_processing_gold_feature_lstm


def main(snapshotdate):
    print("\n\n--- Starting Gold LSTM Feature N-CMAPSS job ---\n\n")

    # Define input/output directories
    snapshot_date_str = snapshotdate
    gold_label_dir = f"datamart/gold/label/n_cmapss/snapshot_date={snapshot_date_str}"
    gold_feature_dir = f"datamart/gold/feature/lstm/n_cmapss/snapshot_date={snapshot_date_str}"

    # Create output directories for each split
    for split in ['train', 'val', 'test', 'oot']:
        os.makedirs(f"{gold_feature_dir}/{split}", exist_ok=True)

    print(f"Input directory: {gold_label_dir}")
    print(f"Output directory: {gold_feature_dir}")

    # Configuration
    SELECTED_FEATURES = [
        'HPC Outlet Pressure',
        'LPT Coolant Bleed',
        'Fan Inlet Pressure',
        'Demanded Fan Speed',
        'Fan Speed',
        'Core Speed',
        'Pressure in Bypass Duct',
        'Fuel Flow Ratio',
        'LPT Outlet Temperature',
        'Altitude',
        'Mach Number',
        'Throttle Resolver Angle'
    ]

    SEQUENCE_LENGTH = 30  # Lookback window

    # TODO: Implement LSTM feature engineering
    # Tasks:
    # 1. Load train/val/test/oot DataFrames from gold label layer
    # 2. Select 12 features from SELECTED_FEATURES
    # 3. For each split:
    #    a. Create Darts TimeSeries objects (one per engine unit)
    #    b. Separate covariates (features) and targets (RUL_Clipped)
    # 4. Normalize features:
    #    - Fit Scaler on training covariates
    #    - Transform val/test/oot covariates
    # 5. Normalize targets:
    #    - Fit separate Scaler on training targets
    #    - Transform val/test/oot targets
    # 6. Save TimeSeries objects:
    #    - train/covariates.pkl, train/targets.pkl, train/units.pkl
    #    - val/covariates.pkl, val/targets.pkl, val/units.pkl
    #    - test/covariates.pkl, test/targets.pkl, test/units.pkl
    #    - oot/covariates.pkl, oot/targets.pkl, oot/units.pkl
    # 7. Save scalers:
    #    - feature_scaler.pkl
    #    - target_scaler.pkl

    # Expected output structure per engine:
    # - Covariates: TimeSeries (time_steps, 12)
    # - Targets: TimeSeries (time_steps, 1)
    # - All values normalized to [0, 1]

    # Call processing function (to be implemented in utils)
    # data_processing_gold_feature_lstm.process_gold_feature_lstm(
    #     gold_label_dir,
    #     gold_feature_dir,
    #     snapshot_date_str,
    #     SELECTED_FEATURES,
    #     SEQUENCE_LENGTH
    # )

    print("\n\n--- Completed Gold LSTM Feature N-CMAPSS job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gold LSTM Feature engineering for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
