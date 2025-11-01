#!/usr/bin/env python3
"""
Gold Layer - XGBoost Feature Store: Create time-windowed features for XGBoost model
Usage:
    python3 gold_feature_xgboost_ncmapss.py --snapshotdate "2023-01-01"
"""

import os
import argparse
import pandas as pd
import numpy as np
import gc
# from utils import data_processing_gold_feature_xgboost


def main(snapshotdate):
    print("\n\n--- Starting Gold XGBoost Feature N-CMAPSS job ---\n\n")

    # Define input/output directories
    snapshot_date_str = snapshotdate
    gold_label_base_dir = f"datamart/gold/label_base/n_cmapss/snapshot_date={snapshot_date_str}"
    gold_feature_xgboost_dir = f"datamart/gold/feature/xgboost/n_cmapss/snapshot_date={snapshot_date_str}"

    # Create output directories for each split
    for split in ['train', 'val', 'test', 'oot']:
        os.makedirs(f"{gold_feature_xgboost_dir}/{split}", exist_ok=True)

    print(f"Input directory: {gold_label_base_dir}")
    print(f"Output directory: {gold_feature_xgboost_dir}")

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

    WINDOWS = {
        'short': 5,    # Last 5 cycles - immediate trends
        'medium': 15,  # Last 15 cycles - medium-term trends
        'long': 30     # Last 30 cycles - long-term trends (matches LSTM)
    }

    # TODO: Implement XGBoost feature engineering
    # Tasks:
    # 1. Load train/val/test/oot DataFrames from gold label base layer
    # 2. For each split:
    #    a. Select 12 base features from SELECTED_FEATURES
    #    b. Convert features to float32 (memory optimization)
    #    c. For each of 12 features:
    #       - Create rolling mean (3 windows × 12 = 36 features)
    #         * {feature}_mean_short (window=5)
    #         * {feature}_mean_medium (window=15)
    #         * {feature}_mean_long (window=30)
    #       - Create rolling std (3 windows × 12 = 36 features)
    #         * {feature}_std_short (window=5)
    #         * {feature}_std_medium (window=15)
    #         * {feature}_std_long (window=30)
    #       - Create acceleration: mean_short - mean_medium (12 features)
    #         * {feature}_acceleration
    #    d. Process features in batches (3 at a time) with gc.collect()
    # 3. Save engineered DataFrames:
    #    - train/features.parquet (26.9M rows × 96 features)
    #    - val/features.parquet (6.9M rows × 96 features)
    #    - test/features.parquet (7.3M rows × 96 features)
    #    - oot/features.parquet (13.7M rows × 96 features)
    # 4. Save feature metadata:
    #    - feature_columns.txt (list of 96 feature names)

    # Expected output:
    # - Original features: 12
    # - Rolling means: 36 (12 features × 3 windows)
    # - Rolling stds: 36 (12 features × 3 windows)
    # - Acceleration: 12 (12 features × 1)
    # - Total: 96 engineered features + 5 metadata columns = 101 columns

    # Example engineered features:
    # - Fan Speed_mean_short
    # - Fan Speed_mean_medium
    # - Fan Speed_mean_long
    # - Fan Speed_std_short
    # - Fan Speed_std_medium
    # - Fan Speed_std_long
    # - Fan Speed_acceleration

    # Call processing function (to be implemented in utils)
    # data_processing_gold_feature_xgboost.process_gold_feature_xgboost(
    #     gold_label_base_dir,
    #     gold_feature_xgboost_dir,
    #     snapshot_date_str,
    #     SELECTED_FEATURES,
    #     WINDOWS
    # )

    print("\n\n--- Completed Gold XGBoost Feature N-CMAPSS job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gold XGBoost Feature engineering for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
