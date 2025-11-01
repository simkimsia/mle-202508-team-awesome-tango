"""
Gold Feature XGBoost Processing
Creates time-windowed features for XGBoost.

This module transforms raw sensor data into engineered features using
rolling statistics across multiple time windows for tabular modeling.
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
import gc


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
    'short': 5,    # Last 5 cycles
    'medium': 15,  # Last 15 cycles
    'long': 30     # Last 30 cycles
}


def process_gold_feature_xgboost(
    gold_label_base_dir: str,
    gold_feature_xgboost_dir: str,
    snapshot_date_str: str
):
    """
    Create time-windowed features for XGBoost.

    Args:
        gold_label_base_dir: Path to base label parquet files
        gold_feature_xgboost_dir: Output directory for XGBoost features
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load train/val/test/oot DataFrames
        2. Convert features to float32 (memory optimization)
        3. For each of 12 features, create:
           - Rolling mean (3 windows) → 36 features
           - Rolling std (3 windows) → 36 features
           - Acceleration (mean_short - mean_medium) → 12 features
        4. Process features in batches with garbage collection
        5. Save engineered DataFrames

    Output:
        - train/features.parquet (26.9M rows × 96 features, float32)
        - val/features.parquet (6.9M rows × 96 features)
        - test/features.parquet (7.3M rows × 96 features)
        - oot/features.parquet (13.7M rows × 96 features)
        - feature_columns.txt (list of 96 feature names)
    """
    print(f"Processing Gold Feature XGBoost for snapshot date: {snapshot_date_str}")
    print(f"Loading base labels from {gold_label_base_dir}")

    # Create output directories
    base_output_dir = os.path.join(
        gold_feature_xgboost_dir,
        "n_cmapss",
        f"snapshot_date={snapshot_date_str}"
    )

    for split in ['train', 'val', 'test', 'oot']:
        os.makedirs(os.path.join(base_output_dir, split), exist_ok=True)

    # Input directory
    input_dir = os.path.join(
        gold_label_base_dir,
        "n_cmapss",
        f"snapshot_date={snapshot_date_str}"
    )

    # TODO: Load base DataFrames
    # df_train = pd.read_parquet(os.path.join(input_dir, 'df_train.parquet'))
    # df_val = pd.read_parquet(os.path.join(input_dir, 'df_val.parquet'))
    # df_test = pd.read_parquet(os.path.join(input_dir, 'df_test.parquet'))
    # df_oot = pd.read_parquet(os.path.join(input_dir, 'df_oot.parquet'))

    # TODO: Convert selected features to float32
    # for col in SELECTED_FEATURES:
    #     df_train[col] = df_train[col].astype('float32')

    # TODO: Create rolling features
    # feature_dfs = []
    # feature_columns = []
    #
    # For each feature and window:
    # - Create rolling mean
    # - Create rolling std
    # - Create acceleration (difference between window means)
    #
    # for feature in SELECTED_FEATURES:
    #     for window_name, window_size in WINDOWS.items():
    #         # Rolling mean
    #         col_name = f"{feature}_mean_{window_name}"
    #         df_train[col_name] = df_train.groupby('unit')[feature].transform(
    #             lambda x: x.rolling(window=window_size, min_periods=1).mean()
    #         )
    #         feature_columns.append(col_name)
    #
    #         # Rolling std
    #         col_name = f"{feature}_std_{window_name}"
    #         df_train[col_name] = df_train.groupby('unit')[feature].transform(
    #             lambda x: x.rolling(window=window_size, min_periods=1).std()
    #         )
    #         feature_columns.append(col_name)
    #
    # # Acceleration features
    # for feature in SELECTED_FEATURES:
    #     col_name = f"{feature}_acceleration"
    #     df_train[col_name] = (
    #         df_train[f"{feature}_mean_short"] - df_train[f"{feature}_mean_medium"]
    #     )
    #     feature_columns.append(col_name)

    # TODO: Extract only engineered feature columns
    # train_features = df_train[feature_columns].astype('float32')
    # val_features = df_val[feature_columns].astype('float32')
    # test_features = df_test[feature_columns].astype('float32')
    # oot_features = df_oot[feature_columns].astype('float32')

    # TODO: Save as parquet files
    # train_features.to_parquet(
    #     os.path.join(base_output_dir, 'train', 'features.parquet'),
    #     index=False
    # )
    # val_features.to_parquet(
    #     os.path.join(base_output_dir, 'val', 'features.parquet'),
    #     index=False
    # )
    # test_features.to_parquet(
    #     os.path.join(base_output_dir, 'test', 'features.parquet'),
    #     index=False
    # )
    # oot_features.to_parquet(
    #     os.path.join(base_output_dir, 'oot', 'features.parquet'),
    #     index=False
    # )

    # TODO: Save feature column names
    # with open(os.path.join(base_output_dir, 'feature_columns.txt'), 'w') as f:
    #     f.write('\n'.join(feature_columns))

    print(f"Gold Feature XGBoost written successfully to {base_output_dir}")
    return base_output_dir
