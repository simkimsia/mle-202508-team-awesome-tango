"""
Gold Feature LSTM Processing
Creates LSTM-compatible time series features.

This module transforms raw sensor data into normalized TimeSeries objects
suitable for LSTM model training with sequence modeling.
"""

import os
import pickle
from datetime import datetime
import pandas as pd
# from darts import TimeSeries
# from darts.dataprocessing.transformers import Scaler


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


def process_gold_feature_lstm(
    gold_label_base_dir: str,
    gold_feature_lstm_dir: str,
    snapshot_date_str: str
):
    """
    Create LSTM-compatible time series features.

    Args:
        gold_label_base_dir: Path to base label parquet files
        gold_feature_lstm_dir: Output directory for LSTM features
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load train/val/test/oot DataFrames
        2. Select 12 features from SELECTED_FEATURES
        3. For each split:
           - Group by engine unit
           - Create Darts TimeSeries objects (one per engine)
           - Each TimeSeries contains 12 feature values indexed by time
        4. Normalize features using Scaler:
           - Fit scaler on training features
           - Transform val/test/oot features
        5. Save TimeSeries objects and scaler

    Output:
        - train/covariates.pkl (list of 39 TimeSeries, shape: (time_steps, 12))
        - val/covariates.pkl (list of 7 TimeSeries)
        - test/covariates.pkl (list of 9 TimeSeries)
        - oot/covariates.pkl (list of 19 TimeSeries)
        - feature_scaler.pkl (fitted Scaler)
        - train/units.pkl, val/units.pkl, test/units.pkl, oot/units.pkl
    """
    print(f"Processing Gold Feature LSTM for snapshot date: {snapshot_date_str}")
    print(f"Loading base labels from {gold_label_base_dir}")

    # Create output directories
    base_output_dir = os.path.join(
        gold_feature_lstm_dir,
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

    # TODO: Select features
    # feature_cols = SELECTED_FEATURES

    # TODO: Convert to TimeSeries objects
    # For each split, group by unit and create TimeSeries with selected features
    # train_covariates = []
    # train_units = []
    # for unit in df_train['unit'].unique():
    #     df_unit = df_train[df_train['unit'] == unit].sort_values('time')
    #     ts = TimeSeries.from_dataframe(
    #         df_unit,
    #         time_col='time',
    #         value_cols=feature_cols
    #     )
    #     train_covariates.append(ts)
    #     train_units.append(unit)

    # TODO: Normalize using Scaler
    # scaler = Scaler()
    # train_covariates_normalized = scaler.fit_transform(train_covariates)
    # val_covariates_normalized = scaler.transform(val_covariates)
    # test_covariates_normalized = scaler.transform(test_covariates)
    # oot_covariates_normalized = scaler.transform(oot_covariates)

    # TODO: Save TimeSeries objects
    # with open(os.path.join(base_output_dir, 'train', 'covariates.pkl'), 'wb') as f:
    #     pickle.dump(train_covariates_normalized, f)
    # with open(os.path.join(base_output_dir, 'train', 'units.pkl'), 'wb') as f:
    #     pickle.dump(train_units, f)

    # TODO: Save scaler
    # with open(os.path.join(base_output_dir, 'feature_scaler.pkl'), 'wb') as f:
    #     pickle.dump(scaler, f)

    print(f"Gold Feature LSTM written successfully to {base_output_dir}")
    return base_output_dir
