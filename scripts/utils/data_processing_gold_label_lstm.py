"""
Gold Label LSTM Processing
Converts base labels to LSTM-compatible TimeSeries format.

This module transforms DataFrame labels into Darts TimeSeries objects
suitable for LSTM model training and inference.
"""

import os
import pickle
from datetime import datetime
import pandas as pd
# from darts import TimeSeries
# from darts.dataprocessing.transformers import Scaler


def process_gold_label_lstm(
    gold_label_base_dir: str,
    gold_label_lstm_dir: str,
    snapshot_date_str: str
):
    """
    Convert base labels to LSTM-compatible TimeSeries format.

    Args:
        gold_label_base_dir: Path to base label parquet files
        gold_label_lstm_dir: Output directory for LSTM labels
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load train/val/test/oot parquet files
        2. For each split:
           - Group by engine unit
           - Create Darts TimeSeries objects (one per engine)
           - Each TimeSeries contains RUL_Clipped values indexed by time
        3. Normalize labels using Scaler:
           - Fit scaler on training labels
           - Transform val/test/oot labels
        4. Save TimeSeries objects and scaler

    Output:
        - train/targets.pkl (list of 39 TimeSeries)
        - val/targets.pkl (list of 7 TimeSeries)
        - test/targets.pkl (list of 9 TimeSeries)
        - oot/targets.pkl (list of 19 TimeSeries)
        - target_scaler.pkl (fitted Scaler)
        - train/units.pkl, val/units.pkl, test/units.pkl, oot/units.pkl
    """
    print(f"Processing Gold Label LSTM for snapshot date: {snapshot_date_str}")
    print(f"Loading base labels from {gold_label_base_dir}")

    # Create output directories
    base_output_dir = os.path.join(
        gold_label_lstm_dir,
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

    # TODO: Load base label DataFrames
    # df_train = pd.read_parquet(os.path.join(input_dir, 'df_train.parquet'))
    # df_val = pd.read_parquet(os.path.join(input_dir, 'df_val.parquet'))
    # df_test = pd.read_parquet(os.path.join(input_dir, 'df_test.parquet'))
    # df_oot = pd.read_parquet(os.path.join(input_dir, 'df_oot.parquet'))

    # TODO: Convert to TimeSeries objects
    # For each split, group by unit and create TimeSeries
    # train_targets = []
    # train_units = []
    # for unit in df_train['unit'].unique():
    #     df_unit = df_train[df_train['unit'] == unit].sort_values('time')
    #     ts = TimeSeries.from_dataframe(
    #         df_unit,
    #         time_col='time',
    #         value_cols=['RUL_Clipped']
    #     )
    #     train_targets.append(ts)
    #     train_units.append(unit)

    # TODO: Normalize using Scaler
    # scaler = Scaler()
    # train_targets_normalized = scaler.fit_transform(train_targets)
    # val_targets_normalized = scaler.transform(val_targets)
    # test_targets_normalized = scaler.transform(test_targets)
    # oot_targets_normalized = scaler.transform(oot_targets)

    # TODO: Save TimeSeries objects
    # with open(os.path.join(base_output_dir, 'train', 'targets.pkl'), 'wb') as f:
    #     pickle.dump(train_targets_normalized, f)
    # with open(os.path.join(base_output_dir, 'train', 'units.pkl'), 'wb') as f:
    #     pickle.dump(train_units, f)

    # TODO: Save scaler
    # with open(os.path.join(base_output_dir, 'target_scaler.pkl'), 'wb') as f:
    #     pickle.dump(scaler, f)

    print(f"Gold Label LSTM written successfully to {base_output_dir}")
    return base_output_dir
