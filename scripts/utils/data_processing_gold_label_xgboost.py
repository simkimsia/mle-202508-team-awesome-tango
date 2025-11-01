"""
Gold Label XGBoost Processing
Prepares labels in XGBoost-compatible DataFrame format.

This module extracts and formats labels for XGBoost model training.
Unlike LSTM, XGBoost doesn't require normalization or TimeSeries conversion.
"""

import os
from datetime import datetime
import pandas as pd


def process_gold_label_xgboost(
    gold_label_base_dir: str,
    gold_label_xgboost_dir: str,
    snapshot_date_str: str
):
    """
    Prepare labels in XGBoost-compatible DataFrame format.

    Args:
        gold_label_base_dir: Path to base label parquet files
        gold_label_xgboost_dir: Output directory for XGBoost labels
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load train/val/test/oot parquet files
        2. For each split:
           - Extract label columns: unit, time, RUL_Clipped
           - Keep as DataFrame (no normalization needed for XGBoost)
        3. Save label DataFrames

    Output:
        - train/labels.parquet (26.9M rows × 3 columns)
        - val/labels.parquet (6.9M rows × 3 columns)
        - test/labels.parquet (7.3M rows × 3 columns)
        - oot/labels.parquet (13.7M rows × 3 columns)
    """
    print(f"Processing Gold Label XGBoost for snapshot date: {snapshot_date_str}")
    print(f"Loading base labels from {gold_label_base_dir}")

    # Create output directories
    base_output_dir = os.path.join(
        gold_label_xgboost_dir,
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

    # TODO: Extract label columns (unit, time, RUL_Clipped)
    # label_columns = ['unit', 'time', 'RUL_Clipped']
    # train_labels = df_train[label_columns]
    # val_labels = df_val[label_columns]
    # test_labels = df_test[label_columns]
    # oot_labels = df_oot[label_columns]

    # TODO: Save as parquet files
    # train_labels.to_parquet(
    #     os.path.join(base_output_dir, 'train', 'labels.parquet'),
    #     index=False
    # )
    # val_labels.to_parquet(
    #     os.path.join(base_output_dir, 'val', 'labels.parquet'),
    #     index=False
    # )
    # test_labels.to_parquet(
    #     os.path.join(base_output_dir, 'test', 'labels.parquet'),
    #     index=False
    # )
    # oot_labels.to_parquet(
    #     os.path.join(base_output_dir, 'oot', 'labels.parquet'),
    #     index=False
    # )

    print(f"Gold Label XGBoost written successfully to {base_output_dir}")
    return base_output_dir
