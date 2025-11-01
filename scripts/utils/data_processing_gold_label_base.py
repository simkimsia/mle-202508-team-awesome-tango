"""
Gold Label Base Processing
Creates RUL_Clipped labels and performs train/val/test/oot splits.

This module processes silver layer data to create the base labels that will be
used by both LSTM and XGBoost models. It implements the splitting strategy
defined in the pipeline architecture.
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np


# Configuration
RUL_CLIP_MAX = 90
TRAIN_SETS = [1, 3, 4, 5, 6]  # Internal datasets
OOT_SETS = [2, 7]              # Out-of-time datasets


def process_gold_label_base(
    silver_dir: str,
    gold_label_base_dir: str,
    snapshot_date_str: str
):
    """
    Process silver data to create base labels with train/val/test/oot splits.

    Args:
        silver_dir: Path to silver layer parquet files
        gold_label_base_dir: Output directory for base labels
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load silver DataFrame
        2. Create RUL_Clipped = clip(Remaining Useful Life, upper=90)
        3. Split data by engine units:
           - Separate OOT sets (datasets 2 & 7)
           - Internal split: 85% train+val, 15% test
           - Train/val split: 85% train, 15% val
        4. Save split DataFrames as parquet
        5. Save split metadata (unit lists)

    Output:
        - df_train.parquet (~39 engines, 26.9M rows)
        - df_val.parquet (~7 engines, 6.9M rows)
        - df_test.parquet (~9 engines, 7.3M rows)
        - df_oot.parquet (~19 engines, 13.7M rows)
        - split_metadata.json (unit lists for each split)
    """
    print(f"Processing Gold Label Base for snapshot date: {snapshot_date_str}")
    print(f"Loading silver data from {silver_dir}")

    # Create output directory
    output_dir = os.path.join(
        gold_label_base_dir,
        "n_cmapss",
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # TODO: Load silver DataFrame
    # df = pd.read_parquet(silver_dir)

    # TODO: Create RUL_Clipped column
    # df['RUL_Clipped'] = np.clip(df['Remaining_Useful_Life'], 0, RUL_CLIP_MAX)

    # TODO: Split by datasets (OOT vs Internal)
    # df_oot = df[df['dataset'].isin(OOT_SETS)]
    # df_internal = df[df['dataset'].isin(TRAIN_SETS)]

    # TODO: Split internal data into train/val/test
    # Get unique units
    # units_internal = df_internal['unit'].unique()
    # Perform 85/15 split for test
    # Perform 85/15 split on remaining for train/val

    # TODO: Save DataFrames
    # df_train.to_parquet(os.path.join(output_dir, 'df_train.parquet'))
    # df_val.to_parquet(os.path.join(output_dir, 'df_val.parquet'))
    # df_test.to_parquet(os.path.join(output_dir, 'df_test.parquet'))
    # df_oot.to_parquet(os.path.join(output_dir, 'df_oot.parquet'))

    # TODO: Save split metadata
    # split_metadata = {
    #     'train_units': train_units.tolist(),
    #     'val_units': val_units.tolist(),
    #     'test_units': test_units.tolist(),
    #     'oot_units': oot_units.tolist()
    # }
    # with open(os.path.join(output_dir, 'split_metadata.json'), 'w') as f:
    #     json.dump(split_metadata, f, indent=2)

    print(f"Gold Label Base written successfully to {output_dir}")
    return output_dir
