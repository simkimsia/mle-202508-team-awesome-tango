"""
Gold Label Base Processing
Creates RUL_Clipped labels for temporal splitting strategy.

This module processes silver layer data to create the base labels that will be
used by both LSTM and XGBoost models. Each snapshot date is processed independently,
and temporal splitting (train/val/test/oot) is handled at the DAG level by selecting
different snapshot dates.
"""

import os
import pandas as pd
import numpy as np


# Configuration
RUL_CLIP_MAX = 90


def process_gold_label_base(
    silver_dir: str,
    gold_label_base_dir: str,
    snapshot_date_str: str
):
    """
    Process silver data to create base labels with RUL_Clipped column.

    Args:
        silver_dir: Path to silver layer parquet files for this snapshot date
        gold_label_base_dir: Output directory for base labels
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load silver DataFrame for the snapshot date
        2. Create RUL_Clipped = clip(Remaining_Useful_Life, upper=90)
        3. Save complete dataset as parquet

    Output:
        - data.parquet (complete dataset with RUL_Clipped for this snapshot date)

    Note:
        Train/val/test/oot splitting is done temporally at the DAG level:
        - Train: Earlier snapshot dates (e.g., Jan 1-3)
        - Val: Middle snapshot date (e.g., Jan 4)
        - Test: Middle snapshot date (e.g., Jan 5)
        - OOT: Later snapshot dates (e.g., Jan 6+)
    """
    print(f"Processing Gold Label Base for snapshot date: {snapshot_date_str}")
    print(f"Loading silver data from {silver_dir}")

    # Load silver DataFrame
    df = pd.read_parquet(silver_dir)
    print(f"Loaded {len(df):,} rows from silver layer")

    # Create RUL_Clipped column
    df['RUL_Clipped'] = np.clip(df['Remaining_Useful_Life'], 0, RUL_CLIP_MAX)
    print(f"Created RUL_Clipped column (clipped at {RUL_CLIP_MAX})")

    # Create output directory
    output_dir = os.path.join(
        gold_label_base_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save complete dataset
    output_path = os.path.join(output_dir, 'data.parquet')
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")

    print(f"Gold Label Base written successfully to {output_dir}")
    return output_dir
