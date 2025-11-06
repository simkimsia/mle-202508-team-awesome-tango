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
        1. Load data from gold_label_base (handles both partitioned and single file)
        2. Extract label columns: unit, time, RUL_Clipped
        3. Save as single labels.parquet for this snapshot date
    Output:
        - labels.parquet (all data for this snapshot date)
    Note:
        Temporal splitting (train/val/test/oot) is handled at DAG level via
        snapshot date selection, not via separate files.
    """
    print(f"Processing Gold Label XGBoost for snapshot date: {snapshot_date_str}")
    print(f"Loading base labels from {gold_label_base_dir}")
    output_dir = os.path.join(
        gold_label_xgboost_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)
    input_dir = os.path.join(
        gold_label_base_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    single_file_path = os.path.join(input_dir, 'data.parquet')
    if os.path.exists(single_file_path):
        print(f"Loading data from {single_file_path}")
        df = pd.read_parquet(single_file_path)
    else:
        print(f"Loading partitioned data from {input_dir}")
        df = pd.read_parquet(input_dir)
    print(f"Loaded {len(df):,} rows")
    label_columns = ['unit', 'time', 'RUL_Clipped', 'dataset']
    missing = [col for col in label_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")
    print(f"Extracting label columns: {label_columns}")
    labels = df[label_columns]
    output_path = os.path.join(output_dir, 'labels.parquet')
    print(f"Saving labels to {output_path}...")
    labels.to_parquet(output_path, index=False)
    print(f"  ✓ Saved {len(labels):,} rows")
    print(f"Gold Label XGBoost written successfully to {output_dir}")
    return output_dir