#!/usr/bin/env python3
"""
Gold Layer - XGBoost Label Store: Extract labels in DataFrame format for XGBoost
Usage:
    python3 gold_label_xgboost_ncmapss.py --snapshotdate "2023-01-01"
"""

import os
import argparse
import pandas as pd
# from utils import data_processing_gold_label_xgboost


def main(snapshotdate):
    print("\n\n--- Starting Gold XGBoost Label N-CMAPSS job ---\n\n")

    # Define input/output directories
    snapshot_date_str = snapshotdate
    gold_label_base_dir = f"datamart/gold/label_base/n_cmapss/snapshot_date={snapshot_date_str}"
    gold_label_xgboost_dir = f"datamart/gold/label/xgboost/n_cmapss/snapshot_date={snapshot_date_str}"

    # Create output directories for each split
    for split in ['train', 'val', 'test', 'oot']:
        os.makedirs(f"{gold_label_xgboost_dir}/{split}", exist_ok=True)

    print(f"Input directory: {gold_label_base_dir}")
    print(f"Output directory: {gold_label_xgboost_dir}")

    # TODO: Implement XGBoost label extraction
    # Tasks:
    # 1. Load base label DataFrames (df_train, df_val, df_test, df_oot)
    # 2. For each split:
    #    a. Extract relevant columns: unit, time, RUL_Clipped
    #    b. Keep as DataFrame (no conversion needed)
    #    c. No normalization (XGBoost handles raw values)
    # 3. Save outputs:
    #    - train/labels.parquet (26.9M rows × 3 columns)
    #    - val/labels.parquet (6.9M rows × 3 columns)
    #    - test/labels.parquet (7.3M rows × 3 columns)
    #    - oot/labels.parquet (13.7M rows × 3 columns)

    # Expected output structure:
    # DataFrame with columns:
    # - unit: str (e.g., "DS01_001")
    # - time: int (cycle number)
    # - RUL_Clipped: float (0-90)

    # Example:
    # df_train_labels = df_train[['unit', 'time', 'RUL_Clipped']]
    # df_train_labels.to_parquet(f"{gold_label_xgboost_dir}/train/labels.parquet")

    # Call processing function (to be implemented in utils)
    # data_processing_gold_label_xgboost.process_gold_label_xgboost(
    #     gold_label_base_dir,
    #     gold_label_xgboost_dir,
    #     snapshot_date_str
    # )

    print("\n\n--- Completed Gold XGBoost Label N-CMAPSS job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gold XGBoost Label extraction for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
