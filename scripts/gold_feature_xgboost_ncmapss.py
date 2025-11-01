#!/usr/bin/env python3
"""
Gold Layer - XGBoost Feature Store: Create windowed features for XGBoost model
Usage:
    python3 gold_feature_xgboost_ncmapss.py --snapshotdate "2023-01-01"
"""

import os
import argparse
from utils import data_processing_gold_feature_xgboost


def main(snapshotdate):
    print("\n\n--- Starting Gold LSTM Feature N-CMAPSS job ---\n\n")

    # Define input/output directories
    snapshot_date_str = snapshotdate
    gold_label_base_dir = f"datamart/gold/label_base/n_cmapss"
    gold_feature_xgboost_dir = f"datamart/gold/feature/xgboost/n_cmapss"

    print(f"Input directory: {gold_label_base_dir}")
    print(f"Output directory: {gold_feature_xgboost_dir}")

    # Process XGBoost features
    # Creates TimeSeries objects with selected features for this snapshot date
    # Temporal splitting (train/val/test/oot) is handled at DAG level
    data_processing_gold_feature_xgboost.process_gold_feature_lstm(
        gold_label_base_dir,
        gold_feature_lstm_dir,
        snapshot_date_str
    )

    print("\n\n--- Completed Gold XGBoost Feature N-CMAPSS job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gold XGBoost Feature engineering for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
