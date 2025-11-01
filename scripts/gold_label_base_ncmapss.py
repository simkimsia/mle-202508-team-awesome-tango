#!/usr/bin/env python3
"""
Gold Layer - Base Label Store: Create RUL_Clipped labels for temporal splitting
Usage:
    python3 gold_label_base_ncmapss.py --snapshotdate "2023-01-01"
"""

import os
import argparse
from utils import data_processing_gold_label_base


def main(snapshotdate):
    print("\n\n--- Starting Gold Base Label N-CMAPSS job ---\n\n")

    # Define input/output directories
    snapshot_date_str = snapshotdate
    silver_dir = f"datamart/silver/n_cmapss/snapshot_date={snapshot_date_str}"
    gold_label_base_dir = f"datamart/gold/label_base/n_cmapss"

    print(f"Input directory: {silver_dir}")
    print(f"Output directory: {gold_label_base_dir}")

    # Process gold label base layer
    # Creates RUL_Clipped column and saves complete dataset for this snapshot date
    # Temporal splitting (train/val/test/oot) is handled at DAG level
    data_processing_gold_label_base.process_gold_label_base(
        silver_dir,
        gold_label_base_dir,
        snapshot_date_str
    )

    print("\n\n--- Completed Gold Base Label N-CMAPSS job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gold Base Label creation for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
