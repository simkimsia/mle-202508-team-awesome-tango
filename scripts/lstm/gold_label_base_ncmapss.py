"""
Gold Layer - Base Label Store: Create RUL_Clipped labels for temporal splitting
Usage:
    python3 gold_label_base_ncmapss.py --snapshotdate "2023-01-01" [--no-partition]
Options:
    --snapshotdate: Required. Snapshot date in YYYY-MM-DD format
    --no-partition: Optional. Disable partitioning (saves as single file, requires more memory)
Note:
    By default, data is partitioned by 'dataset' to keep files small and memory-friendly.
    This is recommended for laptops with limited resources.
"""
import os
import argparse
from utils import data_processing_gold_label_base
def main(snapshotdate, use_partitioning=True):
    print("\n\n--- Starting Gold Base Label N-CMAPSS job ---\n\n")
    snapshot_date_str = snapshotdate
    silver_dir = f"../datamart/silver/n_cmapss/snapshot_date={snapshot_date_str}"
    gold_label_base_dir = f"../datamart/gold/label_base/n_cmapss"
    print(f"Input directory: {silver_dir}")
    print(f"Output directory: {gold_label_base_dir}")
    data_processing_gold_label_base.process_gold_label_base(
        silver_dir,
        gold_label_base_dir,
        snapshot_date_str,
        use_partitioning=use_partitioning
    )
    print("\n\n--- Completed Gold Base Label N-CMAPSS job ---\n\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gold Base Label creation for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--no-partition", action="store_true",
                       help="Disable partitioning (saves single file, requires more memory)")
    args = parser.parse_args()
    main(args.snapshotdate, use_partitioning=not args.no_partition)