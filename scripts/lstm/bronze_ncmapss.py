from datetime import datetime
import argparse
import os

import utils.data_processing_bronze_ncmapss

def main(snapshotdate):
    print(f'\n[BRONZE-LSTM] ===== Starting Bronze Ingestion Job =====')
    print(f'[BRONZE-LSTM] Snapshot Date: {snapshotdate}')
    print(f'[BRONZE-LSTM] Processing raw H5 data to Bronze layer (Parquet format)')
    snapshot_date_str = snapshotdate
    bronze_lms_directory = "../datamart/bronze/"
    if not os.path.exists(bronze_lms_directory):
        os.makedirs(bronze_lms_directory)
        print(f'[BRONZE-LSTM] Created bronze directory: {bronze_lms_directory}')
    print(f'[BRONZE-LSTM] Calling bronze processing utility...')
    utils.data_processing_bronze_ncmapss.process_bronze_table(
        snapshot_date_str,
        bronze_lms_directory
    )
    print(f'[BRONZE-LSTM] ===== Completed Bronze Ingestion Job =====\n')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Bronze layer ingestion from H5 to Parquet")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()
    main(args.snapshotdate)