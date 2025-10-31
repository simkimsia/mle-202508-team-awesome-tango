#!/usr/bin/env python3
"""
Driver script to create Silver table from Bronze data.
Usage:
    python3 silver_table_1.py --snapshotdate "2023-01-01"
"""

import os
import argparse
import pyspark
from utils import data_processing_silver_table

def main(snapshotdate):
    print("\n\n--- Starting Silver job ---\n\n")

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_table_job") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    snapshot_date_str = snapshotdate
    bronze_dir = f"datamart/bronze/n_cmapss/snapshot_date={snapshot_date_str}"
    silver_dir = f"datamart/silver/n_cmapss/"

    os.makedirs(silver_dir, exist_ok=True)

    data_processing_silver_table.process_silver_table(
        spark,
        bronze_dir,
        silver_dir,
        snapshot_date_str
    )

    spark.stop()
    print("\n\n--- Completed Silver job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Silver data transformation")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
