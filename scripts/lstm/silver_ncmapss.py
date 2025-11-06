import argparse
import os

import pyspark

from utils import data_processing_silver_ncmapss

def main(snapshotdate):
    print(f"\n[SILVER-LSTM] ===== Starting Silver Transformation Job =====")
    print(f"[SILVER-LSTM] Snapshot Date: {snapshotdate}")
    print(f"[SILVER-LSTM] Cleaning and standardizing Bronze data to Silver layer")
    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_table_job") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print(f"[SILVER-LSTM] Initialized Spark session with 4GB memory allocation")
    snapshot_date_str = snapshotdate
    bronze_dir = f"../datamart/bronze/n_cmapss/snapshot_date={snapshot_date_str}"
    silver_dir = f"../datamart/silver/n_cmapss/"
    os.makedirs(silver_dir, exist_ok=True)
    print(f"[SILVER-LSTM] Input: {bronze_dir}")
    print(f"[SILVER-LSTM] Output: {silver_dir}")
    print(f"[SILVER-LSTM] Processing data quality checks and normalization...")
    data_processing_silver_ncmapss.process_silver_table(
        spark,
        bronze_dir,
        silver_dir,
        snapshot_date_str
    )
    spark.stop()
    print(f"[SILVER-LSTM] Stopped Spark session")
    print(f"[SILVER-LSTM] ===== Completed Silver Transformation Job =====\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Silver layer data cleaning and standardization")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()
    main(args.snapshotdate)