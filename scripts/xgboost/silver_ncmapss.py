import argparse
import os

import pyspark

from utils import data_processing_silver_ncmapss

def main(snapshotdate):
    print("\n\n--- Starting Silver job ---\n\n")
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
    snapshot_date_str = snapshotdate
    bronze_dir = f"../datamart/bronze/n_cmapss/snapshot_date={snapshot_date_str}"
    silver_dir = f"../datamart/silver/n_cmapss/"
    os.makedirs(silver_dir, exist_ok=True)
    data_processing_silver_ncmapss.process_silver_table(
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