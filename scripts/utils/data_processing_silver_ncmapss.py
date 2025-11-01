import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType, StringType, ArrayType, DateType
)


# --------------------------------
# Schema Definition (example)
# --------------------------------
def process_silver_table(spark: SparkSession, bronze_dir: str, silver_dir: str, snapshot_date_str: str):
    print(f"Reading Bronze data from {bronze_dir}")
    df_bronze = spark.read.parquet(bronze_dir)
    print(f"Bronze rows: {df_bronze.count()}")

    df_expanded = df_bronze \
        .withColumn("Altitude", F.col("W")[0]) \
        .withColumn("Mach_Number", F.col("W")[1]) \
        .withColumn("TRA", F.col("W")[2]) \
        .withColumn("T2", F.col("W")[3]) \
        .withColumn("T24", F.col("X_s")[0]) \
        .withColumn("T30", F.col("X_s")[1]) \
        .withColumn("T48", F.col("X_s")[2]) \
        .withColumn("T50", F.col("X_s")[3]) \
        .withColumn("P15", F.col("X_s")[4]) \
        .withColumn("P2", F.col("X_s")[5]) \
        .withColumn("P21", F.col("X_s")[6]) \
        .withColumn("P24", F.col("X_s")[7]) \
        .withColumn("Ps30", F.col("X_s")[8]) \
        .withColumn("P40", F.col("X_s")[9]) \
        .withColumn("P50", F.col("X_s")[10]) \
        .withColumn("Nf", F.col("X_s")[11]) \
        .withColumn("Nc", F.col("X_s")[12]) \
        .withColumn("Wf", F.col("X_s")[13]) \
        .withColumn("T40", F.col("X_v")[0]) \
        .withColumn("P30", F.col("X_v")[1]) \
        .withColumn("P45", F.col("X_v")[2]) \
        .withColumn("W21", F.col("X_v")[3]) \
        .withColumn("W22", F.col("X_v")[4]) \
        .withColumn("W25", F.col("X_v")[5]) \
        .withColumn("W31", F.col("X_v")[6]) \
        .withColumn("W32", F.col("X_v")[7]) \
        .withColumn("W48", F.col("X_v")[8]) \
        .withColumn("W50", F.col("X_v")[9]) \
        .withColumn("SmFan", F.col("X_v")[10]) \
        .withColumn("SmLPC", F.col("X_v")[11]) \
        .withColumn("SmHPC", F.col("X_v")[12]) \
        .withColumn("phi", F.col("X_v")[13]) \
        .withColumn("T1", F.col("T")[0]) \
        .withColumn("T2_deg", F.col("T")[1]) \
        .withColumn("T3_deg", F.col("T")[2]) \
        .withColumn("dataset", F.col("A")[0].cast("int")) \
        .withColumn("unit_orig", F.col("A")[1].cast("int")) \
        .withColumn("time", F.col("A")[2].cast("int")) \
        .withColumn("Fc", F.col("A")[3].cast("int")) \
        .withColumn("hs", F.col("A")[4].cast("int")) \
        .withColumn("Remaining_Useful_Life", F.col("Y")[0]) \
        .withColumn("snapshot_date", F.to_date(F.col("snapshot_date"), "yyyy-MM-dd")) \
        .select(
            "dataset", "unit_orig", "time", "Fc", "hs",
            "Altitude", "Mach_Number", "TRA", "T2",
            "T24", "T30", "T48", "T50",
            "P15", "P2", "P21", "P24", "Ps30", "P40", "P50",
            "Nf", "Nc", "Wf",
            "T40", "P30", "P45", "W21", "W22", "W25", "W31", "W32", "W48", "W50", "SmFan", "SmLPC", "SmHPC", "phi",
            "T1", "T2_deg", "T3_deg",
            "Remaining_Useful_Life",
            "snapshot_date"
        )

    # Write parquet with consistent schema
    # Disable dictionary encoding to prevent schema conflicts downstream
    df_expanded.write\
        .mode("overwrite")\
        .option("partitionOverwriteMode", "dynamic")\
        .option("parquet.enable.dictionary", "false")\
        .option("compression", "snappy")\
        .partitionBy("snapshot_date")\
        .parquet(silver_dir)
    print(f"Silver table written successfully to {silver_dir}")
    return silver_dir