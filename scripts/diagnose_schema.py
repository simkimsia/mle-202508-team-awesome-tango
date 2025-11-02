#!/usr/bin/env python3
"""
Diagnostic script to check parquet schema for both dates
"""
import os
import pyarrow.parquet as pq
import pandas as pd

def check_schema(snapshot_date):
    """Check the schema for a given snapshot date"""
    print(f"\n{'='*60}")
    print(f"Checking schema for {snapshot_date}")
    print(f"{'='*60}")

    base_dir = f"/opt/airflow/datamart/gold/label_base/n_cmapss/snapshot_date={snapshot_date}"

    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return

    print(f"✓ Found directory: {base_dir}")

    # Check for single file
    single_file = os.path.join(base_dir, "data.parquet")
    if os.path.exists(single_file):
        print(f"✓ Found single file: {single_file}")
        table = pq.read_table(single_file)
        print(f"\nSingle file schema:")
        print(table.schema)
        return

    # Check for partitioned data
    partitions = sorted([d for d in os.listdir(base_dir) if d.startswith("dataset=")])
    if partitions:
        print(f"✓ Found {len(partitions)} partitions: {partitions}")

        for partition in partitions:
            partition_file = os.path.join(base_dir, partition, "data.parquet")
            if os.path.exists(partition_file):
                print(f"\n  Partition: {partition}")
                print(f"  File: {partition_file}")

                # Read with pyarrow to see actual schema
                table = pq.read_table(partition_file)
                print(f"  Schema in file:")
                for field in table.schema:
                    print(f"    - {field.name}: {field.type}")

                # Check if dataset column exists and its type
                if 'dataset' in table.column_names:
                    dataset_col = table.column('dataset')
                    print(f"  Dataset column type: {dataset_col.type}")
                    print(f"  Dataset unique values: {dataset_col.unique().to_pylist()}")

        # Now try reading the entire directory and see what happens
        print(f"\n  Attempting to read entire partitioned directory...")
        try:
            df = pd.read_parquet(base_dir)
            print(f"  ✓ Successfully read {len(df):,} rows")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Dataset dtype: {df['dataset'].dtype}")
        except Exception as e:
            print(f"  ❌ Error reading partitioned directory:")
            print(f"     {type(e).__name__}: {e}")

if __name__ == "__main__":
    check_schema("2025-01-01")
    check_schema("2025-01-02")
