"""
Gold Label Base Processing
Creates RUL_Clipped labels for temporal splitting strategy.

This module processes silver layer data to create the base labels that will be
used by both LSTM and XGBoost models. Each snapshot date is processed independently,
and temporal splitting (train/val/test/oot) is handled at the DAG level by selecting
different snapshot dates.

Supports batch processing for memory-constrained environments.
"""

import os
import pandas as pd
import numpy as np
import glob


# Configuration
RUL_CLIP_MAX = 90


def process_gold_label_base(
    silver_dir: str,
    gold_label_base_dir: str,
    snapshot_date_str: str,
    use_partitioning: bool = True
):
    """
    Process silver data to create base labels with RUL_Clipped column.

    Args:
        silver_dir: Path to silver layer parquet files for this snapshot date
        gold_label_base_dir: Output directory for base labels
        snapshot_date_str: Snapshot date in YYYY-MM-DD format
        use_partitioning: If True, keeps data partitioned by dataset for memory efficiency.
                         If False, saves as single file. Default: True

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load silver parquet files (supports both single file and directory)
        2. Process in batches if partitioned
        3. Create RUL_Clipped = clip(Remaining_Useful_Life, upper=90)
        4. Save with partitioning by 'dataset' to keep files manageable

    Output:
        If use_partitioning=True (default):
            - dataset=1/data.parquet
            - dataset=2/data.parquet
            - ... (one file per dataset, memory-friendly)

        If use_partitioning=False:
            - data.parquet (single file, requires more memory)

    Note:
        Train/val/test/oot splitting is done temporally at the DAG level:
        - Train: Earlier snapshot dates (e.g., Jan 1-3)
        - Val: Middle snapshot date (e.g., Jan 4)
        - Test: Middle snapshot date (e.g., Jan 5)
        - OOT: Later snapshot dates (e.g., Jan 6+)
    """
    print(f"Processing Gold Label Base for snapshot date: {snapshot_date_str}")
    print(f"Loading silver data from {silver_dir}")
    print(f"Partitioning mode: {'Enabled (memory-friendly)' if use_partitioning else 'Disabled (single file)'}")

    # Create output directory
    output_dir = os.path.join(
        gold_label_base_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load silver DataFrame
    df = pd.read_parquet(silver_dir)
    print(f"Loaded {len(df):,} rows from silver layer")

    # Create RUL_Clipped column
    df['RUL_Clipped'] = np.clip(df['Remaining_Useful_Life'], 0, RUL_CLIP_MAX)
    print(f"Created RUL_Clipped column (clipped at {RUL_CLIP_MAX})")

    if use_partitioning:
        # Save partitioned by dataset for memory efficiency
        # Each dataset will be in its own file
        print("Saving with partitioning by 'dataset' column...")

        datasets = sorted(df['dataset'].unique())
        print(f"Found {len(datasets)} datasets: {datasets}")

        for dataset_id in datasets:
            df_dataset = df[df['dataset'] == dataset_id]
            dataset_output_dir = os.path.join(output_dir, f"dataset={dataset_id}")
            os.makedirs(dataset_output_dir, exist_ok=True)

            output_path = os.path.join(dataset_output_dir, 'data.parquet')
            # Write with consistent schema to avoid PyArrow type conflicts
            # Disable dictionary encoding to prevent schema conflicts
            df_dataset.to_parquet(
                output_path,
                index=False,
                engine='pyarrow',
                compression='snappy',
                use_dictionary=False
            )
            print(f"  Saved dataset {dataset_id}: {len(df_dataset):,} rows to {output_path}")

        print(f"Saved {len(df):,} total rows across {len(datasets)} partitions")
    else:
        # Save as single file
        output_path = os.path.join(output_dir, 'data.parquet')
        df.to_parquet(
            output_path,
            index=False,
            engine='pyarrow',
            compression='snappy',
            use_dictionary=False
        )
        print(f"Saved {len(df):,} rows to {output_path}")

    print(f"Gold Label Base written successfully to {output_dir}")
    return output_dir
