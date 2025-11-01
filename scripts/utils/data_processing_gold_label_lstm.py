"""
Gold Label LSTM Processing
Converts base labels to LSTM-compatible TimeSeries format.

This module transforms DataFrame labels into Darts TimeSeries objects
suitable for LSTM model training and inference. Each snapshot date is processed
independently, with temporal splitting handled at the DAG level.
"""

import os
import glob
import pickle
import pandas as pd
from darts import TimeSeries


def process_gold_label_lstm(
    gold_label_base_dir: str,
    gold_label_lstm_dir: str,
    snapshot_date_str: str
):
    """
    Convert base labels to LSTM-compatible TimeSeries format.

    Args:
        gold_label_base_dir: Path to base label parquet files
        gold_label_lstm_dir: Output directory for LSTM labels
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load data.parquet for this snapshot date
        2. Group by engine unit
        3. Create Darts TimeSeries objects (one per engine)
           - Each TimeSeries contains RUL_Clipped values indexed by time
        4. Save TimeSeries objects and unit list

    Output:
        - targets.pkl (list of TimeSeries, one per engine unit)
        - units.pkl (list of unit identifiers)

    Note:
        Normalization is handled at model training time across all training
        snapshot dates. Temporal splitting (train/val/test/oot) is done at
        the DAG level by selecting different snapshot dates.
    """
    print(f"Processing Gold Label LSTM for snapshot date: {snapshot_date_str}")
    print(f"Loading base labels from {gold_label_base_dir}")

    # Input directory
    input_dir = os.path.join(
        gold_label_base_dir,
        f"snapshot_date={snapshot_date_str}"
    )

    # Process partitions incrementally to avoid OOM
    # Find all parquet files (either partitioned or single file)
    parquet_files = glob.glob(os.path.join(input_dir, '**/data.parquet'), recursive=True)

    if not parquet_files:
        # Try single file
        parquet_files = glob.glob(os.path.join(input_dir, '*.parquet'))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    print(f"Found {len(parquet_files)} partition files")

    # Process each partition separately to keep memory usage low
    targets = []
    units = []
    total_rows = 0

    for i, parquet_file in enumerate(sorted(parquet_files), 1):
        print(f"Processing partition {i}/{len(parquet_files)}: {os.path.basename(os.path.dirname(parquet_file))}")

        # Load partition
        df_part = pd.read_parquet(parquet_file, engine='pyarrow')

        # Convert categorical/dictionary columns to their base types
        for col in df_part.columns:
            if pd.api.types.is_categorical_dtype(df_part[col]):
                df_part[col] = df_part[col].astype(df_part[col].cat.categories.dtype)

        total_rows += len(df_part)

        # Create unique unit identifier matching notebook format: DS{dataset:02d}_{unit_orig:03d}
        # This prevents conflicts where same unit_orig exists in different datasets
        # Example: dataset=1, unit_orig=1 → 'DS01_001'
        # Use vectorized string formatting for better performance
        df_part['unit'] = (
            'DS' +
            df_part['dataset'].astype(int).astype(str).str.zfill(2) +
            '_' +
            df_part['unit_orig'].astype(int).astype(str).str.zfill(3)
        )

        # Convert to TimeSeries objects for this partition
        # Group by unique unit identifier and create one TimeSeries per engine
        for unit in sorted(df_part['unit'].unique()):
            df_unit = df_part[df_part['unit'] == unit].sort_values('time')

            # Remove duplicate time values (keep first occurrence)
            df_unit = df_unit.drop_duplicates(subset=['time'], keep='first')

            # Create TimeSeries with time index and RUL_Clipped values
            # Use freq=1 to specify uniform time steps
            ts = TimeSeries.from_dataframe(
                df_unit,
                time_col='time',
                value_cols=['RUL_Clipped'],
                fill_missing_dates=True,
                freq=1
            )

            targets.append(ts)
            units.append(unit)

        print(f"  Partition {i}: Created {len(df_part['unit'].unique())} TimeSeries objects from {len(df_part):,} rows")

        # Free memory
        del df_part

    print(f"Processed {total_rows:,} total rows from {len(parquet_files)} partitions")
    print(f"Created {len(targets)} TimeSeries objects (one per engine unit)")

    # Create output directory
    output_dir = os.path.join(
        gold_label_lstm_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save TimeSeries objects
    with open(os.path.join(output_dir, 'targets.pkl'), 'wb') as f:
        pickle.dump(targets, f)
    print(f"Saved targets to {os.path.join(output_dir, 'targets.pkl')}")

    # Save unit list
    with open(os.path.join(output_dir, 'units.pkl'), 'wb') as f:
        pickle.dump(units, f)
    print(f"Saved {len(units)} unit IDs to {os.path.join(output_dir, 'units.pkl')}")

    print(f"Gold Label LSTM written successfully to {output_dir}")
    return output_dir
