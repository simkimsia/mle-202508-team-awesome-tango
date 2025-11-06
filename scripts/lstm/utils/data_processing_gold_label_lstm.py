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
    input_dir = os.path.join(
        gold_label_base_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    parquet_files = glob.glob(os.path.join(input_dir, '**/data.parquet'), recursive=True)
    if not parquet_files:
        parquet_files = glob.glob(os.path.join(input_dir, '*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    print(f"Found {len(parquet_files)} partition files")
    targets = []
    units = []
    total_rows = 0
    for i, parquet_file in enumerate(sorted(parquet_files), 1):
        partition_name = os.path.basename(os.path.dirname(parquet_file))
        print(f"Processing partition {i}/{len(parquet_files)}: {partition_name}")
        dataset_num = None
        if 'dataset=' in partition_name:
            dataset_num = int(partition_name.split('dataset=')[1])
        elif partition_name.startswith('dataset'):
            dataset_num = int(partition_name.replace('dataset', ''))
        if dataset_num is None:
            raise ValueError(f"Could not extract dataset number from partition path: {partition_name}")
        print(f"  Dataset number: {dataset_num}")
        df_part = pd.read_parquet(parquet_file, engine='pyarrow')
        if i == 1:
            print(f"  Columns in parquet: {list(df_part.columns)}")
            print(f"  Sample of first row:")
            print(f"    unit_orig: {df_part['unit_orig'].iloc[0] if 'unit_orig' in df_part.columns else 'MISSING'}")
            print(f"    time: {df_part['time'].iloc[0] if 'time' in df_part.columns else 'MISSING'}")
            print(f"    Unique unit_orig values: {df_part['unit_orig'].nunique() if 'unit_orig' in df_part.columns else 'MISSING'}")
            print(f"    Shape: {df_part.shape}")
        for col in df_part.columns:
            if pd.api.types.is_categorical_dtype(df_part[col]):
                df_part[col] = df_part[col].astype(df_part[col].cat.categories.dtype)
        total_rows += len(df_part)
        df_part['unit'] = (
            'DS' +
            str(dataset_num).zfill(2) +
            '_' +
            df_part['unit_orig'].astype(int).astype(str).str.zfill(3)
        )
        unique_units = sorted(df_part['unit'].unique())
        if i == 1:
            print(f"  First 5 units found: {unique_units[:5]}")
            print(f"  Filtering for unit '{unique_units[0]}'...")
            test_unit = unique_units[0]
            test_df = df_part[df_part['unit'] == test_unit]
            print(f"  Result: {len(test_df)} rows")
            print(f"  Time range: {test_df['time'].min()} to {test_df['time'].max()}" if len(test_df) > 0 else "  Time range: N/A")
            print(f"  Time unique values count: {test_df['time'].nunique()}")
            print(f"  First 10 time values: {test_df['time'].head(10).tolist()}")
            print(f"  Last 10 time values: {test_df['time'].tail(10).tolist()}")
            print(f"  Sample 'unit' column values from df_part: {df_part['unit'].head(10).tolist()}")
            print(f"  Value counts for 'unit' column (first 10): {df_part['unit'].value_counts().head(10).to_dict()}")
        for unit in unique_units:
            df_unit = df_part[df_part['unit'] == unit].sort_values('time')
            ts = TimeSeries.from_values(
                values=df_unit['RUL_Clipped'].values.reshape(-1, 1)
            )
            if len(targets) < 3:
                print(f"    Unit {unit}: {len(df_unit)} rows → TimeSeries length {len(ts)}")
            targets.append(ts)
            units.append(unit)
        print(f"  Partition {i}: Created {len(df_part['unit'].unique())} TimeSeries objects from {len(df_part):,} rows")
        del df_part
    print(f"Processed {total_rows:,} total rows from {len(parquet_files)} partitions")
    print(f"Created {len(targets)} TimeSeries objects (one per engine unit)")
    output_dir = os.path.join(
        gold_label_lstm_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'targets.pkl'), 'wb') as f:
        pickle.dump(targets, f)
    print(f"Saved targets to {os.path.join(output_dir, 'targets.pkl')}")
    with open(os.path.join(output_dir, 'units.pkl'), 'wb') as f:
        pickle.dump(units, f)
    print(f"Saved {len(units)} unit IDs to {os.path.join(output_dir, 'units.pkl')}")
    print(f"Gold Label LSTM written successfully to {output_dir}")
    return output_dir