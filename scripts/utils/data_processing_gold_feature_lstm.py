"""
Gold Feature LSTM Processing
Creates LSTM-compatible time series features.

This module transforms raw sensor data into normalized TimeSeries objects
suitable for LSTM model training with sequence modeling. Each snapshot date
is processed independently, with temporal splitting handled at the DAG level.
"""

import os
import glob
import pickle
import pandas as pd
import pyarrow.parquet as pq
from darts import TimeSeries


# Selected features for LSTM model (12 features)
# Must match notebook features for model inference compatibility
SELECTED_FEATURES = [
    'P30',      # HPC Outlet Pressure
    'W31',      # LPT Coolant Bleed
    'P15',      # Fan Inlet Pressure
    'W50',      # Demanded Fan Speed
    'Nf',       # Fan Speed
    'Nc',       # Core Speed
    'Ps30',     # Pressure in Bypass Duct
    'phi',      # Fuel Flow Ratio
    'T50',      # LPT Outlet Temperature
    'Altitude', # Altitude
    'Mach_Number',  # Mach Number
    'TRA'       # Throttle Resolver Angle
]


def process_gold_feature_lstm(
    gold_label_base_dir: str,
    gold_feature_lstm_dir: str,
    snapshot_date_str: str
):
    """
    Create LSTM-compatible time series features.

    Args:
        gold_label_base_dir: Path to base label parquet files
        gold_feature_lstm_dir: Output directory for LSTM features
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load data.parquet for this snapshot date
        2. Select features from SELECTED_FEATURES
        3. Group by engine unit
        4. Create Darts TimeSeries objects (one per engine)
           - Each TimeSeries contains feature values indexed by time
        5. Save TimeSeries objects and unit list

    Output:
        - covariates.pkl (list of TimeSeries, shape: (time_steps, 12))
        - units.pkl (list of unit identifiers)

    Note:
        Normalization is handled at model training time across all training
        snapshot dates. Temporal splitting (train/val/test/oot) is done at
        the DAG level by selecting different snapshot dates.
    """
    print(f"Processing Gold Feature LSTM for snapshot date: {snapshot_date_str}")
    print(f"Loading base labels from {gold_label_base_dir}")

    # Input directory
    input_dir = os.path.join(
        gold_label_base_dir,
        f"snapshot_date={snapshot_date_str}"
    )

    # Load base DataFrame
    # Supports both partitioned (dataset=X/data.parquet) and single file (data.parquet)
    # Handle schema conflicts between partitions (e.g., int32 vs dictionary-encoded)
    try:
        df = pd.read_parquet(input_dir, engine='pyarrow')
    except Exception as e:
        # If schema conflict, read partitions individually and concatenate
        print(f"Schema conflict detected, reading partitions individually...")

        # Find all parquet files in the directory
        parquet_files = glob.glob(os.path.join(input_dir, '**/data.parquet'), recursive=True)

        if not parquet_files:
            # Try single file
            parquet_files = glob.glob(os.path.join(input_dir, '*.parquet'))

        print(f"Found {len(parquet_files)} partition files")

        # Read each partition and ensure consistent types
        dfs = []
        for parquet_file in parquet_files:
            df_part = pd.read_parquet(parquet_file, engine='pyarrow')

            # Convert categorical/dictionary columns to their base types
            for col in df_part.columns:
                if pd.api.types.is_categorical_dtype(df_part[col]):
                    df_part[col] = df_part[col].astype(df_part[col].cat.categories.dtype)

            dfs.append(df_part)

        # Concatenate all partitions
        df = pd.concat(dfs, ignore_index=True)
        print(f"Concatenated {len(dfs)} partitions")

    print(f"Loaded {len(df):,} rows from base labels")

    # Verify selected features exist in DataFrame
    missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing_features:
        print(f"WARNING: Missing features: {missing_features}")
        feature_cols = [f for f in SELECTED_FEATURES if f in df.columns]
    else:
        feature_cols = SELECTED_FEATURES

    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Create unique unit identifier matching notebook format: DS{dataset:02d}_{unit_orig:03d}
    # This prevents conflicts where same unit_orig exists in different datasets
    # Example: dataset=1, unit_orig=1 → 'DS01_001'
    df['unit'] = df.apply(lambda row: f"DS{int(row['dataset']):02d}_{int(row['unit_orig']):03d}", axis=1)

    # Convert to TimeSeries objects
    # Group by unique unit identifier and create one TimeSeries per engine
    covariates = []
    units = []

    for unit in sorted(df['unit'].unique()):
        df_unit = df[df['unit'] == unit].sort_values('time')

        # Remove duplicate time values (keep first occurrence)
        df_unit = df_unit.drop_duplicates(subset=['time'], keep='first')

        # Create TimeSeries with time index and feature values
        # Use freq=1 to specify uniform time steps
        ts = TimeSeries.from_dataframe(
            df_unit,
            time_col='time',
            value_cols=feature_cols,
            fill_missing_dates=True,
            freq=1
        )

        covariates.append(ts)
        units.append(unit)

    print(f"Created {len(covariates)} TimeSeries objects (one per engine unit)")

    # Create output directory
    output_dir = os.path.join(
        gold_feature_lstm_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save TimeSeries objects
    with open(os.path.join(output_dir, 'covariates.pkl'), 'wb') as f:
        pickle.dump(covariates, f)
    print(f"Saved covariates to {os.path.join(output_dir, 'covariates.pkl')}")

    # Save unit list
    with open(os.path.join(output_dir, 'units.pkl'), 'wb') as f:
        pickle.dump(units, f)
    print(f"Saved {len(units)} unit IDs to {os.path.join(output_dir, 'units.pkl')}")

    print(f"Gold Feature LSTM written successfully to {output_dir}")
    return output_dir
