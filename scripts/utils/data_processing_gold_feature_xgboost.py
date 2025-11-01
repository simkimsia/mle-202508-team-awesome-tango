"""
Gold Feature XGBoost Processing
Creates XGBoost-compatible tabular features.

This module transforms raw sensor data into a flat, tabular DataFrame
suitable for XGBoost model training. It includes multi-scale rolling
window features (mean, std, acceleration) to capture temporal dynamics
in a static format.
"""

import os
import glob
import pandas as pd
import pyarrow.parquet as pq

# Selected features for the model (12 unique features)
SELECTED_FEATURES = [
    'P30',        # HPC Outlet Pressure
    'W32',        # LPT Coolant Bleed
    'P2',        # Fan Inlet Pressure
    'W50',         # Demanded Fan Speed
    'Nf',          # Fan Speed
    'Nc',         # Core Speed
    'P15',       # Pressure in Bypass Duct
    'phi',        # Fuel Flow Ratio
    'T50',        # LPT Outlet Temperature
    'Altitude',   # Altitude
    'Mach_Number',# Mach Number
    'TRA'         # Throttle Resolver Angle
]


def create_multiscale_features(df, features):
    """
    Create rolling window features at multiple time scales (MEMORY EFFICIENT).
    Processes data in chunks and uses float32 to reduce memory usage.
    
    Args:
        df: DataFrame with columns ['unit', 'time', features...]
        features: List of sensor feature names
    
    Returns:
        DataFrame with additional rolling window features
    """
    
    # Define time windows (in cycles)
    windows = {
        'short': 5,      # Last 5 cycles
        'medium': 15,    # Last 15 cycles
        'long': 30       # Last 30 cycles
    }
    
    print("Creating multi-scale rolling window features (MEMORY EFFICIENT MODE)...")
    print(f"Windows: {windows}")
    print(f"Features: {len(features)} sensors")
    print(f"Input data shape: {df.shape}")
    print(f"Estimated memory: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    # Convert to float32 to save memory (float64 uses 2x memory)
    print("\n💾 Converting to float32 to reduce memory usage...")
    df_work = df.copy()
    for col in features:
        if col in df_work.columns and df_work[col].dtype == 'float64':
            df_work[col] = df_work[col].astype('float32')
    
    # Group by engine unit
    grouped = df_work.groupby('unit')
    
    # Process features in batches to avoid memory explosion
    batch_size = 3  # Process 3 sensors at a time
    all_new_dfs = []
    
    for batch_start in range(0, len(features), batch_size):
        batch_features = features[batch_start:batch_start + batch_size]
        print(f"\n📦 Processing batch {batch_start//batch_size + 1}/{(len(features)-1)//batch_size + 1}: {batch_features}")
        
        new_columns = {}
        
        for window_name, window_size in windows.items():
            print(f"   Window: {window_name} (size={window_size})")
            
            for sensor in batch_features:
                if sensor not in df_work.columns:
                    continue
                
                # Rolling aggregations per engine unit
                mean_col = grouped[sensor].transform(
                    lambda x: x.rolling(window=window_size, min_periods=1).mean()
                ).astype('float32')
                
                std_col = grouped[sensor].transform(
                    lambda x: x.rolling(window=window_size, min_periods=1).std().fillna(0)
                ).astype('float32')
                
                # Only keep mean and std to save memory
                new_columns[f"{sensor}_mean_{window_name}"] = mean_col
                new_columns[f"{sensor}_std_{window_name}"] = std_col
        
        # Add cross-window features for this batch
        for sensor in batch_features:
            if sensor not in df_work.columns:
                continue
            
            # Ensure required mean columns exist before calculating acceleration
            mean_short_col = f"{sensor}_mean_short"
            mean_medium_col = f"{sensor}_mean_medium"
            if mean_short_col in new_columns and mean_medium_col in new_columns:
                new_columns[f"{sensor}_acceleration"] = (
                    new_columns[mean_short_col] - 
                    new_columns[mean_medium_col]
                ).astype('float32')
            
        # --- Function completion ---
        # Create a DataFrame from the new columns for this batch
        batch_df = pd.DataFrame(new_columns)
        all_new_dfs.append(batch_df)
        # -------------------------

    # --- Function completion ---
    print("\n✅ All batches processed. Concatenating features...")
    
    if not all_new_dfs:
        print("No features were processed.")
        return df_work

    # Concatenate all batch DataFrames horizontally
    features_df = pd.concat(all_new_dfs, axis=1)
    
    # Combine with the original DataFrame
    # Indices are aligned because .transform() preserves the original index
    df_final = pd.concat([df_work, features_df], axis=1)
    
    print(f"Final DataFrame shape: {df_final.shape}")
    print(f"Final memory: {df_final.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    return df_final
    # -------------------------


def process_gold_feature_xgboost(
    gold_label_base_dir: str,
    gold_feature_xgboost_dir: str,
    snapshot_date_str: str
):
    """
    Create XGBoost-compatible tabular features with rolling windows.

    MEMORY OPTIMIZED: Processes each dataset partition separately to avoid OOM.

    Args:
        gold_label_base_dir: Path to base label parquet files
        gold_feature_xgboost_dir: Output directory for XGBoost features
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Detect if data is partitioned by dataset
        2. Process each partition separately (lower memory usage)
        3. Apply 'create_multiscale_features' to each partition
        4. Save partitioned features.parquet files
    """
    print(f"Processing Gold Feature XGBoost for snapshot date: {snapshot_date_str}")
    print(f"Loading base labels from {gold_label_base_dir}")

    # Input directory
    input_dir = os.path.join(
        gold_label_base_dir,
        f"snapshot_date={snapshot_date_str}"
    )

    # Create output directory
    output_dir = os.path.join(
        gold_feature_xgboost_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Check if data is partitioned (dataset=X subdirectories exist)
    partition_dirs = glob.glob(os.path.join(input_dir, 'dataset=*'))

    if partition_dirs:
        print(f"\n📦 PARTITIONED MODE: Found {len(partition_dirs)} dataset partitions")
        print("Processing each dataset separately to minimize memory usage...")

        for partition_dir in sorted(partition_dirs):
            dataset_name = os.path.basename(partition_dir)  # e.g., "dataset=1"
            dataset_id = dataset_name.split('=')[1]

            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")

            # Load this partition
            partition_file = os.path.join(partition_dir, 'data.parquet')
            if not os.path.exists(partition_file):
                print(f"⚠️  Skipping {dataset_name}: data.parquet not found")
                continue

            df_partition = pd.read_parquet(partition_file, engine='pyarrow')
            print(f"Loaded {len(df_partition):,} rows from {dataset_name}")

            # Process this partition
            df_features = _process_partition(df_partition, dataset_name)

            # Save this partition
            partition_output_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(partition_output_dir, exist_ok=True)
            output_file = os.path.join(partition_output_dir, 'features.parquet')

            df_features.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
            print(f"✅ Saved {dataset_name} features ({df_features.shape}) to {output_file}")

            # Free memory
            del df_partition
            del df_features

        print(f"\n{'='*60}")
        print(f"✅ All {len(partition_dirs)} partitions processed successfully")
        print(f"{'='*60}")

    else:
        # Single file mode (fallback)
        print("\n📄 SINGLE FILE MODE: Processing entire dataset at once")
        single_file = os.path.join(input_dir, 'data.parquet')

        if not os.path.exists(single_file):
            raise FileNotFoundError(f"No data found at {input_dir}")

        df = pd.read_parquet(single_file, engine='pyarrow')
        print(f"Loaded {len(df):,} rows from single file")

        # Process entire dataset
        df_features = _process_partition(df, "all_data")

        # Save as single file
        output_file = os.path.join(output_dir, 'features.parquet')
        df_features.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
        print(f"✅ Saved features ({df_features.shape}) to {output_file}")

    print(f"\nGold Feature XGBoost written successfully to {output_dir}")
    return output_dir


def _process_partition(df: pd.DataFrame, partition_name: str) -> pd.DataFrame:
    """
    Process a single partition (dataset) to create XGBoost features.

    Args:
        df: DataFrame for this partition
        partition_name: Name of the partition (for logging)

    Returns:
        DataFrame with engineered features
    """
    # Verify selected features exist in DataFrame
    missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing_features:
        print(f"WARNING: Missing features in {partition_name}: {missing_features}")
        feature_cols = [f for f in SELECTED_FEATURES if f in df.columns]
    else:
        feature_cols = SELECTED_FEATURES

    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Verify unit column exists (should be created by gold_label_base)
    if 'unit' not in df.columns:
        print(f"WARNING: 'unit' column missing in {partition_name}, creating it...")
        df['unit'] = df.apply(lambda row: f"DS{int(row['dataset']):02d}_{int(row['unit_orig']):03d}", axis=1)
    else:
        print(f"✓ Using existing 'unit' column ({df['unit'].nunique()} unique units)")

    # Apply multi-scale feature engineering
    print(f"\n🚀 Applying multi-scale feature engineering for {partition_name}...")
    df_features = create_multiscale_features(df, feature_cols)
    print(f"Feature engineering complete for {partition_name}.")

    return df_features
