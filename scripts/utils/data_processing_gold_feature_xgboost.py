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

    Args:
        gold_label_base_dir: Path to base label parquet files
        gold_feature_xgboost_dir: Output directory for XGBoost features
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load data.parquet for this snapshot date (handles partitions)
        2. Create a unique 'unit' identifier
        3. Select features from SELECTED_FEATURES
        4. Apply 'create_multiscale_features' to generate rolling stats
        5. Save the final, flat DataFrame to 'features.parquet'
    """
    print(f"Processing Gold Feature XGBoost for snapshot date: {snapshot_date_str}")
    print(f"Loading base labels from {gold_label_base_dir}")

    # Input directory
    input_dir = os.path.join(
        gold_label_base_dir,
        f"snapshot_date={snapshot_date_str}"
    )

    # Load base DataFrame
    # (Reusing the robust data loading logic from your LSTM util)
    try:
        df = pd.read_parquet(input_dir, engine='pyarrow')
    except Exception as e:
        print(f"Schema conflict detected, reading partitions individually...")
        parquet_files = glob.glob(os.path.join(input_dir, '**/data.parquet'), recursive=True)
        if not parquet_files:
            parquet_files = glob.glob(os.path.join(input_dir, '*.parquet'))
        
        print(f"Found {len(parquet_files)} partition files")
        dfs = []
        for parquet_file in parquet_files:
            df_part = pd.read_parquet(parquet_file, engine='pyarrow')
            for col in df_part.columns:
                if pd.api.types.is_categorical_dtype(df_part[col]):
                    df_part[col] = df_part[col].astype(df_part[col].cat.categories.dtype)
            dfs.append(df_part)
        
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

    # Create unique unit identifier
    df['unit'] = df.apply(lambda row: f"DS{int(row['dataset']):02d}_{int(row['unit_orig']):03d}", axis=1)

    # --- XGBoost Feature Engineering ---
    # Apply the multi-scale feature creation function to the entire DataFrame
    print("\n🚀 Applying multi-scale feature engineering for XGBoost...")
    df_features = create_multiscale_features(df, feature_cols)
    print("Feature engineering complete.")
    # ---------------------------------

    # Create output directory
    output_dir = os.path.join(
        gold_feature_xgboost_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the final DataFrame as a single Parquet file
    output_file = os.path.join(output_dir, 'features.parquet')
    try:
        df_features.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
        print(f"\n✅ Saved XGBoost features ({df_features.shape}) to {output_file}")
    except Exception as e:
        print(f"\n❌ Error saving Parquet file: {e}")
        print("Attempting to save with engine='fastparquet'...")
        try:
            df_features.to_parquet(output_file, index=False, engine='fastparquet', compression='snappy')
            print(f"\n✅ Saved XGBoost features ({df_features.shape}) with fastparquet to {output_file}")
        except Exception as e2:
            print(f"\n❌ Failed to save with fastparquet as well: {e2}")

    print(f"Gold Feature XGBoost written successfully to {output_dir}")
    return output_dir
