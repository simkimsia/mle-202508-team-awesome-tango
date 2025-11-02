#!/usr/bin/env python3
"""
Inference - XGBoost RUL Predictions
Usage:
    python3 inference_xgboost_ncmapss.py --snapshotdate "2023-01-01"
"""

import os
import argparse
import pickle
import pandas as pd
import glob


def run_xgboost_inference(
    gold_feature_xgboost_dir: str,
    model_path: str,
    inference_output_dir: str,
    snapshot_date_str: str
):
    """
    Run XGBoost inference on engineered features.

    Args:
        gold_feature_xgboost_dir: Path to XGBoost feature store
        model_path: Path to trained XGBoost model pickle file
        inference_output_dir: Output directory for predictions
        snapshot_date_str: Snapshot date in YYYY-MM-DD format

    Returns:
        str: Path to the output directory

    Processing Steps:
        1. Load pre-trained XGBoost model
        2. Read engineered features from feature store
        3. Extract feature columns (exclude metadata)
        4. Make RUL predictions
        5. Save predictions with metadata
    """
    print(f"\n{'='*60}")
    print(f"XGBoost Inference - N-CMAPSS")
    print(f"{'='*60}")
    print(f"Snapshot date: {snapshot_date_str}")
    print(f"Model path: {model_path}")
    print(f"Feature directory: {gold_feature_xgboost_dir}")

    # Load pre-trained XGBoost model
    print(f"\n📦 Loading XGBoost model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(model_path, 'rb') as f:
        xgb_model = pickle.load(f)
    print(f"✓ Model loaded successfully")

    # Input directory for features
    input_dir = os.path.join(
        gold_feature_xgboost_dir,
        f"snapshot_date={snapshot_date_str}"
    )

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Feature directory not found: {input_dir}")

    # Create output directory
    output_dir = os.path.join(
        inference_output_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Check if data is partitioned (dataset=X subdirectories exist)
    partition_dirs = glob.glob(os.path.join(input_dir, 'dataset=*'))

    # Metadata columns to exclude from features
    metadata_cols = ['unit', 'time', 'Remaining Useful Life', 'RUL_Clipped', 'dataset']

    if partition_dirs:
        print(f"\n📦 PARTITIONED MODE: Found {len(partition_dirs)} dataset partitions")

        all_predictions = []

        for partition_dir in sorted(partition_dirs):
            dataset_name = os.path.basename(partition_dir)  # e.g., "dataset=1"
            dataset_id = dataset_name.split('=')[1]

            print(f"\n{'-'*60}")
            print(f"Processing {dataset_name}")
            print(f"{'-'*60}")

            # Load features for this partition
            partition_file = os.path.join(partition_dir, 'features.parquet')
            if not os.path.exists(partition_file):
                print(f"⚠️  Skipping {dataset_name}: features.parquet not found")
                continue

            df_features = pd.read_parquet(partition_file, engine='pyarrow')
            print(f"Loaded {len(df_features):,} rows from {dataset_name}")

            # Extract feature columns (all columns except metadata)
            feature_cols = [col for col in df_features.columns if col not in metadata_cols]
            print(f"Using {len(feature_cols)} feature columns for prediction")

            # Prepare input features
            X = df_features[feature_cols].values.astype('float32')

            # Make predictions
            print(f"🔮 Running inference...")
            y_pred = xgb_model.predict(X)

            # Create predictions dataframe with metadata
            predictions_df = pd.DataFrame({
                'unit': df_features['unit'],
                'time': df_features['time'],
                'dataset': df_features['dataset'],
                'RUL_Predicted': y_pred.astype('float32')
            })

            # Include actual RUL if available
            if 'RUL_Clipped' in df_features.columns:
                predictions_df['RUL_Actual'] = df_features['RUL_Clipped'].astype('float32')

                # Calculate prediction error
                predictions_df['Prediction_Error'] = (
                    predictions_df['RUL_Predicted'] - predictions_df['RUL_Actual']
                ).astype('float32')

                # Calculate metrics for this partition
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                import numpy as np

                rmse = np.sqrt(mean_squared_error(
                    predictions_df['RUL_Actual'],
                    predictions_df['RUL_Predicted']
                ))
                mae = mean_absolute_error(
                    predictions_df['RUL_Actual'],
                    predictions_df['RUL_Predicted']
                )

                print(f"✓ Predictions complete for {dataset_name}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE:  {mae:.4f}")
            else:
                print(f"✓ Predictions complete for {dataset_name}")
                print(f"  (No ground truth available)")

            all_predictions.append(predictions_df)

        # Combine all partitions
        print(f"\n{'='*60}")
        print(f"Combining predictions from all partitions...")
        combined_predictions = pd.concat(all_predictions, ignore_index=True)

        # Calculate overall metrics if ground truth is available
        if 'RUL_Actual' in combined_predictions.columns:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np

            overall_rmse = np.sqrt(mean_squared_error(
                combined_predictions['RUL_Actual'],
                combined_predictions['RUL_Predicted']
            ))
            overall_mae = mean_absolute_error(
                combined_predictions['RUL_Actual'],
                combined_predictions['RUL_Predicted']
            )

            print(f"\n📊 Overall Metrics:")
            print(f"  Total predictions: {len(combined_predictions):,}")
            print(f"  RMSE: {overall_rmse:.4f}")
            print(f"  MAE:  {overall_mae:.4f}")

        # Save combined predictions
        output_file = os.path.join(output_dir, 'predictions.parquet')
        combined_predictions.to_parquet(
            output_file,
            index=False,
            engine='pyarrow',
            compression='snappy'
        )
        print(f"\n✅ Saved predictions to {output_file}")

    else:
        # Single file mode (fallback)
        print("\n📄 SINGLE FILE MODE: Processing entire dataset at once")
        single_file = os.path.join(input_dir, 'features.parquet')

        if not os.path.exists(single_file):
            raise FileNotFoundError(f"No features found at {input_dir}")

        df_features = pd.read_parquet(single_file, engine='pyarrow')
        print(f"Loaded {len(df_features):,} rows from single file")

        # Extract feature columns
        feature_cols = [col for col in df_features.columns if col not in metadata_cols]
        print(f"Using {len(feature_cols)} feature columns for prediction")

        # Prepare input features
        X = df_features[feature_cols].values.astype('float32')

        # Make predictions
        print(f"\n🔮 Running inference...")
        y_pred = xgb_model.predict(X)

        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'unit': df_features['unit'],
            'time': df_features['time'],
            'RUL_Predicted': y_pred.astype('float32')
        })

        # Include dataset if available
        if 'dataset' in df_features.columns:
            predictions_df['dataset'] = df_features['dataset']

        # Include actual RUL if available
        if 'RUL_Clipped' in df_features.columns:
            predictions_df['RUL_Actual'] = df_features['RUL_Clipped'].astype('float32')
            predictions_df['Prediction_Error'] = (
                predictions_df['RUL_Predicted'] - predictions_df['RUL_Actual']
            ).astype('float32')

            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np

            rmse = np.sqrt(mean_squared_error(
                predictions_df['RUL_Actual'],
                predictions_df['RUL_Predicted']
            ))
            mae = mean_absolute_error(
                predictions_df['RUL_Actual'],
                predictions_df['RUL_Predicted']
            )

            print(f"\n📊 Metrics:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")

        # Save predictions
        output_file = os.path.join(output_dir, 'predictions.parquet')
        predictions_df.to_parquet(
            output_file,
            index=False,
            engine='pyarrow',
            compression='snappy'
        )
        print(f"\n✅ Saved predictions to {output_file}")

    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"{'='*60}\n")

    return output_dir


def main(snapshotdate):
    print("\n\n--- Starting XGBoost Inference N-CMAPSS job ---\n\n")

    # Define paths (use absolute paths for Airflow Docker environment)
    base_dir = "/opt/airflow/scripts"
    gold_feature_xgboost_dir = os.path.join(base_dir, "datamart/gold/feature/xgboost/n_cmapss")
    model_path = os.path.join(base_dir, "model_bank/xgboost_rul_model.pkl")
    inference_output_dir = os.path.join(base_dir, "datamart/inference/xgboost/n_cmapss")

    # Run inference
    run_xgboost_inference(
        gold_feature_xgboost_dir=gold_feature_xgboost_dir,
        model_path=model_path,
        inference_output_dir=inference_output_dir,
        snapshot_date_str=snapshotdate
    )

    print("\n\n--- Completed XGBoost Inference N-CMAPSS job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost inference for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
