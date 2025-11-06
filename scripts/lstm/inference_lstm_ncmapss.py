"""
Inference - LSTM RUL Predictions using Darts BlockRNNModel
Usage:
    python3 inference_lstm_ncmapss.py --snapshotdate "2023-01-01"
This script follows the LSTM pattern from the notebook:
1. Load Darts BlockRNNModel (LSTM with past_covariates)
2. Load feature TimeSeries and label TimeSeries from gold layer
3. Use historical_forecasts() for rolling predictions
4. Inverse transform predictions back to original scale
5. Save predictions with metadata
"""
import os
import argparse
import pickle
import pandas as pd
import numpy as np
import glob
import warnings
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*lower than the number of series.*')
warnings.filterwarnings('ignore', message='.*does not have many workers.*')
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import Scaler
def load_lstm_model(model_path: str):
    """
    Load Darts BlockRNNModel from pickle and checkpoint files.
    Args:
        model_path: Path to model pickle file (e.g., 'darts_lstm_model.pkl')
    Returns:
        BlockRNNModel: Loaded LSTM model ready for inference
    Note:
        Darts models require TWO files:
        - model.pkl: Model structure/configuration
        - model.pkl.ckpt: Trained weights (PyTorch Lightning checkpoint)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    checkpoint_path = model_path + '.ckpt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    print(f"📦 Loading LSTM model from {model_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    import torch
    try:
        print(f"🔍 Checking checkpoint file format...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, BlockRNNModel):
            print(f"⚠️  ERROR: Checkpoint file contains a BlockRNNModel object instead of checkpoint dict")
            print(f"   This happens when model was saved with pickle.dump() instead of model.save()")
            print(f"\n🔧 WORKAROUND: Loading model directly from checkpoint file...")
            print(f"✓ Model loaded from checkpoint file")
            return checkpoint
        elif isinstance(checkpoint, dict):
            needs_fix = False
            if "pytorch-lightning_version" not in checkpoint:
                needs_fix = True
                print(f"⚠️  Checkpoint missing 'pytorch-lightning_version' key")
                checkpoint["pytorch-lightning_version"] = "2.0.0"
            if "epoch" not in checkpoint:
                checkpoint["epoch"] = 0
                needs_fix = True
                print(f"⚠️  Checkpoint missing 'epoch' key")
            if "global_step" not in checkpoint:
                checkpoint["global_step"] = 0
                needs_fix = True
                print(f"⚠️  Checkpoint missing 'global_step' key")
            if needs_fix:
                print(f"🔧 Saving fixed checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)
                print(f"✓ Checkpoint compatibility fixed")
            else:
                print(f"✓ Checkpoint is compatible")
            model = BlockRNNModel.load(model_path, map_location='cpu')
            print(f"✓ Model loaded successfully (mapped to CPU)")
            return model
        else:
            print(f"⚠️  Unknown checkpoint format (type: {type(checkpoint).__name__})")
            raise ValueError(f"Unexpected checkpoint type: {type(checkpoint).__name__}")
    except Exception as e:
        print(f"⚠️  Error during checkpoint check: {e}")
        print(f"   Attempting standard Darts load method with CPU mapping...")
        try:
            model = BlockRNNModel.load(model_path, map_location='cpu')
            print(f"✓ Model loaded successfully (mapped to CPU)")
            return model
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            print(f"\n💡 The checkpoint file appears to be corrupted or incompatible.")
            print(f"   Please retrain the model using model.save() method:")
            print(f"   >>> lstm_model.save('{model_path}')")
            raise
def load_timeseries_data(gold_feature_lstm_dir: str, gold_label_lstm_dir: str, snapshot_date_str: str):
    """
    Load TimeSeries data from gold LSTM feature and label stores.
    Args:
        gold_feature_lstm_dir: Path to LSTM feature store
        gold_label_lstm_dir: Path to LSTM label store
        snapshot_date_str: Snapshot date in YYYY-MM-DD format
    Returns:
        tuple: (covariates_list, targets_list, units_list, metadata_df)
    Note:
        The gold layer saves covariates and targets as lists of TimeSeries objects:
        - covariates.pkl: List of feature TimeSeries (one per unit)
        - targets.pkl: List of RUL TimeSeries (one per unit)
        - units.pkl: List of unit identifiers
    """
    feature_dir = os.path.join(gold_feature_lstm_dir, f"snapshot_date={snapshot_date_str}")
    label_dir = os.path.join(gold_label_lstm_dir, f"snapshot_date={snapshot_date_str}")
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    print(f"\n📂 Loading TimeSeries data")
    print(f"   Features: {feature_dir}")
    print(f"   Labels: {label_dir}")
    covariates_path = os.path.join(feature_dir, 'covariates.pkl')
    if not os.path.exists(covariates_path):
        raise FileNotFoundError(f"Covariates file not found: {covariates_path}")
    with open(covariates_path, 'rb') as f:
        covariates_list = pickle.load(f)
    units_path = os.path.join(feature_dir, 'units.pkl')
    if not os.path.exists(units_path):
        raise FileNotFoundError(f"Units file not found: {units_path}")
    with open(units_path, 'rb') as f:
        units_list = pickle.load(f)
    targets_path = os.path.join(label_dir, 'targets.pkl')
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"Targets file not found: {targets_path}")
    with open(targets_path, 'rb') as f:
        targets_list = pickle.load(f)
    if not (len(covariates_list) == len(targets_list) == len(units_list)):
        raise ValueError(
            f"Mismatch in data lengths: "
            f"covariates={len(covariates_list)}, "
            f"targets={len(targets_list)}, "
            f"units={len(units_list)}"
        )
    metadata_records = []
    for idx, (unit, target) in enumerate(zip(units_list, targets_list)):
        dataset = unit.split('_')[0] if '_' in unit else 'unknown'
        metadata_records.append({
            'dataset': dataset,
            'unit': unit,
            'n_timesteps': len(target)
        })
    metadata_df = pd.DataFrame(metadata_records)
    print(f"✓ Loaded {len(covariates_list)} units")
    print(f"   Total timesteps: {metadata_df['n_timesteps'].sum():,}")
    return covariates_list, targets_list, units_list, metadata_df
def run_lstm_inference(
    lstm_model,
    covariates_list,
    targets_list,
    units_list,
    metadata_df,
    scaler_target,
    sequence_length: int,
    inference_output_dir: str,
    snapshot_date_str: str
):
    """
    Run LSTM inference using historical_forecasts for rolling predictions.
    Args:
        lstm_model: Trained Darts BlockRNNModel
        covariates_list: List of covariate TimeSeries (past_covariates)
        targets_list: List of target TimeSeries (for ground truth)
        units_list: List of unit IDs
        metadata_df: DataFrame with unit metadata
        scaler_target: Darts Scaler for inverse transforming predictions
        sequence_length: Input chunk length used during training
        inference_output_dir: Output directory for predictions
        snapshot_date_str: Snapshot date in YYYY-MM-DD format
    Returns:
        str: Path to output directory
    Note:
        Uses historical_forecasts() with:
        - start=SEQUENCE_LENGTH: Skip first N timesteps (need history)
        - forecast_horizon=1: Predict one step ahead
        - stride=1: Make prediction at every timestep
        - retrain=False: Use pre-trained model (no retraining)
    """
    import torch
    if torch.cuda.is_available():
        lstm_model.trainer_params = {"accelerator": "gpu", "devices": 1}
        print(f"✓ Model configured for GPU inference ({torch.cuda.get_device_name(0)})")
    else:
        lstm_model.trainer_params = {"accelerator": "cpu", "devices": 1}
        print(f"⚠️  GPU not available, using CPU inference")
    print(f"\n{'='*80}")
    print(f"LSTM Inference - N-CMAPSS")
    print(f"{'='*60}")
    print(f"Snapshot date: {snapshot_date_str}")
    print(f"Sequence length: {sequence_length}")
    print(f"Units to process: {len(units_list)}")
    output_dir = os.path.join(
        inference_output_dir,
        f"snapshot_date={snapshot_date_str}"
    )
    os.makedirs(output_dir, exist_ok=True)
    all_predictions = []
    for idx, (cov, target, unit_id) in enumerate(zip(covariates_list, targets_list, units_list)):
        print(f"\n{'-'*60}")
        print(f"Processing Unit {unit_id} ({idx+1}/{len(units_list)})")
        print(f"{'-'*60}")
        try:
            print(f"🔮 Running inference with historical_forecasts()...")
            pred_scaled = lstm_model.historical_forecasts(
                series=target,
                past_covariates=cov,
                start=sequence_length,
                forecast_horizon=1,
                stride=1,
                retrain=False,          # Don't retrain - use pre-trained model
                verbose=False
            )
            if isinstance(pred_scaled, list):
                pred_series = pred_scaled[0] if len(pred_scaled) > 0 else TimeSeries.from_values(np.array([]))
            else:
                pred_series = pred_scaled
            if len(pred_series) == 0:
                print(f"⚠️  Skipping Unit {unit_id}: No predictions generated")
                continue
            preds_scaled_vals = pred_series.values().flatten()
            actuals_scaled_vals = target.slice_intersect(pred_series).values().flatten()
            print(f"   Generated {len(preds_scaled_vals)} predictions")
            preds_vals = scaler_target.inverse_transform(
                preds_scaled_vals.reshape(-1, 1)
            ).flatten()
            actuals_vals = scaler_target.inverse_transform(
                actuals_scaled_vals.reshape(-1, 1)
            ).flatten()
            rmse = np.sqrt(np.mean((actuals_vals - preds_vals) ** 2))
            mae = np.mean(np.abs(actuals_vals - preds_vals))
            print(f"✓ Unit {unit_id} complete")
            print(f"  RMSE: {rmse:.4f} cycles")
            print(f"  MAE:  {mae:.4f} cycles")
            time_indices = target.time_index[sequence_length:sequence_length+len(preds_vals)]
            unit_metadata = metadata_df[metadata_df['unit'] == unit_id].iloc[0]
            predictions_df = pd.DataFrame({
                'unit': unit_id,
                'dataset': unit_metadata['dataset'],
                'time': range(sequence_length, sequence_length + len(preds_vals)),
                'RUL_Predicted': preds_vals.astype('float32'),
                'RUL_Actual': actuals_vals.astype('float32'),
                'Prediction_Error': (preds_vals - actuals_vals).astype('float32')
            })
            all_predictions.append(predictions_df)
        except Exception as e:
            print(f"⚠️  Error processing Unit {unit_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    if not all_predictions:
        raise RuntimeError("No predictions generated for any unit!")
    print(f"\n{'='*60}")
    print(f"Combining predictions from {len(all_predictions)} units...")
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    overall_rmse = np.sqrt(np.mean(combined_predictions['Prediction_Error'] ** 2))
    overall_mae = np.mean(np.abs(combined_predictions['Prediction_Error']))
    print(f"\n📊 Overall Metrics:")
    print(f"  Total predictions: {len(combined_predictions):,}")
    print(f"  RMSE: {overall_rmse:.4f} cycles")
    print(f"  MAE:  {overall_mae:.4f} cycles")
    output_file = os.path.join(output_dir, 'predictions.parquet')
    combined_predictions.to_parquet(
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
    print("\n\n--- Starting LSTM Inference N-CMAPSS job ---\n\n")
    base_dir = "/opt/airflow/scripts"
    gold_feature_lstm_dir = os.path.join(base_dir, "datamart/gold/feature/lstm/n_cmapss")
    gold_label_lstm_dir = os.path.join(base_dir, "datamart/gold/label/lstm/n_cmapss")
    model_path = os.path.join(base_dir, "model_bank/darts_lstm_model.pkl")
    scaler_target_path = os.path.join(base_dir, "model_bank/darts_target_scaler.pkl")
    scaler_cov_path = os.path.join(base_dir, "model_bank/darts_covariate_scaler.pkl")
    inference_output_dir = os.path.join(base_dir, "datamart/inference/lstm/n_cmapss")
    SEQUENCE_LENGTH = 30
    lstm_model = load_lstm_model(model_path)
    print(f"\n📦 Loading target scaler from {scaler_target_path}")
    if not os.path.exists(scaler_target_path):
        raise FileNotFoundError(
            f"Target scaler not found at {scaler_target_path}\n"
            f"Please save the scaler during training with:\n"
            f"  with open('{scaler_target_path}', 'wb') as f:\n"
            f"      pickle.dump(scaler_target, f)"
        )
    with open(scaler_target_path, 'rb') as f:
        scaler_target = pickle.load(f)
    print(f"✓ Target scaler loaded successfully")
    print(f"\n📦 Loading covariate scaler from {scaler_cov_path}")
    if not os.path.exists(scaler_cov_path):
        raise FileNotFoundError(
            f"Covariate scaler not found at {scaler_cov_path}\n"
            f"Please ensure the scaler was saved during training"
        )
    with open(scaler_cov_path, 'rb') as f:
        scaler_covariates = pickle.load(f)
    print(f"✓ Covariate scaler loaded successfully")
    covariates_list, targets_list, units_list, metadata_df = load_timeseries_data(
        gold_feature_lstm_dir,
        gold_label_lstm_dir,
        snapshotdate
    )
    print(f"\n⚙️  Scaling covariates with trained sklearn scaler...")
    covariates_scaled = []
    for cov in covariates_list:
        values = cov.values()
        scaled_values = scaler_covariates.transform(values)
        cov_scaled = TimeSeries.from_values(scaled_values)
        covariates_scaled.append(cov_scaled)
    print(f"✓ Scaled {len(covariates_scaled)} covariate TimeSeries")
    print(f"\n⚙️  Scaling targets with trained sklearn scaler...")
    targets_scaled = []
    for target in targets_list:
        values = target.values()
        scaled_values = scaler_target.transform(values)
        target_scaled = TimeSeries.from_values(scaled_values)
        targets_scaled.append(target_scaled)
    print(f"✓ Scaled {len(targets_scaled)} target TimeSeries")
    run_lstm_inference(
        lstm_model=lstm_model,
        covariates_list=covariates_scaled,
        targets_list=targets_scaled,
        units_list=units_list,
        metadata_df=metadata_df,
        scaler_target=scaler_target,
        sequence_length=SEQUENCE_LENGTH,
        inference_output_dir=inference_output_dir,
        snapshot_date_str=snapshotdate
    )
    print("\n\n--- Completed LSTM Inference N-CMAPSS job ---\n\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSTM inference for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)