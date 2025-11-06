"""
Model monitoring for LSTM RUL predictions on the N-CMAPSS dataset.
For a given snapshot date this script:
1. Loads inference predictions from LSTM model.
2. Computes RMSE and MAE, plus basic run statistics.
3. Appends metrics to a history table for longitudinal tracking.
4. Generates a simple visual report (PNG + HTML) for quick review.
The script is designed to be invoked from Airflow:
    python3 monitor_lstm_ncmapss.py --snapshotdate "2025-01-09"
"""

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from mlflow_config import init_mlflow, log_inference_metrics

    MLFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  MLflow not available. Monitoring will proceed without MLflow logging.")
    MLFLOW_AVAILABLE = False


def _load_baseline_config() -> Dict[str, Dict[str, float]]:
    """Load baseline metrics and thresholds for drift detection."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model_bank",
        "lstm_monitoring_baseline.json",
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Baseline monitoring config not found at {config_path}. "
            "Create the JSON file with baseline metrics and thresholds."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    required_sections = ["baseline", "thresholds"]
    missing_sections = [
        section for section in required_sections if section not in config
    ]
    if missing_sections:
        raise ValueError(
            f"Baseline config missing section(s): {missing_sections}. "
            "Expected keys: 'baseline', 'thresholds'."
        )
    return config


def _build_base_paths() -> Dict[str, str]:
    """Return all base directories used by the monitoring script."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        "predictions": os.path.join(base_dir, "../datamart/inference/lstm/n_cmapss"),
        "metrics": os.path.join(base_dir, "../datamart/monitoring/lstm/n_cmapss"),
        "reports": os.path.join(base_dir, "../reports/monitoring/lstm"),
    }


def _path_for_snapshot(base_path: str, snapshot_date: str, filename: str) -> str:
    """Construct a snapshot-specific path inside a partitioned directory."""
    return os.path.join(base_path, f"snapshot_date={snapshot_date}", filename)


def _files_exist(predictions_path: str) -> bool:
    """Check that required predictions parquet file exists."""
    if not os.path.exists(predictions_path):
        print("\n⚠️  Monitoring skipped - predictions not found:")
        print(f"   - {predictions_path}")
        print("Ensure LSTM inference step has completed.\n")
        return False
    return True


def _load_inputs(predictions_path: str) -> pd.DataFrame:
    """
    Load LSTM predictions from parquet file.
    Expected columns:
        - unit: Engine unit identifier
        - dataset: Dataset ID
        - time: Time cycle
        - RUL_Predicted: Predicted RUL value
        - RUL_Actual: Ground truth RUL value
        - Prediction_Error: RUL_Predicted - RUL_Actual
    """
    print(f"📂 Loading predictions from {predictions_path}")
    df = pd.read_parquet(predictions_path, engine="pyarrow")
    required_cols = ["unit", "time", "RUL_Predicted", "RUL_Actual", "Prediction_Error"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Predictions missing required columns: {missing_cols}")
    print(f"   Loaded {len(df):,} predictions from {df['unit'].nunique()} engines")
    return df


def _compute_metrics(predictions_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute RMSE and MAE from predictions.
    Args:
        predictions_df: DataFrame with RUL_Predicted, RUL_Actual, Prediction_Error
    Returns:
        Dictionary with rmse, mae, mean_pred, std_pred, mean_actual, std_actual
    """
    errors = predictions_df["Prediction_Error"].values
    actuals = predictions_df["RUL_Actual"].values
    preds = predictions_df["RUL_Predicted"].values
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    return {
        "rmse": rmse,
        "mae": mae,
        "mean_pred": float(np.mean(preds)),
        "std_pred": float(np.std(preds)),
        "mean_actual": float(np.mean(actuals)),
        "std_actual": float(np.std(actuals)),
        "n_predictions": len(predictions_df),
        "n_engines": predictions_df["unit"].nunique(),
    }


def _detect_drift(metrics: Dict[str, float], baseline_config: Dict) -> Dict[str, bool]:
    """
    Compare current metrics against baseline thresholds.
    Returns:
        Dictionary with drift flags for each metric
    """
    baseline = baseline_config["baseline"]
    thresholds = baseline_config["thresholds"]
    drift_flags = {}
    rmse_change_pct = abs(metrics["rmse"] - baseline["rmse"]) / baseline["rmse"] * 100
    drift_flags["rmse_drift"] = rmse_change_pct > thresholds["rmse_pct_change"]
    mae_change_pct = abs(metrics["mae"] - baseline["mae"]) / baseline["mae"] * 100
    drift_flags["mae_drift"] = mae_change_pct > thresholds["mae_pct_change"]
    mean_pred_change_pct = (
        abs(metrics["mean_pred"] - baseline.get("mean_pred", metrics["mean_pred"]))
        / baseline.get("mean_pred", 1.0)
        * 100
    )
    drift_flags["distribution_drift"] = mean_pred_change_pct > thresholds.get(
        "distribution_pct_change", 20.0
    )
    return drift_flags


def _append_metrics_to_history(
    metrics: Dict[str, float],
    drift_flags: Dict[str, bool],
    snapshot_date: str,
    metrics_dir: str,
) -> None:
    """Append current metrics to history table for longitudinal tracking."""
    os.makedirs(metrics_dir, exist_ok=True)
    history_path = os.path.join(metrics_dir, "metrics_history.parquet")
    # Create new row
    new_row = pd.DataFrame(
        [
            {
                "snapshot_date": snapshot_date,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **metrics,
                **drift_flags,
            }
        ]
    )
    # Append to history
    if os.path.exists(history_path):
        history_df = pd.read_parquet(history_path, engine="pyarrow")
        history_df = pd.concat([history_df, new_row], ignore_index=True)
    else:
        history_df = new_row
    history_df.to_parquet(history_path, index=False, engine="pyarrow")
    print(f"✓ Metrics appended to {history_path}")


def _generate_report(
    predictions_df: pd.DataFrame,
    metrics: Dict[str, float],
    drift_flags: Dict[str, bool],
    snapshot_date: str,
    reports_dir: str,
    baseline_config: Dict,
) -> Tuple[str, str]:
    """
    Generate visual monitoring report as PNG and HTML.
    Returns:
        Tuple of (png_path, html_path)
    """
    os.makedirs(reports_dir, exist_ok=True)
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"LSTM Model Monitoring Report - {snapshot_date}",
        fontsize=16,
        fontweight="bold",
    )
    # 1. Prediction Error Distribution
    ax = axes[0, 0]
    ax.hist(predictions_df["Prediction_Error"], bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
    ax.axvline(
        predictions_df["Prediction_Error"].mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean Error: {predictions_df['Prediction_Error'].mean():.4f}",
    )
    ax.set_xlabel("Prediction Error (cycles)")
    ax.set_ylabel("Frequency")
    ax.set_title("Prediction Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 2. Predicted vs Actual RUL
    ax = axes[0, 1]
    ax.scatter(
        predictions_df["RUL_Actual"], predictions_df["RUL_Predicted"], alpha=0.5, s=10
    )
    max_val = max(
        predictions_df["RUL_Actual"].max(), predictions_df["RUL_Predicted"].max()
    )
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Predicted vs Actual RUL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 3. Per-Engine RMSE
    ax = axes[1, 0]
    per_engine = (
        predictions_df.groupby("unit")
        .apply(lambda x: np.sqrt(np.mean(x["Prediction_Error"] ** 2)))
        .sort_values()
    )
    ax.barh(range(len(per_engine)), per_engine.values, alpha=0.7)
    ax.axvline(
        metrics["rmse"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall RMSE: {metrics['rmse']:.4f}",
    )
    ax.set_xlabel("RMSE (cycles)")
    ax.set_ylabel("Engine Index")
    ax.set_title("Per-Engine RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 4. Metrics Summary with Drift Indicators
    ax = axes[1, 1]
    ax.axis("off")
    # Build metrics text
    metrics_text = f"""
    📊 METRICS SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RMSE:              {metrics["rmse"]:.4f} cycles
    MAE:               {metrics["mae"]:.4f} cycles
    Mean Predicted:    {metrics["mean_pred"]:.4f} cycles
    Std Predicted:     {metrics["std_pred"]:.4f} cycles
    Mean Actual:       {metrics["mean_actual"]:.4f} cycles
    Std Actual:        {metrics["std_actual"]:.4f} cycles
    Total Predictions: {metrics["n_predictions"]:,}
    Engines:           {metrics["n_engines"]}
    🚨 DRIFT DETECTION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RMSE Drift:        {"⚠️ YES" if drift_flags.get("rmse_drift", False) else "✓ NO"}
    MAE Drift:         {"⚠️ YES" if drift_flags.get("mae_drift", False) else "✓ NO"}
    Distribution:      {"⚠️ YES" if drift_flags.get("distribution_drift", False) else "✓ NO"}
    """
    ax.text(
        0.1,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    plt.tight_layout()
    png_path = os.path.join(reports_dir, f"monitoring_{snapshot_date}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"✓ Report saved to {png_path}")
    plt.close()
    html_path = os.path.join(reports_dir, f"monitoring_{snapshot_date}.html")
    with open(png_path, "rb") as f:
        png_b64 = base64.b64encode(f.read()).decode("utf-8")
    drift_status = (
        "🚨 DRIFT DETECTED" if any(drift_flags.values()) else "✅ NO DRIFT DETECTED"
    )
    drift_color = "red" if any(drift_flags.values()) else "green"
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LSTM Monitoring Report - {snapshot_date}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color:
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color:
                border-bottom: 3px solid
                padding-bottom: 10px;
            }}
            .status {{
                font-size: 24px;
                font-weight: bold;
                color: {drift_color};
                margin: 20px 0;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 30px 0;
            }}
            .metric-card {{
                background:
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid
            }}
            .metric-label {{
                color:
                font-size: 14px;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color:
            }}
            img {{
                width: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .drift-section {{
                background:
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid
                margin: 20px 0;
            }}
            .timestamp {{
                color:
                font-size: 14px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 LSTM Model Monitoring Report</h1>
            <p><strong>Snapshot Date:</strong> {snapshot_date}</p>
            <p class="status">{drift_status}</p>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">RMSE (Root Mean Square Error)</div>
                    <div class="metric-value">{metrics["rmse"]:.4f} cycles</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">MAE (Mean Absolute Error)</div>
                    <div class="metric-value">{metrics["mae"]:.4f} cycles</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Predictions</div>
                    <div class="metric-value">{metrics["n_predictions"]:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Engines Processed</div>
                    <div class="metric-value">{metrics["n_engines"]}</div>
                </div>
            </div>
            <div class="drift-section">
                <h3>🚨 Drift Detection Results</h3>
                <ul>
                    <li><strong>RMSE Drift:</strong> {"⚠️ Detected" if drift_flags.get("rmse_drift", False) else "✓ Not detected"}</li>
                    <li><strong>MAE Drift:</strong> {"⚠️ Detected" if drift_flags.get("mae_drift", False) else "✓ Not detected"}</li>
                    <li><strong>Distribution Drift:</strong> {"⚠️ Detected" if drift_flags.get("distribution_drift", False) else "✓ Not detected"}</li>
                </ul>
            </div>
            <h2>📈 Visualization</h2>
            <img src="data:image/png;base64,{png_b64}" alt="Monitoring Charts">
            <p class="timestamp">Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
        </div>
    </body>
    </html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✓ HTML report saved to {html_path}")
    return png_path, html_path


def main(snapshotdate: str):
    """Main monitoring workflow for LSTM model."""
    print("\n\n--- Starting LSTM Monitoring N-CMAPSS job ---\n\n")
    print(f"Snapshot date: {snapshotdate}")
    try:
        baseline_config = _load_baseline_config()
        print(
            f"✓ Loaded baseline config (RMSE: {baseline_config['baseline']['rmse']:.4f}, MAE: {baseline_config['baseline']['mae']:.4f})"
        )
    except FileNotFoundError:
        print("⚠️  No baseline config found. Will proceed without drift detection.")
        baseline_config = {
            "baseline": {"rmse": 0.15, "mae": 0.05, "mean_pred": 45.0},
            "thresholds": {
                "rmse_pct_change": 20.0,
                "mae_pct_change": 20.0,
                "distribution_pct_change": 20.0,
            },
        }
    paths = _build_base_paths()
    predictions_path = _path_for_snapshot(
        paths["predictions"], snapshotdate, "predictions.parquet"
    )
    if not _files_exist(predictions_path):
        print("Exiting monitoring job.")
        return
    predictions_df = _load_inputs(predictions_path)
    print("\n📊 Computing metrics...")
    metrics = _compute_metrics(predictions_df)
    print(f"   RMSE: {metrics['rmse']:.4f} cycles")
    print(f"   MAE:  {metrics['mae']:.4f} cycles")
    print(f"   Predictions: {metrics['n_predictions']:,}")
    print(f"   Engines: {metrics['n_engines']}")
    drift_flags = _detect_drift(metrics, baseline_config)
    if any(drift_flags.values()):
        print("\n⚠️  DRIFT DETECTED:")
        for key, value in drift_flags.items():
            if value:
                print(f"   - {key}: YES")
    else:
        print("\n✅ No drift detected")
    _append_metrics_to_history(metrics, drift_flags, snapshotdate, paths["metrics"])
    print("\n📈 Generating monitoring report...")
    png_path, html_path = _generate_report(
        predictions_df,
        metrics,
        drift_flags,
        snapshotdate,
        paths["reports"],
        baseline_config,
    )
    if MLFLOW_AVAILABLE:
        try:
            print("\n📊 Logging metrics to MLflow...")
            init_mlflow()
            log_inference_metrics(
                snapshot_date=snapshotdate,
                metrics=metrics,
                drift_flags=drift_flags,
                model_name="lstm",
            )
            print("✓ Metrics logged to MLflow")
        except Exception as e:
            print(f"⚠️  MLflow logging failed: {e}")
    print("\n✅ Monitoring complete!")
    print("\nReports generated:")
    print(f"  - PNG: {png_path}")
    print(f"  - HTML: {html_path}")
    print("\n\n--- Completed LSTM Monitoring N-CMAPSS job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor LSTM model performance for N-CMAPSS"
    )
    parser.add_argument(
        "--snapshotdate", type=str, required=True, help="Snapshot date (YYYY-MM-DD)"
    )
    args = parser.parse_args()
    main(args.snapshotdate)
