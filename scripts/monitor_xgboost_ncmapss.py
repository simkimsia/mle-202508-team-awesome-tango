#!/usr/bin/env python3
"""
Model monitoring for XGBoost RUL predictions on the N-CMAPSS dataset.

For a given snapshot date this script:
1. Loads inference predictions and gold labels.
2. Computes RMSE and MAE, plus basic run statistics.
3. Appends metrics to a history table for longitudinal tracking.
4. Generates a simple visual report (PNG + HTML) for quick review.

The script is designed to be invoked from Airflow:

    python3 monitor_xgboost_ncmapss.py --snapshotdate "2025-01-09"
"""

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_baseline_config() -> Dict[str, Dict[str, float]]:
    """Load baseline metrics and thresholds for drift detection."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model_bank",
        "xgboost_monitoring_baseline.json",
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Baseline monitoring config not found at {config_path}. "
            "Create the JSON file with baseline metrics and thresholds."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_sections = ["baseline", "thresholds"]
    missing_sections = [section for section in required_sections if section not in config]
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
        "predictions": os.path.join(
            base_dir, "datamart/inference/xgboost/n_cmapss"
        ),
        "labels": os.path.join(base_dir, "datamart/gold/label/xgboost/n_cmapss"),
        "metrics": os.path.join(base_dir, "datamart/monitoring/xgboost/n_cmapss"),
        "reports": os.path.join(base_dir, "docs/monitoring/xgboost"),
    }


def _path_for_snapshot(base_path: str, snapshot_date: str, filename: str) -> str:
    """Construct a snapshot-specific path inside a partitioned directory."""
    return os.path.join(base_path, f"snapshot_date={snapshot_date}", filename)


def _files_exist(predictions_path: str, labels_path: str) -> bool:
    """Check that both required parquet files exist."""
    missing = []
    if not os.path.exists(predictions_path):
        missing.append(predictions_path)
    if not os.path.exists(labels_path):
        missing.append(labels_path)

    if missing:
        print("\n⚠️  Monitoring skipped - required data missing:")
        for path in missing:
            print(f"   - {path}")
        print("Ensure inference and label generation steps have completed.\n")
        return False
    return True


def _load_inputs(
    predictions_path: str, labels_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions and labels from parquet files."""
    print(f"Loading predictions from {predictions_path}")
    df_pred = pd.read_parquet(predictions_path, engine="pyarrow")
    print(f"  ✓ Loaded {len(df_pred):,} prediction rows")

    print(f"Loading labels from {labels_path}")
    df_label = pd.read_parquet(labels_path, engine="pyarrow")
    print(f"  ✓ Loaded {len(df_label):,} label rows")

    return df_pred, df_label


def _merge_predictions_labels(
    df_pred: pd.DataFrame, df_label: pd.DataFrame
) -> pd.DataFrame:
    """Merge predictions with labels to obtain actual RUL values."""
    merge_cols = ["unit", "time"]
    if "dataset" in df_pred.columns and "dataset" in df_label.columns:
        merge_cols.append("dataset")

    # Drop any previously attached ground-truth columns; monitoring pulls labels afresh.
    df_pred = df_pred.drop(columns=[col for col in ["RUL_Actual", "Prediction_Error"] if col in df_pred.columns])

    missing_pred_cols = [col for col in merge_cols if col not in df_pred.columns]
    if missing_pred_cols:
        raise ValueError(
            f"Predictions parquet missing merge columns {missing_pred_cols}. "
            f"Available columns: {list(df_pred.columns)}"
        )

    missing_label_cols = [col for col in merge_cols if col not in df_label.columns]
    if missing_label_cols:
        raise ValueError(
            f"Labels parquet missing merge columns {missing_label_cols}. "
            f"Available columns: {list(df_label.columns)}"
        )

    if "RUL_Clipped" in df_label.columns:
        df_label = df_label.rename(columns={"RUL_Clipped": "RUL_Actual"})
    elif "RUL_Actual" not in df_label.columns:
        raise KeyError(
            "Labels parquet must contain either 'RUL_Clipped' or 'RUL_Actual'. "
            f"Available columns: {list(df_label.columns)}"
        )

    # Enforce uniqueness on the merge keys to satisfy a 1:1 join.
    pred_before = len(df_pred)
    df_pred = df_pred.drop_duplicates(subset=merge_cols, keep="first")
    pred_dropped = pred_before - len(df_pred)
    if pred_dropped > 0:
        print(f"  • Dropped {pred_dropped:,} duplicate prediction rows based on {merge_cols}")

    label_before = len(df_label)
    df_label = df_label.drop_duplicates(subset=merge_cols, keep="first")
    label_dropped = label_before - len(df_label)
    if label_dropped > 0:
        print(f"  • Dropped {label_dropped:,} duplicate label rows based on {merge_cols}")

    merged = df_pred.merge(
        df_label[merge_cols + ["RUL_Actual"]],
        on=merge_cols,
        how="inner",
        validate="1:1",
    )

    rows_with_actuals = merged["RUL_Actual"].notna().sum()
    if rows_with_actuals == 0:
        raise ValueError(
            "Merged predictions contain no rows with actual RUL values. "
            "Confirm that labels include 'RUL_Clipped' for this snapshot."
        )

    merged["RUL_Actual"] = merged["RUL_Actual"].astype("float32")
    return merged


def _calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute monitoring metrics from the merged dataframe."""
    if "RUL_Actual" not in df.columns or "RUL_Predicted" not in df.columns:
        raise ValueError(
            "Merged dataframe must contain 'RUL_Predicted' and 'RUL_Actual' columns."
        )

    residuals = df["RUL_Predicted"] - df["RUL_Actual"]
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "prediction_count": int(len(df)),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std(ddof=0)),
    }

    print("\n📊 Monitoring metrics")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  Rows: {metrics['prediction_count']:,}")

    return metrics


def _evaluate_drift(
    metrics: Dict[str, float], baseline_config: Dict[str, Dict[str, float]]
) -> Dict[str, str]:
    """Assess drift status by comparing against baseline thresholds."""
    baseline_rmse = baseline_config["baseline"]["rmse"]
    baseline_mae = baseline_config["baseline"]["mae"]
    thresh = baseline_config["thresholds"]

    rmse = metrics["rmse"]
    mae = metrics["mae"]

    rmse_warning = baseline_rmse * thresh.get("rmse_warning_multiplier", 1.15)
    rmse_critical = baseline_rmse * thresh.get("rmse_critical_multiplier", 1.30)
    mae_warning = baseline_mae * thresh.get("mae_warning_multiplier", 1.20)
    mae_critical = baseline_mae * thresh.get("mae_critical_multiplier", 1.40)

    status = "ok"
    reason = "Within baseline thresholds."

    if rmse > rmse_critical:
        status = "critical"
        reason = (
            f"RMSE {rmse:.2f} exceeds critical threshold ({rmse_critical:.2f})."
        )
    elif rmse > rmse_warning:
        status = "warning"
        reason = f"RMSE {rmse:.2f} exceeds warning threshold ({rmse_warning:.2f})."
    elif mae > mae_critical:
        status = "critical"
        reason = (
            f"MAE {mae:.2f} exceeds critical threshold ({mae_critical:.2f})."
        )
    elif mae > mae_warning:
        status = "warning"
        reason = f"MAE {mae:.2f} exceeds warning threshold ({mae_warning:.2f})."

    print(f"\n🚦 Drift status: {status.upper()} – {reason}")

    return {
        "status": status,
        "reason": reason,
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
        "rmse_warning_threshold": rmse_warning,
        "rmse_critical_threshold": rmse_critical,
        "mae_warning_threshold": mae_warning,
        "mae_critical_threshold": mae_critical,
    }


def _update_metrics_history(
    metrics_base: str,
    snapshot_date: str,
    metrics: Dict[str, float],
    drift_info: Dict[str, str],
) -> pd.DataFrame:
    """Append snapshot metrics to the history parquet file."""
    os.makedirs(metrics_base, exist_ok=True)
    history_path = os.path.join(metrics_base, "metrics_history.parquet")

    record = {
        "snapshot_date": snapshot_date,
        **metrics,
        "drift_status": drift_info["status"],
        "drift_reason": drift_info["reason"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if os.path.exists(history_path):
        history = pd.read_parquet(history_path, engine="pyarrow")
        history = history[history["snapshot_date"] != snapshot_date]
        history = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
    else:
        history = pd.DataFrame([record])

    history.sort_values("snapshot_date", inplace=True)
    history.to_parquet(history_path, index=False, engine="pyarrow", compression="snappy")
    print(f"\n✅ Metrics history updated at {history_path}")
    return history


def _persist_snapshot_artifacts(
    metrics_base: str,
    snapshot_date: str,
    metrics: Dict[str, float],
    merged_df: pd.DataFrame,
    drift_info: Dict[str, str],
) -> None:
    """Persist snapshot-specific metrics and residuals for auditing."""
    snapshot_dir = os.path.join(metrics_base, f"snapshot_date={snapshot_date}")
    os.makedirs(snapshot_dir, exist_ok=True)

    metrics_path = os.path.join(snapshot_dir, "metrics.json")
    metrics_payload = {**metrics, "drift": drift_info}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    residuals_path = os.path.join(snapshot_dir, "residuals.parquet")
    residuals_df = merged_df[
        ["unit", "time", "RUL_Predicted", "RUL_Actual"]
    ].copy()
    residuals_df["residual"] = (
        residuals_df["RUL_Predicted"] - residuals_df["RUL_Actual"]
    ).astype("float32")
    residuals_df.to_parquet(
        residuals_path, index=False, engine="pyarrow", compression="snappy"
    )

    print(f"  • Snapshot metrics saved to {metrics_path}")
    print(f"  • Residuals saved to {residuals_path}")


def _generate_report(
    reports_base: str,
    snapshot_date: str,
    history: pd.DataFrame,
    merged_df: pd.DataFrame,
    drift_info: Dict[str, str],
) -> None:
    """Create a PNG plot and HTML report summarising metrics."""
    os.makedirs(reports_base, exist_ok=True)
    png_path = os.path.join(reports_base, f"monitoring_{snapshot_date}.png")
    html_path = os.path.join(reports_base, f"monitoring_{snapshot_date}.html")

    # Prepare figure with two panels: metric history and residual histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if len(history) > 0:
        history_sorted = history.sort_values("snapshot_date")
        axes[0].plot(
            history_sorted["snapshot_date"],
            history_sorted["rmse"],
            marker="o",
            label="RMSE",
        )
        axes[0].plot(
            history_sorted["snapshot_date"],
            history_sorted["mae"],
            marker="s",
            label="MAE",
        )
        axes[0].set_title("Error Metrics Over Time")
        axes[0].set_xlabel("Snapshot Date")
        axes[0].set_ylabel("Cycles (RUL)")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].legend()
    else:
        axes[0].text(
            0.5,
            0.5,
            "No history available yet",
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[0].set_axis_off()

    residuals = merged_df["RUL_Predicted"] - merged_df["RUL_Actual"]
    axes[1].hist(residuals, bins=30, color="#4472C4", edgecolor="black")
    axes[1].set_title(f"Residuals (Predicted - Actual)\n{snapshot_date}")
    axes[1].set_xlabel("Residual (cycles)")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"  • Monitoring figure saved to {png_path}")

    # Embed the PNG into an HTML report
    with open(png_path, "rb") as f_img:
        encoded_img = base64.b64encode(f_img.read()).decode("utf-8")

    status_color = {
        "ok": "#2E7D32",
        "warning": "#F9A825",
        "critical": "#C62828",
    }.get(drift_info["status"], "#424242")

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>XGBoost Monitoring Report – {snapshot_date}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1 {{ margin-bottom: 0.5rem; }}
    p {{ margin-top: 0.5rem; }}
    img {{ max-width: 100%; height: auto; }}
    .banner {{
      padding: 12px 16px;
      border-radius: 6px;
      color: #fff;
      margin-bottom: 1rem;
      font-weight: 600;
    }}
    .metadata {{ font-size: 0.9rem; color: #555; }}
  </style>
</head>
<body>
  <h1>XGBoost Monitoring Report</h1>
  <div class="banner" style="background-color: {status_color};">
    Drift status: {drift_info["status"].upper()} – {drift_info["reason"]}
  </div>
  <div class="metadata">
    <p><strong>Snapshot date:</strong> {snapshot_date}</p>
    <p><strong>Generated:</strong> {datetime.now(timezone.utc).isoformat()}</p>
  </div>
  <img src="data:image/png;base64,{encoded_img}" alt="Monitoring chart">
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f_html:
        f_html.write(html_content)

    print(f"  • HTML report saved to {html_path}")


def main(snapshotdate: str) -> None:
    print("\n\n--- Starting XGBoost monitoring job ---\n")
    paths = _build_base_paths()
    baseline_config = _load_baseline_config()

    predictions_path = _path_for_snapshot(
        paths["predictions"], snapshotdate, "predictions.parquet"
    )
    labels_path = _path_for_snapshot(paths["labels"], snapshotdate, "labels.parquet")

    if not _files_exist(predictions_path, labels_path):
        return

    df_pred, df_label = _load_inputs(predictions_path, labels_path)
    merged_df = _merge_predictions_labels(df_pred, df_label)

    metrics = _calculate_metrics(merged_df)
    drift_info = _evaluate_drift(metrics, baseline_config)
    history = _update_metrics_history(paths["metrics"], snapshotdate, metrics, drift_info)
    _persist_snapshot_artifacts(paths["metrics"], snapshotdate, metrics, merged_df, drift_info)
    _generate_report(paths["reports"], snapshotdate, history, merged_df, drift_info)

    print("\n--- Completed XGBoost monitoring job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor XGBoost RUL predictions for the N-CMAPSS dataset"
    )
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
