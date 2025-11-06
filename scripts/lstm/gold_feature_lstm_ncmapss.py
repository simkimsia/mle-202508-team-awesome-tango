import argparse
import os

from utils import data_processing_gold_feature_lstm

def main(snapshotdate):
    print(f"\n[GOLD-FEATURE-LSTM] ===== Starting Gold LSTM Feature Engineering Job =====")
    print(f"[GOLD-FEATURE-LSTM] Snapshot Date: {snapshotdate}")
    print(f"[GOLD-FEATURE-LSTM] Creating TimeSeries objects with 12 selected sensor features")
    snapshot_date_str = snapshotdate
    gold_label_base_dir = f"../datamart/gold/label_base/n_cmapss"
    gold_feature_lstm_dir = f"../datamart/gold/feature/lstm/n_cmapss"
    print(f"[GOLD-FEATURE-LSTM] Input directory: {gold_label_base_dir}")
    print(f"[GOLD-FEATURE-LSTM] Output directory: {gold_feature_lstm_dir}")
    print(f"[GOLD-FEATURE-LSTM] Processing LSTM feature TimeSeries...")
    data_processing_gold_feature_lstm.process_gold_feature_lstm(
        gold_label_base_dir,
        gold_feature_lstm_dir,
        snapshot_date_str
    )
    print(f"[GOLD-FEATURE-LSTM] ===== Completed Gold LSTM Feature Engineering Job =====\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Gold layer feature engineering for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()
    main(args.snapshotdate)