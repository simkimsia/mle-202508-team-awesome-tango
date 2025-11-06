"""
Gold Layer - LSTM Label Store: Convert base labels to TimeSeries format for LSTM
Usage:
    python3 gold_label_lstm_ncmapss.py --snapshotdate "2023-01-01"
"""
import os
import argparse
from utils import data_processing_gold_label_lstm
def main(snapshotdate):
    print("\n\n--- Starting Gold LSTM Label N-CMAPSS job ---\n\n")
    snapshot_date_str = snapshotdate
    gold_label_base_dir = f"../datamart/gold/label_base/n_cmapss"
    gold_label_lstm_dir = f"../datamart/gold/label/lstm/n_cmapss"
    print(f"Input directory: {gold_label_base_dir}")
    print(f"Output directory: {gold_label_lstm_dir}")
    data_processing_gold_label_lstm.process_gold_label_lstm(
        gold_label_base_dir,
        gold_label_lstm_dir,
        snapshot_date_str
    )
    print("\n\n--- Completed Gold LSTM Label N-CMAPSS job ---\n\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gold LSTM Label formatting for N-CMAPSS")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)