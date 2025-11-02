import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import ShortCircuitOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def should_run_inference(ds, **kwargs):
    """
    Check if inference should run based on:
    1. Date >= 2025-01-08
    2. Model artifact exists
    3. Feature store has data for this snapshot date

    Returns:
        bool: True if inference should run, False to skip
    """
    import glob
    from datetime import datetime

    # Parse the execution date (ds is in YYYY-MM-DD format)
    execution_date = datetime.strptime(ds, "%Y-%m-%d")
    cutoff_date = datetime(2025, 1, 8)

    print(f"\n{'=' * 60}")
    print("Checking inference prerequisites...")
    print(f"{'=' * 60}")
    print(f"Execution date: {execution_date.strftime('%Y-%m-%d')}")
    print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")

    # Check 1: Date must be >= 2025-01-08
    if execution_date < cutoff_date:
        print(
            f"❌ Date check FAILED: {execution_date.strftime('%Y-%m-%d')} < {cutoff_date.strftime('%Y-%m-%d')}"
        )
        print(
            f"   Skipping inference for dates before {cutoff_date.strftime('%Y-%m-%d')}"
        )
        return False
    else:
        print(
            f"✓ Date check PASSED: {execution_date.strftime('%Y-%m-%d')} >= {cutoff_date.strftime('%Y-%m-%d')}"
        )

    # Check 2: Model artifact exists
    model_path = "/opt/airflow/scripts/model_bank/xgboost_rul_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model check FAILED: Model not found at {model_path}")
        print("   Skipping inference - model artifact missing")
        return False
    else:
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(
            f"✓ Model check PASSED: Found model at {model_path} ({model_size_mb:.2f} MB)"
        )

    # Check 3: Feature store has data for this snapshot date
    feature_dir = (
        f"/opt/airflow/scripts/datamart/gold/feature/xgboost/n_cmapss/snapshot_date={ds}"
    )

    if not os.path.exists(feature_dir):
        print(f"❌ Feature check FAILED: Feature directory not found at {feature_dir}")
        print("   Skipping inference - features not generated yet")
        return False

    # Check for either partitioned data or single file
    partition_dirs = glob.glob(os.path.join(feature_dir, "dataset=*"))
    single_file = os.path.join(feature_dir, "features.parquet")

    if partition_dirs:
        # Check that at least one partition has data
        has_data = False
        for partition_dir in partition_dirs:
            partition_file = os.path.join(partition_dir, "features.parquet")
            if os.path.exists(partition_file):
                has_data = True
                break

        if has_data:
            print(
                f"✓ Feature check PASSED: Found {len(partition_dirs)} dataset partitions"
            )
        else:
            print("❌ Feature check FAILED: No feature files found in partitions")
            return False
    elif os.path.exists(single_file):
        file_size_mb = os.path.getsize(single_file) / (1024 * 1024)
        print(
            f"✓ Feature check PASSED: Found features at {single_file} ({file_size_mb:.2f} MB)"
        )
    else:
        print(f"❌ Feature check FAILED: No feature files found in {feature_dir}")
        return False

    print(f"\n{'=' * 60}")
    print("✅ All checks PASSED - Proceeding with inference")
    print(f"{'=' * 60}\n")
    return True


with DAG(
    "dag",
    default_args=default_args,
    description="data pipeline run daily",
    schedule_interval="0 0 * * *",  # At 00:00 on everyday
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 8),
    catchup=True,
) as dag:
    # --- 0. Start / End Markers ---
    start_pipeline = DummyOperator(task_id="start_pipeline")
    end_pipeline = DummyOperator(task_id="end_pipeline")

    # --- 1. Bronze Layer: Raw Data Loading ---
    bronze_ncmapss = BashOperator(
        task_id="bronze_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 bronze_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- 2. Silver Layer: Cleaned & Standardized Data ---
    silver_ncmapss = BashOperator(
        task_id="silver_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 silver_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- 3. Gold Layer: Base Label Store ---
    gold_label_base_ncmapss = BashOperator(
        task_id="gold_label_base_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 gold_label_base_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- 4. Gold Layer: LSTM Label Store ---
    # TODO: Waiting for teammate to provide LSTM scaler
    # gold_label_lstm_ncmapss = BashOperator(
    #     task_id="gold_label_lstm_ncmapss",
    #     bash_command=(
    #         "cd /opt/airflow/scripts && "
    #         "python3 gold_label_lstm_ncmapss.py "
    #         '--snapshotdate "{{ ds }}"'
    #     ),
    # )

    # --- 5. Gold Layer: XGBoost Label Store ---
    gold_label_xgboost_ncmapss = BashOperator(
        task_id="gold_label_xgboost_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 gold_label_xgboost_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- 6. Gold Layer: LSTM Feature Store ---
    # gold_feature_lstm_ncmapss = BashOperator(
    #     task_id="gold_feature_lstm_ncmapss",
    #     bash_command=(
    #         "cd /opt/airflow/scripts && "
    #         "python3 gold_feature_lstm_ncmapss.py "
    #         '--snapshotdate "{{ ds }}"'
    #     ),
    # )

    # --- 7. Gold Layer: XGBoost Feature Store ---
    gold_feature_xgboost_ncmapss = BashOperator(
        task_id="gold_feature_xgboost_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 gold_feature_xgboost_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- 8. Inference Gate: Check if inference should run ---
    check_inference_prerequisites = ShortCircuitOperator(
        task_id="check_inference_prerequisites",
        python_callable=should_run_inference,
        provide_context=True,
    )

    # --- 9. Inference: XGBoost RUL Predictions ---
    inference_xgboost_ncmapss = BashOperator(
        task_id="inference_xgboost_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 inference_xgboost_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- Task Dependencies ---
    # Linear: Bronze → Silver → Gold Label Base
    start_pipeline >> bronze_ncmapss >> silver_ncmapss >> gold_label_base_ncmapss

    # Parallel: Gold Label Base → [LSTM Label, XGBoost Label]
    # gold_label_base_ncmapss >> [gold_label_lstm_ncmapss, gold_label_xgboost_ncmapss]
    gold_label_base_ncmapss >> [gold_label_xgboost_ncmapss]

    # Sequential: LSTM Label → LSTM Features
    # gold_label_lstm_ncmapss >> gold_feature_lstm_ncmapss

    # Sequential: XGBoost Label → XGBoost Features → Check → XGBoost Inference
    (
        gold_label_xgboost_ncmapss
        >> gold_feature_xgboost_ncmapss
        >> check_inference_prerequisites
        >> inference_xgboost_ncmapss
    )

    # Converge: Both feature stores + inference → End
    # [gold_feature_lstm_ncmapss, inference_xgboost_ncmapss] >> end_pipeline
    [inference_xgboost_ncmapss] >> end_pipeline
