from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "dag",
    default_args=default_args,
    description="data pipeline run daily",
    schedule_interval="0 0 * * *",  # At 00:00 on everyday
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 2),
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
    gold_label_lstm_ncmapss = BashOperator(
        task_id="gold_label_lstm_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 gold_label_lstm_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

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
    gold_feature_lstm_ncmapss = BashOperator(
        task_id="gold_feature_lstm_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 gold_feature_lstm_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- 7. Gold Layer: XGBoost Feature Store ---
    gold_feature_xgboost_ncmapss = BashOperator(
        task_id="gold_feature_xgboost_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 gold_feature_xgboost_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- Task Dependencies ---
    # Linear: Bronze → Silver → Gold Label Base
    start_pipeline >> bronze_ncmapss >> silver_ncmapss >> gold_label_base_ncmapss

    # Parallel: Gold Label Base → [LSTM Label, XGBoost Label]
    gold_label_base_ncmapss >> [gold_label_lstm_ncmapss, gold_label_xgboost_ncmapss]

    # Sequential: LSTM Label → LSTM Features
    gold_label_lstm_ncmapss >> gold_feature_lstm_ncmapss

    # Sequential: XGBoost Label → XGBoost Features
    gold_label_xgboost_ncmapss >> gold_feature_xgboost_ncmapss

    # Converge: Both feature stores → End
    [gold_feature_lstm_ncmapss, gold_feature_xgboost_ncmapss] >> end_pipeline
