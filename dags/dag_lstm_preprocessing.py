from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
with DAG(
    "lstm_02_preprocessing",
    default_args=default_args,
    description="LSTM Pipeline Stage 2: Data cleaning and TimeSeries creation",
    schedule_interval="0 1 1 * *",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 10, 31),
    catchup=True,
    tags=["lstm", "preprocessing", "silver", "gold"],
) as dag:
    start = DummyOperator(task_id="start_preprocessing")
    print(f"[STAGE 2] Starting LSTM preprocessing pipeline for date {{{{ ds }}}}")
    silver_ncmapss = BashOperator(
        task_id="silver_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/lstm && "
            "echo '[SILVER] Processing silver layer for LSTM' && "
            "python3 silver_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    gold_label_base = BashOperator(
        task_id="gold_label_base_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/lstm && "
            "echo '[GOLD-LABEL-BASE] Creating base labels with RUL clipping' && "
            "python3 gold_label_base_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    gold_label_lstm = BashOperator(
        task_id="gold_label_lstm_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/lstm && "
            "echo '[GOLD-LABEL-LSTM] Converting labels to TimeSeries format' && "
            "python3 gold_label_lstm_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    gold_feature_lstm = BashOperator(
        task_id="gold_feature_lstm_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/lstm && "
            "echo '[GOLD-FEATURE-LSTM] Creating LSTM feature TimeSeries with 12 selected features' && "
            "python3 gold_feature_lstm_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    end = DummyOperator(task_id="end_preprocessing")
    print(f"[STAGE 2] Completed LSTM preprocessing pipeline for date {{{{ ds }}}}")
    start >> silver_ncmapss >> gold_label_base >> gold_label_lstm >> gold_feature_lstm >> end