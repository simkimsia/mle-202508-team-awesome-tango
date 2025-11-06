"""
LSTM Pipeline - Stage 4: Monitoring
===================================
This DAG handles LSTM model performance monitoring and drift detection.
Generates HTML/PNG reports and logs metrics to history.
Schedule: Monthly, triggered after inference completes
Metrics: RMSE, MAE, per-engine performance, drift detection
Outputs: HTML reports, PNG charts, metrics history
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
default_args = {
    "owner": "airflow",
    "depends_on_past": True,  # Wait for inference to complete
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
with DAG(
    "lstm_04_monitoring",
    default_args=default_args,
    description="LSTM Pipeline Stage 4: Model performance monitoring",
    schedule_interval="0 3 1 * *",  # Monthly at 03:00 (after inference)
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 10, 31),
    catchup=True,
    tags=["lstm", "monitoring"],
) as dag:
    start = DummyOperator(task_id="start_monitoring")
    monitor_lstm = BashOperator(
        task_id="monitor_lstm_metrics",
        bash_command=(
            "cd /opt/airflow/scripts/lstm && "
            "python3 monitor_lstm_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    end = DummyOperator(task_id="end_monitoring")
    start >> monitor_lstm >> end