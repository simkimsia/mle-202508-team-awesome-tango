"""
LSTM Pipeline - Stage 4: Monitoring
===================================
This DAG handles LSTM model performance monitoring and drift detection.
Generates HTML/PNG reports and logs metrics to history.
Schedule: Daily, triggered after inference completes
Metrics: RMSE, MAE, per-engine performance, drift detection
Outputs: HTML reports, PNG charts, metrics history
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from dag_config import CATCHUP, DEFAULT_ARGS, END_DATE, SCHEDULE_INTERVAL, START_DATE

default_args = {
    **DEFAULT_ARGS,
    "depends_on_past": True,  # Wait for inference to complete
}

with DAG(
    "lstm_04_monitoring",
    default_args=default_args,
    description="LSTM Pipeline Stage 4: Model performance monitoring",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=START_DATE,
    end_date=END_DATE,
    catchup=CATCHUP,
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
