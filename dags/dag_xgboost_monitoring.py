"""
XGBoost Pipeline - Stage 4: Monitoring
======================================
This DAG handles model performance monitoring and drift detection.
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
    "xgboost_04_monitoring",
    default_args=default_args,
    description="XGBoost Pipeline Stage 4: Model performance monitoring",
    schedule_interval="0 0 * * *",  # daily
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 10),  # cut off at 10th Jan
    catchup=True,
    tags=["xgboost", "monitoring"],
) as dag:
    start = DummyOperator(task_id="start_monitoring")
    monitor_xgboost = BashOperator(
        task_id="monitor_xgboost_metrics",
        bash_command=(
            "cd /opt/airflow/scripts/xgboost && "
            "python3 monitor_xgboost_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    end = DummyOperator(task_id="end_monitoring")
    start >> monitor_xgboost >> end