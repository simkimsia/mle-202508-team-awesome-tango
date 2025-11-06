"""
XGBoost Pipeline - Stage 4: Monitoring
======================================
This DAG handles model performance monitoring and drift detection.
Generates HTML/PNG reports and logs metrics to history.
Schedule: Daily, triggered after inference completes
Metrics: RMSE, MAE, per-engine performance, drift detection
Outputs: HTML reports, PNG charts, metrics history
"""
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

from dag_config import DEFAULT_ARGS, SCHEDULE_INTERVAL, START_DATE, END_DATE, CATCHUP

default_args = {
    **DEFAULT_ARGS,
    "depends_on_past": True,  # Wait for inference to complete
}

with DAG(
    "xgboost_04_monitoring",
    default_args=default_args,
    description="XGBoost Pipeline Stage 4: Model performance monitoring",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=START_DATE,
    end_date=END_DATE,
    catchup=CATCHUP,
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