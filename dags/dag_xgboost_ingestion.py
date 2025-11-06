"""
XGBoost Pipeline - Stage 1: Data Ingestion
=========================================
This DAG handles raw data loading from H5 files into the Bronze layer.
Runs monthly to ingest new snapshot data.
Schedule: Monthly on the 1st at 00:00
Datasets: N-CMAPSS (NASA Commercial Modular Aero-Propulsion System Simulation)
"""
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
    "xgboost_01_ingestion",
    default_args=default_args,
    description="XGBoost Pipeline Stage 1: Raw data ingestion to Bronze layer",
    schedule_interval="0 0 * * *",  # daily
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 10),  # cut off at 10th Jan
    catchup=True,
    tags=["xgboost", "ingestion", "bronze"],
) as dag:
    start = DummyOperator(task_id="start_ingestion")
    bronze_ncmapss = BashOperator(
        task_id="bronze_ncmapss",
        bash_command=(
            "cd /opt/airflow/scripts/xgboost && "
            "python3 bronze_ncmapss.py "
            '--snapshotdate "{{ ds }}"'
        ),
    )
    end = DummyOperator(task_id="end_ingestion")
    start >> bronze_ncmapss >> end